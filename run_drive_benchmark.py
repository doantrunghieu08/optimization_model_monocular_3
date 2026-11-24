"""Run Harmony4D brute-force configs whose inputs are Google Drive file IDs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

import numpy as np
import yaml

from brute_force_runner import run_brute_force
from config_loader import load_config
from fusion_pipeline.aotr_benchmark import evaluation_protocol, report_csv_name
from preprocessing_data.label_persons import run as label_persons


ROOT = Path(__file__).resolve().parent
DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{25,}")


def is_drive_segment(segment):
    # video is optional: absent/commented -> not downloaded (only the frame size is ever used)
    return DRIVE_ID.fullmatch(segment["ground_truth_dir"]) and all(
        DRIVE_ID.fullmatch(str(camera.get("pkl", ""))) and (not camera.get("video") or DRIVE_ID.fullmatch(str(camera["video"])))
        for camera in segment["cameras"]
    )


def download(file_id, path):
    import gdown
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        return
    params = {"id": file_id, "export": "download", "confirm": "t"}
    for url in ("https://drive.usercontent.google.com/download", "https://drive.google.com/uc"):
        with requests.get(url, params=params, headers={"Accept-Encoding": "identity"},
                          stream=True, timeout=(15, 60)) as response:
            if response.status_code != 200 or "attachment" not in response.headers.get("Content-Disposition", "").lower():
                continue
            partial = path.with_suffix(path.suffix + ".http-part")
            with partial.open("wb") as target:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    target.write(chunk)
            size = partial.stat().st_size
            content_length = response.headers.get("Content-Length")
            if content_length and size != int(content_length):
                raise RuntimeError(f"Incomplete download for {path.name}: {size}/{content_length} bytes")
            if size:
                partial.replace(path)
                break
    if not path.is_file():
        gdown.download(id=file_id, output=str(path), use_cookies=False, quiet=True, resume=True)
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"Cannot download {file_id} to {path.name}")
    print(f"[Drive] {path.name} downloaded ({path.stat().st_size:,} bytes)", flush=True)


def extract_poses(archive_path, poses_dir):
    poses_dir.mkdir(parents=True, exist_ok=True)
    persons = set()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            if not re.fullmatch(r"\d+\.npy", name):
                continue
            target = poses_dir / name
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            persons.update(str(person).lower() for person in np.load(target, allow_pickle=True).item())
    return sorted(persons)


def prepare_segment_inputs(segment, dataset_dir, *, local_only=False):
    name = segment["name"].removesuffix(f"_{segment['target_person']}")
    archive = dataset_dir / "poses3d.zip"
    cameras = []
    downloads = []
    for camera in segment["cameras"]:
        cam_dir = dataset_dir / "exo" / camera["id"]
        stem = f"harmony4D_{name}_{camera['id']}"
        pkl = cam_dir / f"{stem}_wham" / f"wham_{stem}_ffmpeg.pkl"
        video = cam_dir / "images" / f"{stem}_ffmpeg.mp4"
        downloads.append((camera["pkl"], pkl))
        if camera.get("video"):
            downloads.append((camera["video"], video))
        cameras.append({"id": camera["id"], "pkl": str(pkl), "video": str(video) if camera.get("video") else None})

    if local_only:
        required = [archive, *(path for _, path in downloads)]
        missing = [path.name for path in required if not path.is_file() or not path.stat().st_size]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} of {len(required)} local inputs: {', '.join(missing)}")
    else:
        download(segment["ground_truth_dir"], archive)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(download, file_id, path) for file_id, path in downloads]
            for future in futures:
                future.result()
    return cameras


def stage_segment(segment, dataset_dir, *, local_only=False):
    cameras = prepare_segment_inputs(segment, dataset_dir, local_only=local_only)
    poses_dir = dataset_dir / "processed_data" / "poses3d"
    persons = extract_poses(dataset_dir / "poses3d.zip", poses_dir)
    if segment["target_person"] not in persons:
        raise ValueError(f"Ground truth for {segment['name']} has no {segment['target_person']}")
    # The Drive configs have no track mapping. Use GT only to assign track IDs.
    label_persons(dataset_dir, persons=persons)
    return {**segment, "ground_truth_dir": str(poses_dir), "cameras": cameras, "_persons": persons}


def segment_protocol(config, segment):
    person = segment.get("target_person")
    if not person:
        return evaluation_protocol(config)
    return evaluation_protocol({**config, "inputs": {**config.get("inputs", {}), "target_person": person}})


def _sequence_name(segment_name):
    return re.sub(r"_aria\d+$", "", segment_name)


def _expand_person_segments(seq_name, seq_segments, persons):
    base = seq_segments[0]
    by_person = {s.get("target_person"): s for s in seq_segments}
    expanded = list(seq_segments)
    for person in persons:
        if person not in by_person:
            expanded.append({**base, "name": f"{seq_name}_{person}", "target_person": person})
    return expanded


def _cached_persons(dataset_dir, seq_segments):
    report = dataset_dir / "person_labels_report.json"
    if report.is_file():
        try:
            persons = json.loads(report.read_text(encoding="utf-8")).get("persons") or []
            if persons:
                return persons
        except Exception:
            pass
    return [s["target_person"] for s in seq_segments if s.get("target_person")]


def report_complete(report, segment, protocol):
    if not report.is_file():
        return False
    with report.open(encoding="utf-8-sig", newline="") as source:
        pairs = {(row["Cam Master"], row["Cam Slave"]) for row in csv.DictReader(source)
                 if row.get("Segment") == segment["name"] and row.get("Protocol") == protocol}
    cameras = [camera["id"] for camera in segment["cameras"]]
    return all((master, slave) in pairs for master in cameras for slave in cameras if master != slave)


def run_segments(base_config, segments, report_dir, *, workers=6):
    config = {**base_config, "paths": {
        key: str(report_dir / "work" / key) if key.endswith("_output_dir") else value
        for key, value in base_config["paths"].items()
    }}
    return run_brute_force(
        config, {"segments": segments}, local_report_dir=report_dir, workers=workers
    )


def run_segment(base_config, segment, report_dir, *, workers=6):
    return run_segments(base_config, [segment], report_dir, workers=workers)[segment["name"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, help="Default: all Drive brute-force configs in configs/")
    parser.add_argument("--workers", type=int, default=6, help="Parallel pair workers (default: 6)")
    parser.add_argument("--reverse", action="store_true", help="Run configs and cameras from bottom to top")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--download-only", action="store_true", help="Cache inputs without launching the benchmark")
    modes.add_argument("--local-only", action="store_true", help="Evaluate using cached inputs only")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    segments = [segment for path in args.configs or sorted((ROOT / "configs").glob("*.yml"))
                for segment in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("segments", [])
                if is_drive_segment(segment)]
    if args.reverse:
        segments = [{**s, "cameras": list(reversed(s["cameras"]))} for s in reversed(segments)]
    if not segments:
        parser.error("No segments with Drive file IDs found")

    for segment in segments:
        name = segment["name"]
        if not re.fullmatch(r"\d{3}_[a-z0-9]+_aria\d+", name) or any(
            not re.fullmatch(r"cam\d+", camera["id"]) for camera in segment["cameras"]
        ):
            raise ValueError(f"Invalid Harmony4D segment or camera name in {name!r}")

    grouped = {}
    for segment in segments:
        grouped.setdefault(_sequence_name(segment["name"]), []).append(segment)

    if not args.download_only:
        config = load_config(ROOT / "configs" / "pipeline.yml")
        # These per-pair overrides were selected on Hugging.
        config["fusion"].setdefault("master_protection", {})["pair_modes"] = {}
    reports = ROOT / "output" / "drive_benchmark"
    datasets = ROOT / "input" / "drive_benchmark"
    downloaded = 0
    final_checks = []
    for seq_name, seq_segments in grouped.items():
        primary = seq_segments[0]
        dataset_dir = datasets / seq_name
        if not dataset_dir.exists() and (datasets / f"{seq_name}_aria01").exists():
            dataset_dir = datasets / f"{seq_name}_aria01"
        report_dir = reports / seq_name
        if not report_dir.exists() and (reports / f"{seq_name}_aria01").exists():
            report_dir = reports / f"{seq_name}_aria01"
        if not args.download_only:
            known_persons = _cached_persons(dataset_dir, seq_segments)
            known_segments = _expand_person_segments(seq_name, seq_segments, known_persons)
            if (dataset_dir / "person_labels_report.json").is_file() and all(
                report_complete(report_dir / report_csv_name(config), seg, segment_protocol(config, seg))
                for seg in known_segments
            ):
                print(f"[Drive] {seq_name} ({', '.join(known_persons)}): already complete", flush=True)
                final_checks.extend((report_dir, seg) for seg in known_segments)
                continue
        print(f"[Drive] Starting {seq_name} | cache={dataset_dir}", flush=True)
        try:
            if args.download_only:
                cameras = prepare_segment_inputs(primary, dataset_dir)
                downloaded += 1
                print(f"[Drive] {seq_name}: {1 + 2 * len(cameras)} input files ready", flush=True)
                continue
            staged = stage_segment(primary, dataset_dir, local_only=args.local_only)
            persons = staged.pop("_persons", None) or [s.get("target_person") for s in seq_segments]
            local_segments = _expand_person_segments(
                seq_name,
                [{**staged, "name": s["name"], "target_person": s["target_person"]} for s in seq_segments],
                persons,
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dataset_dir / "person_labels_report.json", report_dir)
            results = run_segments(config, local_segments, report_dir, workers=args.workers)
            cameras = len(primary["cameras"])
            for seg in local_segments:
                print(f"[Drive] {seg['name']}: {len(results.get(seg['name'], []))}/{cameras * (cameras - 1)} pairs reported", flush=True)
            final_checks.extend((report_dir, seg) for seg in local_segments)
        except Exception as exc:
            print(f"[Drive] {seq_name}: {exc}", flush=True)
            final_checks.extend((report_dir, seg) for seg in seq_segments)
    if args.download_only:
        print(f"[Drive] Inputs ready for {downloaded}/{len(grouped)} datasets", flush=True)
        return int(downloaded != len(grouped))
    return 0 if all(report_complete(rdir / report_csv_name(config), seg, segment_protocol(config, seg))
                    for rdir, seg in final_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Check a Harmony4D-style sequence before any experiment is run on it.

Reports, per sequence and per camera, what the pipeline needs and what is broken or
missing, then prints the exact commands still to run (labelling, identity mask,
brute-force config). It never modifies data and never runs the fusion.

Checks
  layout       exo/camXX/ with a WHAM pkl and a video (mp4 or images/*.jpg); duplicated
               "images (1)" folders
  mapping      *_person_mapping.txt present; which persons each camera covers
  2D input     tracking_results*.pth (pipeline input; without it AOTR uses its 3D surrogate)
  GT leakage   processed_data/poses2d and bbox are derived from 3D GT -> must never be inputs
  GT           processed_data/poses3d: numeric files, download duplicates such as
               "00059(1).npy", frame gaps, persons per file, keypoint shape
  calibration  colmap/workplace cameras.txt (intrinsics) and images.txt (extrinsics)
  sync         WHAM frame ids shared between cameras and with the GT
  identity     identity mask / brute-force config already present in configs/
  alignment    (--gt-align, reads GT, QA only) PA-MPJPE of the mapped track vs GT at frame
               lag 0 and at the best lag in [-L, L]; a best lag != 0 means WHAM frames are not
               aligned with the GT (the MPI problem)

Usage
  python -m preprocessing_data.check_dataset input/043_Grappling2 [--gt-align] [--json out.json]
  python -m preprocessing_data.check_dataset input/new_harmony4d_root --all   # every sequence below
Exit code 1 if any ERROR was found.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GT_SHAPE = (17, 4)  # Harmony4D COCO-17 keypoints: x, y, z, confidence


class Report:
    def __init__(self, name):
        self.name, self.items, self.facts = name, [], {}

    def add(self, level, area, msg):
        self.items.append({"level": level, "area": area, "msg": msg})

    def ok(self, area, msg):
        self.add("OK", area, msg)

    def warn(self, area, msg):
        self.add("WARN", area, msg)

    def error(self, area, msg):
        self.add("ERROR", area, msg)

    @property
    def n_errors(self):
        return sum(i["level"] == "ERROR" for i in self.items)

    def to_dict(self):
        return {"sequence": self.name, "items": self.items, "facts": self.facts}


# ----------------------------------------------------------------------------- discovery
def find_sequences(path):
    """A sequence is a directory with exo/ (and usually processed_data/)."""
    path = Path(path)
    if (path / "exo").is_dir():
        return [path]
    return sorted(p.parent for p in path.glob("**/exo") if p.is_dir())


def discover_cameras(seq_dir):
    cams = []
    for cam_dir in sorted(p for p in (seq_dir / "exo").glob("cam*") if p.is_dir()):
        pkls = sorted(cam_dir.glob("**/wham_*.pkl"))
        mp4s = sorted((cam_dir / "images").glob("*.mp4")) if (cam_dir / "images").is_dir() else []
        jpgs = sorted((cam_dir / "images").glob("*.jpg")) if (cam_dir / "images").is_dir() else []
        dup_dirs = sorted(p.name for p in cam_dir.glob("images (*)") if p.is_dir())
        if not mp4s:
            mp4s = sorted(cam_dir.glob("images*/*.mp4"))
        cams.append({"id": cam_dir.name, "dir": cam_dir, "pkls": pkls, "mp4": mp4s, "jpg_count": len(jpgs),
                     "dup_image_dirs": dup_dirs,
                     "tracking": sorted(pkls[0].parent.glob("tracking_results*.pth")) if pkls else []})
    return cams


# ----------------------------------------------------------------------------- GT
def gt_files(seq_dir):
    d = seq_dir / "processed_data" / "poses3d"
    files = sorted(d.glob("*.npy")) if d.is_dir() else []
    numeric = [p for p in files if p.stem.isdigit()]
    return d, numeric, [p.name for p in files if not p.stem.isdigit()]


def frame_gaps(fids):
    fids = sorted(fids)
    return [(a, b) for a, b in zip(fids, fids[1:]) if b - a > 1]


def check_gt(seq_dir, rep, persons):
    d, numeric, dups = gt_files(seq_dir)
    if not d.is_dir():
        rep.error("gt", f"missing {d.relative_to(seq_dir)}")
        return {}
    if not numeric:
        rep.error("gt", "no numeric NNNNN.npy files in poses3d")
        return {}
    if dups:
        rep.warn("gt", f"{len(dups)} non-numeric file(s) ignored by the loader, e.g. {dups[:3]}")
    fids = [int(p.stem) - 1 for p in numeric]  # loader convention: 00001.npy -> frame id 0
    gaps = frame_gaps(fids)
    per_person = {p: set() for p in persons}
    bad_shape, unreadable, low_conf = [], [], 0
    other_persons = set()
    for p, fid in zip(numeric, fids):
        try:
            item = np.load(p, allow_pickle=True).item()
        except Exception:
            unreadable.append(p.name)
            continue
        for key, kp in item.items():
            k = str(key).lower()
            arr = np.asarray(kp)
            if k not in per_person:
                other_persons.add(k)
                continue
            if arr.shape != EXPECTED_GT_SHAPE:
                bad_shape.append((p.name, k, arr.shape))
                continue
            per_person[k].add(fid)
            if arr.shape[1] >= 4 and np.mean(arr[:, 3] > 0) < 0.5:
                low_conf += 1
    rep.facts["gt"] = {"files": len(numeric), "frame_range": [min(fids), max(fids)], "gaps": gaps[:20],
                       "frames_per_person": {k: len(v) for k, v in per_person.items()}}
    rep.ok("gt", f"{len(numeric)} frames, ids {min(fids)}..{max(fids)}, "
                 f"per person {rep.facts['gt']['frames_per_person']}")
    if gaps:
        rep.warn("gt", f"{len(gaps)} gap(s) in GT frame ids, e.g. {gaps[:3]}")
    if unreadable:
        rep.error("gt", f"{len(unreadable)} unreadable npy, e.g. {unreadable[:3]}")
    if bad_shape:
        rep.error("gt", f"{len(bad_shape)} keypoint arrays with shape != {EXPECTED_GT_SHAPE}, e.g. {bad_shape[:2]}")
    if other_persons:
        rep.warn("gt", f"GT has persons not in --persons: {sorted(other_persons)}")
    for k, v in per_person.items():
        if not v:
            rep.error("gt", f"no GT for {k}")
    if low_conf:
        rep.warn("gt", f"{low_conf} person-frames with < 50% joints of positive confidence")
    return per_person


# ----------------------------------------------------------------------------- cameras
def _load_tracks(pkl):
    from compat import load_joblib_compat
    return load_joblib_compat(pkl)


def check_cameras(seq_dir, rep, persons, gt_frames):
    from compat import parse_person_mapping
    cams = discover_cameras(seq_dir)
    if not cams:
        rep.error("layout", "no exo/cam*/ directories")
        return []
    rep.facts["cameras"] = {}
    usable = []
    for c in cams:
        info = {}
        if not c["pkls"]:
            rep.warn("layout", f"{c['id']}: no WHAM pkl (wham_*.pkl) -> camera unusable until WHAM is run")
            rep.facts["cameras"][c["id"]] = info
            continue
        if len(c["pkls"]) > 1:
            rep.warn("layout", f"{c['id']}: {len(c['pkls'])} WHAM pkls, using {c['pkls'][0].name}")
        pkl = c["pkls"][0]
        info["pkl"] = str(pkl)
        if not c["mp4"] and not c["jpg_count"]:
            rep.warn("layout", f"{c['id']}: no video (images/*.mp4 or images/*.jpg); only needed for visualization")
        if c["dup_image_dirs"]:
            rep.warn("layout", f"{c['id']}: duplicated image folders {c['dup_image_dirs']} (check which one is real)")
        mapping = parse_person_mapping(pkl)
        info["mapping"] = mapping
        if not mapping:
            rep.warn("mapping", f"{c['id']}: no *_person_mapping.txt -> run label_persons (see next steps)")
        try:
            raw = _load_tracks(pkl)
        except Exception as exc:
            rep.error("layout", f"{c['id']}: cannot load {pkl.name}: {type(exc).__name__}: {exc}")
            continue
        tracks = {str(k): v for k, v in raw.items() if isinstance(v, dict) and "frame_ids" in v}
        info["tracks"] = len(tracks)
        frames = {}
        for p in persons:
            tids = [t for t, who in mapping.items() if who == p and t in tracks]
            frames[p] = sorted({int(f) for t in tids for f in np.asarray(tracks[t]["frame_ids"]).astype(int)})
            if mapping and not tids:
                rep.warn("mapping", f"{c['id']}: no track mapped to {p}")
            if len(tids) > 1:
                info.setdefault("fragmented", {})[p] = tids
        unknown = [t for t in mapping if t not in tracks]
        if unknown:
            rep.error("mapping", f"{c['id']}: mapping names tracks not in the pkl: {unknown}")
        info["frames"] = {p: len(v) for p, v in frames.items()}
        info["gt_overlap"] = {p: len(set(frames[p]) & gt_frames.get(p, set())) for p in persons}
        info["all_frame_ids"] = sorted({int(f) for t in tracks.values() for f in np.asarray(t["frame_ids"]).astype(int)})
        info["tracking_results"] = [x.name for x in c["tracking"]]
        for p in persons:
            if mapping and frames[p] and info["gt_overlap"][p] == 0:
                rep.error("sync", f"{c['id']}: {p} track frames never overlap GT frame ids")
        rep.facts["cameras"][c["id"]] = {k: v for k, v in info.items() if k != "all_frame_ids"}
        usable.append({**c, "pkl": pkl, "mapping": mapping, "frame_ids": set(info["all_frame_ids"])})
    no2d = [c["id"] for c in usable if not c["tracking"]]
    if no2d:
        rep.warn("2d", f"{len(no2d)}/{len(usable)} cameras without tracking_results*.pth -> AOTR uses its 3D "
                       f"surrogate there: {no2d}")
    n_map = sum(bool(c["mapping"]) for c in usable)
    msg =f"{len(cams)} cameras, {len(usable)} with WHAM, {n_map} with person mapping"
    if n_map < 2:
        rep.error("layout", msg + " -> fewer than 2 usable cameras, no camera pair can run")
    else:
        rep.ok("layout", msg)
    return usable


def check_sync(rep, cams):
    if len(cams) < 2:
        return
    no_overlap = [(a["id"], b["id"]) for i, a in enumerate(cams) for b in cams[i + 1:]
                  if not (a["frame_ids"] & b["frame_ids"])]
    if no_overlap:
        rep.warn("sync", f"{len(no_overlap)} camera pair(s) share no WHAM frame id (skipped by the runner), "
                         f"e.g. {no_overlap[:3]}")
    else:
        rep.ok("sync", "every camera pair shares WHAM frame ids")


def check_gt_leak(seq_dir, rep):
    for sub in ("poses2d", "bbox"):
        d = seq_dir / "processed_data" / sub
        if d.is_dir():
            rep.warn("gt-leak", f"processed_data/{sub} exists: it is derived from the 3D GT -> never use it "
                                "as a pipeline input (2D confidence, boxes, person labels)")


def check_calibration(seq_dir, rep):
    w = seq_dir / "colmap" / "workplace"
    cams_txt, imgs_txt = w / "cameras.txt", w / "images.txt"
    calib = {"intrinsics": cams_txt.is_file(), "extrinsics": imgs_txt.is_file()}
    if cams_txt.is_file():
        n = sum(1 for line in cams_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip() and not line.startswith("#"))
        calib["n_intrinsics"] = n
    rep.facts["calibration"] = calib
    if calib["extrinsics"]:
        rep.ok("calib", "colmap images.txt present: real extrinsics -> world-frame MPJPE and a calibrated "
                        "oracle (triangulation) are possible")
    elif calib["intrinsics"]:
        rep.warn("calib", "only colmap cameras.txt (intrinsics); no images.txt -> no extrinsics, MPJPE stays "
                          "sequence-aligned")
    else:
        rep.warn("calib", "no colmap calibration found")


def check_configs(seq_dir, rep, cams):
    stems = {c["pkl"].stem for c in cams}
    masks = []
    for f in sorted((ROOT / "configs").glob("identity_mask_*.json")):
        try:
            margins = json.loads(f.read_text(encoding="utf-8")).get("margins", {})
        except Exception:
            continue
        cover = stems & set(margins)
        if cover:
            masks.append((f.name, len(cover)))
    brutes = []
    rel = str(seq_dir.resolve().relative_to(ROOT)).replace("\\", "/") if seq_dir.resolve().is_relative_to(ROOT) else None
    for f in sorted((ROOT / "configs").glob("brute_force*.yml")):
        if rel and rel in f.read_text(encoding="utf-8", errors="ignore"):
            brutes.append(f.name)
    rep.facts["configs"] = {"identity_masks": masks, "brute_configs": brutes}
    if masks:
        rep.ok("identity", f"identity mask covers cameras: {masks}")
    else:
        rep.warn("identity", "no identity mask covers this sequence -> build_identity_mask (see next steps)")
    if brutes:
        rep.ok("identity", f"brute-force config: {brutes}")
    else:
        rep.warn("identity", "no configs/brute_force*.yml references this sequence")
    return masks, brutes


# ----------------------------------------------------------------------------- alignment (reads GT)
def check_alignment(seq_dir, rep, cams, persons, max_lag=10, n_samples=40):
    from compat import resolve_person_payload
    from evaluation_pipeline.executor import _compute_pa_mpjpe, _load_ground_truth, _parse_new_gt
    from keypoints_map import load_keypoints3d_map

    kmap = load_keypoints3d_map(str(ROOT / "configs/keypoints3D_map.yml"))
    names = {k["name"] for k in kmap["keypoints"]}
    reg_idx = {k["name"]: k["regressor_index"] for k in kmap["keypoints"]}
    keys = [k for k in kmap["priority1"] if k != "neck"]
    J = np.load(ROOT / "models/J_regressor_body25_extended.npy")
    truth_dir = seq_dir / "processed_data" / "poses3d"
    gt = {p: {f: _parse_new_gt(it, names, kmap)
              for f, it in _load_ground_truth(truth_dir / "_.json", truth_dir=truth_dir, target_person=p).items()}
          for p in persons}
    out = {}
    for c in cams:
        if not c["mapping"]:
            continue
        raw = _load_tracks(c["pkl"])
        for p in persons:
            track = resolve_person_payload(raw, pkl_path=c["pkl"], target_person=p)
            if track is None:
                continue
            fids = np.asarray(track["frame_ids"]).astype(int)
            rows = [r for r, f in enumerate(fids) if f in gt[p]]
            if len(rows) < 5:
                continue
            rows = [rows[i] for i in np.linspace(0, len(rows) - 1, min(n_samples, len(rows))).astype(int)]
            preds = {r: (lambda jt: {n: jt[i] for n, i in reg_idx.items()})(J @ np.asarray(track["verts"][r]))
                     for r in rows}
            med = {}
            for lag in range(-max_lag, max_lag + 1):
                v = [_compute_pa_mpjpe(preds[r], gt[p][fids[r] + lag], keys)[0]
                     for r in rows if fids[r] + lag in gt[p]]
                if len(v) >= max(5, len(rows) // 2):
                    med[lag] = float(np.median(v))
            if 0 not in med:
                continue
            best = min(med, key=med.get)
            out[f"{c['id']}/{p}"] = {"lag0_pa_median": round(med[0], 1), "best_lag": best,
                                     "best_pa_median": round(med[best], 1)}
            if best != 0 and med[0] - med[best] > 5.0:
                rep.error("align", f"{c['id']}/{p}: WHAM frames look shifted vs GT: PA {med[0]:.0f} mm at lag 0, "
                                   f"{med[best]:.0f} mm at lag {best:+d}")
            elif med[0] > 120.0:
                rep.warn("align", f"{c['id']}/{p}: median PA {med[0]:.0f} mm at lag 0 (wrong person / bad track?)")
    rep.facts["alignment"] = out
    if out and not any(i["area"] == "align" and i["level"] == "ERROR" for i in rep.items):
        rep.ok("align", f"no frame shift found on {len(out)} camera/person tracks (lag window ±{max_lag})")


# ----------------------------------------------------------------------------- main
def next_steps(seq_dir, rep, cams, masks, brutes):
    rel = seq_dir.resolve()
    rel = str(rel.relative_to(ROOT) if rel.is_relative_to(ROOT) else rel).replace("\\", "/")
    slug = re.sub(r"[^0-9a-z]+", "_", seq_dir.name.lower()).strip("_")
    brute = f"configs/brute_force_{slug}.yml"
    steps = []
    if any(not c["mapping"] for c in cams) or not brutes:
        steps.append(f"python -m preprocessing_data.label_persons {rel} --brute-config {brute}")
    if not masks:
        steps.append(f"python build_identity_mask.py {brutes and 'configs/' + brutes[0] or brute} "
                     f"configs/identity_mask_{slug}.json")
    steps.append(f"python run_method_comparison.py run --brute {brutes and 'configs/' + brutes[0] or brute} "
                 f"--mask {masks and 'configs/' + masks[0][0] or f'configs/identity_mask_{slug}.json'} "
                 f"--out output/method_comparison/{slug} --worker 0 --n-workers 1")
    return steps


def check_sequence(seq_dir, persons, gt_align=False, max_lag=10):
    seq_dir = Path(seq_dir)
    rep = Report(seq_dir.name)
    gt_frames = check_gt(seq_dir, rep, persons)
    cams = check_cameras(seq_dir, rep, persons, gt_frames)
    check_sync(rep, cams)
    check_gt_leak(seq_dir, rep)
    check_calibration(seq_dir, rep)
    masks, brutes = check_configs(seq_dir, rep, cams)
    if gt_align and gt_frames:
        check_alignment(seq_dir, rep, cams, persons, max_lag=max_lag)
    rep.facts["next_steps"] = next_steps(seq_dir, rep, cams, masks, brutes)
    return rep


def print_report(rep):
    print(f"\n=== {rep.name} ===")
    for level in ("ERROR", "WARN", "OK"):
        for i in rep.items:
            if i["level"] == level:
                print(f"  [{level:5}] {i['area']:8} {i['msg']}")
    cams = rep.facts.get("cameras", {})
    if cams:
        print("  camera  tracks  frames/person  GT-overlap/person  2D")
        for cid, info in cams.items():
            print(f"  {cid:6}  {info.get('tracks', '-'):>6}  {str(info.get('frames', '-')):<28} "
                  f"{str(info.get('gt_overlap', '-')):<28} {'yes' if info.get('tracking_results') else 'no'}")
    print("  next steps:")
    for s in rep.facts.get("next_steps", []):
        print(f"    {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="a sequence dir (with exo/) or, with --all, a root containing sequences")
    ap.add_argument("--all", action="store_true", help="check every sequence below path")
    ap.add_argument("--persons", nargs="+", default=["aria01", "aria02"])
    ap.add_argument("--gt-align", action="store_true", help="also test WHAM<->GT frame alignment (reads GT)")
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--json", help="write the full report as JSON")
    a = ap.parse_args()
    seqs = find_sequences(a.path) if a.all else [Path(a.path)]
    if not seqs:
        raise SystemExit(f"no sequence (dir with exo/) under {a.path}")
    reports = [check_sequence(s, [p.lower() for p in a.persons], a.gt_align, a.max_lag) for s in seqs]
    for r in reports:
        print_report(r)
    total = sum(r.n_errors for r in reports)
    print(f"\n{len(reports)} sequence(s), {total} error(s), "
          f"{sum(i['level'] == 'WARN' for r in reports for i in r.items)} warning(s)")
    if a.json:
        Path(a.json).write_text(json.dumps([r.to_dict() for r in reports], indent=1, default=str), encoding="utf-8")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()

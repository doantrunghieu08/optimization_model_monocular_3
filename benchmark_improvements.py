"""Benchmark baseline hard replacement against guarded fusion variants.

Examples:
    python benchmark_improvements.py --pairs 28 --seg seg_4
    python benchmark_improvements.py --pairs 28 --seg seg_4 --direction reverse
    python benchmark_improvements.py --mode i2_i7 --pairs 10
"""

import argparse
import copy
import csv
import json
import math
import os
import statistics
import sys
import traceback
from datetime import datetime
from itertools import combinations
from pathlib import Path


WORKSPACE_DIR = Path(__file__).parent.resolve()
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from config_loader import absolutize_config_paths, load_config, set_env_from_filename
from evaluation_pipeline.executor import run_evaluation
from fusion_pipeline.executor import run_fusion
from pose_pipeline.executor import run_pose_export
from preprocess_pipeline.executor import run_preprocess


SEGMENTS_BASE = WORKSPACE_DIR / "input" / "Seq1" / "imageSequence" / "Segments"
VIDEO_BASE = WORKSPACE_DIR / "input" / "Seq1" / "imageSequence"
GT_DIR = str(WORKSPACE_DIR / "input" / "Seq1" / "GT")
CONFIG_PATH = WORKSPACE_DIR / "configs" / "pipeline.yml"
OUTPUT_DIR = WORKSPACE_DIR / "benchmark_results"
CAMERA_IDS = ("cam0", "cam1", "cam2", "cam4", "cam5", "cam6", "cam7", "cam8")

VARIANTS = {
    "confidence": {"enabled": True, "confidence_delta_cap": 0.05, "blend_mode": "confidence", "alignment_mode": "sequence_root", "selector": "occlusion"},
    "body_pose": {"enabled": False, "shared_body_pose_enabled": True},
    "i2_only": {"enabled": True, "confidence_delta_cap": 0.10, "blend_mode": "hard"},
    "i7_only": {"enabled": True, "confidence_delta_cap": 0.05, "blend_mode": "confidence"},
    "i2_i7": {"enabled": True, "confidence_delta_cap": 0.10, "blend_mode": "confidence"},
}
BASELINE = {"enabled": True, "confidence_delta_cap": 0.05, "blend_mode": "hard", "alignment_mode": "frame", "selector": "confidence"}


def build_pairs(direction: str) -> list[tuple[str, str]]:
    forward = list(combinations(CAMERA_IDS, 2))
    if direction == "forward":
        return forward
    reverse = [(right, left) for left, right in forward]
    return reverse if direction == "reverse" else forward + reverse


def build_pair_config(seg: str, cam_master: str, cam_slave: str, base_cfg: dict):
    master_id = cam_master.removeprefix("cam")
    slave_id = cam_slave.removeprefix("cam")
    master_pkl = SEGMENTS_BASE / f"video_{master_id}_{seg}" / f"video_{master_id}_{seg}.pkl"
    slave_pkl = SEGMENTS_BASE / f"video_{slave_id}_{seg}" / f"video_{slave_id}_{seg}.pkl"
    missing = [str(path) for path in (master_pkl, slave_pkl) if not path.exists()]
    if missing:
        return None, f"Missing input: {', '.join(missing)}"

    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("inputs", {}).update({
        "cam1_pkl": str(master_pkl),
        "cam2_pkl": str(slave_pkl),
        "camera1_video": str(VIDEO_BASE / f"video_{master_id}.avi"),
        "camera2_video": str(VIDEO_BASE / f"video_{slave_id}.avi"),
        "ground_truth_dir": GT_DIR,
    })
    cfg.setdefault("runtime", {}).update({"clean_output": True, "stage": "evaluation"})
    cfg.setdefault("visualization", {})["enabled"] = False
    cfg.setdefault("learnable", {})["enabled"] = False
    cfg.setdefault("learnable_extra", {})["enabled"] = False
    return absolutize_config_paths(cfg, WORKSPACE_DIR), None


def with_variant(config: dict, variant: dict) -> dict:
    cfg = copy.deepcopy(config)
    variant = dict(variant)
    cfg.setdefault("fusion", {})["shared_body_pose_enabled"] = variant.pop("shared_body_pose_enabled", False)
    cfg["fusion"].setdefault("correction", {}).update(variant)
    return cfg


def parse_metric(eval_dir: Path, metric: str, camera: str, prefix: str) -> float:
    path = eval_dir / f"{metric}_{camera}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    column = f"{prefix}_priority1_mm"
    average = next((row for row in rows if row.get("Frame") == "AVERAGE"), None)
    if average is None or column not in average:
        raise ValueError(f"Missing {column} AVERAGE in {path}")
    return float(average[column])


def read_metrics(config: dict) -> dict:
    eval_dir = Path(config["paths"]["evaluation_output_dir"])
    result = {}
    for camera in ("cam1", "cam2"):
        result[camera] = {}
        for metric in ("MPJPE", "PA-MPJPE"):
            result[camera][metric] = {
                "posed": parse_metric(eval_dir, metric, camera, "posed"),
                "fused": parse_metric(eval_dir, metric, camera, "fused"),
            }
    return result


def read_diagnostics(config: dict) -> dict:
    metadata_dir = Path(config["paths"]["fused_output_dir"]) / "metadata"
    totals = {
        "cam1": {"changed": 0, "weighted_displacement": 0.0},
        "cam2": {"changed": 0, "weighted_displacement": 0.0},
        "K1": 0,
        "K2": 0,
        "frames": 0,
    }
    for path in metadata_dir.glob("fused_data_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        totals["frames"] += 1
        totals["K1"] += len(data.get("K1", []))
        totals["K2"] += len(data.get("K2", []))
        for output_camera, metadata_camera in (("cam1", "camera1"), ("cam2", "camera2")):
            change = data.get("changes", {}).get(metadata_camera, {})
            count = int(change.get("changed_joint_count", 0))
            totals[output_camera]["changed"] += count
            totals[output_camera]["weighted_displacement"] += count * float(change.get("mean_displacement_mm", 0.0))
    for camera in ("cam1", "cam2"):
        count = totals[camera]["changed"]
        totals[camera]["mean_displacement_mm"] = totals[camera].pop("weighted_displacement") / count if count else 0.0
    return totals


def run_variant(config: dict, label: str) -> dict:
    correction = config["fusion"]["correction"]
    print(f"    [{label}] cap={correction['confidence_delta_cap']} blend={correction['blend_mode']}")
    run_fusion(config)
    run_evaluation(config)
    return {"metrics": read_metrics(config), "diagnostics": read_diagnostics(config)}


def run_pair(seg: str, master: str, slave: str, base_cfg: dict, mode: str) -> list[dict]:
    pair = f"{master}-{slave}"
    config, error = build_pair_config(seg, master, slave, base_cfg)
    if error:
        print(f"  [SKIP] {pair}: {error}")
        return []

    print(f"  [PREPARE] {pair}")
    try:
        config.setdefault("pose_export", {})["shared_body_pose"] = mode == "body_pose"
        run_preprocess(config, extract_frames=False)
        run_pose_export(config)
        baseline = run_variant(with_variant(config, BASELINE), "baseline")
        improved = run_variant(with_variant(config, VARIANTS[mode]), mode)
    except Exception as exc:
        traceback.print_exc()
        print(f"  [FAILED] {pair}: {exc}")
        return []

    rows = []
    physical_camera = {"cam1": master, "cam2": slave}
    for camera in ("cam1", "cam2"):
        base_metrics = baseline["metrics"][camera]
        new_metrics = improved["metrics"][camera]
        base_diag = baseline["diagnostics"][camera]
        new_diag = improved["diagnostics"][camera]
        row = {
            "pair": pair,
            "role": camera,
            "camera": physical_camera[camera],
            "seg": seg,
            "mode": mode,
            "posed_mpjpe": base_metrics["MPJPE"]["posed"],
            "posed_pa": base_metrics["PA-MPJPE"]["posed"],
            "baseline_mpjpe": base_metrics["MPJPE"]["fused"],
            "baseline_pa": base_metrics["PA-MPJPE"]["fused"],
            "improved_mpjpe": new_metrics["MPJPE"]["fused"],
            "improved_pa": new_metrics["PA-MPJPE"]["fused"],
            "baseline_changed": base_diag["changed"],
            "improved_changed": new_diag["changed"],
            "baseline_displacement_mm": base_diag["mean_displacement_mm"],
            "improved_displacement_mm": new_diag["mean_displacement_mm"],
            "improved_k1": improved["diagnostics"]["K1"],
            "improved_k2": improved["diagnostics"]["K2"],
            "frames": improved["diagnostics"]["frames"],
        }
        row["delta_mpjpe_mm"] = row["baseline_mpjpe"] - row["improved_mpjpe"]
        row["delta_pa_mm"] = row["baseline_pa"] - row["improved_pa"]
        row["delta_mpjpe_pct"] = 100.0 * row["delta_mpjpe_mm"] / row["baseline_mpjpe"]
        row["delta_pa_pct"] = 100.0 * row["delta_pa_mm"] / row["baseline_pa"]
        row["delta_vs_posed_mpjpe_mm"] = row["posed_mpjpe"] - row["improved_mpjpe"]
        row["delta_vs_posed_pa_mm"] = row["posed_pa"] - row["improved_pa"]
        rows.append(row)
        print(
            f"    {camera}/{physical_camera[camera]}: "
            f"MPJPE {row['baseline_mpjpe']:.2f}->{row['improved_mpjpe']:.2f} "
            f"({row['delta_mpjpe_mm']:+.2f} mm), "
            f"PA {row['baseline_pa']:.2f}->{row['improved_pa']:.2f} "
            f"({row['delta_pa_mm']:+.2f} mm), changed={row['improved_changed']}"
        )
    return rows


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def write_results(rows: list[dict], csv_path: Path, summary_path: Path, args) -> None:
    if not rows:
        summary_path.write_text("No valid results.\n", encoding="utf-8")
        return

    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    posed_mp_delta, posed_mp_ci = mean_ci95([row["delta_vs_posed_mpjpe_mm"] for row in rows])
    posed_pa_delta, posed_pa_ci = mean_ci95([row["delta_vs_posed_pa_mm"] for row in rows])
    legacy_mp_delta, legacy_mp_ci = mean_ci95([row["delta_mpjpe_mm"] for row in rows])
    legacy_pa_delta, legacy_pa_ci = mean_ci95([row["delta_pa_mm"] for row in rows])
    avg = lambda key: statistics.fmean(row[key] for row in rows)
    summary = [
        "=" * 72,
        f"BENCHMARK SUMMARY - {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Mode: {args.mode} | Direction: {args.direction} | Segment: {args.seg}",
        f"Pair-camera samples: {len(rows)} ({len(rows) // 2} pairs)",
        "=" * 72,
        "",
        "Primary comparison (raw posed -> improved fusion):",
        f"MPJPE     : {avg('posed_mpjpe'):.2f} -> {avg('improved_mpjpe'):.2f} mm "
        f"(delta {posed_mp_delta:+.2f} +/- {posed_mp_ci:.2f} mm, descriptive 95% CI)",
        f"PA-MPJPE  : {avg('posed_pa'):.2f} -> {avg('improved_pa'):.2f} mm "
        f"(delta {posed_pa_delta:+.2f} +/- {posed_pa_ci:.2f} mm, descriptive 95% CI)",
        "",
        "Legacy comparison (hard replacement -> improved fusion):",
        f"MPJPE     : {avg('baseline_mpjpe'):.2f} -> {avg('improved_mpjpe'):.2f} mm "
        f"(delta {legacy_mp_delta:+.2f} +/- {legacy_mp_ci:.2f} mm, descriptive 95% CI)",
        f"PA-MPJPE  : {avg('baseline_pa'):.2f} -> {avg('improved_pa'):.2f} mm "
        f"(delta {legacy_pa_delta:+.2f} +/- {legacy_pa_ci:.2f} mm, descriptive 95% CI)",
        f"Changed joint-frames: {avg('baseline_changed'):.1f} -> {avg('improved_changed'):.1f}",
        f"Mean displacement   : {avg('baseline_displacement_mm'):.2f} -> {avg('improved_displacement_mm'):.2f} mm",
        "",
        "Per pair and evaluated camera:",
    ]
    for row in rows:
        summary.append(
            f"  {row['pair']:12s} {row['role']}/{row['camera']:4s} "
            f"MPJPE {row['posed_mpjpe']:7.2f}->{row['improved_mpjpe']:7.2f} ({row['delta_vs_posed_mpjpe_mm']:+6.2f}), "
            f"legacy delta={row['delta_mpjpe_mm']:+6.2f}; "
            f"PA {row['posed_pa']:7.2f}->{row['improved_pa']:7.2f} ({row['delta_vs_posed_pa_mm']:+6.2f}), "
            f"legacy delta={row['delta_pa_mm']:+6.2f}; changed={row['improved_changed']}"
        )
    summary.extend(("", f"CSV: {csv_path}"))
    text = "\n".join(summary) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    print("\n" + text)


def main():
    parser = argparse.ArgumentParser(description="Benchmark isolated fusion variants on both cameras")
    parser.add_argument("--pairs", type=int, default=28, help="Maximum directed camera pairs")
    parser.add_argument("--seg", default="seg_4")
    parser.add_argument("--config", default=None, help="Ablation filename used to load alpha/beta/method")
    parser.add_argument("--mode", choices=sorted(VARIANTS), default="confidence")
    parser.add_argument("--direction", choices=("forward", "reverse", "both"), default="forward")
    args = parser.parse_args()

    if args.config:
        os.environ["NOTEBOOK_NAME"] = args.config
        set_env_from_filename(args.config)
    base_cfg = load_config(CONFIG_PATH)
    pairs = build_pairs(args.direction)[: max(0, args.pairs)]
    print(f"[Benchmark] mode={args.mode}, segment={args.seg}, pairs={len(pairs)}")

    rows = []
    for master, slave in pairs:
        rows.extend(run_pair(args.seg, master, slave, base_cfg, args.mode))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"benchmark_{timestamp}_{args.mode}_{args.direction}"
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    summary_path = OUTPUT_DIR / f"{stem}_summary.txt"
    write_results(rows, csv_path, summary_path, args)
    print(f"[Benchmark] Done: {csv_path}")


if __name__ == "__main__":
    main()

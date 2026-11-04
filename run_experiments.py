"""Ablation runner for AOTR experiments:
A - Baseline AOTR: Consensus weight 0.12, Anatomy ON
B - No Advantage: Consensus weight 0.0, Anatomy ON
C - No Anatomy: Consensus weight 0.12, Anatomy OFF
D - Combined B & C: Consensus weight 0.0, Anatomy OFF
"""
import argparse
import copy
import csv
import sys
import yaml
from pathlib import Path
import numpy as np

from config_loader import load_config
from fusion_pipeline.aotr_benchmark import report_csv_name, evaluation_protocol
import brute_force_runner

WS_DIR = Path(__file__).parent.resolve()

EXPERIMENTS = {
    "A": {
        "name": "Bản hiện tại",
        "consensus_weight": 0.12,
        "anatomy": True,
        "purpose": "Baseline AOTR hiện tại",
    },
    "B": {
        "name": "Bỏ sửa không có lợi thế",
        "consensus_weight": 0.0,
        "anatomy": True,
        "purpose": "Đo tác hại của consensus floor",
    },
    "C": {
        "name": "Bỏ hiệu chuẩn xương",
        "consensus_weight": 0.12,
        "anatomy": False,
        "purpose": "Đo ảnh hưởng của anatomical calibration",
    },
    "D": {
        "name": "Kết hợp B và C",
        "consensus_weight": 0.0,
        "anatomy": False,
        "purpose": "Kiểm tra tương tác hai cơ chế",
    },
}


def build_config(base_cfg, exp_key):
    exp = EXPERIMENTS[exp_key]
    cfg = copy.deepcopy(base_cfg)
    cfg["fusion"]["aotr"]["three_d_consensus_weight"] = exp["consensus_weight"]
    cfg["fusion"]["aotr"]["sequence_stability"]["anatomical_calibration"]["enabled"] = exp["anatomy"]
    return cfg


def get_csv_path(base_cfg, exp_key):
    cfg = build_config(base_cfg, exp_key)
    return WS_DIR / "output" / report_csv_name(cfg)


def parse_report_metrics(csv_path):
    p = Path(csv_path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Rank") != "End"]
    if not rows:
        return None

    def avg(col):
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "N/A", "nan")]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "pairs": len(rows),
        "pa_base": avg("pa-mpjpe.all_baseline"),
        "pa_method": avg("pa-mpjpe.all_method"),
        "d_pa": avg("% delta_pa-mpjpe"),
        "mpjpe_base": avg("mpjpe.all_baseline"),
        "mpjpe_method": avg("mpjpe.all_method"),
        "d_mpjpe": avg("% delta_mpjpe"),
        "accel_base": avg("accel.baseline"),
        "accel_method": avg("accel.method"),
        "d_accel": avg("% delta_accel"),
        "mble_base": avg("mble.baseline"),
        "mble_method": avg("mble.method"),
        "d_mble": avg("% delta_mble"),
    }


def print_summary_table(base_cfg):
    print("\n" + "=" * 115)
    print(f"{'Exp':<4} | {'Consensus':<9} | {'Anatomy':<7} | {'PA-MPJPE (mm)':<16} | {'Δ PA (%)':<10} | {'Accel (Δ%)':<14} | {'MBLE (Δ%)':<14} | {'Pairs':<6} | {'Status'}")
    print("=" * 115)

    for k, info in EXPERIMENTS.items():
        csv_file = get_csv_path(base_cfg, k)
        m = parse_report_metrics(csv_file)
        cw = f"{info['consensus_weight']:.2f}"
        anat = "Bật" if info["anatomy"] else "Tắt"

        if m is None:
            print(f"{k:<4} | {cw:<9} | {anat:<7} | {'-':<16} | {'-':<10} | {'-':<14} | {'-':<14} | {'0':<6} | Chưa chạy")
        else:
            pa_str = f"{m['pa_base']:.1f} -> {m['pa_method']:.1f}"
            d_pa_str = f"{m['d_pa']:+.2f}%"
            accel_str = f"{m['d_accel']:+.2f}%"
            mble_str = f"{m['d_mble']:+.2f}%"
            status = "Hoàn thành" if m["pairs"] == 56 else f"Đang chạy ({m['pairs']}/56)"
            print(f"{k:<4} | {cw:<9} | {anat:<7} | {pa_str:<16} | {d_pa_str:<10} | {accel_str:<14} | {mble_str:<14} | {m['pairs']:<6} | {status}")
    print("=" * 115 + "\n")


def run_experiment(exp_key, base_cfg, brute_cfg=None):
    info = EXPERIMENTS[exp_key]
    cfg = build_config(base_cfg, exp_key)
    csv_file = get_csv_path(base_cfg, exp_key)
    print(f"\n=======================================================")
    print(f"[*] BẮT ĐẦU THỬ NGHIỆM {exp_key}: {info['name']}")
    print(f"    - Consensus weight: {info['consensus_weight']}")
    print(f"    - Anatomy calibration: {'Bật' if info['anatomy'] else 'Tắt'}")
    print(f"    - Mục đích: {info['purpose']}")
    print(f"    - CSV đích: output/{csv_file.name}")
    print(f"=======================================================\n")
    brute_force_runner.run_brute_force(base_cfg=cfg, brute_cfg=brute_cfg)


def filter_brute_cfg(cams_filter):
    with open(WS_DIR / "configs/brute_force.yml", "r", encoding="utf-8") as f:
        brute_cfg = yaml.safe_load(f)
    if not cams_filter:
        return brute_cfg
    cams_set = set(c.strip() for c in cams_filter.split(","))
    for seg in brute_cfg.get("segments", []):
        seg["cameras"] = [c for c in seg.get("cameras", []) if c.get("id") in cams_set]
    return brute_cfg


def main():
    parser = argparse.ArgumentParser(description="Chạy bộ thử nghiệm AOTR (A, B, C, D)")
    parser.add_argument("--exp", nargs="+", choices=["A", "B", "C", "D"], default=["B", "C", "D"],
                        help="Danh sách thử nghiệm cần chạy (mặc định B C D vì A đã có kết quả)")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Giới hạn camera ID (ví dụ: 'cam04,cam05') để kiểm tra nhanh")
    parser.add_argument("--summary", action="store_true", help="Chỉ in bảng tổng hợp kết quả hiện có")
    args = parser.parse_args()

    base_cfg = load_config(WS_DIR / "configs/pipeline.yml")

    if args.summary:
        print_summary_table(base_cfg)
        return

    print_summary_table(base_cfg)
    brute_cfg = filter_brute_cfg(args.pairs) if args.pairs else None

    for exp_key in args.exp:
        run_experiment(exp_key, base_cfg, brute_cfg=brute_cfg)

    print_summary_table(base_cfg)


if __name__ == "__main__":
    main()

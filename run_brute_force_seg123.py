import os
import sys
import yaml
import csv
import time
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Set environment variables for best config matching Colab/GG Drive setup
os.environ["NOTEBOOK_NAME"] = "ablation_hieuDT_belief_fusion_H260912RayCasting_optical_global_kinematic_huber_alpha1E_1_beta85E_2.ipynb"
os.environ["ALPHA"] = "0.1"
os.environ["BETA"] = "0.85"
os.environ["LOCAL_METHOD"] = "optical_aware_belief"
os.environ["GLOBAL"] = "true"
os.environ["KINEMATIC_CONSTRAINTS"] = "true"
os.environ["LOSS_TYPE"] = "huber"

from config_loader import load_config
from brute_force_runner import _evaluate_camera_pair, _build_report_rows

def main():
    print("=" * 110, flush=True)
    print(" BẮT ĐẦU CHẠY THỰC NGHIỆM VÉT CẠN (SEG 1, SEG 2, SEG 3)", flush=True)
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(" Report Format: EXACT Google Drive / Google Sheets format (_build_report_rows)", flush=True)
    print("=" * 110, flush=True)

    with open(WORKSPACE / "configs/brute_force.yml", "r", encoding="utf-8") as f:
        brute_cfg = yaml.safe_load(f)

    base_cfg = load_config(WORKSPACE / "configs/pipeline.yml")

    target_segments = ["seg_1", "seg_2", "seg_3"]

    all_camera_pairs = []
    for seg in brute_cfg.get("segments", []):
        seg_name = seg["name"]
        if seg_name in target_segments:
            cameras = seg.get("cameras", [])
            for i in range(len(cameras)):
                for j in range(len(cameras)):
                    if i != j:
                        pklA = WORKSPACE / cameras[i]["pkl"]
                        pklB = WORKSPACE / cameras[j]["pkl"]
                        if pklA.exists() and pklB.exists():
                            all_camera_pairs.append((seg_name, seg["ground_truth_dir"], cameras[i], cameras[j]))

    total_pairs = len(all_camera_pairs)
    print(f"\n[+] Tổng số cặp camera HỢP LỆ (Seg 1, 2, 3): {total_pairs} cặp", flush=True)
    print("-" * 110, flush=True)

    all_results = {}
    report_csv = WORKSPACE / "brute_force_local_report.csv"
    report_full_csv = WORKSPACE / "brute_force_seg123_full_report.csv"

    for idx, (seg_name, seg_gt_dir, camA, camB) in enumerate(all_camera_pairs, start=1):
        camA_id, camB_id = camA["id"], camB["id"]
        print(f"\n[{idx}/{total_pairs}] Segment={seg_name} | Master={camA_id} -> Slave={camB_id}...", flush=True)

        t0 = time.time()
        res = _evaluate_camera_pair(camA, camB, base_cfg, str(WORKSPACE / seg_gt_dir), WORKSPACE, seg_name)
        elapsed = time.time() - t0

        if seg_name not in all_results:
            all_results[seg_name] = []
        all_results[seg_name].append(res)

        # Periodically save full report CSV matching GG Drive structure
        joint_keys = set()
        for s_res in all_results.values():
            for r in s_res:
                joint_keys.update(r.get("joints", {}).keys())

        rows = _build_report_rows(all_results, sorted(list(joint_keys)))

        with open(report_full_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

        # Also save simple report CSV
        with open(report_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "Segment", "Master", "Slave", "Raw_MPJPE", "Fusion_MPJPE", "Delta_MPJPE_Pct",
                "Raw_PA_MPJPE", "Fusion_PA_MPJPE", "Delta_PA_MPJPE_Pct", "MBLE_mm", "Accel_Error", "Status"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s_name, s_res in all_results.items():
                for r in s_res:
                    writer.writerow({
                        "Segment": s_name,
                        "Master": r.get("master"),
                        "Slave": r.get("supplement"),
                        "Raw_MPJPE": round(r.get("old_mpjpe", 0.0), 2),
                        "Fusion_MPJPE": round(r.get("mpjpe", 0.0), 2),
                        "Delta_MPJPE_Pct": round(r.get("% delta_mpjpe", 0.0), 2),
                        "Raw_PA_MPJPE": round(r.get("old_pa_mpjpe", 0.0), 2),
                        "Fusion_PA_MPJPE": round(r.get("pa_mpjpe", 0.0), 2),
                        "Delta_PA_MPJPE_Pct": round(r.get("% delta_pa_mpjpe", 0.0), 2),
                        "MBLE_mm": round(r.get("mble", 0.0), 2),
                        "Accel_Error": round(r.get("accel", 0.0), 2),
                        "Status": "PASSED"
                    })

        print(f" -> Completed in {elapsed:.1f}s | Fusion MPJPE={res.get('mpjpe', 0):.2f}mm (Δ {res.get('% delta_mpjpe', 0):+.2f}%) | PA-MPJPE={res.get('pa_mpjpe', 0):.2f}mm (Δ {res.get('% delta_pa_mpjpe', 0):+.2f}%)", flush=True)

    print("\n" + "=" * 110, flush=True)
    print(f" CHẠY SEG 1, 2, 3 HOÀN TẤT ({total_pairs} CẶP)!", flush=True)
    print(f" File báo cáo chi tiết (Google Drive Format): {report_full_csv}", flush=True)
    print(f" File báo cáo vắn tắt: {report_csv}", flush=True)
    print("=" * 110, flush=True)

if __name__ == "__main__":
    main()

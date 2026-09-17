import os
import sys
import yaml
import json
import csv
import time
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# 1. Set environment variables to best config
os.environ["NOTEBOOK_NAME"] = "ablation_hieuDT_belief_fusion_H260912RayCasting_optical_global_kinematic_huber_alpha1E_1_beta85E_2.ipynb"
os.environ["ALPHA"] = "0.1"
os.environ["BETA"] = "0.85"
os.environ["LOCAL_METHOD"] = "optical_aware_belief"
os.environ["GLOBAL"] = "true"
os.environ["KINEMATIC_CONSTRAINTS"] = "true"
os.environ["LOSS_TYPE"] = "huber"

from config_loader import load_config, set_env_from_filename
from brute_force_runner import _evaluate_camera_pair, extract_set_name

def run_local_brute_force():
    print("=" * 110, flush=True)
    print(" BẮT ĐẦU CHẠY VÉT CẠN THỰC NGHIỆM (LOCAL BRUTE-FORCE RUNNER - BEST CONFIG)", flush=True)
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(" Config: Ray-Casting 3D | Optical-Aware Belief | Global Skeleton | Sequence Root | Huber Loss", flush=True)
    print("=" * 110, flush=True)
    
    with open(WORKSPACE / "configs/brute_force.yml", "r", encoding="utf-8") as f:
        brute_cfg = yaml.safe_load(f)
        
    base_cfg = load_config(WORKSPACE / "configs/pipeline.yml")
    gt_dir = str(WORKSPACE / "input/Seq1/GT")
    
    # Enable all segments (seg_1 to seg_11)
    segments_to_run = []
    
    all_camera_pairs = []
    for seg_idx, seg in enumerate(brute_cfg.get("segments", [])):
        seg_name = seg["name"]
        if seg_name in segments_to_run or len(segments_to_run) == 0:
            cameras = seg.get("cameras", [])
            for i in range(len(cameras)):
                for j in range(len(cameras)):
                    if i != j:
                        pklA = WORKSPACE / cameras[i]["pkl"]
                        pklB = WORKSPACE / cameras[j]["pkl"]
                        if pklA.exists() and pklB.exists():
                            all_camera_pairs.append((seg_name, seg["ground_truth_dir"], cameras[i], cameras[j]))
                        
    total_pairs = len(all_camera_pairs)
    print(f"\n[+] Tổng số cặp camera HỢP LỆ cần quét thực nghiệm: {total_pairs} cặp", flush=True)
    print("-" * 110, flush=True)
    
    results = []
    report_csv = WORKSPACE / "brute_force_local_report.csv"
    
    fieldnames = [
        "Segment", "Master", "Slave", "Raw_MPJPE", "Fusion_MPJPE", "Delta_MPJPE_Pct",
        "Raw_PA_MPJPE", "Fusion_PA_MPJPE", "Delta_PA_MPJPE_Pct", "MBLE_mm", "Accel_Error", "Status"
    ]
    
    with open(report_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()
        
        for idx, (seg_name, seg_gt_dir, camA, camB) in enumerate(all_camera_pairs, start=1):
            camA_id, camB_id = camA["id"], camB["id"]
            print(f"\n[{idx}/{total_pairs}] Running Segment={seg_name} | Master={camA_id} -> Slave={camB_id}...", flush=True)
            
            t0 = time.time()
            res = _evaluate_camera_pair(camA, camB, base_cfg, str(WORKSPACE / seg_gt_dir), WORKSPACE, seg_name)
            elapsed = time.time() - t0
            
            raw_m = res.get("old_mpjpe", float("inf"))
            fused_m = res.get("mpjpe", float("inf"))
            d_m = res.get("% delta_mpjpe", 0.0)
            
            raw_pa = res.get("old_pa_mpjpe", float("inf"))
            fused_pa = res.get("pa_mpjpe", float("inf"))
            d_pa = res.get("% delta_pa_mpjpe", 0.0)
            
            mble = res.get("mble", float("inf"))
            accel = res.get("accel", float("inf"))
            
            if fused_m == float("inf"):
                print(f" -> Skipped writing invalid result for {camA_id}->{camB_id}", flush=True)
                continue

            row_dict = {
                "Segment": seg_name,
                "Master": camA_id,
                "Slave": camB_id,
                "Raw_MPJPE": round(raw_m, 2),
                "Fusion_MPJPE": round(fused_m, 2),
                "Delta_MPJPE_Pct": round(d_m, 2),
                "Raw_PA_MPJPE": round(raw_pa, 2),
                "Fusion_PA_MPJPE": round(fused_pa, 2),
                "Delta_PA_MPJPE_Pct": round(d_pa, 2),
                "MBLE_mm": round(mble, 2),
                "Accel_Error": round(accel, 2),
                "Status": "PASSED"
            }
            
            writer.writerow(row_dict)
            csvfile.flush()
            
            print(f" -> Completed in {elapsed:.1f}s | Fusion MPJPE={fused_m:.2f}mm (Δ {d_m:+.2f}%) | PA-MPJPE={fused_pa:.2f}mm (Δ {d_pa:+.2f}%)", flush=True)

    print("\n" + "=" * 110, flush=True)
    print(f" CHẠY THỰC NGHIỆM VÉT CẠN HOÀN TẤT ({total_pairs} CẶP)!", flush=True)
    print(f" File báo cáo chi tiết được lưu tại: {report_csv}", flush=True)
    print("=" * 110, flush=True)

if __name__ == "__main__":
    run_local_brute_force()

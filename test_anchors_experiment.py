import sys
import csv
from pathlib import Path
from config_loader import load_config, absolutize_config_paths
from fusion_pipeline.executor import run_fusion
from evaluation_pipeline.executor import run_evaluation

import fusion_pipeline.config
import fusion_pipeline.detector
import fusion_pipeline.correction

ws = Path(__file__).parent.resolve()
cfg = absolutize_config_paths(load_config(ws / "configs/pipeline.yml"), ws)
cfg["runtime"]["clean_output"] = True
cfg["visualization"]["enabled"] = False
cfg["learnable"]["enabled"] = False
cfg["learnable_extra"]["enabled"] = False

def set_anchors(anchors):
    fusion_pipeline.config.NON_REPLACEABLE_ANCHORS = set(anchors)
    fusion_pipeline.detector.NON_REPLACEABLE_ANCHORS = set(anchors)
    fusion_pipeline.correction.NON_REPLACEABLE_ANCHORS = set(anchors)

def read_avg(fpath):
    if not Path(fpath).exists():
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        for row in r:
            if row and row[0] == "AVERAGE":
                return dict(zip(h, row))
    return {}

experiments = [
    ("Baseline (4 shoulders/hips)", {"left_hip", "left_shoulder", "right_hip", "right_shoulder"}),
    ("Empty set (Bo hoan toan)", set()),
    ("Pelvis + Neck only", {"pelvis", "neck"}),
]

results = []
for name, anchors in experiments:
    print(f"\n>>> Running: {name}")
    set_anchors(anchors)
    out_fused = ws / f"output/test_fused_{len(results)}"
    out_eval = ws / f"output/test_eval_{len(results)}"
    cfg["paths"]["fused_output_dir"] = str(out_fused)
    cfg["paths"]["evaluation_output_dir"] = str(out_eval)
    run_fusion(cfg)
    run_evaluation(cfg)
    
    mpjpe = read_avg(out_eval / "MPJPE_cam1.csv")
    pa = read_avg(out_eval / "PA-MPJPE_cam1.csv")
    
    results.append({
        "name": name,
        "mpjpe": mpjpe.get("fused_priority1_mm", "N/A"),
        "pa_mpjpe": pa.get("fused_priority1_mm", "N/A"),
        "posed_mpjpe": mpjpe.get("posed_priority1_mm", "N/A"),
        "posed_pa": pa.get("posed_priority1_mm", "N/A"),
    })

print("\n" + "="*70)
print(f"{'Experiment':<32} | {'MPJPE (mm)':<15} | {'PA-MPJPE (mm)':<15}")
print("="*70)
for r in results:
    print(f"{r['name']:<32} | {r['mpjpe']:<15} | {r['pa_mpjpe']:<15}")
print("="*70)


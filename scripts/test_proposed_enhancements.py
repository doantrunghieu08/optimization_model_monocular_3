import os
import sys
import yaml
import copy
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

os.environ["NOTEBOOK_NAME"] = "ablation_test_proposed_enhancements.ipynb"
os.environ["ALPHA"] = "0.01"
os.environ["BETA"] = "0.8"
os.environ["LOCAL_METHOD"] = "optical_aware_belief"
os.environ["GLOBAL"] = "true"
os.environ["KINEMATIC_CONSTRAINTS"] = "true"
os.environ["LOSS_TYPE"] = "huber"

from src.core.config_loader import load_config
from brute_force_runner import _evaluate_camera_pair, resolve_existing_path

def get_valid_pairs():
    with open(WORKSPACE / "configs/brute_force.yml", "r", encoding="utf-8") as f:
        brute_cfg = yaml.safe_load(f)
        
    valid_pairs = []
    for seg in brute_cfg.get("segments", []):
        seg_name = seg["name"]
        gt_dir = seg["ground_truth_dir"]
        cameras = seg.get("cameras", [])
        for i in range(len(cameras)):
            for j in range(len(cameras)):
                if i != j:
                    camA, camB = cameras[i], cameras[j]
                    pklA = resolve_existing_path(camA["pkl"], WORKSPACE)
                    pklB = resolve_existing_path(camB["pkl"], WORKSPACE)
                    if pklA and pklB:
                        valid_pairs.append((seg_name, gt_dir, camA, camB))
    return valid_pairs

def run_benchmark():
    valid_pairs = get_valid_pairs()
    if not valid_pairs:
        print("[!] Không tìm thấy cặp camera hợp lệ nào!")
        return

    # Select 2 distinct test pairs for evaluation
    selected_pairs = valid_pairs[:2]
    
    base_cfg = load_config(WORKSPACE / "configs/pipeline.yml")
    base_cfg["runtime"]["clean_output"] = True
    base_cfg["learnable_extra"]["enabled"] = True

    configs_to_test = [
        ("aligned_averaging", {"method": "aligned_averaging"}),
        ("proposed (original)", {"method": "proposed", "opt_override": {"pre_fuse_f_points": False, "cross_view_lambda": 0.0}}),
        ("proposed + Sol 1 (Pre-Fusion)", {"method": "proposed", "opt_override": {"pre_fuse_f_points": True, "cross_view_lambda": 0.0}}),
        ("proposed + Sol 1 + Sol 2 (Full)", {"method": "proposed", "opt_override": {"pre_fuse_f_points": True, "cross_view_lambda": 1.0}}),
    ]

    print("=" * 100)
    print("🚀 SO SÁNH HIỆU NĂNG CÁC GIẢI PHÁP TRÊN NHÁNH PROPOSED")
    print("=" * 100)

    summary_table = []

    for seg_name, seg_gt_dir, camA, camB in selected_pairs:
        gt_dir = str(WORKSPACE / seg_gt_dir)
        pair_str = f"{seg_name} | {camA['id']} -> {camB['id']}"
        print(f"\n▶ ĐANG ĐÁNH GIÁ CẶP: {pair_str}")

        for name, spec in configs_to_test:
            cfg = copy.deepcopy(base_cfg)
            cfg["fusion"]["method"] = spec["method"]
            if "opt_override" in spec:
                for k, v in spec["opt_override"].items():
                    cfg["fusion"]["optimization"][k] = v

            res = _evaluate_camera_pair(camA, camB, cfg, gt_dir, WORKSPACE, seg_name)
            if res:
                summary_table.append({
                    "pair": pair_str,
                    "method": name,
                    "mpjpe": res.get("mpjpe", float("inf")),
                    "pa_mpjpe": res.get("pa_mpjpe", float("inf")),
                    "mble": res.get("mble", float("inf")),
                    "accel": res.get("accel", float("inf")),
                    "delta_mpjpe": res.get("% delta_mpjpe", 0.0),
                    "delta_pa": res.get("% delta_pa_mpjpe", 0.0),
                })

    print("\n" + "=" * 100)
    print("📊 BẢNG KẾT QUẢ SO SÁNH TỔNG HỢP:")
    print(f"{'Pair':<35} | {'Method':<32} | {'MPJPE (mm)':<10} | {'PA-MPJPE':<10} | {'MBLE':<8} | {'Accel':<8}")
    print("-" * 100)
    for r in summary_table:
        print(f"{r['pair']:<35} | {r['method']:<32} | {r['mpjpe']:<10.2f} | {r['pa_mpjpe']:<10.2f} | {r['mble']:<8.2f} | {r['accel']:<8.2f}")
    print("=" * 100)

if __name__ == "__main__":
    run_benchmark()

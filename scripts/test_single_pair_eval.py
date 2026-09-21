import os
import sys
import yaml
import random
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

os.environ["NOTEBOOK_NAME"] = "ablation_hieuDT_belief_fusion_H260912RayCasting_optical_global_kinematic_huber_alpha1E_1_beta85E_2.ipynb"
os.environ["ALPHA"] = "0.1"
os.environ["BETA"] = "0.85"
os.environ["LOCAL_METHOD"] = "optical_aware_belief"
os.environ["GLOBAL"] = "true"
os.environ["KINEMATIC_CONSTRAINTS"] = "true"
os.environ["LOSS_TYPE"] = "huber"

from src.core.config_loader import load_config
from brute_force_runner import _evaluate_camera_pair, resolve_existing_path

def get_all_valid_camera_pairs():
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

def run_test(target_pair_str=None):
    base_cfg = load_config(WORKSPACE / "configs/pipeline.yml")
    base_cfg["runtime"]["clean_output"] = True
    base_cfg["learnable_extra"]["enabled"] = True
    
    valid_pairs = get_all_valid_camera_pairs()
    if not valid_pairs:
        print("[!] Không tìm thấy cặp camera hợp lệ nào có file PKL tồn tại!")
        return None
        
    if target_pair_str and target_pair_str != "random":
        matched = [item for item in valid_pairs if f"{item[2]['id']}_{item[3]['id']}" == target_pair_str]
        if matched:
            selected = matched[0]
        else:
            print(f"[!] Cặp '{target_pair_str}' không khả dụng, chọn ngẫu nhiên 1 cặp thay thế...")
            selected = random.choice(valid_pairs)
    else:
        selected = random.choice(valid_pairs)
        
    seg_name, seg_gt_dir, camA, camB = selected
    gt_dir = str(WORKSPACE / seg_gt_dir)

    print("=" * 80)
    print(f"🎲 CHẠY THỰC NGHIỆM ĐÁNH GIÁ NGẪU NHIÊN: Segment={seg_name} | {camA['id']} -> {camB['id']}")
    print("Config selector =", base_cfg["fusion"]["correction"]["selector"])
    print("Config belief_delta_cap =", base_cfg["fusion"]["correction"]["belief_delta_cap"])
    print("=" * 80)

    res = _evaluate_camera_pair(camA, camB, base_cfg, gt_dir, WORKSPACE, seg_name)

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ CHI TIẾT (Segment={seg_name} | {camA['id']} -> {camB['id']}):")
    print(f" - Raw (Old) MPJPE   : {res.get('old_mpjpe', 0):.2f} mm")
    print(f" - Fusion MPJPE      : {res.get('mpjpe', 0):.2f} mm (Δ {res.get('% delta_mpjpe', 0):+.2f}%)")
    print(f" - Raw (Old) PA-MPJPE: {res.get('old_pa_mpjpe', 0):.2f} mm")
    print(f" - Fusion PA-MPJPE   : {res.get('pa_mpjpe', 0):.2f} mm (Δ {res.get('% delta_pa_mpjpe', 0):+.2f}%)")
    print(f" - MBLE (Bone Error) : {res.get('mble', 0):.2f} mm")
    print(f" - Accel Error       : {res.get('accel', 0):.2f} mm/frame^2")
    print("=" * 80)
    return res

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "random"
    run_test(target)

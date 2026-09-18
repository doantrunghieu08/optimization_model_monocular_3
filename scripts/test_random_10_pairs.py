import os
import sys
import yaml
import random
import time
import copy
import shutil
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

from src.core.config_loader import load_config, absolutize_config_paths
from src.pipelines.orchestrator import run_pipeline
from brute_force_runner import _parse_pipeline_results, extract_set_name, resolve_existing_path

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

def _evaluate_isolated_pair(camA, camB, base_cfg, gt_dir, workspace, seg_name, pair_idx):
    current_set = extract_set_name(camA["pkl"])
    pklA_path = resolve_existing_path(camA["pkl"], workspace)
    pklB_path = resolve_existing_path(camB["pkl"], workspace)
    gt_path = resolve_existing_path(gt_dir, workspace)

    if not pklA_path or not pklB_path:
        return None

    try:
        pklA_rel = str(pklA_path.relative_to(workspace))
    except ValueError:
        pklA_rel = str(pklA_path)

    try:
        pklB_rel = str(pklB_path.relative_to(workspace))
    except ValueError:
        pklB_rel = str(pklB_path)

    camA_copy = dict(camA, pkl=pklA_rel)
    camB_copy = dict(camB, pkl=pklB_rel)
    resolved_gt_dir = str(gt_path) if gt_path else gt_dir

    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("inputs", {})
    cfg["inputs"].update({
        "ground_truth_dir": resolved_gt_dir, "cam1_pkl": camA_copy["pkl"], "camera1_video": camA_copy["video"],
        "cam2_pkl": camB_copy["pkl"], "camera2_video": camB_copy["video"]
    })
    temp_dir = f"output/temp_eval_isolated_{pair_idx}"
    cfg.setdefault("paths", {})
    cfg["paths"].update({
        "preprocess_output_dir": f"{temp_dir}/preprocess_results",
        "pose_output_dir": f"{temp_dir}/pose_results",
        "fused_output_dir": f"{temp_dir}/fused_results",
        "learnable_output_dir": f"{temp_dir}/learnable_results",
        "learnable_extra_output_dir": f"{temp_dir}/learnable_extra_results",
        "visualization_output_dir": f"{temp_dir}/visualize_results",
        "evaluation_output_dir": f"{temp_dir}/evaluation_results",
    })
    cfg.setdefault("visualization", {})["enabled"] = False
    cfg.setdefault("runtime", {})["clean_output"] = True
    cfg["runtime"]["stage"] = "visualization"
    cfg.setdefault("learnable", {})["device"] = "cuda"
    cfg.setdefault("learnable_extra", {})["device"] = "cuda"
    config = absolutize_config_paths(cfg, workspace)

    try:
        run_pipeline(config, stage_override=None)
        res = _parse_pipeline_results(config, current_set, camA["id"], camB["id"], seg_name)
        # dọn dẹp thư mục tạm sau khi hoàn tất
        temp_path = workspace / temp_dir
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        return res
    except Exception as e:
        print(f"Lỗi khi chạy cặp {camA['id']}-{camB['id']}: {e}")
        return None

def run_10_random_pairs_eval(target_count=10):
    base_cfg = load_config(WORKSPACE / "configs/pipeline.yml")
    base_cfg["learnable_extra"]["enabled"] = False
    
    all_pairs = get_all_valid_camera_pairs()
    if not all_pairs:
        print("[!] Không tìm thấy cặp camera hợp lệ nào!")
        return
        
    random.seed(int(time.time()))
    shuffled_pairs = list(all_pairs)
    random.shuffle(shuffled_pairs)
    
    print("=" * 110)
    print(f"🎲 BẮT ĐẦU CHẠY THỰC NGHIỆM ĐÁNH GIÁ TRÊN {target_count} CẶP CAMERA NGẪU NHIÊN (ISOLATED PATHS)")
    print("=" * 110)
    
    results = []
    for idx, (seg_name, seg_gt_dir, camA, camB) in enumerate(shuffled_pairs, start=1):
        if len(results) >= target_count:
            break
        print(f"\n--- [{len(results)+1}/{target_count}] Evaluated: Segment={seg_name} | Master={camA['id']} -> Slave={camB['id']} ---")
        gt_dir = str(WORKSPACE / seg_gt_dir)
        res = _evaluate_isolated_pair(camA, camB, base_cfg, gt_dir, WORKSPACE, seg_name, idx)
        if res and res.get("mpjpe", float("inf")) != float("inf"):
            results.append(res)
            d_m = res.get("% delta_mpjpe", 0.0)
            d_pa = res.get("% delta_pa_mpjpe", 0.0)
            print(f" -> Success ({camA['id']}->{camB['id']}): MPJPE={res['mpjpe']:.2f}mm (Δ {d_m:+.2f}%), PA-MPJPE={res['pa_mpjpe']:.2f}mm (Δ {d_pa:+.2f}%)")
        else:
            print(f" -> Skipped invalid or mismatched pair: {camA['id']}->{camB['id']}")
            
    print("\n" + "=" * 110)
    print(f"📊 BẢNG TỔNG HỢP KẾT QUẢ {len(results)} CẶP CAMERA NGẪU NHIÊN:")
    print("=" * 110)
    print(f"{'STT':<4} | {'Segment':<8} | {'Cặp Cam':<10} | {'Raw MPJPE':<10} | {'Fused MPJPE':<12} | {'Δ MPJPE %':<10} | {'Raw PA':<10} | {'Fused PA':<10} | {'Δ PA %':<10} | {'MBLE (mm)':<10}")
    print("-" * 110)
    
    sum_d_m, sum_d_pa = 0.0, 0.0
    for idx, r in enumerate(results, start=1):
        seg = r.get("segment", r.get("set", "N/A"))
        pair_str = f"{r.get('master')}-{r.get('supplement')}"
        raw_m = r.get("old_mpjpe", 0.0)
        fused_m = r.get("mpjpe", 0.0)
        d_m = r.get("% delta_mpjpe", 0.0)
        
        raw_pa = r.get("old_pa_mpjpe", 0.0)
        fused_pa = r.get("pa_mpjpe", 0.0)
        d_pa = r.get("% delta_pa_mpjpe", 0.0)
        mble = r.get("mble", 0.0)
        
        sum_d_m += d_m
        sum_d_pa += d_pa
        
        print(f"{idx:<4} | {seg:<8} | {pair_str:<10} | {raw_m:<10.2f} | {fused_m:<12.2f} | {d_m:<+10.2f}% | {raw_pa:<10.2f} | {fused_pa:<10.2f} | {d_pa:<+10.2f}% | {mble:<10.2f}")
        
    print("-" * 110)
    if results:
        avg_d_m = sum_d_m / len(results)
        avg_d_pa = sum_d_pa / len(results)
        print(f"TRUNG BÌNH {len(results)} CẶP: Δ MPJPE % = {avg_d_m:+.2f}% | Δ PA-MPJPE % = {avg_d_pa:+.2f}%")
    print("=" * 110)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_10_random_pairs_eval(n)

import yaml
import csv
import itertools
from pathlib import Path

from config_loader import load_config, absolutize_config_paths
from pipeline import run_pipeline

def parse_average_csv(csv_path: str, target_column: str) -> float:
    path = Path(csv_path)
    if not path.exists():
        return float('inf')
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            target_idx = header.index(target_column)
        except ValueError:
            return float('inf')
        
        for row in reader:
            if row and row[0] == "AVERAGE":
                return float(row[target_idx])
    return float('inf')

def run_brute_force():
    WORKSPACE_DIR = Path(__file__).parent.resolve()
    brute_config_path = WORKSPACE_DIR / "configs" / "brute_force.yml"
    pipeline_config_path = WORKSPACE_DIR / "configs" / "pipeline.yml"

    with open(brute_config_path, "r", encoding="utf-8") as f:
        brute_cfg = yaml.safe_load(f)
        
    with open(pipeline_config_path, "r", encoding="utf-8") as f:
        base_pipeline_cfg = yaml.safe_load(f)

    all_results = {}

    for segment in brute_cfg.get("segments", []):
        seg_name = segment["name"]
        print(f"=== Bắt đầu vét cạn cho Segment: {seg_name} ===")
        gt_dir = str(WORKSPACE_DIR / segment["ground_truth_dir"])
        cameras = segment.get("cameras", [])
        
        if len(cameras) < 2:
            print(f"Segment {seg_name} không đủ 2 camera, bỏ qua.")
            continue

        results = []
        # Chỉnh hợp chập 2 (có thứ tự)
        for camA, camB in itertools.permutations(cameras, 2):
            print(f"\n--- Đang chạy cặp: Master={camA['id']} | Supplement={camB['id']} ---")
            
            import copy
            config = copy.deepcopy(base_pipeline_cfg)
            
            # Cập nhật inputs
            config.setdefault("inputs", {})
            config["inputs"]["ground_truth_dir"] = gt_dir
            config["inputs"]["cam1_pkl"] = camA["pkl"]
            config["inputs"]["camera1_video"] = camA["video"]
            config["inputs"]["cam2_pkl"] = camB["pkl"]
            config["inputs"]["camera2_video"] = camB["video"]
            
            # Tắt visualization để tiết kiệm thời gian
            config.setdefault("visualization", {})["enabled"] = False
            config.setdefault("runtime", {})["clean_output"] = True
            config["runtime"]["stage"] = "visualization"
            
            # Kích hoạt GPU (CUDA) cho module Learnable để tăng tốc
            config.setdefault("learnable", {})["device"] = "cuda"
            config.setdefault("learnable_extra", {})["device"] = "cuda"

            # Absolutize paths
            config = absolutize_config_paths(config, WORKSPACE_DIR)
            
            # Chạy toàn bộ pipeline
            try:
                run_pipeline(config, stage_override=None)
                
                # Đọc kết quả từ CSV của camera1 (Cam Master)
                eval_dir = Path(config["paths"]["evaluation_output_dir"])
                mpjpe_csv = eval_dir / "MPJPE_cam1.csv"
                pa_mpjpe_csv = eval_dir / "PA-MPJPE_cam1.csv"
                
                mpjpe = parse_average_csv(mpjpe_csv, "fusion-learnable_priority1_mm")
                pa_mpjpe = parse_average_csv(pa_mpjpe_csv, "fusion-learnable_priority1_mm")
                
                avg_score = (mpjpe + pa_mpjpe) / 2.0
                
                results.append({
                    "master": camA["id"],
                    "supplement": camB["id"],
                    "mpjpe": mpjpe,
                    "pa_mpjpe": pa_mpjpe,
                    "score": avg_score
                })
                print(f"Kết quả cặp {camA['id']}-{camB['id']}: MPJPE={mpjpe:.2f}, PA-MPJPE={pa_mpjpe:.2f}")
            except Exception as e:
                import traceback
                print(f"Lỗi khi chạy cặp {camA['id']}-{camB['id']}: {e}")
                traceback.print_exc()
                results.append({
                    "master": camA["id"],
                    "supplement": camB["id"],
                    "mpjpe": float('inf'),
                    "pa_mpjpe": float('inf'),
                    "score": float('inf')
                })
        
        # Xếp hạng
        results.sort(key=lambda x: x["score"])
        all_results[seg_name] = results
        
    # Tạo CSV Report
    generate_csv_report(all_results, str(WORKSPACE_DIR / "brute_force_report.csv"))


def generate_csv_report(all_results, out_path):
    import csv
    with open(out_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Segment', 'Rank', 'Cam Master', 'Cam Bo Sung', 'MPJPE (mm)', 'PA-MPJPE (mm)', 'Avg Score'])
        
        for seg_name, results in all_results.items():
            for rank, res in enumerate(results, start=1):
                mpjpe_str = f"{res['mpjpe']:.2f}" if res['mpjpe'] != float('inf') else "N/A"
                pa_mpjpe_str = f"{res['pa_mpjpe']:.2f}" if res['pa_mpjpe'] != float('inf') else "N/A"
                score_str = f"{res['score']:.2f}" if res['score'] != float('inf') else "N/A"
                
                writer.writerow([
                    seg_name,
                    rank,
                    res['master'],
                    res['supplement'],
                    mpjpe_str,
                    pa_mpjpe_str,
                    score_str
                ])
                
    print(f"\nĐã xuất báo cáo CSV tại: {out_path}")


if __name__ == "__main__":
    run_brute_force()

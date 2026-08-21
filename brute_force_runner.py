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

def load_existing_results(csv_path: str) -> dict:
    existing = {}
    path = Path(csv_path)
    if not path.exists():
        return existing
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return existing
            
        try:
            seg_idx = header.index('Segment')
            master_idx = header.index('Cam Master')
            supp_idx = header.index('Cam Bo Sung')
            mpjpe_idx = header.index('MPJPE (mm)')
            pa_mpjpe_idx = header.index('PA-MPJPE (mm)')
            score_idx = header.index('Avg Score')
        except ValueError:
            return existing

        for row in reader:
            if len(row) > score_idx:
                seg = row[seg_idx]
                master = row[master_idx]
                supplement = row[supp_idx]
                score_str = row[score_idx]
                if score_str != "N/A":
                    existing[(seg, master, supplement)] = {
                        "mpjpe": float(row[mpjpe_idx]) if row[mpjpe_idx] != "N/A" else float('inf'),
                        "pa_mpjpe": float(row[pa_mpjpe_idx]) if row[pa_mpjpe_idx] != "N/A" else float('inf'),
                        "score": float(score_str)
                    }
    return existing

def run_brute_force():
    WORKSPACE_DIR = Path(__file__).parent.resolve()
    brute_config_path = WORKSPACE_DIR / "configs" / "brute_force.yml"
    pipeline_config_path = WORKSPACE_DIR / "configs" / "pipeline.yml"

    with open(brute_config_path, "r", encoding="utf-8") as f:
        brute_cfg = yaml.safe_load(f)
        
    with open(pipeline_config_path, "r", encoding="utf-8") as f:
        base_pipeline_cfg = yaml.safe_load(f)

    report_csv_path = WORKSPACE_DIR / "brute_force_report.csv"
    existing_results = load_existing_results(str(report_csv_path))
    if existing_results:
        print(f"Đã tải {len(existing_results)} kết quả từ lần chạy trước.")

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
            
            # Kiểm tra xem cặp này đã có kết quả chưa
            if (seg_name, camA["id"], camB["id"]) in existing_results:
                print("Bỏ qua cặp này do đã có kết quả từ lần chạy trước.")
                res = existing_results[(seg_name, camA["id"], camB["id"])]
                results.append({
                    "master": camA["id"],
                    "supplement": camB["id"],
                    "mpjpe": res["mpjpe"],
                    "pa_mpjpe": res["pa_mpjpe"],
                    "score": res["score"]
                })
                continue

            if not (WORKSPACE_DIR / camA["pkl"]).exists() or not (WORKSPACE_DIR / camB["pkl"]).exists():
                print(f"Bỏ qua cặp này do thiếu file pkl đầu vào.")
                results.append({
                    "master": camA["id"],
                    "supplement": camB["id"],
                    "mpjpe": float('inf'),
                    "pa_mpjpe": float('inf'),
                    "score": float('inf')
                })
                continue

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
                
                target_col = "fusion-learnable_priority1_mm" if config.get("learnable", {}).get("enabled", True) else "fused_priority1_mm"
                mpjpe = parse_average_csv(mpjpe_csv, target_col)
                pa_mpjpe = parse_average_csv(pa_mpjpe_csv, target_col)
                
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
                
            # Ghi online/incremental ngay sau mỗi cặp để không mất dữ liệu nếu bị ngắt
            temp_results = dict(all_results)
            temp_seg_results = list(results)
            temp_seg_results.sort(key=lambda x: x["score"])
            temp_results[seg_name] = temp_seg_results
            generate_csv_report(temp_results, str(report_csv_path), silent=True)
        
        # Xếp hạng
        results.sort(key=lambda x: x["score"])
        all_results[seg_name] = results
        
    # Tạo CSV Report cuối cùng
    generate_csv_report(all_results, str(report_csv_path), silent=False)


def generate_csv_report(all_results, out_path, silent=False):
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
                
    if not silent:
        print(f"\nĐã xuất báo cáo CSV tại: {out_path}")

if __name__ == "__main__":
    run_brute_force()

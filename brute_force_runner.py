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

def parse_detailed_csv(csv_path: str, prefix: str) -> tuple[float, dict[str, float]]:
    path = Path(csv_path)
    if not path.exists():
        return float('inf'), {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        target_idx = -1
        joint_indices = {}
        
        for i, h in enumerate(header):
            if h == f"{prefix}_priority1_mm":
                target_idx = i
            elif h.startswith(f"{prefix}_") and h.endswith("_mm") and not h.endswith("priority1_mm") and not h.endswith("priority2_mm"):
                joint_name = h[len(f"{prefix}_") : -len("_mm")]
                joint_indices[joint_name] = i
                
        if target_idx == -1:
            return float('inf'), {}
            
        for row in reader:
            if row and row[0] == "AVERAGE":
                mean_val = float(row[target_idx])
                joints_dict = {}
                for j_name, j_idx in joint_indices.items():
                    try:
                        joints_dict[j_name] = float(row[j_idx])
                    except ValueError:
                        pass
                return mean_val, joints_dict
    return float('inf'), {}

def extract_local_belief(metadata_dir: Path) -> tuple[float, float]:
    import json
    if not metadata_dir.exists():
        return 0.0, 0.0
    cam1_beliefs = []
    cam2_beliefs = []
    for meta_file in metadata_dir.glob("fused_data_*.json"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                conf = meta.get("joint_confidence", {})
                c1 = conf.get("camera1", {})
                c2 = conf.get("camera2", {})
                if c1:
                    cam1_beliefs.append(sum(c1.values()) / len(c1))
                if c2:
                    cam2_beliefs.append(sum(c2.values()) / len(c2))
        except Exception:
            pass
    b1 = sum(cam1_beliefs) / len(cam1_beliefs) if cam1_beliefs else 0.0
    b2 = sum(cam2_beliefs) / len(cam2_beliefs) if cam2_beliefs else 0.0
    return b1, b2

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
            supp_idx = header.index('Cam Slave')
            mpjpe_idx = header.index('MPJPE (mm)')
            pa_mpjpe_idx = header.index('PA-MPJPE (mm)')
            score_idx = header.index('Avg Score')
            
            # Extract optional new columns
            lbm_idx = header.index('local_belief Master') if 'local_belief Master' in header else -1
            lbs_idx = header.index('local_belief Slave') if 'local_belief Slave' in header else -1
            old_mpjpe_idx = header.index('Old MPJPE') if 'Old MPJPE' in header else -1
            old_pa_mpjpe_idx = header.index('Old PA-MPJPE') if 'Old PA-MPJPE' in header else -1
            d_mpjpe_idx = header.index('Delta_MPJPE') if 'Delta_MPJPE' in header else -1
            d_pa_mpjpe_idx = header.index('Delta_PA-MPJPE') if 'Delta_PA-MPJPE' in header else -1
            
            # Find per-joint columns dynamically
            joint_cols = {}
            for i, h in enumerate(header):
                if h.startswith("MPJPE_") or h.startswith("PA-MPJPE_"):
                    joint_cols[h] = i
                    
        except ValueError:
            return existing

        for row in reader:
            if len(row) > score_idx:
                seg = row[seg_idx]
                master = row[master_idx]
                supplement = row[supp_idx]
                score_str = row[score_idx]
                if score_str != "N/A":
                    res_dict = {
                        "mpjpe": float(row[mpjpe_idx]) if row[mpjpe_idx] != "N/A" else float('inf'),
                        "pa_mpjpe": float(row[pa_mpjpe_idx]) if row[pa_mpjpe_idx] != "N/A" else float('inf'),
                        "score": float(score_str),
                        "local_belief_master": float(row[lbm_idx]) if lbm_idx != -1 and row[lbm_idx] != "N/A" else 0.0,
                        "local_belief_slave": float(row[lbs_idx]) if lbs_idx != -1 and row[lbs_idx] != "N/A" else 0.0,
                        "old_mpjpe": float(row[old_mpjpe_idx]) if old_mpjpe_idx != -1 and row[old_mpjpe_idx] != "N/A" else float('inf'),
                        "old_pa_mpjpe": float(row[old_pa_mpjpe_idx]) if old_pa_mpjpe_idx != -1 and row[old_pa_mpjpe_idx] != "N/A" else float('inf'),
                        "delta_mpjpe": float(row[d_mpjpe_idx]) if d_mpjpe_idx != -1 and row[d_mpjpe_idx] != "N/A" else 0.0,
                        "delta_pa_mpjpe": float(row[d_pa_mpjpe_idx]) if d_pa_mpjpe_idx != -1 and row[d_pa_mpjpe_idx] != "N/A" else 0.0,
                        "joints": {}
                    }
                    for j_name, j_idx in joint_cols.items():
                        if j_idx < len(row) and row[j_idx] != "N/A":
                            res_dict["joints"][j_name] = float(row[j_idx])
                    
                    existing[(seg, master, supplement)] = res_dict
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
                
                target_prefix = "fusion-learnable" if config.get("learnable", {}).get("enabled", True) else "fused"
                
                mpjpe, mpjpe_joints = parse_detailed_csv(mpjpe_csv, target_prefix)
                pa_mpjpe, pa_mpjpe_joints = parse_detailed_csv(pa_mpjpe_csv, target_prefix)
                
                old_mpjpe, _ = parse_detailed_csv(mpjpe_csv, "posed")
                old_pa_mpjpe, _ = parse_detailed_csv(pa_mpjpe_csv, "posed")
                
                avg_score = (mpjpe + pa_mpjpe) / 2.0
                delta_mpjpe = old_mpjpe - mpjpe if (old_mpjpe != float('inf') and mpjpe != float('inf')) else 0.0
                delta_pa_mpjpe = old_pa_mpjpe - pa_mpjpe if (old_pa_mpjpe != float('inf') and pa_mpjpe != float('inf')) else 0.0
                
                b1, b2 = extract_local_belief(Path(config["paths"]["fused_output_dir"]) / "metadata")
                
                joint_metrics = {}
                for k, v in mpjpe_joints.items():
                    joint_metrics[f"MPJPE_{k}"] = v
                for k, v in pa_mpjpe_joints.items():
                    joint_metrics[f"PA-MPJPE_{k}"] = v
                
                results.append({
                    "master": camA["id"],
                    "supplement": camB["id"],
                    "mpjpe": mpjpe,
                    "pa_mpjpe": pa_mpjpe,
                    "score": avg_score,
                    "local_belief_master": b1,
                    "local_belief_slave": b2,
                    "old_mpjpe": old_mpjpe,
                    "old_pa_mpjpe": old_pa_mpjpe,
                    "delta_mpjpe": delta_mpjpe,
                    "delta_pa_mpjpe": delta_pa_mpjpe,
                    "joints": joint_metrics
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
                    "score": float('inf'),
                    "local_belief_master": 0.0,
                    "local_belief_slave": 0.0,
                    "old_mpjpe": float('inf'),
                    "old_pa_mpjpe": float('inf'),
                    "delta_mpjpe": 0.0,
                    "delta_pa_mpjpe": 0.0,
                    "joints": {}
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
        
        # Determine all possible joint columns dynamically
        all_joint_keys = set()
        for seg_results in all_results.values():
            for res in seg_results:
                if "joints" in res:
                    all_joint_keys.update(res["joints"].keys())
        all_joint_keys = sorted(list(all_joint_keys))
        
        header = ['Segment', 'Rank', 'Cam Master', 'Cam Slave', 
                  'MPJPE (mm)', 'PA-MPJPE (mm)', 'Avg Score',
                  'local_belief Master', 'local_belief Slave',
                  'Old MPJPE', 'Old PA-MPJPE', 'Delta_MPJPE', 'Delta_PA-MPJPE'] + all_joint_keys
                  
        writer.writerow(header)
        
        for seg_name, results in all_results.items():
            for rank, res in enumerate(results, start=1):
                def fmt(v):
                    return f"{v:.2f}" if v != float('inf') else "N/A"
                    
                row = [
                    seg_name,
                    rank,
                    res['master'],
                    res.get('supplement', 'N/A'),
                    fmt(res.get('mpjpe', float('inf'))),
                    fmt(res.get('pa_mpjpe', float('inf'))),
                    fmt(res.get('score', float('inf'))),
                    fmt(res.get('local_belief_master', 0.0)),
                    fmt(res.get('local_belief_slave', 0.0)),
                    fmt(res.get('old_mpjpe', float('inf'))),
                    fmt(res.get('old_pa_mpjpe', float('inf'))),
                    fmt(res.get('delta_mpjpe', 0.0)),
                    fmt(res.get('delta_pa_mpjpe', 0.0))
                ]
                
                # Append per-joint columns
                for jk in all_joint_keys:
                    j_val = res.get("joints", {}).get(jk, float('inf'))
                    row.append(fmt(j_val))
                    
                writer.writerow(row)
                
    if not silent:
        print(f"\nĐã xuất báo cáo CSV tại: {out_path}")

if __name__ == "__main__":
    run_brute_force()

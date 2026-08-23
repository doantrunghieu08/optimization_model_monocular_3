#version 260823
import os
import platform
import getpass
import yaml
import csv
import itertools
import json
from pathlib import Path
from datetime import datetime

VIDEO_FOLDER = "imageSequence"

try:
    from google.colab import auth
    from google.auth import default
    import gspread
except ImportError:
    print("Cảnh báo: Không tìm thấy thư viện google colab/gspread.")

from config_loader import load_config, absolutize_config_paths
from pipeline import run_pipeline

GC_CLIENT = None

def get_gspread_client():
    global GC_CLIENT
    if GC_CLIENT is None:
        auth.authenticate_user()
        creds, _ = default()
        GC_CLIENT = gspread.authorize(creds)
    return GC_CLIENT

def get_system_metadata() -> tuple[str, str, str]:
    try:
        username = getpass.getuser()
    except Exception:
        username = os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))
    os_version = platform.platform()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return os_version, username, timestamp

def extract_set_name(pkl_path: str) -> str:
    parts = Path(pkl_path).parts
    if VIDEO_FOLDER in parts:
        try:
            idx = parts.index(VIDEO_FOLDER)
            if idx >= 2:
                return f"{parts[idx - 2]}/{parts[idx - 1]}"
            elif idx == 1:
                return parts[0]
        except ValueError:
            pass
    return "Unknown_Set"

def _build_format_requests(sheet_id: int, total_rows: int, total_cols: int) -> list:
    reqs = []
    # Reset background
    reqs.append({"repeatCell": {
        "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": total_rows, 
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {"backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}}},
        "fields": "userEnteredFormat.backgroundColor"
    }})
    # Format Header
    reqs.append({"repeatCell": {
        "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1, 
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {"textFormat": {"bold": True}, 
                                       "backgroundColor": {"red": 0.85, "green": 0.85, "blue": 0.85}}},
        "fields": "userEnteredFormat(textFormat,backgroundColor)"
    }})
    # Format Even Rows
    for r in range(1, total_rows, 2):
        reqs.append({"repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": r, "endRowIndex": r + 1, 
                      "startColumnIndex": 0, "endColumnIndex": total_cols},
            "cell": {"userEnteredFormat": {"backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.6}}},
            "fields": "userEnteredFormat.backgroundColor"
        }})
    return reqs

def decorate(worksheet, total_rows: int, total_cols: int):
    try:
        worksheet.freeze(rows=1)
        requests = _build_format_requests(worksheet.id, total_rows, total_cols)
        worksheet.spreadsheet.batch_update({"requests": requests})
    except Exception as e:
        print(f"Lỗi khi trang trí Google Sheets: {e}")

def parse_average_csv(csv_path: str, target_column: str) -> float:
    path = Path(csv_path)
    if not path.exists(): return float('inf')
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            target_idx = header.index(target_column)
        except ValueError: return float('inf')
        for row in reader:
            if row and row[0] == "AVERAGE": return float(row[target_idx])
    return float('inf')

def parse_detailed_csv(csv_path: str, prefix: str) -> tuple[float, dict[str, float]]:
    path = Path(csv_path)
    if not path.exists(): return float('inf'), {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        t_idx, j_indices = -1, {}
        for i, h in enumerate(header):
            if h == f"{prefix}_priority1_mm": t_idx = i
            elif h.startswith(f"{prefix}_") and h.endswith("_mm") and "priority" not in h:
                j_indices[h[len(f"{prefix}_") : -len("_mm")]] = i
        if t_idx == -1: return float('inf'), {}
        for row in reader:
            if row and row[0] == "AVERAGE":
                j_dict = {name: float(row[idx]) for name, idx in j_indices.items() if row[idx]}
                return float(row[t_idx]), j_dict
    return float('inf'), {}

def extract_local_belief(metadata_dir: Path) -> tuple[str, str]:
    if not metadata_dir.exists(): return "[]", "[]"
    
    c1_acc = {}
    c2_acc = {}
    count = 0
    
    for meta_file in metadata_dir.glob("fused_data_*.json"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                conf = json.load(f).get("joint_confidence", {})
                c1, c2 = conf.get("camera1", {}), conf.get("camera2", {})
                
                # Nếu c1, c2 là list, chuyển thành dict với key là index để dễ xử lý
                if isinstance(c1, list): c1 = {str(i): v for i, v in enumerate(c1)}
                if isinstance(c2, list): c2 = {str(i): v for i, v in enumerate(c2)}
                
                if c1:
                    for k, v in c1.items():
                        c1_acc[k] = c1_acc.get(k, 0.0) + v
                if c2:
                    for k, v in c2.items():
                        c2_acc[k] = c2_acc.get(k, 0.0) + v
            count += 1
        except Exception: pass
        
    if count == 0:
        return "[]", "[]"
        
    # Tính trung bình, làm tròn 2 chữ số và chuyển thành List.
    # (Sắp xếp các key theo thứ tự số để đảm bảo đúng thứ tự khớp từ 0, 1, 2...)
    def sort_key(k): return int(k) if str(k).isdigit() else k
    
    b1_list = [round(c1_acc[k] / count, 2) for k in sorted(c1_acc.keys(), key=sort_key)]
    b2_list = [round(c2_acc[k] / count, 2) for k in sorted(c2_acc.keys(), key=sort_key)]
    
    # Trả về luôn chuỗi string (vd: "[0.85, 0.91]") để ghi vào Excel
    return str(b1_list), str(b2_list)

def _get_sheet_data(sheet_name: str) -> tuple[list, list]:
    try:
        gc = get_gspread_client()
        data = gc.open(sheet_name).sheet1.get_all_values()
        if not data or len(data) < 2: return [], []
        return data[0], data[1:]
    except Exception: return [], []

def _get_header_indices(header: list) -> dict:
    idx = {}
    keys = ['Segment', 'Cam Master', 'Cam Slave', 'MPJPE (mm)', 'PA-MPJPE (mm)', 'Avg Score', 
            'local_belief Master', 'local_belief Slave', 'Old MPJPE', 'Old PA-MPJPE', 
            '% Delta_MPJPE', '% Delta_PA-MPJPE', 'OS Version', 'Username', 'Timestamp']
    for k in keys:
        idx[k] = header.index(k) if k in header else -1
    idx['joints'] = {h: i for i, h in enumerate(header) if h.startswith(("MPJPE_", "PA-MPJPE_"))}
    return idx

def _parse_history_row(row: list, idx: dict) -> tuple:
    s_idx = idx['Avg Score']
    if s_idx >= len(row) or row[s_idx] in ("N/A", ""): return None, None

    def get_val(col, default="N/A"):
        return row[idx[col]] if idx.get(col, -1) != -1 and idx[col] < len(row) else default
    
    def sf(col_name): 
        v = row[idx[col_name]] if idx[col_name] != -1 else "N/A"
        return float(v) if v not in ("N/A", "") else float('inf')
    
    key = (row[idx['Segment']], row[idx['Cam Master']], row[idx['Cam Slave']])
    # Tính trung bình và làm tròn trước khi đưa vào res
    avg_h1 = [round(val / context.count_of_frames, 2) for val in context.H1]
    avg_h2 = [round(val / context.count_of_frames, 2) for val in context.H2]
    
    res = {
        "mpjpe": sf('MPJPE (mm)'), "pa_mpjpe": sf('PA-MPJPE (mm)'), "score": float(row[s_idx]),
        "local_belief_master": str(avg_h1) if avg_h1 else 0.0,
        "local_belief_slave": str(avg_h2) if avg_h2 else 0.0,
        "old_mpjpe": sf('Old MPJPE'), "old_pa_mpjpe": sf('Old PA-MPJPE'),
        "% delta_mpjpe": sf('% Delta_MPJPE') if sf('% Delta_MPJPE') != float('inf') else 0.0,
        "% delta_pa_mpjpe": sf('% Delta_PA-MPJPE') if sf('% Delta_PA-MPJPE') != float('inf') else 0.0,
        "os_version": get_val('OS Version'), "username": get_val('Username'), "timestamp": get_val('Timestamp'),
        "joints": {jn: float(row[ji]) for jn, ji in idx['joints'].items() if ji < len(row) and row[ji] not in ("N/A", "")}
    }
    return key, res

def load_existing_spreadsheet_results(sheet_name: str) -> dict:
    existing = {}
    header, rows = _get_sheet_data(sheet_name)
    if not header: return existing
    
    idx = _get_header_indices(header)
    if idx['Segment'] == -1: return existing

    for row in rows:
        key, res = _parse_history_row(row, idx)
        if key and key[0] != "N/A": existing[key] = res
    return existing

def _build_report_rows(all_results: dict, joint_keys: list) -> list:
    header = ['Set', 'Segment', 'Rank', 'Cam Master', 'Cam Slave', 'MPJPE (mm)', 'PA-MPJPE (mm)', 
              'Avg Score', 'local_belief Master', 'local_belief Slave', 'Old MPJPE', 
              'Old PA-MPJPE', '% Delta_MPJPE', '% Delta_PA-MPJPE', 
              'OS Version', 'Username', 'Timestamp'] + joint_keys
    rows = [header]
    
    for seg_name, results in all_results.items():
        for rank, res in enumerate(results, start=1):
            def fmt(v): return round(float(v), 2) if v != float('inf') else "N/A"
            row = [
                res.get('set', 'Unknown_Set'), seg_name, rank, res['master'], res.get('supplement', 'N/A'),
                fmt(res.get('mpjpe', float('inf'))), fmt(res.get('pa_mpjpe', float('inf'))),
                fmt(res.get('score', float('inf'))), fmt(res.get('local_belief_master', 0.0)),
                fmt(res.get('local_belief_slave', 0.0)), fmt(res.get('old_mpjpe', float('inf'))),
                fmt(res.get('old_pa_mpjpe', float('inf'))), fmt(res.get('% delta_mpjpe', 0.0)),
                fmt(res.get('% delta_pa_mpjpe', 0.0)), res.get('os_version', 'N/A'), 
                res.get('username', 'N/A'), res.get('timestamp', 'N/A')
            ]
            row.extend([fmt(res.get("joints", {}).get(jk, float('inf'))) for jk in joint_keys])
            rows.append(row)
    return rows

def generate_spreadsheet_report(all_results, sheet_name, worksheet_title=None, silent=False):
    gc = get_gspread_client()
    try: sh = gc.open(sheet_name)
    except gspread.exceptions.SpreadsheetNotFound: sh = gc.create(sheet_name)
    worksheet = sh.sheet1

    all_joint_keys = set()
    for seg_results in all_results.values():
        for res in seg_results:
            all_joint_keys.update(res.get("joints", {}).keys())
    
    rows_to_insert = _build_report_rows(all_results, sorted(list(all_joint_keys)))

    worksheet.clear()
    try: worksheet.update(values=rows_to_insert, range_name="A1")
    except TypeError: worksheet.update(rows_to_insert)

    if not silent:
        print(f"\nĐã xuất báo cáo ra Google Spreadsheet thành công!\n🔗 Xem file tại: {sh.url}")

def _setup_pipeline_config(base_cfg, gt_dir, camA, camB, workspace):
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("inputs", {})
    cfg["inputs"].update({
        "ground_truth_dir": gt_dir, "cam1_pkl": camA["pkl"], "camera1_video": camA["video"],
        "cam2_pkl": camB["pkl"], "camera2_video": camB["video"]
    })
    cfg.setdefault("visualization", {})["enabled"] = False
    cfg.setdefault("runtime", {})["clean_output"] = True
    cfg["runtime"]["stage"] = "visualization"
    cfg.setdefault("learnable", {})["device"] = "cuda"
    cfg.setdefault("learnable_extra", {})["device"] = "cuda"
    return absolutize_config_paths(cfg, workspace)

def _parse_pipeline_results(config: dict, current_set: str, camA_id: str, camB_id: str) -> dict:
    eval_dir = Path(config["paths"]["evaluation_output_dir"])
    t_pref = "fusion-learnable" if config.get("learnable", {}).get("enabled", True) else "fused"
    
    mpjpe, m_jts = parse_detailed_csv(eval_dir / "MPJPE_cam1.csv", t_pref)
    pa_mpjpe, pa_jts = parse_detailed_csv(eval_dir / "PA-MPJPE_cam1.csv", t_pref)
    old_m, _ = parse_detailed_csv(eval_dir / "MPJPE_cam1.csv", "posed")
    old_pa, _ = parse_detailed_csv(eval_dir / "PA-MPJPE_cam1.csv", "posed")
    
    pd_m = (old_m - mpjpe)*100/old_m if (old_m != float('inf') and mpjpe != float('inf')) else 0.0
    pd_pa = (old_pa - pa_mpjpe)*100/old_pa if (old_pa != float('inf') and pa_mpjpe != float('inf')) else 0.0
    b1, b2 = extract_local_belief(Path(config["paths"]["fused_output_dir"]) / "metadata")
    
    joint_metrics = {f"MPJPE_{k}": v for k, v in m_jts.items()}
    joint_metrics.update({f"PA-MPJPE_{k}": v for k, v in pa_jts.items()})
    os_v, usr, ts = get_system_metadata()

    return {
        "set": current_set, "master": camA_id, "supplement": camB_id, "mpjpe": mpjpe, 
        "pa_mpjpe": pa_mpjpe, "score": (mpjpe + pa_mpjpe) / 2.0, "local_belief_master": b1,
        "local_belief_slave": b2, "old_mpjpe": old_m, "old_pa_mpjpe": old_pa,
        "% delta_mpjpe": pd_m, "% delta_pa_mpjpe": pd_pa, "joints": joint_metrics,
        "os_version": os_v, "username": usr, "timestamp": ts
    }

def _evaluate_camera_pair(camA, camB, base_cfg, gt_dir, workspace) -> dict:
    current_set = extract_set_name(camA["pkl"])
    os_v, usr, ts = get_system_metadata()
    if not (workspace / camA["pkl"]).exists() or not (workspace / camB["pkl"]).exists():
        print("Bỏ qua cặp này do thiếu file pkl đầu vào.")
        return {"set": current_set, "master": camA["id"], "supplement": camB["id"], "score": float('inf'),
                "os_version": os_v, "username": usr, "timestamp": ts}
    try:
        config = _setup_pipeline_config(base_cfg, gt_dir, camA, camB, workspace)
        run_pipeline(config, stage_override=None)
        res = _parse_pipeline_results(config, current_set, camA["id"], camB["id"])
        print(f"Kết quả {camA['id']}-{camB['id']}: MPJPE={res['mpjpe']:.2f}, PA-MPJPE={res['pa_mpjpe']:.2f}")
        return res
    except Exception as e:
        import traceback
        print(f"Lỗi khi chạy cặp {camA['id']}-{camB['id']}: {e}")
        traceback.print_exc()
        return {"set": current_set, "master": camA["id"], "supplement": camB["id"], "score": float('inf'),
                "os_version": os_v, "username": usr, "timestamp": ts}

def run_brute_force():
    WS_DIR = Path(__file__).parent.resolve()
    with open(WS_DIR / "configs/brute_force.yml", "r", encoding="utf-8") as f: brute_cfg = yaml.safe_load(f)
    with open(WS_DIR / "configs/pipeline.yml", "r", encoding="utf-8") as f: base_cfg = yaml.safe_load(f)

    sh_name = "Brute_Force_Report_Pipeline v260823"
    ws_title = f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    existing = load_existing_spreadsheet_results(sh_name)
    all_res = {}

    for seg in brute_cfg.get("segments", []):
        seg_name, cameras = seg["name"], seg.get("cameras", [])
        if len(cameras) < 2: continue
        print(f"=== Bắt đầu vét cạn cho Segment: {seg_name} ===")
        
        results = []
        for cA, cB in itertools.permutations(cameras, 2):
            print(f"\n--- Master={cA['id']} | Supplement={cB['id']} ---")
            if (seg_name, cA["id"], cB["id"]) in existing:
                res = existing[(seg_name, cA["id"], cB["id"])]
                res.update({"set": res.get("set", extract_set_name(cA["pkl"])), "master": cA["id"], "supplement": cB["id"]})
                results.append(res)
                continue
            
            res = _evaluate_camera_pair(cA, cB, base_cfg, str(WS_DIR / seg["ground_truth_dir"]), WS_DIR)
            results.append(res)
            
            temp_res = dict(all_res)
            temp_res[seg_name] = sorted(results, key=lambda x: x.get("score", float('inf')))
            generate_spreadsheet_report(temp_res, sh_name, ws_title, silent=True)

        all_res[seg_name] = sorted(results, key=lambda x: x.get("score", float('inf')))
    generate_spreadsheet_report(all_res, sh_name, ws_title, silent=False)

if __name__ == "__main__":
    run_brute_force()

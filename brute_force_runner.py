#version 260826_fixed
import os
import platform
import getpass
import yaml
import csv
import itertools
import json
from pathlib import Path
from datetime import datetime
import time
import threading
import queue
from config_loader import load_config

VIDEO_FOLDER = "imageSequence"

_COLAB_AVAILABLE = False
try:
    from google.colab import auth
    from google.auth import default
    import gspread
    _COLAB_AVAILABLE = True
except ImportError:
    auth = None
    default = None
    gspread = None
    print("Cảnh báo: Không tìm thấy thư viện google colab/gspread. "
          "Các tính năng Google Sheets sẽ không khả dụng.")

from config_loader import load_config, absolutize_config_paths, set_env_from_filename, get_notebook_name
from pipeline import run_pipeline

GC_CLIENT = None

def get_gspread_client():
    global GC_CLIENT
    if not _COLAB_AVAILABLE:
        raise RuntimeError(
            "brute_force_runner requires google.colab, google.auth, and gspread. "
            "These are only available in a Google Colab environment. "
            "Install gspread and google-auth manually, or run this script inside Colab."
        )
    if GC_CLIENT is None:
        auth.authenticate_user()
        creds, _ = default()
        GC_CLIENT = gspread.authorize(creds)
    return GC_CLIENT

def get_system_metadata() -> tuple[str, str, str]:
    try:
        username = os.environ.get('RUNNER_NAME', getpass.getuser())
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
    reqs.append({"repeatCell": {
        "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": total_rows, 
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {"backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}}},
        "fields": "userEnteredFormat.backgroundColor"
    }})
    reqs.append({"repeatCell": {
        "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1, 
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {"textFormat": {"bold": True}, 
                                       "backgroundColor": {"red": 0.85, "green": 0.85, "blue": 0.85}}},
        "fields": "userEnteredFormat(textFormat,backgroundColor)"
    }})
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

def parse_frame_metric_csv(csv_path: str, module: str, target_column: str) -> float:
    path = Path(csv_path)
    if not path.exists(): return float('inf')
    frame_values = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get("Module") != module or not row.get(target_column):
                continue
            try:
                frame_values[row["Frame"]] = float(row[target_column])
            except (KeyError, ValueError):
                continue
    return sum(frame_values.values()) / len(frame_values) if frame_values else float('inf')

def extract_local_belief(metadata_dir: Path) -> tuple[str, str, int, int]:
    if not metadata_dir.exists(): return "[]", "[]", 0, 0
    c1_acc, c2_acc, count = {}, {}, 0
    occluded_master = occluded_slave = 0
    for meta_file in metadata_dir.glob("fused_data_*.json"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                conf = metadata.get("joint_confidence", {})
                c1, c2 = conf.get("camera1", {}), conf.get("camera2", {})
                if isinstance(c1, list): c1 = {str(i): v for i, v in enumerate(c1)}
                if isinstance(c2, list): c2 = {str(i): v for i, v in enumerate(c2)}
                if c1:
                    for k, v in c1.items(): c1_acc[k] = c1_acc.get(k, 0.0) + v
                if c2:
                    for k, v in c2.items(): c2_acc[k] = c2_acc.get(k, 0.0) + v
                occluded_master += sum(visible is False for visible in metadata.get("vis1", {}).values())
                occluded_slave += sum(visible is False for visible in metadata.get("vis2", {}).values())
            count += 1
        except Exception: pass
    if count == 0: return "[]", "[]", 0, 0
    def sort_key(k): return int(k) if str(k).isdigit() else k
    b1_list = [round(c1_acc[k] / count, 2) for k in sorted(c1_acc.keys(), key=sort_key)]
    b2_list = [round(c2_acc[k] / count, 2) for k in sorted(c2_acc.keys(), key=sort_key)]
    return str(b1_list), str(b2_list), occluded_master, occluded_slave

def _get_sheet_data(sheet_name: str) -> tuple[list, list, str | None, bool]:
    try:
        gc = get_gspread_client()
        sh = gc.open(sheet_name)
        worksheets = sh.worksheets()
        if not worksheets:
            return [], [], None, False
        latest_worksheet = worksheets[-1] 
        data = latest_worksheet.get_all_values()
        ws_title = latest_worksheet.title
        if not data or len(data) < 2: return [], [], ws_title, False
        has_end_marker = any(row and row[0] == "End" for row in data[1:])
        return data[0], data[1:], ws_title, has_end_marker
    except Exception: 
        return [], [], None, False

def _get_header_indices(header: list) -> dict:
    idx = {}
    keys = [
        'Set', 'Segment', 'Rank', 'Cam Master', 'Cam Slave', 'Alpha', 'Beta',
        'Global Belief', 'Local Method', 'Kinematic Constraints', 'Loss Type',
        'Optimization Enabled', 'Orientation Correction', 'Learnable', 'Learnable Extra',
        'MPJPE', 'PA-MPJPE', 'MBLE',
        'Fusion MBLE', 'LE MBLE', 'Old MBLE',
        'Accel Error (mm/frame^2)', 'GT Accel Error (mm/frame^2)',
        'Fusion Accel Error (mm/frame^2)', 'LE Accel Error (mm/frame^2)',
        'Old Accel Error (mm/frame^2)',
        'LE MPJPE Master', 'LE PA-MPJPE Master',
        'belief Master', 'belief Slave', 'Occluded Joint-Frames Master',
        'Occluded Joint-Frames Slave',
        'Old MPJPE', 'Old PA-MPJPE', '% Δ_MPJPE', '% Δ_PA-MPJPE', 
        'OS Version', 'Username', 'Timestamp'
    ]
    for k in keys:
        if k in header:
            idx[k] = header.index(k)
        elif k == '% Δ_MPJPE' and '% d_MPJPE%' in header:
            idx[k] = header.index('% d_MPJPE%')
        elif k == '% Δ_PA-MPJPE' and 'd_PA-MPJPE' in header:
            idx[k] = header.index('d_PA-MPJPE')
        elif k in ('belief Master', 'belief Slave') and f'local_{k}' in header:
            idx[k] = header.index(f'local_{k}')
        elif k == 'Accel Error (mm/frame^2)' and 'Accel' in header:
            idx[k] = header.index('Accel')
        else:
            idx[k] = -1
    idx['joints'] = {h: i for i, h in enumerate(header) if h.startswith(("MPJPE_", "PA-MPJPE_"))}
    return idx

def _parse_history_row(row: list, idx: dict) -> tuple:
    if row[idx['Segment']] == "End": return None, None
    def get_val(col, default="N/A"):
        return row[idx[col]] if idx.get(col, -1) != -1 and idx[col] < len(row) else default
    def sf(col_name): 
        v = row[idx[col_name]] if idx.get(col_name, -1) != -1 else "N/A"
        v = str(v).strip().replace(',', '.')
        return float(v) if v not in ("N/A", "") else float('inf')
    
    key = (row[idx['Segment']], row[idx['Cam Master']], row[idx['Cam Slave']])
    res = {
        "alpha": sf('Alpha'), "beta": sf('Beta'),
        "global_belief": get_val('Global Belief'),
        "local_method": get_val('Local Method'),
        "kinematic_constraints": get_val('Kinematic Constraints'),
        "loss_type": get_val('Loss Type'),
        "optimization_enabled": get_val('Optimization Enabled'),
        "orientation_correction": get_val('Orientation Correction'),
        "learnable_enabled": get_val('Learnable'),
        "learnable_extra_enabled": get_val('Learnable Extra'),
        "mpjpe": sf('MPJPE'), "pa_mpjpe": sf('PA-MPJPE'),
        "mble": sf('MBLE'), "accel": sf('Accel Error (mm/frame^2)'),
        "fusion_mble": sf('Fusion MBLE'), "le_mble": sf('LE MBLE'),
        "old_mble": sf('Old MBLE'),
        "gt_accel_error": sf('GT Accel Error (mm/frame^2)'),
        "fusion_accel_error": sf('Fusion Accel Error (mm/frame^2)'),
        "le_accel_error": sf('LE Accel Error (mm/frame^2)'),
        "old_accel_error": sf('Old Accel Error (mm/frame^2)'),
        "le_mpjpe_master": get_val('LE MPJPE Master', "N/A"),
        "le_pa_mpjpe_master": get_val('LE PA-MPJPE Master', "N/A"),
        "belief_master": get_val('belief Master', "[]"),
        "belief_slave": get_val('belief Slave', "[]"),
        "occluded_joint_frames_master": sf('Occluded Joint-Frames Master'),
        "occluded_joint_frames_slave": sf('Occluded Joint-Frames Slave'),
        "old_mpjpe": sf('Old MPJPE'), "old_pa_mpjpe": sf('Old PA-MPJPE'),
        "% delta_mpjpe": sf('% Δ_MPJPE') if sf('% Δ_MPJPE') != float('inf') else 0.0,
        "% delta_pa_mpjpe": sf('% Δ_PA-MPJPE') if sf('% Δ_PA-MPJPE') != float('inf') else 0.0,
        "os_version": get_val('OS Version'), "username": get_val('Username'), "timestamp": get_val('Timestamp'),
        "joints": {jn: float(str(row[ji]).strip().replace(',', '.')) for jn, ji in idx['joints'].items() if ji < len(row) and row[ji] not in ("N/A", "")}
    }
    return key, res

def load_existing_spreadsheet_results(sheet_name: str) -> tuple[dict, str | None, bool]:
    existing = {}
    header, rows, ws_title, has_end_marker = _get_sheet_data(sheet_name)
    if not header: return existing, ws_title, has_end_marker
    idx = _get_header_indices(header)
    required = (
        'Segment', 'Alpha', 'Beta', 'Global Belief', 'Local Method',
        'Kinematic Constraints', 'Loss Type', 'MBLE', 'Accel Error (mm/frame^2)',
        'Optimization Enabled', 'Orientation Correction', 'Learnable', 'Learnable Extra',
        'Fusion MBLE', 'LE MBLE', 'Old MBLE', 'GT Accel Error (mm/frame^2)',
        'Fusion Accel Error (mm/frame^2)', 'LE Accel Error (mm/frame^2)',
        'Old Accel Error (mm/frame^2)',
        'Occluded Joint-Frames Master', 'Occluded Joint-Frames Slave',
    )
    if any(idx[column] == -1 for column in required): return existing, None, has_end_marker
    for row in rows:
        key, res = _parse_history_row(row, idx)
        if key and key[0] != "N/A": existing[key] = res
    return existing, ws_title, has_end_marker

def _build_report_rows(all_results: dict, joint_keys: list) -> list:
    header = ['Set', 'Segment', 'Rank', 'Cam Master', 'Cam Slave', 'Alpha', 'Beta',
              'Global Belief', 'Local Method', 'Kinematic Constraints', 'Loss Type',
              'Optimization Enabled', 'Orientation Correction', 'Learnable', 'Learnable Extra',
              'MPJPE', 'PA-MPJPE', 'MBLE', 'Accel Error (mm/frame^2)',
              'Fusion MBLE', 'LE MBLE', 'Old MBLE',
              'GT Accel Error (mm/frame^2)', 'Fusion Accel Error (mm/frame^2)',
              'LE Accel Error (mm/frame^2)', 'Old Accel Error (mm/frame^2)',
              'LE MPJPE Master', 'LE PA-MPJPE Master', 
              'belief Master', 'belief Slave', 'Occluded Joint-Frames Master',
              'Occluded Joint-Frames Slave', 'Old MPJPE',
              'Old PA-MPJPE', '% Δ_MPJPE', '% Δ_PA-MPJPE', 
              'OS Version', 'Username', 'Timestamp'] + joint_keys
    rows = [header]
    
    def fmt(v): return round(float(v), 2) if v != float('inf') else "N/A"
    
    for seg_name, results in all_results.items():
        for rank, res in enumerate(results, start=1):
            row = [
                res.get('set', 'Unknown_Set'), seg_name, rank, res['master'], res.get('supplement', 'N/A'),
                res.get('alpha', 'N/A'), res.get('beta', 'N/A'),
                res.get('global_belief', 'N/A'), res.get('local_method', 'N/A'),
                res.get('kinematic_constraints', 'N/A'),
                res.get('loss_type', 'N/A'),
                res.get('optimization_enabled', 'N/A'),
                res.get('orientation_correction', 'N/A'),
                res.get('learnable_enabled', 'N/A'),
                res.get('learnable_extra_enabled', 'N/A'),
                fmt(res.get('mpjpe', float('inf'))), fmt(res.get('pa_mpjpe', float('inf'))),
                fmt(res.get('mble', float('inf'))), fmt(res.get('accel', float('inf'))),
                fmt(res.get('fusion_mble', float('inf'))), fmt(res.get('le_mble', float('inf'))),
                fmt(res.get('old_mble', float('inf'))),
                fmt(res.get('gt_accel_error', float('inf'))),
                fmt(res.get('fusion_accel_error', float('inf'))),
                fmt(res.get('le_accel_error', float('inf'))),
                fmt(res.get('old_accel_error', float('inf'))),
                res.get('le_mpjpe_master', 'N/A'), res.get('le_pa_mpjpe_master', 'N/A'),
                res.get('belief_master', "[]"), res.get('belief_slave', "[]"),
                fmt(res.get('occluded_joint_frames_master', float('inf'))),
                fmt(res.get('occluded_joint_frames_slave', float('inf'))),
                fmt(res.get('old_mpjpe', float('inf'))), fmt(res.get('old_pa_mpjpe', float('inf'))), 
                fmt(res.get('% delta_mpjpe', 0.0)), fmt(res.get('% delta_pa_mpjpe', 0.0)), 
                res.get('os_version', 'N/A'), res.get('username', 'N/A'), res.get('timestamp', 'N/A')
            ]
            row.extend([fmt(res.get("joints", {}).get(jk, float('inf'))) for jk in joint_keys])
            rows.append(row)
            
    return rows

def _get_or_create_worksheet(sheet_name: str, worksheet_title: str = None, silent: bool = False):
    gc = get_gspread_client()
    try:
        sh = gc.open(sheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        sh = gc.create(sheet_name)
        try:
            sh.share('', perm_type='anyone', role='reader')
            if not silent: print(f"[+] Đã tạo Spreadsheet '{sheet_name}' và cấp quyền Public.")
        except Exception as e:
            if not silent: print(f"[-] Không thể tự động cấp quyền Public. Lỗi: {e}")

    if worksheet_title:
        try:
            worksheet = sh.worksheet(worksheet_title)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=worksheet_title, rows="1000", cols="50")
    else:
        worksheet = sh.sheet1
    return sh, worksheet

def generate_spreadsheet_report(all_results, sheet_name, worksheet_title=None, silent=False, is_final=False):
    sh, worksheet = _get_or_create_worksheet(sheet_name, worksheet_title, silent)
    all_joint_keys = set()
    for seg_results in all_results.values():
        for res in seg_results:
            all_joint_keys.update(res.get("joints", {}).keys())
    
    rows_to_insert = _build_report_rows(all_results, sorted(list(all_joint_keys)))
    if is_final and len(rows_to_insert) > 0:
        rows_to_insert.append(["End"] * len(rows_to_insert[0]))

    worksheet.clear()
    try:
        worksheet.update(values=rows_to_insert, range_name="A1")
    except TypeError:
        worksheet.update(rows_to_insert)

    decorate(worksheet, len(rows_to_insert), len(rows_to_insert[0]))
    if not silent: print(f"\nĐã xuất báo cáo ra Google Spreadsheet thành công!\n🔗 Xem file tại: {sh.url}")

def generate_spreadsheet_report_safe(all_results, sheet_name, worksheet_title=None, silent=False, is_final=False, max_retries=5):
    for attempt in range(max_retries):
        try:
            generate_spreadsheet_report(all_results, sheet_name, worksheet_title, silent, is_final)
            return
        except gspread.exceptions.APIError as e:
            if hasattr(e, 'response') and e.response.status_code == 429:
                wait_time = (attempt + 1) * 15 
                print(f"\n[!] Vượt quá giới hạn Google API (429). Đang chờ {wait_time}s trước khi thử lại...")
                time.sleep(wait_time)
            else:
                raise e
    print("[-] Đã thử lại nhiều lần nhưng không thể ghi Google Sheet do nghẽn API.")

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

def _parse_pipeline_results(config: dict, current_set: str, camA_id: str, camB_id: str, seg_name: str) -> dict:
    eval_dir = Path(config["paths"]["evaluation_output_dir"])
    t_pref = "fusion-learnable" if config.get("learnable", {}).get("enabled", True) else "fused"
    belief_cfg = config["fusion"]["belief"]
    
    mpjpe, m_jts = parse_detailed_csv(eval_dir / "MPJPE_cam1.csv", t_pref)
    pa_mpjpe, pa_jts = parse_detailed_csv(eval_dir / "PA-MPJPE_cam1.csv", t_pref)
    mble_csv = eval_dir / "MBLE_cam1.csv"
    mble = parse_frame_metric_csv(mble_csv, t_pref, "Frame_MBLE_mm")
    fusion_mble = parse_frame_metric_csv(mble_csv, "fused", "Frame_MBLE_mm")
    le_mble = parse_frame_metric_csv(mble_csv, "only_learnable", "Frame_MBLE_mm")
    old_mble = parse_frame_metric_csv(mble_csv, "posed", "Frame_MBLE_mm")
    accel = parse_frame_metric_csv(eval_dir / "Accel_cam1.csv", t_pref, "Frame_Accel_Error_mm_frame2")
    accel_csv = eval_dir / "Accel_cam1.csv"
    fusion_accel_error = parse_frame_metric_csv(accel_csv, "fused", "Frame_Accel_Error_mm_frame2")
    le_accel_error = parse_frame_metric_csv(accel_csv, "only_learnable", "Frame_Accel_Error_mm_frame2")
    # "posed" is reconstructed directly from the synchronized WHAM PKL pose/trans/betas.
    old_accel_error = parse_frame_metric_csv(accel_csv, "posed", "Frame_Accel_Error_mm_frame2")
    gt_accel_error = 0.0 if old_accel_error != float('inf') else float('inf')
    old_m, _ = parse_detailed_csv(eval_dir / "MPJPE_cam1.csv", "posed")
    old_pa, _ = parse_detailed_csv(eval_dir / "PA-MPJPE_cam1.csv", "posed")
    
    pd_m = (old_m - mpjpe)*100/old_m if (old_m != float('inf') and mpjpe != float('inf')) else 0.0
    pd_pa = (old_pa - pa_mpjpe)*100/old_pa if (old_pa != float('inf') and pa_mpjpe != float('inf')) else 0.0
    b1, b2, occluded_master, occluded_slave = extract_local_belief(
        Path(config["paths"]["fused_output_dir"]) / "metadata"
    )
    
    joint_metrics = {f"MPJPE_{k}": v for k, v in m_jts.items()}
    joint_metrics.update({f"PA-MPJPE_{k}": v for k, v in pa_jts.items()})
    os_v, usr, ts = get_system_metadata()

    # Trích xuất ngay lập tức metrics LE từ biến môi trường của lần chạy hiện tại
    raw_env = os.environ.get("LEARNABLE_EXTRA_METRICS", "{}")
    le_data = json.loads(raw_env) if raw_env.strip() else {}
    seg_metrics = le_data.get(seg_name, le_data)
    # Cam Master (camA) luôn luôn được pipeline gán vào role "camera1"
    cam1_metrics = seg_metrics.get("camera1", {}) if isinstance(seg_metrics, dict) else {}
    le_mpjpe = cam1_metrics.get("MPJPE", "N/A")
    le_pa_mpjpe = cam1_metrics.get("PA-MPJPE", "N/A")

    return {
        "set": current_set, "master": camA_id, "supplement": camB_id,
        "alpha": belief_cfg["alpha"], "beta": belief_cfg["beta"], "mpjpe": mpjpe,
        "global_belief": belief_cfg["global"], "local_method": belief_cfg["local_method"],
        "kinematic_constraints": config["fusion"]["optimization"]["use_kinematic_constraints"],
        "loss_type": config["fusion"]["optimization"]["loss_type"],
        "optimization_enabled": config["fusion"]["optimization"]["enabled"],
        "orientation_correction": config["fusion"]["correction"]["orientation_enabled"],
        "learnable_enabled": config["learnable"]["enabled"],
        "learnable_extra_enabled": config["learnable_extra"]["enabled"],
        "pa_mpjpe": pa_mpjpe, "mble": mble, "accel": accel,
        "fusion_mble": fusion_mble, "le_mble": le_mble, "old_mble": old_mble,
        "gt_accel_error": gt_accel_error, "fusion_accel_error": fusion_accel_error,
        "le_accel_error": le_accel_error, "old_accel_error": old_accel_error,
        "le_mpjpe_master": le_mpjpe, "le_pa_mpjpe_master": le_pa_mpjpe,
        "belief_master": b1,
        "belief_slave": b2,
        "occluded_joint_frames_master": occluded_master,
        "occluded_joint_frames_slave": occluded_slave,
        "old_mpjpe": old_m, "old_pa_mpjpe": old_pa,
        "% delta_mpjpe": pd_m, "% delta_pa_mpjpe": pd_pa, "joints": joint_metrics,
        "os_version": os_v, "username": usr, "timestamp": ts
    }

def _evaluate_camera_pair(camA, camB, base_cfg, gt_dir, workspace, seg_name: str) -> dict:
    current_set = extract_set_name(camA["pkl"])
    os_v, usr, ts = get_system_metadata()
    if not (workspace / camA["pkl"]).exists() or not (workspace / camB["pkl"]).exists():
        print("Bỏ qua cặp này do thiếu file pkl đầu vào.")
        return {"set": current_set, "master": camA["id"], "supplement": camB["id"], 
                "os_version": os_v, "username": usr, "timestamp": ts}
    try:
        config = _setup_pipeline_config(base_cfg, gt_dir, camA, camB, workspace)
        run_pipeline(config, stage_override=None)
        res = _parse_pipeline_results(config, current_set, camA["id"], camB["id"], seg_name)

        d_mpjpe = res.get('% delta_mpjpe', 0.0)
        d_pa_mpjpe = res.get('% delta_pa_mpjpe', 0.0)
        le_mpjpe = res.get('le_mpjpe_master', 0.0)
        le_pa_mpjpe = res.get('le_pa_mpjpe_master', 0.0)
        mble = res.get('mble', float('inf'))
        accel = res.get('accel', float('inf'))

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[{current_time}] Kết quả {camA['id']}-{camB['id']}: "
              f"MPJPE={res['mpjpe']:.2f} (Δ {d_mpjpe:+.2f}%), "
              f"PA-MPJPE={res['pa_mpjpe']:.2f} (Δ {d_pa_mpjpe:+.2f}%), "
              f"MBLE={mble:.2f}, Accel={accel:.2f}",
              f"LE MPJPE={le_mpjpe} (LE PA-MPJPE {le_pa_mpjpe}), "
             )
        return res
    except Exception as e:
        import traceback
        print(f"Lỗi khi chạy cặp {camA['id']}-{camB['id']}: {e}")
        traceback.print_exc()
        return {"set": current_set, "master": camA["id"], "supplement": camB["id"], 
                "os_version": os_v, "username": usr, "timestamp": ts}

def _get_timed_input(timeout: int) -> str | None:
    q = queue.Queue()
    def ask():
        try: q.put(input(">> Tên file của bạn: ").strip())
        except Exception: q.put(None)
    threading.Thread(target=ask, daemon=True).start()
    try: return q.get(timeout=timeout)
    except queue.Empty: return None


def _matches_active_config(result: dict, config: dict) -> bool:
    def as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        return None

    belief = config["fusion"]["belief"]
    optimization = config["fusion"]["optimization"]
    return (
        result.get("alpha") == belief["alpha"]
        and result.get("beta") == belief["beta"]
        and as_bool(result.get("global_belief")) == belief["global"]
        and result.get("local_method") == belief["local_method"]
        and as_bool(result.get("kinematic_constraints")) == optimization["use_kinematic_constraints"]
        and result.get("loss_type") == optimization["loss_type"]
        and as_bool(result.get("optimization_enabled")) == optimization["enabled"]
        and as_bool(result.get("orientation_correction")) == config["fusion"]["correction"]["orientation_enabled"]
        and as_bool(result.get("learnable_enabled")) == config["learnable"]["enabled"]
        and as_bool(result.get("learnable_extra_enabled")) == config["learnable_extra"]["enabled"]
    )

def _archive_old_spreadsheet(default_name: str):
    print(f"\n[+] Đang kiểm tra và lưu trữ file mặc định cũ '{default_name}'...")
    try:
        sh = get_gspread_client().open(default_name)
        time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if getattr(sh, 'creationTime', None):
                time_suffix = datetime.fromisoformat(str(sh.creationTime).replace('Z', '')).strftime("%Y%m%d_%H%M%S")
        except Exception: pass
        archived_name = f"{default_name}_{time_suffix}"
        sh.update_title(archived_name)
        print(f"[+] Đã đổi tên Spreadsheet cũ thành: '{archived_name}'")
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"[*] Không tìm thấy file cũ '{default_name}'. Sẽ tự động tạo file mới.")
    except Exception as e:
        print(f"[-] Không thể đổi tên file cũ ({e}). Sẽ dùng file mặc định hiện tại.")

def get_spreadsheet_name_input(default_name: str = "Brute_Force_Report_Pipeline v260823", timeout: int = 10) -> str:
    print(f"\n[?] Nhập tên file Google Spreadsheet (Mặc định: '{default_name}'):")
    print("    - ENTER / Bỏ trống: Sử dụng tên mặc định")
    print("    - 'new': Đổi tên file cũ (thêm timestamp) & tạo file mới | 'now': Đặt tên vYYMMDD")
    
    user_input = _get_timed_input(timeout)
    if not user_input:
        print(f"\n[!] Quá {timeout}s không nhập liệu/bỏ trống. Dùng mặc định: '{default_name}'")
        return default_name

    cmd = user_input.lower()
    if cmd == "new":
        _archive_old_spreadsheet(default_name)
        return default_name
    elif cmd == "now":
        _, runner_name, _ = get_system_metadata()
        generated_name = f"{runner_name}_Brute_Force_Report_Pipeline v{datetime.now().strftime('%y%m%d')}"
        print(f"\n[+] Tên file tự động khởi tạo: '{generated_name}'")
        return generated_name

    print(f"\n[+] Đã ghi nhận tên file tùy chỉnh: '{user_input}'")
    return user_input

def _process_segment(seg, existing, base_cfg, ws_dir, sh_name, ws_title, all_res, current_idx, total_pairs):
    seg_name, cameras = seg["name"], seg.get("cameras", [])
    seg_total_pairs = len(cameras) * (len(cameras) - 1)
    print(f"\n=== Bắt đầu vét cạn cho Segment: {seg_name} ({seg_total_pairs} cặp) ===")
    
    results = []
    last_update_time = time.time()
    
    for cA, cB in itertools.permutations(cameras, 2):
        current_idx += 1
        print(f"\n--- [Tiến trình: {current_idx}/{total_pairs}] Master={cA['id']} | Supplement={cB['id']} ---")

        if (seg_name, cA["id"], cB["id"]) in existing:
            res = existing[(seg_name, cA["id"], cB["id"])]
            if res.get("mpjpe", float('inf')) != float('inf'):
                print(f"[Bỏ qua] Cặp {cA['id']}-{cB['id']} của {seg_name} đã chạy xong trước đó (MPJPE={res.get('mpjpe')}). Giữ kết quả cũ.")
                res.update({"set": res.get("set", extract_set_name(cA["pkl"])), "master": cA["id"], "supplement": cB["id"]})
                results.append(res)
                continue
            else:
                print(f"[Thử lại] Cặp {cA['id']}-{cB['id']} từng bị lỗi ở lần chạy trước. Đang tiến hành chạy lại...")
        
        res = _evaluate_camera_pair(cA, cB, base_cfg, str(ws_dir / seg["ground_truth_dir"]), ws_dir, seg_name)
        results.append(res)
        
        current_time = time.time()
        if current_time - last_update_time > 30:
            temp_res = dict(all_res)
            temp_res[seg_name] = sorted(
                results, 
                key=lambda x: x.get("% delta_mpjpe", float('-inf')) + x.get("% delta_pa_mpjpe", float('-inf')), 
                reverse=True
            )
            generate_spreadsheet_report_safe(temp_res, sh_name, ws_title, silent=True)
            last_update_time = current_time

    sorted_results = sorted(
        results, 
        key=lambda x: x.get("% delta_mpjpe", float('-inf')) + x.get("% delta_pa_mpjpe", float('-inf')), 
        reverse=True
    )
    return sorted_results, current_idx

def run_brute_force():
    WS_DIR = Path(__file__).parent.resolve()
    with open(WS_DIR / "configs/brute_force.yml", "r", encoding="utf-8") as f: brute_cfg = yaml.safe_load(f)

    # ── Nạp biến môi trường từ tên notebook TRƯỚC khi load config ──────────────
    # Đây là bước bắt buộc: pipeline.yml dùng ${VAR:-default} nên phải set
    # os.environ TRƯỚC khi custom_yaml.load() được gọi bên trong load_config().
    nb_name = get_notebook_name()
    if nb_name:
        print(f"[ENV] Detecting notebook: '{nb_name}'")
        set_env_from_filename(nb_name)
    else:
        print("[ENV] WARNING: Khong the lay ten notebook. Cac tham so se dung gia tri default trong pipeline.yml.")
        print("[ENV] De fix: set os.environ['NOTEBOOK_NAME'] = '<ten_notebook>' truoc khi goi run_brute_force().")

    # Gọi hàm load_config sau khi env vars đã sẵn sàng
    base_cfg = load_config(WS_DIR / "configs/pipeline.yml")
    # Kiểm tra xem file config đã nhận đúng giá trị chưa
    print("Alpha trong config:", base_cfg['fusion']['belief']['alpha'])
    print("Beta trong config:", base_cfg['fusion']['belief']['beta'])
    print("Global belief trong config:", base_cfg['fusion']['belief']['global'])
    print("Local method trong config:", base_cfg['fusion']['belief']['local_method'])
    print("Kinematic constraints trong config:", base_cfg['fusion']['optimization']['use_kinematic_constraints'])
    print("Loss type trong config:", base_cfg['fusion']['optimization']['loss_type'])
    print("Optimization enabled trong config:", base_cfg['fusion']['optimization']['enabled'])
    print("Orientation correction trong config:", base_cfg['fusion']['correction']['orientation_enabled'])
    print("Learnable trong config:", base_cfg['learnable']['enabled'])
    print("Learnable extra trong config:", base_cfg['learnable_extra']['enabled'])
    
    _, runner_name, _ = get_system_metadata()
    default_sh_name = f"{runner_name}_brute_force_pipeline"
    sh_name = get_spreadsheet_name_input(default_name=default_sh_name, timeout=10)
    
    existing, existing_ws_title, has_end_marker = load_existing_spreadsheet_results(sh_name)
    if existing and not all(_matches_active_config(result, base_cfg) for result in existing.values()):
        print("[!] Config hiện tại khác worksheet chưa hoàn thành. Tạo worksheet mới để không trộn kết quả cũ.")
        existing = {}
        existing_ws_title = None
        has_end_marker = False
    if existing_ws_title and not has_end_marker:
        ws_title = existing_ws_title
        print(f"[+] Worksheet (cell/tab) gần nhất '{ws_title}' chưa hoàn thành (chưa có dấu END). Sẽ tiếp tục ghi bổ sung vào worksheet này.")
    else:
        new_ws_title = f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if existing_ws_title and has_end_marker:
            print(f"[+] Worksheet (cell/tab) gần nhất '{existing_ws_title}' đã hoàn tất (có dấu END). Tạo worksheet mới: '{new_ws_title}'.")
            existing = {}
        else:
            print(f"[+] Khởi tạo worksheet (cell/tab) mới: '{new_ws_title}' trong file Google Sheets '{sh_name}'.")
        ws_title = new_ws_title

    all_res = {}
    for (s_name, master, supplement), res in existing.items():
        if s_name not in all_res:
            all_res[s_name] = []
        all_res[s_name].append(res)

    total_pairs = sum(
        len(seg.get("cameras", [])) * (len(seg.get("cameras", [])) - 1)
        for seg in brute_cfg.get("segments", [])
        if len(seg.get("cameras", [])) >= 2
    )
    current_pair_idx = 0 

    for seg in brute_cfg.get("segments", []):
        if len(seg.get("cameras", [])) < 2: continue
        
        sorted_res, current_pair_idx = _process_segment(
            seg, existing, base_cfg, WS_DIR, sh_name, ws_title, all_res, current_pair_idx, total_pairs
        )
        all_res[seg["name"]] = sorted_res
        
    generate_spreadsheet_report_safe(all_res, sh_name, ws_title, silent=False, is_final=True)

if __name__ == "__main__":
    run_brute_force()

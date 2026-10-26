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

HAS_GSPREAD = False
try:
    from google.colab import auth
    from google.auth import default
    import gspread
    HAS_GSPREAD = True
except (ImportError, Exception):
    print("[Local Mode] Chạy môi trường Local (không dùng Google Sheets / Colab). Báo cáo xuất ra file CSV local.")

from config_loader import load_config, absolutize_config_paths
from pipeline import run_pipeline

GC_CLIENT = None

def get_gspread_client():
    global GC_CLIENT
    if not HAS_GSPREAD:
        return None
    if GC_CLIENT is None:
        try:
            auth.authenticate_user()
            creds, _ = default()
            GC_CLIENT = gspread.authorize(creds)
        except Exception:
            return None
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
    if "Segments" in parts:
        try:
            idx = parts.index("Segments")
            if idx >= 1:
                return parts[idx - 1]
        except ValueError:
            pass
    return "Seq1"

def _resolve_input_path(workspace: Path, path_str: str | Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = workspace / p
    if p.exists():
        return p
    clean_str = str(path_str).replace("imageSequence/", "").replace("imageSequence\\", "")
    p_clean = Path(clean_str)
    if not p_clean.is_absolute():
        p_clean = workspace / p_clean
    if p_clean.exists():
        return p_clean
    return p

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
            if row and row[0] == "AVERAGE":
                val = row[target_idx]
                if val in ("N/A", "", "nan"): return float('inf')
                try: return float(val)
                except ValueError: return float('inf')
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
            elif h.startswith(f"{prefix}_") and h.endswith("_mm") and "priority" not in h and not h.endswith(("_occ_mm", "_vis_mm")):
                j_indices[h[len(f"{prefix}_") : -len("_mm")]] = i
        if t_idx == -1: return float('inf'), {}
        for row in reader:
            if row and row[0] == "AVERAGE":
                j_dict = {name: float(row[idx]) for name, idx in j_indices.items() if row[idx] and row[idx] != "N/A"}
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

def extract_local_belief(metadata_dir: Path) -> tuple[str, str]:
    if not metadata_dir.exists(): return "[]", "[]"
    c1_acc, c2_acc, count = {}, {}, 0
    for meta_file in metadata_dir.glob("fused_data_*.json"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                conf = json.load(f).get("joint_confidence", {})
                c1, c2 = conf.get("camera1", {}), conf.get("camera2", {})
                if isinstance(c1, list): c1 = {str(i): v for i, v in enumerate(c1)}
                if isinstance(c2, list): c2 = {str(i): v for i, v in enumerate(c2)}
                if c1:
                    for k, v in c1.items(): c1_acc[k] = c1_acc.get(k, 0.0) + v
                if c2:
                    for k, v in c2.items(): c2_acc[k] = c2_acc.get(k, 0.0) + v
            count += 1
        except Exception: pass
    if count == 0: return "[]", "[]"
    def sort_key(k): return int(k) if str(k).isdigit() else k
    b1_list = [round(c1_acc[k] / count, 2) for k in sorted(c1_acc.keys(), key=sort_key)]
    b2_list = [round(c2_acc[k] / count, 2) for k in sorted(c2_acc.keys(), key=sort_key)]
    return str(b1_list), str(b2_list)

def _get_sheet_data(sheet_name: str) -> tuple[list, list]:
    try:
        gc = get_gspread_client()
        sh = gc.open(sheet_name)
        latest_worksheet = sh.worksheets()[-1] 
        data = latest_worksheet.get_all_values()
        if not data or len(data) < 2: return [], []
        return data[0], data[1:]
    except Exception: 
        return [], []

def _get_header_indices(header: list) -> dict:
    aliases = {
        'Set': ['Set'],
        'Segment': ['Segment'],
        'Rank': ['Rank'],
        'Cam Master': ['Cam Master', 'Master'],
        'Cam Slave': ['Cam Slave', 'Supplement'],
        'Alpha': ['Alpha'],
        'Beta': ['Beta'],
        # MPJPE Group
        'mpjpe.all_baseline': ['mpjpe.all_baseline', 'Old MPJPE'],
        'mpjpe.all_method': ['mpjpe.all_method', 'MPJPE'],
        'mpjpe.all_le': ['mpjpe.all_le', 'LE MPJPE Master', 'LE MPJPE'],
        '% delta_mpjpe': ['% delta_mpjpe', '% Δ_MPJPE', '% d_MPJPE%'],
        'mpjpe.occ_baseline': ['mpjpe.occ_baseline', 'Old MPJPE_occ'],
        'mpjpe.occ_method': ['mpjpe.occ_method', 'MPJPE_occ'],
        'mpjpe.vis_baseline': ['mpjpe.vis_baseline', 'Old MPJPE_vis'],
        'mpjpe.vis_method': ['mpjpe.vis_method', 'MPJPE_vis'],
        # PA-MPJPE Group
        'pa-mpjpe.all_baseline': ['pa-mpjpe.all_baseline', 'Old PA-MPJPE'],
        'pa-mpjpe.all_method': ['pa-mpjpe.all_method', 'PA-MPJPE'],
        'pa-mpjpe.all_le': ['pa-mpjpe.all_le', 'LE PA-MPJPE Master', 'LE PA-MPJPE'],
        '% delta_pa-mpjpe': ['% delta_pa-mpjpe', '% Δ_PA-MPJPE', 'd_PA-MPJPE'],
        'pa-mpjpe.occ_baseline': ['pa-mpjpe.occ_baseline', 'Old PA-MPJPE_occ'],
        'pa-mpjpe.occ_method': ['pa-mpjpe.occ_method', 'PA-MPJPE_occ'],
        'pa-mpjpe.vis_baseline': ['pa-mpjpe.vis_baseline', 'Old PA-MPJPE_vis'],
        'pa-mpjpe.vis_method': ['pa-mpjpe.vis_method', 'PA-MPJPE_vis'],
        # MBLE Group
        'mble.baseline': ['mble.baseline', 'Old MBLE'],
        'mble.method': ['mble.method', 'MBLE', 'Fusion MBLE'],
        'mble.le': ['mble.le', 'LE MBLE'],
        '% delta_mble': ['% delta_mble', '% delta-mble', '% Δ_MBLE', 'd_MBLE'],
        # Accel Group
        'accel.baseline': ['accel.baseline', 'Old Accel Error (mm/frame^2)', 'Old Accel Error'],
        'accel.method': ['accel.method', 'Accel Error (mm/frame^2)', 'Fusion Accel Error (mm/frame^2)', 'Accel'],
        'accel.le': ['accel.le', 'LE Accel Error (mm/frame^2)', 'LE Accel Error'],
        '% delta_accel': ['% delta_accel', '% delta-accel', '% Δ_Accel', 'd_Accel'],
        # Belief Group
        'belief Master': ['belief Master', 'local_belief Master'],
        'belief Slave': ['belief Slave', 'local_belief Slave'],
        # Metadata
        'Timestamp': ['Timestamp'],
        'OS Version': ['OS Version'],
        'Username': ['Username'],
    }
    idx = {}
    for canonical, candidates in aliases.items():
        found = -1
        for cand in candidates:
            if cand in header:
                found = header.index(cand)
                break
        idx[canonical] = found
    idx['joints'] = {
        h: i for i, h in enumerate(header)
        if (h.startswith(("MPJPE_", "PA-MPJPE_", "mpjpe.", "pa-mpjpe."))
            and not h.endswith(("_occ", "_vis", "_baseline", "_method", "_le", "_all")))
    }
    return idx

def _parse_history_row(row: list, idx: dict) -> tuple:
    if idx.get('Segment', -1) == -1 or idx['Segment'] >= len(row): return None, None
    if row[idx['Segment']] == "End": return None, None
    if idx.get('mpjpe.all_method', -1) != -1 and row[idx['mpjpe.all_method']] in ("N/A", "", "inf"): return None, None
    
    def get_val(col, default="N/A"):
        return row[idx[col]] if idx.get(col, -1) != -1 and idx[col] < len(row) else default
    def sf(col_name): 
        v = row[idx[col_name]] if idx.get(col_name, -1) != -1 and idx[col_name] < len(row) else "N/A"
        v = str(v).strip().replace(',', '.')
        return float(v) if v not in ("N/A", "") else float('inf')
    
    key = (row[idx['Segment']], row[idx['Cam Master']], row[idx['Cam Slave']])
    res = {
        "set": get_val('Set', "Unknown_Set"),
        "master": get_val('Cam Master', "N/A"),
        "supplement": get_val('Cam Slave', "N/A"),
        "alpha": sf('Alpha'), "beta": sf('Beta'),
        "mpjpe": sf('mpjpe.all_method'),
        "mpjpe_occ": sf('mpjpe.occ_method'),
        "mpjpe_vis": sf('mpjpe.vis_method'),
        "pa_mpjpe": sf('pa-mpjpe.all_method'),
        "pa_mpjpe_occ": sf('pa-mpjpe.occ_method'),
        "pa_mpjpe_vis": sf('pa-mpjpe.vis_method'),
        "mble": sf('mble.method'),
        "accel": sf('accel.method'),
        "fusion_mble": sf('mble.method'),
        "le_mble": sf('mble.le'),
        "old_mble": sf('mble.baseline'),
        "fusion_accel_error": sf('accel.method'),
        "le_accel_error": sf('accel.le'),
        "old_accel_error": sf('accel.baseline'),
        "le_mpjpe_master": get_val('mpjpe.all_le', "N/A"),
        "le_pa_mpjpe_master": get_val('pa-mpjpe.all_le', "N/A"),
        "belief_master": get_val('belief Master', "[]"),
        "belief_slave": get_val('belief Slave', "[]"),
        "old_mpjpe": sf('mpjpe.all_baseline'),
        "old_mpjpe_occ": sf('mpjpe.occ_baseline'),
        "old_mpjpe_vis": sf('mpjpe.vis_baseline'),
        "old_pa_mpjpe": sf('pa-mpjpe.all_baseline'),
        "old_pa_mpjpe_occ": sf('pa-mpjpe.occ_baseline'),
        "old_pa_mpjpe_vis": sf('pa-mpjpe.vis_baseline'),
        "% delta_mpjpe": (
            round((sf('mpjpe.all_baseline') - sf('mpjpe.all_method')) * 100.0 / sf('mpjpe.all_baseline'), 2)
            if sf('mpjpe.all_baseline') not in (0.0, float('inf')) and sf('mpjpe.all_method') not in (0.0, float('inf'))
            else (sf('% delta_mpjpe') if sf('% delta_mpjpe') != float('inf') else 0.0)
        ),
        "% delta_pa_mpjpe": (
            round((sf('pa-mpjpe.all_baseline') - sf('pa-mpjpe.all_method')) * 100.0 / sf('pa-mpjpe.all_baseline'), 2)
            if sf('pa-mpjpe.all_baseline') not in (0.0, float('inf')) and sf('pa-mpjpe.all_method') not in (0.0, float('inf'))
            else (sf('% delta_pa-mpjpe') if sf('% delta_pa-mpjpe') != float('inf') else 0.0)
        ),
        "% delta_accel": (
            round((sf('accel.baseline') - sf('accel.method')) * 100.0 / sf('accel.baseline'), 2)
            if sf('accel.baseline') not in (0.0, float('inf')) and sf('accel.method') not in (0.0, float('inf'))
            else (sf('% delta_accel') if sf('% delta_accel') != float('inf') else 0.0)
        ),
        "% delta_mble": (
            round((sf('mble.baseline') - sf('mble.method')) * 100.0 / sf('mble.baseline'), 2)
            if sf('mble.baseline') not in (0.0, float('inf')) and sf('mble.method') not in (0.0, float('inf'))
            else (sf('% delta_mble') if sf('% delta_mble') != float('inf') else 0.0)
        ),
        "timestamp": get_val('Timestamp'),
        "os_version": get_val('OS Version'),
        "username": get_val('Username'),
        "joints": {jn: float(str(row[ji]).strip().replace(',', '.')) for jn, ji in idx['joints'].items() if ji < len(row) and row[ji] not in ("N/A", "")}
    }
    return key, res

def load_existing_spreadsheet_results(sheet_name: str) -> dict:
    existing = {}
    if HAS_GSPREAD:
        try:
            header, rows = _get_sheet_data(sheet_name)
            if header:
                idx = _get_header_indices(header)
                required = (
                    'Segment', 'Alpha', 'Beta', 'mble.baseline', 'mble.method',
                    'accel.baseline', 'accel.method',
                )
                if not any(idx.get(column, -1) == -1 for column in required):
                    for row in rows:
                        key, res = _parse_history_row(row, idx)
                        if key and key[0] != "N/A": existing[key] = res
                    return existing
        except Exception:
            pass

    ws_dir = Path(__file__).parent.resolve()
    local_csv = ws_dir / "output" / "brute_force_report.csv"
    if local_csv.exists():
        try:
            with open(local_csv, "r", encoding="utf-8-sig") as f:
                reader = list(csv.reader(f))
            if len(reader) >= 2:
                header, rows = reader[0], reader[1:]
                idx = _get_header_indices(header)
                for row in rows:
                    key, res = _parse_history_row(row, idx)
                    if key and key[0] != "N/A": existing[key] = res
                print(f"[Local] Đã tải {len(existing)} kết quả cũ từ {local_csv.name}")
        except Exception:
            pass
    return existing

def _build_report_rows(all_results: dict, joint_keys: list) -> list:
    header = [
        'Set', 'Segment', 'Rank', 'Cam Master', 'Cam Slave', 'Alpha', 'Beta',
        # Key Improvement Deltas (placed side-by-side)
        '% delta_mpjpe', '% delta_pa-mpjpe', '% delta_accel', '% delta_mble',
        # MPJPE Group
        'mpjpe.all_baseline', 'mpjpe.all_method', 'mpjpe.all_le',
        'mpjpe.occ_baseline', 'mpjpe.occ_method',
        'mpjpe.vis_baseline', 'mpjpe.vis_method',
        # PA-MPJPE Group
        'pa-mpjpe.all_baseline', 'pa-mpjpe.all_method', 'pa-mpjpe.all_le',
        'pa-mpjpe.occ_baseline', 'pa-mpjpe.occ_method',
        'pa-mpjpe.vis_baseline', 'pa-mpjpe.vis_method',
        # MBLE Group
        'mble.baseline', 'mble.method', 'mble.le',
        # Accel Group
        'accel.baseline', 'accel.method', 'accel.le',
        # Belief Group
        'belief Master', 'belief Slave',
        # Metadata
        'Timestamp', 'OS Version', 'Username'
    ] + joint_keys
    rows = [header]
    
    def fmt(v):
        if v is None or v == float('inf') or v == "N/A":
            return "N/A"
        try:
            return round(float(v), 2)
        except (ValueError, TypeError):
            return str(v)
    
    for seg_name, results in all_results.items():
        valid_results = [r for r in results if r and r.get('mpjpe', float('inf')) != float('inf')]
        for rank, res in enumerate(valid_results, start=1):
            row = [
                res.get('set', 'Unknown_Set'), seg_name, rank, res['master'], res.get('supplement', 'N/A'),
                res.get('alpha', 'N/A') if res.get('alpha') != float('inf') else "N/A",
                res.get('beta', 'N/A') if res.get('beta') != float('inf') else "N/A",
                # Key Improvement Deltas
                fmt(res.get('% delta_mpjpe', 0.0)),
                fmt(res.get('% delta_pa-mpjpe', res.get('% delta_pa_mpjpe', 0.0))),
                fmt(res.get('% delta_accel', res.get('% delta-accel', 0.0))),
                fmt(res.get('% delta_mble', res.get('% delta-mble', 0.0))),
                # MPJPE Group
                fmt(res.get('old_mpjpe', float('inf'))),
                fmt(res.get('mpjpe', float('inf'))),
                res.get('le_mpjpe_master', 'N/A'),
                fmt(res.get('old_mpjpe_occ', float('inf'))),
                fmt(res.get('mpjpe_occ', float('inf'))),
                fmt(res.get('old_mpjpe_vis', float('inf'))),
                fmt(res.get('mpjpe_vis', float('inf'))),
                # PA-MPJPE Group
                fmt(res.get('old_pa_mpjpe', float('inf'))),
                fmt(res.get('pa_mpjpe', float('inf'))),
                res.get('le_pa_mpjpe_master', 'N/A'),
                fmt(res.get('old_pa_mpjpe_occ', float('inf'))),
                fmt(res.get('pa_mpjpe_occ', float('inf'))),
                fmt(res.get('old_pa_mpjpe_vis', float('inf'))),
                fmt(res.get('pa_mpjpe_vis', float('inf'))),
                # MBLE Group
                fmt(res.get('old_mble', float('inf'))),
                fmt(res.get('mble', float('inf'))),
                fmt(res.get('le_mble', float('inf'))),
                # Accel Group
                fmt(res.get('old_accel_error', float('inf'))),
                fmt(res.get('accel', float('inf'))),
                fmt(res.get('le_accel_error', float('inf'))),
                # Belief Group
                res.get('belief_master', "[]"),
                res.get('belief_slave', "[]"),
                # Metadata
                res.get('timestamp', 'N/A'),
                res.get('os_version', 'N/A'),
                res.get('username', 'N/A'),
            ]
            row.extend([fmt(res.get("joints", {}).get(jk, float('inf'))) for jk in joint_keys])
            rows.append(row)
            
    return rows

def _get_or_create_worksheet(sheet_name: str, worksheet_title: str = None, silent: bool = False):
    if not HAS_GSPREAD:
        return None, None
    gc = get_gspread_client()
    if gc is None:
        return None, None
    try:
        sh = gc.open(sheet_name)
    except Exception:
        try:
            sh = gc.create(sheet_name)
            try:
                sh.share('', perm_type='anyone', role='reader')
                if not silent: print(f"[+] Đã tạo Spreadsheet '{sheet_name}' và cấp quyền Public.")
            except Exception: pass
        except Exception as e:
            if not silent: print(f"[-] Không thể tạo Google Sheet: {e}")
            return None, None

    if worksheet_title:
        try:
            worksheet = sh.worksheet(worksheet_title)
        except Exception:
            try:
                worksheet = sh.add_worksheet(title=worksheet_title, rows="1000", cols="50")
            except Exception:
                worksheet = sh.sheet1
    else:
        worksheet = sh.sheet1
    return sh, worksheet

def generate_spreadsheet_report(all_results, sheet_name, worksheet_title=None, silent=False, is_final=False):
    all_joint_keys = set()
    for seg_results in all_results.values():
        for res in seg_results:
            all_joint_keys.update(res.get("joints", {}).keys())
    
    rows_to_insert = _build_report_rows(all_results, sorted(list(all_joint_keys)))
    if is_final and len(rows_to_insert) > 0:
        rows_to_insert.append(["End"] * len(rows_to_insert[0]))

    ws_dir = Path(__file__).parent.resolve()
    csv_path = ws_dir / "output" / "brute_force_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_insert)
        if not silent:
            print(f"\n[Local] Đã lưu báo cáo CSV: {csv_path.resolve()}")
    except Exception as e:
        if not silent: print(f"[-] Lỗi khi ghi CSV local: {e}")

    if not HAS_GSPREAD:
        return

    try:
        sh, worksheet = _get_or_create_worksheet(sheet_name, worksheet_title, silent)
        if not worksheet:
            return
        worksheet.clear()
        try:
            worksheet.update(values=rows_to_insert, range_name="A1")
        except TypeError:
            worksheet.update(rows_to_insert)

        decorate(worksheet, len(rows_to_insert), len(rows_to_insert[0]))
        if not silent: print(f"\nĐã xuất báo cáo ra Google Spreadsheet thành công!\n🔗 Xem file tại: {sh.url}")
    except Exception as e:
        if not silent: print(f"[-] Không thể ghi Google Sheet: {e}. Báo cáo đã lưu ở CSV local.")

def generate_spreadsheet_report_safe(all_results, sheet_name, worksheet_title=None, silent=False, is_final=False, max_retries=3):
    try:
        generate_spreadsheet_report(all_results, sheet_name, worksheet_title, silent, is_final)
    except Exception as e:
        if not silent: print(f"[-] Lỗi khi tạo báo cáo: {e}")

def _setup_pipeline_config(base_cfg, gt_dir, camA, camB, workspace):
    import copy
    import torch
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("inputs", {})
    cfg["inputs"].update({
        "ground_truth_dir": str(_resolve_input_path(workspace, gt_dir)),
        "cam1_pkl": str(_resolve_input_path(workspace, camA["pkl"])),
        "camera1_video": str(_resolve_input_path(workspace, camA["video"])),
        "cam2_pkl": str(_resolve_input_path(workspace, camB["pkl"])),
        "camera2_video": str(_resolve_input_path(workspace, camB["video"]))
    })
    cfg.setdefault("visualization", {})["enabled"] = False
    cfg.setdefault("runtime", {})["clean_output"] = True
    cfg["runtime"]["stage"] = "default"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.setdefault("learnable", {})["device"] = device
    cfg.setdefault("learnable_extra", {})["device"] = device
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
    
    mpjpe_occ = parse_average_csv(eval_dir / "MPJPE_cam1.csv", f"{t_pref}_occ_mm")
    mpjpe_vis = parse_average_csv(eval_dir / "MPJPE_cam1.csv", f"{t_pref}_vis_mm")
    pa_mpjpe_occ = parse_average_csv(eval_dir / "PA-MPJPE_cam1.csv", f"{t_pref}_occ_mm")
    pa_mpjpe_vis = parse_average_csv(eval_dir / "PA-MPJPE_cam1.csv", f"{t_pref}_vis_mm")

    old_mpjpe_occ = parse_average_csv(eval_dir / "MPJPE_cam1.csv", "posed_occ_mm")
    old_mpjpe_vis = parse_average_csv(eval_dir / "MPJPE_cam1.csv", "posed_vis_mm")
    old_pa_mpjpe_occ = parse_average_csv(eval_dir / "PA-MPJPE_cam1.csv", "posed_occ_mm")
    old_pa_mpjpe_vis = parse_average_csv(eval_dir / "PA-MPJPE_cam1.csv", "posed_vis_mm")

    pd_m = (old_m - mpjpe)*100/old_m if (old_m != float('inf') and mpjpe != float('inf')) else 0.0
    pd_pa = (old_pa - pa_mpjpe)*100/old_pa if (old_pa != float('inf') and pa_mpjpe != float('inf')) else 0.0
    pd_accel = (old_accel_error - accel)*100/old_accel_error if (old_accel_error not in (float('inf'), 0.0) and accel != float('inf')) else 0.0
    pd_mble = (old_mble - mble)*100/old_mble if (old_mble not in (float('inf'), 0.0) and mble != float('inf')) else 0.0
    b1, b2 = extract_local_belief(Path(config["paths"]["fused_output_dir"]) / "metadata")
    
    joint_metrics = {f"MPJPE_{k}": v for k, v in m_jts.items()}
    joint_metrics.update({f"PA-MPJPE_{k}": v for k, v in pa_jts.items()})
    os_v, usr, ts = get_system_metadata()

    # Trích xuất metrics LE: ưu tiên đọc trực tiếp từ evaluation CSV, fallback sang os.environ
    le_m_val = parse_average_csv(eval_dir / "MPJPE_cam1.csv", "only_learnable_priority1_mm")
    le_pa_val = parse_average_csv(eval_dir / "PA-MPJPE_cam1.csv", "only_learnable_priority1_mm")
    raw_env = os.environ.get("LEARNABLE_EXTRA_METRICS", "{}")
    le_data = json.loads(raw_env) if raw_env.strip() else {}
    seg_metrics = le_data.get(seg_name, le_data)
    cam1_metrics = seg_metrics.get("camera1", {}) if isinstance(seg_metrics, dict) else {}
    le_mpjpe = f"{le_m_val:.2f}" if le_m_val != float('inf') else cam1_metrics.get("MPJPE", "N/A")
    le_pa_mpjpe = f"{le_pa_val:.2f}" if le_pa_val != float('inf') else cam1_metrics.get("PA-MPJPE", "N/A")

    return {
        "set": current_set, "master": camA_id, "supplement": camB_id,
        "alpha": belief_cfg["alpha"], "beta": belief_cfg["beta"], "mpjpe": mpjpe,
        "mpjpe_occ": mpjpe_occ, "mpjpe_vis": mpjpe_vis,
        "pa_mpjpe": pa_mpjpe, "pa_mpjpe_occ": pa_mpjpe_occ, "pa_mpjpe_vis": pa_mpjpe_vis,
        "mble": mble, "accel": accel,
        "fusion_mble": fusion_mble, "le_mble": le_mble, "old_mble": old_mble,
        "gt_accel_error": gt_accel_error, "fusion_accel_error": fusion_accel_error,
        "le_accel_error": le_accel_error, "old_accel_error": old_accel_error,
        "le_mpjpe_master": le_mpjpe, "le_pa_mpjpe_master": le_pa_mpjpe,
        "belief_master": b1,
        "belief_slave": b2, "old_mpjpe": old_m, "old_mpjpe_occ": old_mpjpe_occ, "old_mpjpe_vis": old_mpjpe_vis,
        "old_pa_mpjpe": old_pa, "old_pa_mpjpe_occ": old_pa_mpjpe_occ, "old_pa_mpjpe_vis": old_pa_mpjpe_vis,
        "% delta_mpjpe": pd_m, "% delta_pa_mpjpe": pd_pa, "% delta_pa-mpjpe": pd_pa,
        "% delta_accel": pd_accel, "% delta-accel": pd_accel,
        "% delta_mble": pd_mble, "% delta-mble": pd_mble,
        "joints": joint_metrics,
        "os_version": os_v, "username": usr, "timestamp": ts
    }

def _evaluate_camera_pair(camA, camB, base_cfg, gt_dir, workspace, seg_name: str) -> dict:
    current_set = extract_set_name(camA["pkl"])
    os_v, usr, ts = get_system_metadata()
    camA_pkl = _resolve_input_path(workspace, camA["pkl"])
    camB_pkl = _resolve_input_path(workspace, camB["pkl"])
    if not camA_pkl.exists() or not camB_pkl.exists():
        print(f"Bỏ qua cặp {camA['id']}-{camB['id']} do thiếu file pkl đầu vào.")
        return None
    try:
        config = _setup_pipeline_config(base_cfg, gt_dir, camA, camB, workspace)
        run_pipeline(config, stage_override=None)
        res = _parse_pipeline_results(config, current_set, camA["id"], camB["id"], seg_name)

        d_mpjpe = res.get('% delta_mpjpe', 0.0)
        d_pa_mpjpe = res.get('% delta_pa_mpjpe', 0.0)
        d_accel = res.get('% delta_accel', 0.0)
        d_mble = res.get('% delta_mble', 0.0)
        le_mpjpe = res.get('le_mpjpe_master', 0.0)
        le_pa_mpjpe = res.get('le_pa_mpjpe_master', 0.0)
        mble = res.get('mble', float('inf'))
        accel = res.get('accel', float('inf'))

        def _fmt(v): return f"{v:.2f}" if v != float('inf') else "N/A"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[{current_time}] Kết quả {camA['id']}-{camB['id']}: "
              f"Δ: MPJPE {d_mpjpe:+.2f}%, PA {d_pa_mpjpe:+.2f}%, Accel {d_accel:+.2f}%, MBLE {d_mble:+.2f}% | "
              f"MPJPE={res['mpjpe']:.2f}, PA-MPJPE={res['pa_mpjpe']:.2f}, Accel={accel:.2f}, MBLE={mble:.2f} | "
              f"LE MPJPE={le_mpjpe} (LE PA-MPJPE {le_pa_mpjpe})")
        return res
    except Exception as e:
        import traceback
        print(f"Lỗi khi chạy cặp {camA['id']}-{camB['id']}: {e}")
        traceback.print_exc()
        return None

def _ranking_key(result: dict) -> tuple[float, float, float]:
    mpjpe = float(result.get("% delta_mpjpe", float("-inf")))
    pa_mpjpe = float(result.get("% delta_pa_mpjpe", float("-inf")))
    accel = float(result.get("% delta_accel", float("-inf")))
    mble = float(result.get("% delta_mble", float("-inf")))
    # Pose changes within 0.1 percentage point are a tie; Accel/MBLE then break it.
    return round(min(mpjpe, pa_mpjpe), 1), round(0.6 * mpjpe + 0.4 * pa_mpjpe, 1), accel + mble


def _get_timed_input(timeout: int) -> str | None:
    q = queue.Queue()
    def ask():
        try: q.put(input(">> Tên file của bạn: ").strip())
        except Exception: q.put(None)
    threading.Thread(target=ask, daemon=True).start()
    try: return q.get(timeout=timeout)
    except queue.Empty: return None

def _archive_old_spreadsheet(default_name: str):
    if not HAS_GSPREAD:
        return
    print(f"\n[+] Đang kiểm tra và lưu trữ file mặc định cũ '{default_name}'...")
    try:
        client = get_gspread_client()
        if not client:
            return
        sh = client.open(default_name)
        time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if getattr(sh, 'creationTime', None):
                time_suffix = datetime.fromisoformat(str(sh.creationTime).replace('Z', '')).strftime("%Y%m%d_%H%M%S")
        except Exception: pass
        archived_name = f"{default_name}_{time_suffix}"
        sh.update_title(archived_name)
        print(f"[+] Đã đổi tên Spreadsheet cũ thành: '{archived_name}'")
    except Exception as e:
        print(f"[-] Không thể đổi tên file cũ ({e}).")

def get_spreadsheet_name_input(default_name: str = "Brute_Force_Report_Pipeline v260823", timeout: int = 10) -> str:
    if not HAS_GSPREAD:
        print(f"\n[Local Mode] Báo cáo sẽ được lưu trực tiếp vào: output/brute_force_report.csv")
        return default_name
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
            res.update({"set": res.get("set", extract_set_name(cA["pkl"])), "master": cA["id"], "supplement": cB["id"]})
            results.append(res)
            continue
        
        res = _evaluate_camera_pair(cA, cB, base_cfg, str(ws_dir / seg["ground_truth_dir"]), ws_dir, seg_name)
        if res is not None:
            results.append(res)
        
        temp_res = dict(all_res)
        temp_res[seg_name] = sorted(
            results, 
            key=_ranking_key,
            reverse=True
        )
        generate_spreadsheet_report_safe(temp_res, sh_name, ws_title, silent=True)

    sorted_results = sorted(
        results, 
        key=_ranking_key,
        reverse=True
    )
    return sorted_results, current_idx

def run_brute_force():
    WS_DIR = Path(__file__).parent.resolve()
    with open(WS_DIR / "configs/brute_force.yml", "r", encoding="utf-8") as f: brute_cfg = yaml.safe_load(f)
    # Gọi hàm load_config đã được "nâng cấp" để nó tự động parse biến môi trường
    base_cfg = load_config(WS_DIR / "configs/pipeline.yml")
    # Kiểm tra xem file config đã nhận đúng giá trị chưa
    print("Alpha trong config:", base_cfg['fusion']['belief']['alpha'])
    print("Beta trong config:", base_cfg['fusion']['belief']['beta'])
    
    _, runner_name, _ = get_system_metadata()
    default_sh_name = f"{runner_name}_brute_force_pipeline"
    sh_name = get_spreadsheet_name_input(default_name=default_sh_name, timeout=10)
    
    ws_title = f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    existing = load_existing_spreadsheet_results(sh_name)
    all_res = {}

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

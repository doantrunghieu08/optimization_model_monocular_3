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
EVALUATION_PROTOCOL = "no_gt_v10_master_only_postopt"
from fusion_pipeline.aotr_benchmark import (
    aotr_enabled, cache_compatible, evaluation_protocol,
    report_csv_name, verify_aotr_execution,
)
ACTIVE_EVALUATION_PROTOCOL = EVALUATION_PROTOCOL
ACTIVE_REPORT_CSV_NAME = "brute_force_report.csv"
ACTIVE_LOCAL_REPORT_DIR = None

HAS_GSPREAD = False
try:
    from google.colab import auth
    from google.auth import default
    import gspread
    HAS_GSPREAD = True
except (ImportError, Exception):
    pass  # local mode, announced by run_brute_force (not at import: --workers processes import this file too)

from config_loader import load_config, absolutize_config_paths
from pipeline import run_pipeline

GC_CLIENT = None

def _get_timed_input(timeout: int, prompt: str = ">> Tên file của bạn: ") -> str | None:
    """
    Đọc input có timeout, không tạo thread đọc stdin chạy ngầm.
    Phù hợp với môi trường terminal macOS/Linux.
    """
    import select
    import sys

    print(prompt, end="", flush=True)

    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip()
        return None
    except (OSError, ValueError, AttributeError):
        print("\n[!] Không hỗ trợ timed input trong terminal này.")
        return None
    
def _cleanup_local_input_files(
    brute_cfg: dict,
    workspace: Path,
    timeout: int = 10
):
    """
    Hỏi người dùng có muốn xóa toàn bộ pkl_local và video_local không.

    Quy tắc:
    - Nhập N hoặc n: giữ nguyên.
    - Các trường hợp còn lại, bao gồm timeout: xóa.
    - Chỉ xóa file được khai báo trong pkl_local/video_local.
    - Không xóa thư mục, symlink hoặc file khác.
    """

    print("\n" + "=" * 65)
    print("[CLEANUP] Dọn dẹp dữ liệu đầu vào sau khi brute-force hoàn tất.")
    print("Chương trình sẽ xóa các file pkl_local, video_local và tracking_result_local.")
    print(f"Nhập N hoặc n trong {timeout} giây để GIỮ LẠI các file.")
    print("Nếu không nhập hoặc nhập nội dung khác: chương trình sẽ XÓA.")
    print("=" * 65)

    choice = _get_timed_input(
        timeout,
        prompt=f">> Bạn có muốn giữ lại các file đầu vào không? [N/n để giữ, mặc định xóa] ({timeout}s): "
    )

    if choice in ("N", "n"):
        print("\n[CLEANUP] Người dùng chọn giữ lại dữ liệu.")
        print("[CLEANUP] Không có file đầu vào nào bị xóa.")
        return

    if choice is None:
        print("\n[CLEANUP] Hết thời gian chờ. Tự động xóa dữ liệu đầu vào.")
    else:
        print(f"\n[CLEANUP] Lựa chọn '{choice}' không phải N/n. Tiến hành xóa.")

    # Thu thập tất cả file PKL và video từ cấu hình.
    files_to_delete = set()

    for segment in brute_cfg.get("segments", []):
        for camera in segment.get("cameras", []):

            for key in ("pkl_local", "video_local", "tracking_result_local"):
                value = camera.get(key)

                if not value:
                    continue

                path = Path(str(value)).expanduser()

                if not path.is_absolute():
                    path = workspace / path

                # Chuẩn hóa đường dẫn nhưng không dereference symlink.
                path = Path(os.path.abspath(str(path)))

                files_to_delete.add(path)

    if not files_to_delete:
        print("[CLEANUP] Không tìm thấy pkl_local/video_local trong cấu hình.")
        return

    deleted_count = 0
    missing_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"\n[CLEANUP] Tổng số đường dẫn duy nhất: {len(files_to_delete)}")

    for path in sorted(files_to_delete, key=lambda p: str(p)):

        try:
            # Không xóa symlink để tránh ảnh hưởng dữ liệu bên ngoài.
            if path.is_symlink():
                print(f"[SKIP] Symlink: {path}")
                skipped_count += 1
                continue

            if not path.exists():
                print(f"[MISSING] {path}")
                missing_count += 1
                continue

            # Chỉ xóa file, tuyệt đối không xóa thư mục.
            if not path.is_file():
                print(f"[SKIP] Không phải file: {path}")
                skipped_count += 1
                continue

            path.unlink()

            print(f"[DELETED] {path}")
            deleted_count += 1

        except Exception as e:
            print(f"[ERROR] Không thể xóa {path}: {e}")
            failed_count += 1

    print("\n" + "=" * 65)
    print("[CLEANUP] KẾT QUẢ")
    print(f"  Đã xóa       : {deleted_count}")
    print(f"  Không tồn tại: {missing_count}")
    print(f"  Bỏ qua       : {skipped_count}")
    print(f"  Lỗi          : {failed_count}")
    print("=" * 65)

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


def _resolve_local_or_drive_path(
    workspace: Path,
    config: dict,
    remote_key: str,
    local_key: str,
    description: str = "input",
    required: bool = True,
) -> Path | None:
    """
    Ưu tiên local path nếu tồn tại.
    Nếu không có, thử original path (ví dụ đường dẫn trong workspace).
    Google Drive File ID không được coi là local path.
    """
    local_value = config.get(local_key)
    remote_value = config.get(remote_key)

    # 1. Ưu tiên đường dẫn local
    if local_value:
        local_path = Path(str(local_value)).expanduser()

        if not local_path.is_absolute():
            local_path = workspace / local_path

        if local_path.exists():
            print(f"[Input] {description}: LOCAL -> {local_path}")
            # ponytail: abspath (not resolve) keeps symlinks in their local dir alongside *_person_mapping.txt
            return Path(os.path.abspath(str(local_path)))

    # 2. Thử đường dẫn gốc
    if remote_value:
        remote_str = str(remote_value).strip()

        # File ID thường là chuỗi không chứa dấu phân cách đường dẫn.
        looks_like_drive_id = (
            len(remote_str) >= 20
            and "/" not in remote_str
            and "\\" not in remote_str
        )

        if not looks_like_drive_id:
            remote_path = _resolve_input_path(workspace, remote_str)

            if remote_path.exists():
                print(f"[Input] {description}: ORIGINAL -> {remote_path}")
                return Path(os.path.abspath(str(remote_path)))

    # 3. Không tìm thấy
    if not required:
        return None

    raise FileNotFoundError(
        f"Không tìm thấy {description}.\n"
        f"  {local_key}: {local_value}\n"
        f"  {remote_key}: {remote_value}\n"
        f"Đã kiểm tra local path và original path."
    )


def _resolve_camera_inputs(
    workspace: Path,
    camera: dict,
) -> dict:
    """Phân giải đồng thời PKL và video của một camera."""
    return {
        "pkl": _resolve_local_or_drive_path(
            workspace, camera, "pkl", "pkl_local",
            description=f"{camera['id']} PKL"
        ),
        # Video is optional (frame size only); missing -> None, never downloaded.
        "video": _resolve_local_or_drive_path(
            workspace, camera, "video", "video_local",
            description=f"{camera['id']} video", required=False,
        ),
    }

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
        'Protocol': ['Protocol'],
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
        # Written by _build_report_rows; without them a resumed run rewrote these columns as N/A.
        'Target Person': ['Target Person'],
        'Evaluated Frames': ['Evaluated Frames'],
        'AOTR corrected frames': ['AOTR corrected frames'],
        'AOTR accepted bones': ['AOTR accepted bones'],
        'AOTR fallback frames': ['AOTR fallback frames'],
        'AOTR temporal retention %': ['AOTR temporal retention %'],
        'AOTR residual accel before mm/f2': ['AOTR residual accel before mm/f2'],
        'AOTR residual accel after mm/f2': ['AOTR residual accel after mm/f2'],
        'AOTR anatomy calibrated bones': ['AOTR anatomy calibrated bones'],
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

def _parse_history_row(row: list, idx: dict, expected_protocol=EVALUATION_PROTOCOL) -> tuple:
    # expected_protocol: one protocol or a set (one per target person: aria01 / aria02 segments).
    expected = {expected_protocol} if isinstance(expected_protocol, str) else set(expected_protocol)
    if idx.get('Segment', -1) == -1 or idx['Segment'] >= len(row): return None, None
    if idx.get('Protocol', -1) == -1 or idx['Protocol'] >= len(row) or row[idx['Protocol']] not in expected: return None, None
    if row[idx['Segment']] == "End": return None, None
    if idx.get('mpjpe.all_method', -1) != -1 and row[idx['mpjpe.all_method']] in ("N/A", "", "inf"): return None, None
    
    def get_val(col, default="N/A"):
        return row[idx[col]] if idx.get(col, -1) != -1 and idx[col] < len(row) else default
    def sf(col_name): 
        v = row[idx[col_name]] if idx.get(col_name, -1) != -1 and idx[col_name] < len(row) else "N/A"
        v = str(v).strip().replace(',', '.')
        return float(v) if v not in ("N/A", "") else float('inf')
    def count(col_name):
        # Frame / bone counts come back as int (as written fresh), "N/A" when absent.
        v = sf(col_name)
        if v == float('inf'):
            return "N/A"
        return int(v) if float(v).is_integer() else v
    def delta(col_name, baseline_col, method_col):
        # The stored % delta (from unrounded metrics) first: recomputing it from the 2-decimal
        # metrics moved % delta_accel by 0.01 on resume. Recompute only when it is missing.
        stored = sf(col_name)
        if stored != float('inf'):
            return stored
        base, method = sf(baseline_col), sf(method_col)
        if base not in (0.0, float('inf')) and method not in (0.0, float('inf')):
            return round((base - method) * 100.0 / base, 2)
        return 0.0

    key = (row[idx['Segment']], row[idx['Cam Master']], row[idx['Cam Slave']])
    res = {
        "set": get_val('Set', "Unknown_Set"),
        "protocol": get_val('Protocol'),
        "evaluated_frames": count('Evaluated Frames'),
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
        "aotr_corrected_frames": count('AOTR corrected frames'),
        "aotr_accepted_bones": count('AOTR accepted bones'),
        "aotr_fallback_frames": count('AOTR fallback frames'),
        "aotr_stability_retention": sf('AOTR temporal retention %'),
        "aotr_residual_accel_before": sf('AOTR residual accel before mm/f2'),
        "aotr_residual_accel_after": sf('AOTR residual accel after mm/f2'),
        "aotr_calibrated_bones": count('AOTR anatomy calibrated bones'),
        "belief_master": get_val('belief Master', "[]"),
        "belief_slave": get_val('belief Slave', "[]"),
        "old_mpjpe": sf('mpjpe.all_baseline'),
        "old_mpjpe_occ": sf('mpjpe.occ_baseline'),
        "old_mpjpe_vis": sf('mpjpe.vis_baseline'),
        "old_pa_mpjpe": sf('pa-mpjpe.all_baseline'),
        "old_pa_mpjpe_occ": sf('pa-mpjpe.occ_baseline'),
        "old_pa_mpjpe_vis": sf('pa-mpjpe.vis_baseline'),
        "% delta_mpjpe": delta('% delta_mpjpe', 'mpjpe.all_baseline', 'mpjpe.all_method'),
        "% delta_pa_mpjpe": delta('% delta_pa-mpjpe', 'pa-mpjpe.all_baseline', 'pa-mpjpe.all_method'),
        "% delta_accel": delta('% delta_accel', 'accel.baseline', 'accel.method'),
        "% delta_mble": delta('% delta_mble', 'mble.baseline', 'mble.method'),
        "timestamp": get_val('Timestamp'),
        "os_version": get_val('OS Version'),
        "username": get_val('Username'),
        "joints": {jn: float(str(row[ji]).strip().replace(',', '.')) for jn, ji in idx['joints'].items() if ji < len(row) and row[ji] not in ("N/A", "")}
    }
    person = str(get_val('Target Person', "")).strip()
    if person and person != "N/A":  # rows damaged by older runs get it from their segment (_reuse_cached)
        res["target_person"] = person
    return key, res

def load_existing_spreadsheet_results(sheet_name: str, expected_protocol=None) -> dict:
    expected_protocol = expected_protocol or ACTIVE_EVALUATION_PROTOCOL
    existing = {}
    if ACTIVE_LOCAL_REPORT_DIR is None and HAS_GSPREAD:
        try:
            header, rows = _get_sheet_data(sheet_name)
            if header:
                idx = _get_header_indices(header)
                required = (
                    'Segment', 'Protocol', 'Alpha', 'Beta', 'mble.baseline', 'mble.method',
                    'accel.baseline', 'accel.method',
                )
                if not any(idx.get(column, -1) == -1 for column in required):
                    for row in rows:
                        key, res = _parse_history_row(row, idx, expected_protocol=expected_protocol)
                        if key and key[0] != "N/A": existing[key] = res
                    return existing
        except Exception:
            pass

    ws_dir = Path(__file__).parent.resolve()
    local_csv = (ACTIVE_LOCAL_REPORT_DIR or ws_dir / "output") / ACTIVE_REPORT_CSV_NAME
    if local_csv.exists():
        try:
            with open(local_csv, "r", encoding="utf-8-sig") as f:
                reader = list(csv.reader(f))
            if len(reader) >= 2:
                header, rows = reader[0], reader[1:]
                idx = _get_header_indices(header)
                for row in rows:
                    key, res = _parse_history_row(row, idx, expected_protocol=expected_protocol)
                    if key and key[0] != "N/A": existing[key] = res
                print(f"[Local] Đã tải {len(existing)} kết quả cũ từ {local_csv.name}")
        except Exception:
            pass
    return existing

def _build_report_rows(all_results: dict, joint_keys: list) -> list:
    header = [
        'Set', 'Segment', 'Target Person', 'Protocol', 'Rank', 'Cam Master', 'Cam Slave', 'Evaluated Frames', 'Alpha', 'Beta',
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
        # AOTR activation / improvement coverage
        'AOTR corrected frames', 'AOTR accepted bones', 'AOTR fallback frames',
        'AOTR temporal retention %', 'AOTR residual accel before mm/f2',
        'AOTR residual accel after mm/f2', 'AOTR anatomy calibrated bones',
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
                res.get('set', 'Unknown_Set'), seg_name, res.get('target_person', 'N/A'), res.get('protocol', EVALUATION_PROTOCOL),
                rank, res['master'], res.get('supplement', 'N/A'),
                res.get('evaluated_frames', 'N/A'),
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
                # AOTR diagnostics
                fmt(res.get('aotr_corrected_frames')),
                fmt(res.get('aotr_accepted_bones')),
                fmt(res.get('aotr_fallback_frames')),
                fmt(res.get('aotr_stability_retention')),
                fmt(res.get('aotr_residual_accel_before')),
                fmt(res.get('aotr_residual_accel_after')),
                fmt(res.get('aotr_calibrated_bones')),
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
    csv_path = (ACTIVE_LOCAL_REPORT_DIR or ws_dir / "output") / ACTIVE_REPORT_CSV_NAME
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_path = csv_path.with_suffix(".tmp") if ACTIVE_LOCAL_REPORT_DIR is not None else csv_path
    try:
        with write_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_insert)
        if write_path != csv_path:
            write_path.replace(csv_path)
        if not silent:
            print(f"\n[Local] Đã lưu báo cáo CSV: {csv_path.resolve()}")
    except PermissionError:
        fallback_path = csv_path.with_name(f"{csv_path.stem}_{int(time.time())}.csv")
        with open(fallback_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows_to_insert)
        if not silent:
            print(f"\n[Local] File chính bị khóa (Excel mở?). Đã lưu vào fallback: {fallback_path.resolve()}")
    except Exception as e:
        if not silent: print(f"[-] Lỗi khi ghi CSV local: {e}")

    if ACTIVE_LOCAL_REPORT_DIR is not None or not HAS_GSPREAD:
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

def _resolve_pair_mode(master_cfg: dict, master: str, slave: str) -> str:
    mode = master_cfg.get("pair_modes", {}).get(
        f"{master}->{slave}", master_cfg.get("default_mode", "aggressive")
    )
    if mode not in {"aggressive", "protected", "posed_fallback"}:
        raise ValueError(f"Unsupported fusion pair mode: {mode}")
    return mode


def _setup_pipeline_config(
    base_cfg,
    gt_dir,
    camA,
    camB,
    workspace,
    target_person=None
):
    import copy
    import torch

    cfg = copy.deepcopy(base_cfg)

    # Resolve camera inputs, ưu tiên local
    camA_inputs = _resolve_camera_inputs(workspace, camA)
    camB_inputs = _resolve_camera_inputs(workspace, camB)

    # Ground truth đã được resolve trước khi truyền vào
    gt_path = Path(gt_dir).expanduser()
    if not gt_path.is_absolute():
        gt_path = workspace / gt_path

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy Ground Truth: {gt_path}"
        )

    cfg.setdefault("inputs", {})
    cfg["inputs"].update({
        "ground_truth_dir": os.path.abspath(str(gt_path)),
        "cam1_pkl": str(camA_inputs["pkl"]),
        "camera1_video": str(camA_inputs["video"]) if camA_inputs["video"] else None,
        "cam2_pkl": str(camB_inputs["pkl"]),
        "camera2_video": str(camB_inputs["video"]) if camB_inputs["video"] else None,
    })

    for cam, entry in (("camera1", camA), ("camera2", camB)):  # YAML extrinsic (used only under EXT=1)
        cfg["inputs"].pop(f"{cam}_extrinsic", None)
        if entry.get("extrinsic"):
            cfg["inputs"][f"{cam}_extrinsic"] = entry["extrinsic"]

    if target_person is not None:
        cfg["inputs"]["target_person"] = target_person

    cfg.setdefault("visualization", {})["enabled"] = False
    cfg.setdefault("runtime", {})["clean_output"] = True
    cfg["runtime"]["stage"] = "default"

    master_cfg = cfg.setdefault(
        "fusion", {}
    ).setdefault("master_protection", {})

    pair_mode = _resolve_pair_mode(
        master_cfg, camA["id"], camB["id"]
    )

    master_cfg["enabled"] = (
        master_cfg.get("enabled", True)
        if pair_mode != "disabled" else False
    )

    cfg["fusion"]["pair_mode"] = pair_mode

    print(
        f"[Fusion] Pair mode "
        f"{camA['id']}->{camB['id']}: {pair_mode}"
    )

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

    # Verify that this was a real AOTR execution, not stale legacy output.
    aotr_summary = verify_aotr_execution(config)
    aotr_stability = (aotr_summary or {}).get("sequence_stability", {})
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
        "set": current_set, "protocol": evaluation_protocol(config),
        "target_person": config.get("inputs", {}).get("target_person", "aria01"),
        "evaluated_frames": aotr_summary.get("frames") if aotr_summary else None,
        "master": camA_id, "supplement": camB_id,
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
        "aotr_corrected_frames": aotr_summary["corrected_frames"] if aotr_summary else None,
        "aotr_accepted_bones": aotr_summary["accepted_bone_frames"] if aotr_summary else None,
        "aotr_fallback_frames": aotr_summary["fallback_frames"] if aotr_summary else None,
        "aotr_stability_retention": (
            100.0 * aotr_stability["mean_temporal_retention"]
            if "mean_temporal_retention" in aotr_stability else None
        ),
        "aotr_residual_accel_before": (
            1000.0 * aotr_stability["initial_residual_accel_m_frame2"]
            if "initial_residual_accel_m_frame2" in aotr_stability else None
        ),
        "aotr_residual_accel_after": (
            1000.0 * aotr_stability["final_residual_accel_m_frame2"]
            if "final_residual_accel_m_frame2" in aotr_stability else None
        ),
        "aotr_calibrated_bones": aotr_stability.get("anatomically_calibrated_bones"),
        "os_version": os_v, "username": usr, "timestamp": ts
    }

def _evaluate_camera_pair(camA, camB, base_cfg, gt_dir, workspace, seg_name: str, target_person=None) -> dict:
    os_v, usr, ts = get_system_metadata()
    try:
        camA_inputs = _resolve_camera_inputs(workspace, camA)
        camB_inputs = _resolve_camera_inputs(workspace, camB)

        current_set = extract_set_name(
            str(camA_inputs["pkl"])
        )

        config = _setup_pipeline_config(
            base_cfg, gt_dir, camA, camB,
            workspace, target_person=target_person
        )
        executed = run_pipeline(config, stage_override=None)
        if executed is False:
            return None
        aotr_diag = verify_aotr_execution(config)
        if aotr_diag:
            print(
                f"[AOTR] {camA['id']}->{camB['id']}: "
                f"frames={aotr_diag['frames']}, corrected={aotr_diag['corrected_frames']}, "
                f"accepted_bones={aotr_diag['accepted_bone_frames']}, "
                f"fallback={aotr_diag['fallback_frames']}, "
                f"reasons={aotr_diag.get('reason_counts', {})}"
            )
            if aotr_diag["accepted_bone_frames"] == 0:
                print("[AOTR] WARNING: No bone updates; compare against Master, inspect diagnostics.")
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
    # ponytail: rank primarily by delta PA-MPJPE (highest improvement first)
    pa_gain = float(result.get("% delta_pa_mpjpe") if result.get("% delta_pa_mpjpe") is not None else result.get("% delta_pa-mpjpe", float("-inf")))
    pa_error = float(result.get("pa_mpjpe") if result.get("pa_mpjpe") is not None else float("inf"))
    accel = float(result.get("accel") if result.get("accel") is not None else float("inf"))
    mble = float(result.get("mble") if result.get("mble") is not None else float("inf"))
    return pa_gain, -pa_error, -(accel + mble)



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
        print(f"\n[Local Mode] Báo cáo sẽ được lưu trực tiếp vào: output/{ACTIVE_REPORT_CSV_NAME}")
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

def _segment_protocol(base_cfg, seg) -> str:
    """Protocol of one segment. It includes the target person (and height), so an aria02
    segment has its own protocol; its rows are found again only with it."""
    person = seg.get("target_person")
    if not person:
        return ACTIVE_EVALUATION_PROTOCOL
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("inputs", {})["target_person"] = person
    return evaluation_protocol(cfg)


def _reuse_cached(cached: dict, seg: dict, cA: dict, cB: dict) -> dict:
    """A report row reused as the result of pair cA->cB of seg (sequential and --workers runs)."""
    cached.update({"set": cached.get("set", extract_set_name(cA["pkl"])), "master": cA["id"], "supplement": cB["id"]})
    if not cached.get("target_person") and seg.get("target_person"):
        # The protocol matched this segment's, so the row is for its target person.
        cached["target_person"] = seg["target_person"]
    return cached


def _process_segment(seg, existing, base_cfg, ws_dir, sh_name, ws_title, all_res, current_idx, total_pairs):
    seg_name, cameras = seg["name"], seg.get("cameras", [])
    seg_total_pairs = len(cameras) * (len(cameras) - 1)
    seg_protocol = _segment_protocol(base_cfg, seg)
    print(f"\n=== Bắt đầu vét cạn cho Segment: {seg_name} "
          f"(target_person={seg.get('target_person', 'N/A')}, {seg_total_pairs} cặp) ===")

    results = []
    last_update_time = time.time()

    for cA, cB in itertools.permutations(cameras, 2):
        current_idx += 1
        print(f"\n--- [Tiến trình: {current_idx}/{total_pairs}] Master={cA['id']} | Supplement={cB['id']} ---")

        cached = existing.get((seg_name, cA["id"], cB["id"]))
        if cache_compatible(cached, seg_protocol):
            results.append(_reuse_cached(cached, seg, cA, cB))
            continue
        if cached:
            print(f"[BruteForce] IGNORE stale result with protocol={cached.get('protocol')!r}")
        
        gt_dir = _resolve_local_or_drive_path(
            ws_dir,
            seg,
            "ground_truth_dir",
            "ground_truth_dir_local",
            description=f"{seg_name} Ground Truth"
        )

        res = _evaluate_camera_pair(
            cA, cB, base_cfg, str(gt_dir), ws_dir, seg_name,
            target_person=seg.get("target_person")
        )
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


# ---------------------------------------------------------------------------
# --workers N: N camera pairs at once, one process each.
# A pair is single-threaded Python (AOTR); BLAS/torch threads only add overhead
# (cam01->cam02 on M1: 41 s with 77 s CPU by default, 37.5 s with 37 s CPU on one
# thread), so the workers get one thread each. Every stage cleans and rewrites its
# paths.*_output_dir, so each worker has its own copy of those directories.
# ---------------------------------------------------------------------------
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
WORKERS_DIR_NAME = "brute_force_workers"
_WORKER = {}  # state of one worker process, set by _init_worker


def _report_dir() -> Path:
    return ACTIVE_LOCAL_REPORT_DIR or Path(__file__).parent.resolve() / "output"


def _worker_config(base_cfg: dict, worker_dir: Path) -> dict:
    """base_cfg with every paths.*_output_dir moved into worker_dir (one sub-directory per key)."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    paths = cfg.setdefault("paths", {})
    for key in list(paths):
        if key.endswith("_output_dir"):
            paths[key] = str(Path(worker_dir) / (key[:-len("_output_dir")] + "_results"))
    return cfg


def _pair_log_name(seg_name: str, cam_a: str, cam_b: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(seg_name))
    return f"{safe}__{cam_a}-{cam_b}.log"


def _init_worker(slots, root: str, base_cfg: dict, workspace: str, evaluate):
    import signal
    # Ctrl+C reaches the whole process group: only the parent handles it (and stops the workers).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    slot = slots.get()
    worker_dir = Path(root) / f"w{slot:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _WORKER.update(slot=slot, cfg=_worker_config(base_cfg, worker_dir), workspace=Path(workspace),
                   logs=Path(root) / "logs", evaluate=evaluate)


def _pool_evaluate_pair(seg_name, cam_a, cam_b, gt_dir, target_person):
    """One pair in a worker; everything the pipeline prints goes to one log file per pair."""
    import contextlib
    log_path = _WORKER["logs"] / _pair_log_name(seg_name, cam_a["id"], cam_b["id"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        res = _WORKER["evaluate"](cam_a, cam_b, _WORKER["cfg"], gt_dir, _WORKER["workspace"],
                                  seg_name, target_person=target_person)
    return res, time.perf_counter() - started, str(log_path), _WORKER["slot"]


def _plan_pairs(segments, existing, base_cfg, ws_dir):
    """({segment: cached results}, [(segment, master, slave, gt_dir) still to run]), in runner order."""
    results = {seg["name"]: [] for seg in segments}
    tasks = []
    for seg in segments:
        seg_protocol = _segment_protocol(base_cfg, seg)
        gt_dir = None
        for cA, cB in itertools.permutations(seg.get("cameras", []), 2):
            cached = existing.get((seg["name"], cA["id"], cB["id"]))
            if cache_compatible(cached, seg_protocol):
                results[seg["name"]].append(_reuse_cached(cached, seg, cA, cB))
                continue
            if cached:
                print(f"[BruteForce] IGNORE stale result {seg['name']} {cA['id']}->{cB['id']} "
                      f"with protocol={cached.get('protocol')!r}")
            if gt_dir is None:
                gt_dir = _resolve_local_or_drive_path(
                    ws_dir, seg, "ground_truth_dir", "ground_truth_dir_local",
                    description=f"{seg['name']} Ground Truth")
            tasks.append((seg, cA, cB, str(gt_dir)))
    return results, tasks


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(max(seconds, 0.0))), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def _result_summary(res: dict) -> str:
    def num(key):
        try:
            return float(res.get(key))
        except (TypeError, ValueError):
            return float("inf")

    def show(key):
        value = num(key)
        return "N/A" if value == float("inf") else f"{value:.2f}"

    deltas = ", ".join(f"{label} {num(key):+.2f}%" for label, key in (
        ("MPJPE", "% delta_mpjpe"), ("PA", "% delta_pa_mpjpe"),
        ("Accel", "% delta_accel"), ("MBLE", "% delta_mble")) if num(key) != float("inf"))
    return (f"Δ: {deltas} | MPJPE={show('mpjpe')}, PA-MPJPE={show('pa_mpjpe')}, "
            f"Accel={show('accel')}, MBLE={show('mble')}")


def _run_pairs_parallel(segments, existing, base_cfg, ws_dir, sh_name, ws_title, workers, evaluate=None):
    """Same pairs, cache and report as the sequential loop, on `workers` processes.

    The report (CSV) is rewritten after every finished pair, with the cached rows of every
    segment, so Ctrl+C loses only the pairs that were running. Returns {segment: ranked results}.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results, tasks = _plan_pairs(segments, existing, base_cfg, ws_dir)

    def write_report():
        ranked = {name: sorted(rows, key=_ranking_key, reverse=True) for name, rows in results.items()}
        generate_spreadsheet_report_safe(ranked, sh_name, ws_title, silent=True)
        return ranked

    cached = sum(len(rows) for rows in results.values())
    root = _report_dir() / WORKERS_DIR_NAME
    if not tasks:
        print(f"\n=== Không còn cặp nào cần chạy: {cached} cặp lấy từ cache ===")
        return write_report()
    n = max(1, min(int(workers), len(tasks)))
    print(f"\n=== Song song: {len(tasks)} cặp cần chạy trên {n} worker (1 luồng mỗi worker), "
          f"{cached} cặp lấy từ cache ===")
    print(f"[BruteForce] Log của từng cặp: {root / 'logs'}")

    ctx = mp.get_context("spawn")  # fork is unsafe once torch / BLAS threads exist
    slots = ctx.Queue()
    for slot in range(n):
        slots.put(slot)
    saved_env = {var: os.environ.get(var) for var in THREAD_ENV_VARS}
    os.environ.update({var: "1" for var in THREAD_ENV_VARS})  # read by numpy/torch when a worker starts
    pool = ProcessPoolExecutor(
        max_workers=n, mp_context=ctx, initializer=_init_worker,
        initargs=(slots, str(root), base_cfg, str(ws_dir), evaluate or _evaluate_camera_pair))
    started = time.perf_counter()
    try:
        futures = {
            pool.submit(_pool_evaluate_pair, seg["name"], cA, cB, gt_dir, seg.get("target_person")):
                (seg["name"], cA["id"], cB["id"])
            for seg, cA, cB, gt_dir in tasks
        }
        for done, future in enumerate(as_completed(futures), start=1):
            seg_name, cam_a, cam_b = futures[future]
            eta = (time.perf_counter() - started) / done * (len(tasks) - done)
            head = f"[{done}/{len(tasks)} | còn ~{_format_duration(eta)}] {seg_name} {cam_a}->{cam_b}"
            try:
                res, seconds, log_path, slot = future.result()
            except Exception as exc:
                print(f"{head}: LỖI worker {type(exc).__name__}: {exc}", flush=True)
                continue
            if res is None:
                err_hint = ""
                try:
                    lines = [ln.strip() for ln in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                    if lines:
                        err_hint = f" | {lines[-1]}"
                except Exception:
                    pass
                print(f"{head} (w{slot:02d}, {seconds:.0f}s): không có kết quả, xem {log_path}{err_hint}", flush=True)
                continue
            results[seg_name].append(res)
            # flush: progress must show up even when stdout is redirected to a file
            print(f"{head} (w{slot:02d}, {seconds:.0f}s) {_result_summary(res)}", flush=True)
            write_report()
    except KeyboardInterrupt:
        print("\n[BruteForce] Ctrl+C: dừng các worker. Các cặp đã xong nằm trong báo cáo; "
              "chạy lại lệnh để tiếp tục.")
        for proc in mp.active_children():
            proc.terminate()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for var, value in saved_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
    print(f"[BruteForce] {len(tasks)} cặp xong sau {_format_duration(time.perf_counter() - started)}")
    return write_report()


def run_brute_force(base_cfg=None, brute_cfg=None, *, local_report_dir=None, workers=1):
    global ACTIVE_EVALUATION_PROTOCOL, ACTIVE_REPORT_CSV_NAME, ACTIVE_LOCAL_REPORT_DIR
    if not HAS_GSPREAD:
        print("[Local Mode] Chạy môi trường Local (không dùng Google Sheets / Colab). Báo cáo xuất ra file CSV local.")
    ACTIVE_LOCAL_REPORT_DIR = Path(local_report_dir) if local_report_dir is not None else None
    WS_DIR = Path(__file__).parent.resolve()
    if brute_cfg is None:
        with open(WS_DIR / "configs/brute_force.yml", "r", encoding="utf-8") as f: brute_cfg = yaml.safe_load(f)
    if base_cfg is None:
        base_cfg = load_config(WS_DIR / "configs/pipeline.yml")
    ACTIVE_EVALUATION_PROTOCOL = evaluation_protocol(base_cfg)
    ACTIVE_REPORT_CSV_NAME = report_csv_name(base_cfg)
    print(f"[BruteForce] Active protocol: {ACTIVE_EVALUATION_PROTOCOL}")
    print(f"[BruteForce] Results CSV: {(ACTIVE_LOCAL_REPORT_DIR or WS_DIR / 'output') / ACTIVE_REPORT_CSV_NAME}")
    print(f"[BruteForce] AOTR enabled: {aotr_enabled(base_cfg)}")
    # Kiểm tra xem file config đã nhận đúng giá trị chưa
    print("Alpha trong config:", base_cfg['fusion']['belief']['alpha'])
    print("Beta trong config:", base_cfg['fusion']['belief']['beta'])
    
    _, runner_name, _ = get_system_metadata()
    suffix = ACTIVE_EVALUATION_PROTOCOL.rsplit("__", 1)[-1] if aotr_enabled(base_cfg) else "legacy"
    default_sh_name = f"{runner_name}_AOTR_brute_force_{suffix}" if aotr_enabled(base_cfg) else f"{runner_name}_brute_force_pipeline"
    sh_name = default_sh_name if ACTIVE_LOCAL_REPORT_DIR is not None else get_spreadsheet_name_input(default_name=default_sh_name, timeout=10)
    
    ws_title = f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # One expected protocol per target person, so cached aria02 rows are reused too.
    expected = {ACTIVE_EVALUATION_PROTOCOL} | {
        _segment_protocol(base_cfg, seg) for seg in brute_cfg.get("segments", [])}
    existing = load_existing_spreadsheet_results(sh_name, expected_protocol=expected)
    all_res = {}

    total_pairs = sum(
        len(seg.get("cameras", [])) * (len(seg.get("cameras", [])) - 1)
        for seg in brute_cfg.get("segments", [])
        if len(seg.get("cameras", [])) >= 2
    )
    current_pair_idx = 0

    if int(workers) > 1:
        segments = [seg for seg in brute_cfg.get("segments", []) if len(seg.get("cameras", [])) >= 2]
        all_res = _run_pairs_parallel(segments, existing, base_cfg, WS_DIR, sh_name, ws_title, int(workers))
    else:
        for seg in brute_cfg.get("segments", []):
            if len(seg.get("cameras", [])) < 2: continue

            sorted_res, current_pair_idx = _process_segment(
                seg, existing, base_cfg, WS_DIR, sh_name, ws_title, all_res, current_pair_idx, total_pairs
            )
            all_res[seg["name"]] = sorted_res

    generate_spreadsheet_report_safe(all_res, sh_name, ws_title, silent=False, is_final=True)
    # Dọn dẹp PKL và video sau khi đã hoàn tất báo cáo.
    _cleanup_local_input_files(
        brute_cfg=brute_cfg,
        workspace=WS_DIR,
        timeout=10
    )
    return all_res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Brute Force Combinatorial Evaluation")
    parser.add_argument("--config", type=str, default=None, help="Base pipeline config (default: configs/pipeline.yml)")
    parser.add_argument("--brute-config", type=str, default=None, help="Brute force segments config (default: configs/brute_force.yml)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of camera pairs run at once, one process and one thread each "
                             "(default 1 = sequential). Pair logs: output/brute_force_workers/logs/. "
                             "About one worker per CPU core, minus one or two for the system.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    WS_DIR = Path(__file__).parent.resolve()
    base_cfg = load_config(WS_DIR / args.config) if args.config else None
    brute_cfg = None
    if args.brute_config:
        with open(WS_DIR / args.brute_config, "r", encoding="utf-8") as f:
            brute_cfg = yaml.safe_load(f)
    run_brute_force(base_cfg=base_cfg, brute_cfg=brute_cfg, workers=args.workers)

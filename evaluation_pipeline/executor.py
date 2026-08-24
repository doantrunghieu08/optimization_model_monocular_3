import csv
import json
import re
import sys
import threading
import getpass
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import procrustes
from keypoints_map import load_keypoints3d_map
from config_loader import resolve_inputs

MODULES = ["posed", "fused", "learnable"]
MODULES = ["posed", "fused", "learnable"]

EVALUATION_OUTPUT_MODULE_NAMES = {
    "learnable": "fusion-learnable",
    "learnable_extra": "only_learnable",
}
CAMERAS = ["camera1", "camera2"]
CAMERA_FILE_NAMES = {
    "camera1": "cam1",
    "camera2": "cam2",
}


def _camera_key_from_video_path(video_path: Optional[str]) -> str:
    if not video_path:
        return "camera1"
    stem = Path(video_path).stem
    match = re.search(r"video_(\d+)", stem)
    if match:
        return f"camera{int(match.group(1))}"
    match = re.search(r"camera_(\d+)", stem)
    if match:
        return f"camera{int(match.group(1))}"
    nums = re.findall(r"\d+", stem)
    if nums:
        return f"camera{int(nums[-1])}"
    return "camera1"

def _frame_index_from_path(p: Path) -> int:
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else -1

def _resolve_root_joint(joints: dict) -> np.ndarray:
    if "left_hip" in joints and "right_hip" in joints:
        return (joints["left_hip"] + joints["right_hip"]) / 2.0
    raise ValueError("Missing left_hip or right_hip for root alignment")

def _compute_mpjpe(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> tuple[float, dict[str, float]]:
    pred_root = _resolve_root_joint(pred)
    truth_root = _resolve_root_joint(truth)

    errors = []
    errors_dict = {}
    for k in keys:
        p = pred[k] - pred_root
        t = truth[k] - truth_root
        err = float(np.linalg.norm(p - t) * 1000.0)
        errors.append(err)
        errors_dict[k] = err
    return float(np.mean(errors)), errors_dict

def _compute_pa_mpjpe(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> tuple[float, dict[str, float]]:
    P = np.array([pred[k] for k in keys], dtype=float)
    T = np.array([truth[k] for k in keys], dtype=float)

    P_c = P - np.mean(P, axis=0)
    T_c = T - np.mean(T, axis=0)

    norm_P = np.linalg.norm(P_c)
    norm_T = np.linalg.norm(T_c)
    
    if norm_P < 1e-6 or norm_T < 1e-6:
        return 0.0, {k: 0.0 for k in keys}

    P_c_unit = P_c / norm_P
    T_c_unit = T_c / norm_T

    U, s, Vt = np.linalg.svd(np.dot(T_c_unit.T, P_c_unit))
    R = np.dot(U, Vt)
    
    scale = np.sum(s) * (norm_T / norm_P)
    P_aligned = np.dot(P_c, R.T) * scale + np.mean(T, axis=0)

    errors = []
    errors_dict = {}
    for i, k in enumerate(keys):
        err = float(np.linalg.norm(P_aligned[i] - T[i]) * 1000.0)
        errors.append(err)
        errors_dict[k] = err
        
    return float(np.mean(errors)), errors_dict

def _compute_pck_mm(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> tuple[float, dict[str, float]]:
    # PCK-mm is mean Euclidean distance (absolute, no root align)
    errors = []
    errors_dict = {}
    for k in keys:
        p = pred[k]
        t = truth[k]
        err = float(np.linalg.norm(p - t) * 1000.0)
        errors.append(err)
        errors_dict[k] = err
    return float(np.mean(errors)), errors_dict

def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_frame_map(frame_dir: Path, frame_offset: int = 0) -> dict[int, Path]:
    frame_map = {}
    for path in frame_dir.glob("*.json"):
        frame_idx = _frame_index_from_path(path)
        if frame_idx < 0:
            raise ValueError(f"Cannot extract frame index from {path}")
        frame_idx += frame_offset
        if frame_idx in frame_map:
            raise ValueError(f"Duplicate frame index {frame_idx} in {frame_dir}")
        frame_map[frame_idx] = path
    return frame_map


def _load_segment_gt(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"GT file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    gt_map = {}
    for idx, item in enumerate(data):
        gt_map[idx] = item
        if "frame_id" in item:
            gt_map[int(item["frame_id"])] = item
    return gt_map


def _parse_new_gt(item: dict, canonical_names: set, map_data: dict) -> dict:
    joints = {}
    # Convert from millimeters to meters to match the prediction scale
    pose3d = np.array(item["pose3d"], dtype=float) / 1000.0
    
    # MPI-INF-3DHP 28-joint format indices (correct mapping):
    # [0]=spine3, [1]=spine4, [2]=spine2, [3]=spine, [4]=pelvis,
    # [5]=neck, [6]=head, [7]=head_top,
    # [8]=left_clavicle, [9]=left_shoulder, [10]=left_elbow, [11]=left_wrist, [12]=left_hand,
    # [13]=right_clavicle, [14]=right_shoulder, [15]=right_elbow, [16]=right_wrist, [17]=right_hand,
    # [18]=left_hip, [19]=left_knee, [20]=left_ankle, [21]=left_foot, [22]=left_toe,
    # [23]=right_hip, [24]=right_knee, [25]=right_ankle, [26]=right_foot, [27]=right_toe
    new_gt_indices = {
        "head": 6,
        "neck": 5,
        "pelvis": 4,
        "left_shoulder": 9,   "right_shoulder": 14,
        "left_elbow": 10,     "right_elbow": 15,
        "left_wrist": 11,     "right_wrist": 16,
        "left_hand": 12,      "right_hand": 17,
        "left_hip": 18,       "right_hip": 23,
        "left_knee": 19,      "right_knee": 24,
        "left_ankle": 20,     "right_ankle": 25,
        "left_foot": 21,      "right_foot": 26,
        "left_toe": 22,       "right_toe": 27,
    }
    
    for name, idx in new_gt_indices.items():
        if idx < len(pose3d):
            joints[name] = pose3d[idx]
            
    return joints


def _format_frame_sample(frames: set[int], limit: int = 8) -> str:
    if not frames:
        return "[]"
    ordered = sorted(frames)
    if len(ordered) <= limit:
        return str(ordered)
    head = ", ".join(str(n) for n in ordered[:limit])
    return f"[{head}, ...] (total={len(ordered)})"


def _warn_frame_mismatch(module_name: str, truth_frames: set[int], module_frames: set[int]) -> None:
    extra_truth = truth_frames - module_frames
    extra_module = module_frames - truth_frames
    if not extra_truth and not extra_module:
        return
    print(
        f"[Evaluation] Frame mismatch in {module_name}: "
        f"truth={len(truth_frames)}, module={len(module_frames)}, "
        f"overlap={len(truth_frames & module_frames)}"
    )
    if extra_truth:
        print(f"[Evaluation]   GT-only frames: {_format_frame_sample(extra_truth)}")
    if extra_module:
        print(f"[Evaluation]   Output-only frames: {_format_frame_sample(extra_module)}")

def get_spreadsheet_filename(default_name="evaluation_summary") -> str:
    """Cho người dùng 10s để nhập tên file. Nhập 'now' sẽ gắn thêm YYMMDD."""
    user_input = [None]
    
    def wait_for_input():
        try:
            user_input[0] = input(f"Nhập tên file spreadsheet (mặc định '{default_name}.csv', gõ 'now' để thêm ngày tháng). Bạn có 10s: ")
        except EOFError:
            pass
            
    print("\n--- CHỜ NHẬP TÊN FILE ---")
    t = threading.Thread(target=wait_for_input)
    t.daemon = True
    t.start()
    t.join(10.0) # Đợi tối đa 10 giây
    
    if t.is_alive():
        print(f"\n[Hết 10s] Tự động sử dụng mặc định.")
        final_input = ""
    else:
        final_input = (user_input[0] or "").strip()
        
    if final_input.lower() == "now":
        date_suffix = datetime.now().strftime("%y%m%d")
        return f"{default_name}_{date_suffix}.csv"
    elif final_input:
        if not final_input.endswith(".csv"):
            final_input += ".csv"
        return final_input
    
    return f"{default_name}.csv"

def extract_set_and_segment(video_path: str) -> tuple[str, str]:
    """Trích xuất Set (thư mục cha) và Segment (sau chữ seg) từ đường dẫn video."""
    if not video_path:
        return "UnknownSet", "UnknownSeg"
    p = Path(video_path)
    video_set = p.parent.name
    
    # Tìm chuỗi đứng sau 'seg' trong tên file, VD: video_seg_01 -> 01, nhảy qua dấu _ hoặc -
    match = re.search(r"seg[_-]?([A-Za-z0-9]+)", p.stem, re.IGNORECASE)
    segment = match.group(1) if match else p.stem
    
    return video_set, segment

def _resolve_truth_frame_payload(truth_data: dict, testcase_name: Optional[str], truth_path: Path) -> dict:
    if testcase_name is not None:
        tc_data = truth_data.get(testcase_name)
        if tc_data is None:
            raise ValueError(f"Missing testcase {testcase_name} in {truth_path}")
        if not isinstance(tc_data, dict):
            raise ValueError(f"Invalid testcase payload for {testcase_name} in {truth_path}")
        return tc_data

    camera_keys = [key for key in truth_data if re.fullmatch(r"camera\d+", str(key))]
    if camera_keys:
        return truth_data

    if len(truth_data) != 1:
        raise ValueError(
            f"Expected either camera* keys or exactly one top-level testcase entry in {truth_path} "
            "when evaluation.testcase_name is null"
        )

    tc_data = next(iter(truth_data.values()))
    if not isinstance(tc_data, dict):
        raise ValueError(f"Invalid top-level testcase payload in {truth_path}")
    return tc_data

def run_evaluation(config: dict) -> None:
    eval_cfg = config.get("evaluation", {})
    if not eval_cfg["enabled"]:
        print("[Evaluation] Disabled by config: evaluation.enabled=false")
        return

    paths = config["paths"]
    inputs = resolve_inputs(config)
    map_data = load_keypoints3d_map(paths["keypoints3d_map"])
    canonical_names = set([k["name"] for k in map_data["keypoints"]])
    priority1_names = map_data.get("priority1", [])

    truth_dir = Path(inputs["ground_truth_dir"])
    out_dir = Path(paths["evaluation_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lấy tên file xuất ra (chờ 10s)
    spreadsheet_filename = get_spreadsheet_filename()
    spreadsheet_path = out_dir / spreadsheet_filename

    # 2. Các tham số modules
    module_names = ["posed", "fused"]
    if config.get("learnable", {}).get("enabled", False):
        module_names.append("learnable")
    if config.get("learnable_extra", {}).get("enabled", False):
        module_names.append("learnable_extra")

    module_dirs = {
        "posed": Path(paths["pose_output_dir"]) / "keypoints3d",
        "fused": Path(paths["fused_output_dir"]) / "keypoints3d",
    }
    if "learnable" in module_names:
        module_dirs["learnable"] = Path(paths["learnable_output_dir"]) / "keypoints3d"
    if "learnable_extra" in module_names:
        module_dirs["learnable_extra"] = Path(paths["learnable_extra_output_dir"]) / "keypoints3d"

    # Load dữ liệu GT (Ground Truth)
    cam1_gt_stem = Path(inputs.get("cam1_pkl", "")).stem
    cam2_gt_stem = Path(inputs.get("cam2_pkl", "")).stem
    gt_cam_data = {
        "camera1": _load_segment_gt(truth_dir / f"{cam1_gt_stem}.json"),
        "camera2": _load_segment_gt(truth_dir / f"{cam2_gt_stem}.json")
    }

    # Load Frames
    module_frame_maps = {name: _load_frame_map(mdir) for name, mdir in module_dirs.items()}
    common_frames = set.intersection(*(set(m) for m in module_frame_maps.values()))
    frames = sorted(common_frames)

    # Lưu lại tổng sai số để tính trung bình cho toàn bộ segment
    # Cấu trúc: segment_metrics[cam][mod][metric] = tổng sai số
    segment_metrics = {"camera1": {}, "camera2": {}}
    for cam in ["camera1", "camera2"]:
        for mod in module_names:
            segment_metrics[cam][mod] = {"MPJPE": 0.0, "PA-MPJPE": 0.0, "count": 0}

    # 3. Tính toán trên từng Frame
    for frame in frames:
        metadata_path = Path(paths["pose_output_dir"]) / "metadata" / f"pose_data_{frame}.json"
        if not metadata_path.exists(): continue
        
        metadata = _load_json(metadata_path)
        src_indices = metadata.get("metadata", {}).get("source_frame_indices", {})
        
        for cam, gt_cam in zip(["camera1", "camera2"], [gt_cam_data["camera1"], gt_cam_data["camera2"]]):
            cam_frame_id = src_indices.get(cam)
            if cam_frame_id not in gt_cam: continue

            truth_joints = _parse_new_gt(gt_cam[cam_frame_id], canonical_names, map_data)

            for mod in module_names:
                mod_data = _load_json(module_frame_maps[mod][frame])
                pred_joints = {k: np.array(v, dtype=float) for k, v in mod_data[cam].items()}
                
                valid_priority1 = [k for k in priority1_names if k in set(pred_joints.keys()) & set(truth_joints.keys())]
                
                if valid_priority1:
                    mpjpe_err, _ = _compute_mpjpe(pred_joints, truth_joints, valid_priority1)
                    pa_mpjpe_err, _ = _compute_pa_mpjpe(pred_joints, truth_joints, valid_priority1)
                    
                    segment_metrics[cam][mod]["MPJPE"] += mpjpe_err
                    segment_metrics[cam][mod]["PA-MPJPE"] += pa_mpjpe_err
                    segment_metrics[cam][mod]["count"] += 1

    # 4. Tính giá trị trung bình toàn Segment & Định dạng Output
    video_set, segment_name = extract_set_and_segment(inputs.get("camera1_video", ""))
    cam_master = _camera_key_from_video_path(inputs.get("camera1_video"))
    cam_slave = _camera_key_from_video_path(inputs.get("camera2_video"))
    
    # Chuẩn bị dữ liệu để ghi
    os_version = f"{platform.system()} {platform.release()}"
    username = getpass.getuser()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Mở file CSV theo mode "append" (ghi nối tiếp)
    file_exists = spreadsheet_path.exists()
    with spreadsheet_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Ghi header nếu file mới
        if not file_exists:
            writer.writerow([
                "Set", "Segment", "Rank", "Cam Master", "Cam Slave",
                "fusion_MPJPE", "fusion_PA-MPJPE", 
                "learnable_fusion_MPJPE", "learnable_fusion_PA-MPJPE", 
                "learnable_MPJPE", "learnable_PA-MPJPE", 
                "Old MPJPE", "Old PA-MPJPE", 
                "% F Δ_MPJPE", "% F Δ_PA-MPJPE", 
                "% LF Δ_MPJPE", "% LF Δ_PA-MPJPE", 
                "% L Δ_MPJPE", "% L Δ_PA-MPJPE", 
                "OS Version", "Username", "Timestamp"
            ])

        # Tính toán row cho camera master (Mặc định lấy cam1 làm đại diện tính lỗi tổng hợp)
        # Nếu muốn tính cho cả master và slave độc lập, bạn có thể loop qua ["camera1", "camera2"] để ghi 2 dòng.
        # Ở đây lấy camera1 làm gốc đại diện cho master theo format của bạn.
        cam_eval = "camera1" 
        
        def get_avg(mod_name, metric):
            data = segment_metrics[cam_eval].get(mod_name, {"count": 0})
            return data[metric] / data["count"] if data["count"] > 0 else 0.0

        # Ánh xạ theo tên cột của bạn
        old_mpjpe = get_avg("posed", "MPJPE")
        old_pa_mpjpe = get_avg("posed", "PA-MPJPE")
        
        fusion_mpjpe = get_avg("fused", "MPJPE")
        fusion_pa_mpjpe = get_avg("fused", "PA-MPJPE")
        
        learn_fusion_mpjpe = get_avg("learnable", "MPJPE")
        learn_fusion_pa_mpjpe = get_avg("learnable", "PA-MPJPE")
        
        learn_mpjpe = get_avg("learnable_extra", "MPJPE")
        learn_pa_mpjpe = get_avg("learnable_extra", "PA-MPJPE")

        # Tính Delta (so sánh giữa learnable_fusion và old/posed)
        lf_delta_mpjpe = ((learn_fusion_mpjpe - old_mpjpe) / old_mpjpe * 100) if old_mpjpe > 0 else 0.0
        lf_delta_pa_mpjpe = ((learn_fusion_pa_mpjpe - old_pa_mpjpe) / old_pa_mpjpe * 100) if old_pa_mpjpe > 0 else 0.0

        # Tính Delta (so sánh giữa fusion và old/posed)
        f_delta_mpjpe = ((fusion_mpjpe - old_mpjpe) / old_mpjpe * 100) if old_mpjpe > 0 else 0.0
        f_delta_pa_mpjpe = ((fusion_pa_mpjpe - old_pa_mpjpe) / old_pa_mpjpe * 100) if old_pa_mpjpe > 0 else 0.0

        # Tính Delta (so sánh giữa learnable_fusion và old/posed)
        L_delta_mpjpe = ((learn_mpjpe - old_mpjpe) / old_mpjpe * 100) if old_mpjpe > 0 else 0.0
        L_delta_pa_mpjpe = ((learn_pa_mpjpe - old_pa_mpjpe) / old_pa_mpjpe * 100) if old_pa_mpjpe > 0 else 0.0

        row = [
            video_set, segment_name, "N/A", cam_master, cam_slave,
            f"{fusion_mpjpe:.2f}", f"{fusion_pa_mpjpe:.2f}",
            f"{learn_fusion_mpjpe:.2f}", f"{learn_fusion_pa_mpjpe:.2f}",
            f"{learn_mpjpe:.2f}", f"{learn_pa_mpjpe:.2f}",
            f"{old_mpjpe:.2f}", f"{old_pa_mpjpe:.2f}",
            f"{f_delta_mpjpe:.2f}%", f"{f_delta_pa_mpjpe:.2f}%",
            f"{lf_delta_mpjpe:.2f}%", f"{lf_delta_pa_mpjpe:.2f}%",
            f"{L_delta_mpjpe:.2f}%", f"{L_delta_pa_mpjpe:.2f}%",
            os_version, username, timestamp
        ]
        
        writer.writerow(row)

    print(f"\n[Evaluation] Đã lưu báo cáo tại: {spreadsheet_path}")
"""## 12. Visualization phase"""

import csv
import json
import os 
import re
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
    priority2_names = map_data.get("priority2", [])

    truth_dir = Path(inputs["ground_truth_dir"])
    out_dir = Path(paths["evaluation_output_dir"])
    testcase_name = eval_cfg.get("testcase_name")
    if testcase_name in ("", None):
        testcase_name = None

    metrics_cfg = eval_cfg.get("metrics")
    if not isinstance(metrics_cfg, dict):
        raise ValueError("Missing config section: evaluation.metrics")
    for key in ("pa_mpjpe", "mpjpe", "pck"):
        if key not in metrics_cfg or metrics_cfg[key] is None:
            raise ValueError(f"Missing config evaluation metric flag: evaluation.metrics.{key}")
    metric_enabled = {
        "MPJPE": bool(metrics_cfg["mpjpe"]),
        "PA-MPJPE": bool(metrics_cfg["pa_mpjpe"]),
        "PCK": bool(metrics_cfg["pck"]),
    }
    enabled_metrics = [name for name, enabled in metric_enabled.items() if enabled]
    if not enabled_metrics:
        raise ValueError("At least one evaluation metric must be enabled")

    if config.get("runtime", {}).get("clean_output", True):
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.csv"):
            old.unlink(missing_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

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

    for name, mdir in module_dirs.items():
        if not mdir.exists():
            raise FileNotFoundError(f"Missing required module directory: {mdir}")

    # GT là segment file, mỗi camera có một file riêng.
    cam1_gt_stem = Path(inputs.get("cam1_pkl", "")).stem
    cam2_gt_stem = Path(inputs.get("cam2_pkl", "")).stem
    if not cam1_gt_stem or not cam2_gt_stem:
        raise ValueError("Missing cam1_pkl or cam2_pkl in inputs to determine GT files.")
        
    cam1_gt_path = truth_dir / f"{cam1_gt_stem}.json"
    cam2_gt_path = truth_dir / f"{cam2_gt_stem}.json"
    
    gt_cam1_data = _load_segment_gt(cam1_gt_path)
    gt_cam2_data = _load_segment_gt(cam2_gt_path)
    gt_cam_data = {"camera1": gt_cam1_data, "camera2": gt_cam2_data}

    gt_camera_keys = {
        "camera1": _camera_key_from_video_path(inputs.get("camera1_video")),
        "camera2": _camera_key_from_video_path(inputs.get("camera2_video")),
    }

    module_frame_maps = {}
    module_frame_sets = {}
    for name, mdir in module_dirs.items():
        module_frame_maps[name] = _load_frame_map(mdir)
        module_frame_sets[name] = set(module_frame_maps[name])

    common_frames = None
    for frames_set in module_frame_sets.values():
        if common_frames is None:
            common_frames = frames_set.copy()
        else:
            common_frames &= frames_set

    if not common_frames:
        raise ValueError("No overlapping frames between module outputs.")

    frames = sorted(common_frames)

    # metrics: metric -> cam -> frame -> module -> priority -> value
    results = {metric: {"camera1": {}, "camera2": {}} for metric in enabled_metrics}
    evaluated_frames = []

    for frame in frames:
        # Resolve actual frame_id from metadata
        metadata_path = Path(paths["pose_output_dir"]) / "metadata" / f"pose_data_{frame}.json"
        if not metadata_path.exists():
            continue
        metadata = _load_json(metadata_path)
        src_indices = metadata.get("metadata", {}).get("source_frame_indices", {})
        
        cam1_frame_id = src_indices.get("camera1")
        cam2_frame_id = src_indices.get("camera2")
        
        if cam1_frame_id not in gt_cam1_data or cam2_frame_id not in gt_cam2_data:
            continue
            
        evaluated_frames.append(frame)

        for cam in CAMERAS:
            gt_frame_id = cam1_frame_id if cam == "camera1" else cam2_frame_id
            truth_item = gt_cam_data[cam][gt_frame_id]
            truth_joints = _parse_new_gt(truth_item, canonical_names, map_data)

            for metric in results:
                if frame not in results[metric][cam]:
                    results[metric][cam][frame] = {}

            for mod in module_names:
                mod_path = module_frame_maps[mod][frame]
                mod_data = _load_json(mod_path)
                if cam not in mod_data:
                    raise ValueError(f"Missing {cam} in {mod_path}")

                pred_joints = {k: np.array(v, dtype=float) for k, v in mod_data[cam].items()}
                
                # Filter priority names to only include keys present in both pred and truth
                available_keys = set(pred_joints.keys()) & set(truth_joints.keys())
                valid_priority1 = [k for k in priority1_names if k in available_keys]
                valid_priority2 = [k for k in priority2_names if k in available_keys]
                
                if not valid_priority1:
                    print(f"Warning: No overlapping priority1 keys for {mod} {cam} frame {frame}")

                # compute metrics
                # MPJPE
                if metric_enabled["MPJPE"]:
                    results["MPJPE"][cam][frame].setdefault(mod, {})
                    mean_p1, dict_p1 = _compute_mpjpe(pred_joints, truth_joints, valid_priority1) if valid_priority1 else (0.0, {})
                    results["MPJPE"][cam][frame][mod]["priority1_mm"] = mean_p1
                    results["MPJPE"][cam][frame][mod]["priority1_details"] = dict_p1
                    
                    mean_p2, dict_p2 = _compute_mpjpe(pred_joints, truth_joints, valid_priority2) if valid_priority2 else (0.0, {})
                    results["MPJPE"][cam][frame][mod]["priority2_mm"] = mean_p2
                    results["MPJPE"][cam][frame][mod]["priority2_details"] = dict_p2

                if metric_enabled["PA-MPJPE"]:
                    results["PA-MPJPE"][cam][frame].setdefault(mod, {})
                    mean_p1, dict_p1 = _compute_pa_mpjpe(pred_joints, truth_joints, valid_priority1) if valid_priority1 else (0.0, {})
                    results["PA-MPJPE"][cam][frame][mod]["priority1_mm"] = mean_p1
                    results["PA-MPJPE"][cam][frame][mod]["priority1_details"] = dict_p1
                    
                    mean_p2, dict_p2 = _compute_pa_mpjpe(pred_joints, truth_joints, valid_priority2) if valid_priority2 else (0.0, {})
                    results["PA-MPJPE"][cam][frame][mod]["priority2_mm"] = mean_p2
                    results["PA-MPJPE"][cam][frame][mod]["priority2_details"] = dict_p2

                if metric_enabled["PCK"]:
                    results["PCK"][cam][frame].setdefault(mod, {})
                    mean_p1, dict_p1 = _compute_pck_mm(pred_joints, truth_joints, valid_priority1) if valid_priority1 else (0.0, {})
                    results["PCK"][cam][frame][mod]["priority1_mm"] = mean_p1
                    results["PCK"][cam][frame][mod]["priority1_details"] = dict_p1
                    
                    mean_p2, dict_p2 = _compute_pck_mm(pred_joints, truth_joints, valid_priority2) if valid_priority2 else (0.0, {})
                    results["PCK"][cam][frame][mod]["priority2_mm"] = mean_p2
                    results["PCK"][cam][frame][mod]["priority2_details"] = dict_p2

    module_output_names = {
        mod: EVALUATION_OUTPUT_MODULE_NAMES.get(mod, mod)
        for mod in module_names
    }

    header = ["Frame", "Evaluated_Camera", "Ground_Truth_Camera"]
    for mod in module_names:
        out_name = module_output_names[mod]
        header.append(f"{out_name}_priority1_mm")
        header.append(f"{out_name}_priority2_mm")
        for joint in priority1_names:
            header.append(f"{out_name}_{joint}_mm")

    for metric in enabled_metrics:
        for cam in CAMERAS:
            filename = f"{metric}_{CAMERA_FILE_NAMES[cam]}.csv"
            out_file = out_dir / filename
            with out_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

                avg_sums = {h: 0.0 for h in header[3:]}

                for frame in evaluated_frames:
                    row = [frame, cam, gt_camera_keys[cam]]
                    for mod in module_names:
                        out_name = module_output_names[mod]
                        
                        v1 = results[metric][cam][frame][mod]["priority1_mm"]
                        v2 = results[metric][cam][frame][mod]["priority2_mm"]
                        row.append(f"{v1:.2f}")
                        row.append(f"{v2:.2f}")
                        avg_sums[f"{out_name}_priority1_mm"] += v1
                        avg_sums[f"{out_name}_priority2_mm"] += v2
                        
                        p1_details = results[metric][cam][frame][mod]["priority1_details"]
                        for joint in priority1_names:
                            err_j = p1_details.get(joint, 0.0)
                            row.append(f"{err_j:.2f}")
                            avg_sums[f"{out_name}_{joint}_mm"] += err_j

                n_frames = len(evaluated_frames)
                if n_frames > 0:
                    avg_row = ["AVERAGE", cam, gt_camera_keys[cam]]
                    for h in header[3:]:
                        avg_row.append(f"{(avg_sums[h]/n_frames):.2f}")
                    writer.writerow(avg_row)

    if "learnable_extra" in module_names and len(evaluated_frames) > 0:
        learnable_extra_summary = {}
        target_metrics = ["MPJPE", "PA-MPJPE"]
        
        for cam in CAMERAS:
            cam_dict = {}
            for metric in target_metrics:
                if metric_enabled.get(metric):
                    # Tính tổng lỗi của priority1_mm trên tất cả các frame
                    total_err = sum(
                        results[metric][cam][frame]["learnable_extra"]["priority1_mm"] 
                        for frame in evaluated_frames
                    )
                    # Tính trung bình và ép kiểu sang string (giữ 2 chữ số thập phân)
                    avg_err = total_err / len(evaluated_frames)
                    cam_dict[metric] = f"{avg_err:.2f}"
            
            if cam_dict:
                learnable_extra_summary[cam] = cam_dict

        # Chuyển đổi dictionary sang chuỗi JSON
        summary_str = json.dumps(learnable_extra_summary)
        
        # Lưu vào biến môi trường của OS
        os.environ["LEARNABLE_EXTRA_METRICS"] = summary_str
        print(f"[Evaluation] Đã lưu metrics vào os.environ['LEARNABLE_EXTRA_METRICS']: {summary_str}")
    
    print(f"[Evaluation] Done. Output: {out_dir}")

"""## 12. Visualization phase"""

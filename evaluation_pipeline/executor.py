import csv
from functools import lru_cache
import json
import os 
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import procrustes
from keypoints_map import load_keypoints3d_map
from evaluation_pipeline.extrinsics import (camera_name_from_path, ext_enabled, extrinsic_from_config,
                                           find_colmap_workplace, load_world_to_camera,
                                           transform_truth_map)

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "debug-0e35fc.log"

def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "0e35fc",
        "runId": "gt-switch",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as _df:
        _df.write(json.dumps(payload, default=str) + "\n")
# #endregion

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
        return (np.asarray(joints["left_hip"], dtype=float) + np.asarray(joints["right_hip"], dtype=float)) / 2.0
    raise ValueError("Missing left_hip or right_hip for root alignment")

def _align_camera_to_world_rigid(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str], rotate: bool = True) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Align camera-frame root-centered predicted pose to world-frame root-centered truth pose.

    Applies pure 3D rotation R in SO(3) without changing scale (scale=1.0) or limb lengths.
    rotate=False (EXT=1: truth already in the camera frame) only root-centres both poses.
    """
    pred_root = _resolve_root_joint(pred)
    truth_root = _resolve_root_joint(truth)
    P = np.array([np.asarray(pred[k], dtype=float) - pred_root for k in keys])
    T = np.array([np.asarray(truth[k], dtype=float) - truth_root for k in keys])
    if not rotate:
        return ({k: P[i] for i, k in enumerate(keys)}, {k: T[i] for i, k in enumerate(keys)})
    H = np.dot(T.T, P)
    U, _, Vt = np.linalg.svd(H)
    corr = np.eye(3)
    corr[-1, -1] = np.sign(np.linalg.det(np.dot(U, Vt)))
    R = np.dot(np.dot(U, corr), Vt)
    P_world = np.dot(P, R.T)
    p_dict = {k: P_world[i] for i, k in enumerate(keys)}
    t_dict = {k: T[i] for i, k in enumerate(keys)}
    return p_dict, t_dict

def _compute_mpjpe(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str], rotate: bool = True) -> tuple[float, dict[str, float]]:
    p_dict, t_dict = _align_camera_to_world_rigid(pred, truth, keys, rotate=rotate)
    errors = []
    errors_dict = {}
    for k in keys:
        err = float(np.linalg.norm(p_dict[k] - t_dict[k]) * 1000.0)
        errors.append(err)
        errors_dict[k] = err
    return float(np.mean(errors)), errors_dict

def mpjpe_alignment_requested(eval_cfg: dict) -> str:
    value = str(eval_cfg.get("mpjpe_alignment", "frame")).lower()
    if value not in ("frame", "sequence"):
        raise ValueError(f"evaluation.mpjpe_alignment must be 'frame' or 'sequence', got {value!r}")
    return value


COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _qvec_to_rotmat(qvec) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) -> world-to-camera rotation matrix R_w2c."""
    q = np.asarray(qvec, dtype=float)
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=float)


def _cam_num_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    matches = re.findall(r"cam(\d+)", str(text), re.IGNORECASE)
    return int(matches[-1]) if matches else None


def _find_colmap_images_txt(truth_dir: Path, extra_paths: tuple[Optional[str], ...] = ()) -> Optional[Path]:
    """Locate Harmony4D colmap/workplace/images.txt (or sparse/0/images.txt) near GT/pkl inputs."""
    roots = []
    for raw in (truth_dir, *(Path(p) for p in extra_paths if p)):
        cur = raw if raw.is_dir() else raw.parent
        for _ in range(5):
            if cur not in roots:
                roots.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent
    for root in roots:
        for rel in ("colmap/workplace/images.txt", "colmap/sparse/0/images.txt", "colmap/images.txt"):
            cand = root / rel
            if cand.is_file():
                return cand
    return None


def _orthonormalize_rot(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def _load_colmap_primary_rotation(workplace_dir: Path) -> np.ndarray:
    """Official Harmony4D world->COLMAP rotation (ego_exo_scene.py): scale.npy or aria01 transform."""
    import pickle
    scale_npy = workplace_dir / "scale.npy"
    if scale_npy.is_file():
        return _orthonormalize_rot(np.linalg.inv(np.load(scale_npy))[:3, :3])
    c_from_a = workplace_dir / "colmap_from_aria_transforms.pkl"
    if c_from_a.is_file():
        d = pickle.loads(c_from_a.read_bytes())
        anchor = "aria01" if "aria01" in d else sorted(d)[0]
        return _orthonormalize_rot(np.asarray(d[anchor], dtype=float)[:3, :3])
    a_from_c = workplace_dir / "aria_from_colmap_transforms.pkl"
    if a_from_c.is_file():
        d = pickle.loads(a_from_c.read_bytes())
        anchor = "aria01" if "aria01" in d else sorted(d)[0]
        return _orthonormalize_rot(np.linalg.inv(np.asarray(d[anchor], dtype=float))[:3, :3])
    return np.eye(3)


def _load_colmap_cam_to_world_rotations(images_txt: Path) -> dict[int, np.ndarray]:
    """Parse COLMAP images.txt (+ sibling primary transform) -> {cam_num: R_cam_to_world} (0% GT).

    Matches Harmony4D ego_exo_scene.py: M_w2c = M_raw @ T_primary -> R_cam_to_world = (R_raw @ R_primary).T.
    """
    R_primary = _load_colmap_primary_rotation(images_txt.parent)
    by_cam: dict[int, list[np.ndarray]] = {}
    for raw in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        cid = _cam_num_from_text(" ".join(parts[9:]))
        if cid is None:
            continue
        try:
            qvec = [float(x) for x in parts[1:5]]
        except ValueError:
            continue
        if np.linalg.norm(qvec) < 1e-9:
            continue
        by_cam.setdefault(cid, []).append((_qvec_to_rotmat(qvec) @ R_primary).T)
    return {
        cid: (rots[0] if len(rots) == 1 else _orthonormalize_rot(np.mean(rots, axis=0)))
        for cid, rots in by_cam.items()
    }


def _load_bundled_harmony4d_rotations(paths: tuple[Optional[str], ...]) -> tuple[dict[int, np.ndarray], Optional[str]]:
    """Lookup official Harmony4D extrinsics (0% GT) from configs/harmony4d_extrinsics.json."""
    db_path = Path(__file__).resolve().parents[1] / "configs" / "harmony4d_extrinsics.json"
    if not db_path.is_file():
        return {}, None
    seqs = _load_json(db_path).get("sequences", {})
    seq_name = None
    for raw in paths:
        if not raw:
            continue
        for m in reversed(re.findall(r"\d{3}_[a-z0-9]+", str(raw).lower())):
            if m in seqs:
                seq_name = m
                break
        if seq_name:
            break
    if not seq_name:
        return {}, None
    out = {}
    for cname, mat in seqs[seq_name].get("cameras", {}).items():
        cid = _cam_num_from_text(cname)
        if cid is not None:
            out[cid] = np.asarray(mat, dtype=float)
    return out, seq_name


def _sequence_rotation(frames: list[tuple[dict, dict, list[str]]]) -> np.ndarray:
    """Single rotation R (pred -> truth) over all frames of a sequence, root = hip midpoint."""
    P_all, T_all = [], []
    for pred, truth, keys in frames:
        pr, tr = _resolve_root_joint(pred), _resolve_root_joint(truth)
        P_all.extend(np.asarray(pred[k], dtype=float) - pr for k in keys)
        T_all.extend(np.asarray(truth[k], dtype=float) - tr for k in keys)
    P, T = np.asarray(P_all), np.asarray(T_all)
    U, _, Vt = np.linalg.svd(T.T @ P)
    corr = np.eye(3)
    corr[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    return U @ corr @ Vt


def _compute_mpjpe_fixed_rotation(pred, truth, keys, R) -> tuple[float, dict[str, float]]:
    """Root-centred MPJPE with a given (sequence-level) rotation, no per-frame alignment."""
    pr, tr = _resolve_root_joint(pred), _resolve_root_joint(truth)
    errors = {k: float(np.linalg.norm(R @ (np.asarray(pred[k], dtype=float) - pr)
                                      - (np.asarray(truth[k], dtype=float) - tr)) * 1000.0) for k in keys}
    return float(np.mean(list(errors.values()))), errors


def _compute_pa_mpjpe(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str]) -> tuple[float, dict[str, float]]:
    P = np.array([pred[k] for k in keys], dtype=float)
    T = np.array([truth[k] for k in keys], dtype=float)

    P_c = P - np.mean(P, axis=0)
    T_c = T - np.mean(T, axis=0)

    norm_P = np.linalg.norm(P_c)
    norm_T = np.linalg.norm(T_c)
    
    if norm_P < 1e-6 or norm_T < 1e-6:
        raise ValueError("Cannot compute PA-MPJPE for a degenerate pose")

    P_c_unit = P_c / norm_P
    T_c_unit = T_c / norm_T

    U, s, Vt = np.linalg.svd(np.dot(T_c_unit.T, P_c_unit))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(np.dot(U, Vt)))
    R = np.dot(np.dot(U, correction), Vt)
    
    scale = np.sum(s * np.diag(correction)) * (norm_T / norm_P)
    P_aligned = np.dot(P_c, R.T) * scale + np.mean(T, axis=0)

    errors = []
    errors_dict = {}
    for i, k in enumerate(keys):
        err = float(np.linalg.norm(P_aligned[i] - T[i]) * 1000.0)
        errors.append(err)
        errors_dict[k] = err
        
    return float(np.mean(errors)), errors_dict

def _compute_pck(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], keys: list[str], threshold_mm: float, rotate: bool = True) -> tuple[float, dict[str, float]]:
    p_dict, t_dict = _align_camera_to_world_rigid(pred, truth, keys, rotate=rotate)
    correct = []
    correct_dict = {}
    for k in keys:
        err = float(np.linalg.norm(p_dict[k] - t_dict[k]) * 1000.0)
        value = 100.0 if err <= threshold_mm else 0.0
        correct.append(value)
        correct_dict[k] = value
    return float(np.mean(correct)), correct_dict


def _compute_mble(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], bones: list[list[str]]) -> tuple[float, dict]:
    details = {}
    for start, end in bones:
        if start not in pred or end not in pred or start not in truth or end not in truth:
            continue
        pred_length = float(np.linalg.norm(pred[start] - pred[end]) * 1000.0)
        truth_length = float(np.linalg.norm(truth[start] - truth[end]) * 1000.0)
        details[f"{start}-{end}"] = {
            "pred_length_mm": pred_length,
            "truth_length_mm": truth_length,
            "error_mm": abs(pred_length - truth_length),
        }
    errors = [values["error_mm"] for values in details.values()]
    return (float(np.mean(errors)) if errors else float("nan")), details


def _compute_acceleration_error(
    pred_triplet: tuple[dict, dict, dict],
    truth_triplet: tuple[dict, dict, dict],
    keys: list[str],
    rotation: Optional[np.ndarray] = None,
) -> tuple[float, dict]:
    pred_prev, pred_current, pred_next = pred_triplet
    truth_prev, truth_current, truth_next = truth_triplet
    details = {}
    for key in keys:
        pred_accel = pred_next[key] - 2.0 * pred_current[key] + pred_prev[key]
        if rotation is not None:
            pred_accel = rotation @ pred_accel
        truth_accel = truth_next[key] - 2.0 * truth_current[key] + truth_prev[key]
        details[key] = {
            "pred_accel_mm_frame2": float(np.linalg.norm(pred_accel) * 1000.0),
            "truth_accel_mm_frame2": float(np.linalg.norm(truth_accel) * 1000.0),
            "error_mm_frame2": float(np.linalg.norm(pred_accel - truth_accel) * 1000.0),
        }
    errors = [values["error_mm_frame2"] for values in details.values()]
    return (float(np.mean(errors)) if errors else float("nan")), details

def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _identity_excluded_ids(mask_cfg, pkl_stem: str, person: str) -> set[int]:
    """GT frame ids whose input track follows the wrong person (see build_identity_mask.py).

    mask_cfg: {"path": <json>, "margin_mm": float} or None. Evaluation-set filter only.
    """
    if not mask_cfg or not mask_cfg.get("path"):
        return set()
    data = mask_cfg.get("_data")
    if data is None:
        # path may be one file or a list (one mask per dataset); margins are merged by pkl stem.
        raw_paths = mask_cfg["path"] if isinstance(mask_cfg["path"], (list, tuple)) else [mask_cfg["path"]]
        data = {"margins": {}}
        for raw in raw_paths:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            data["margins"].update(_load_json(path).get("margins", {}))
        mask_cfg["_data"] = data
    margin = float(mask_cfg.get("margin_mm", 10.0))
    per_frame = data.get("margins", {}).get(pkl_stem, {}).get(str(person).lower(), {})
    return {int(fid) for fid, value in per_frame.items() if float(value) > margin}


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


HUGGING_SMPL_INDICES = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8,
    "left_foot": 10, "right_foot": 11,
    "neck": 12, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hand": 22, "right_hand": 23,
}


@lru_cache(maxsize=4)
def _load_npy_gt_cached(search_dir_abs: str, person_key: str, sig: tuple):
    # ponytail: cache parsed Harmony4D .npy GT per worker (each pair loaded 300-600 .npy files x2)
    npy_files = [Path(search_dir_abs) / name for name, _ in sig]
    gt_map = {}
    for idx, p in enumerate(npy_files):
        fid = int(p.stem) - 1 if p.stem.isdigit() else idx
        try:
            d = np.load(p, allow_pickle=True).item()
        except Exception:
            continue
        matched_person = next((k for k in d.keys() if str(k).lower() == person_key), None)
        if matched_person is None and not person_key:
            matched_person = next(iter(d.keys())) if d else None
        if matched_person and matched_person in d:
            pdata = d[matched_person]
            if isinstance(pdata, dict) and "joints" in pdata:
                raw_j = pdata["joints"]
                joints = {k: np.array(raw_j[i], dtype=float) for k, i in HUGGING_SMPL_INDICES.items() if i < len(raw_j)}
                gt_map[fid] = {"joints": joints, "frame_id": fid}
            elif isinstance(pdata, np.ndarray) and pdata.ndim == 2:
                coco_map = {
                    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
                    "left_shoulder": 5, "right_shoulder": 6,
                    "left_elbow": 7, "right_elbow": 8,
                    "left_wrist": 9, "right_wrist": 10,
                    "left_hip": 11, "right_hip": 12,
                    "left_knee": 13, "right_knee": 14,
                    "left_ankle": 15, "right_ankle": 16,
                }
                joints = {k: np.array(pdata[i, :3], dtype=float) for k, i in coco_map.items() if i < len(pdata)}
                gt_map[fid] = {"joints": joints, "frame_id": fid}
    return gt_map


def _load_ground_truth(path: Path, truth_dir: Path = None, target_person: str = "aria01") -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        gt_map = {}
        for idx, item in enumerate(data):
            gt_map[idx] = item
            if "frame_id" in item:
                gt_map[int(item["frame_id"])] = item
        return gt_map

    search_dir = truth_dir or path.parent
    npy_files = sorted(search_dir.glob("*.npy"))
    if not npy_files and (search_dir / "poses3d").exists():
        npy_files = sorted((search_dir / "poses3d").glob("*.npy"))
    if not npy_files and (search_dir / "smpl").exists():
        npy_files = sorted((search_dir / "smpl").glob("*.npy"))

    if npy_files and any(p.stem.isdigit() for p in npy_files):
        # Download duplicates such as "00059(1).npy" would otherwise be keyed by list position.
        skipped = [p.name for p in npy_files if not p.stem.isdigit()]
        if skipped:
            print(f"[Evaluation] GT: ignoring {len(skipped)} non-numeric file(s), e.g. {skipped[:3]}")
        npy_files = [p for p in npy_files if p.stem.isdigit()]

    if npy_files:
        person_key = str(target_person or "aria01").lower()
        if person_key in ("aria1", "aria_1", "aria-1"):
            person_key = "aria01"
        elif person_key in ("aria2", "aria_2", "aria-2"):
            person_key = "aria02"
        resolved_dir = os.path.abspath(str(npy_files[0].parent))
        sig = tuple((p.name, p.stat().st_mtime_ns) for p in npy_files)
        gt_map = _load_npy_gt_cached(resolved_dir, person_key, sig)
        if gt_map:
            return gt_map

    raise FileNotFoundError(f"GT file not found: {path} and no .npy GT in {search_dir}")


def _load_segment_gt(path: Path) -> dict:
    return _load_ground_truth(path)


def _parse_new_gt(item: dict, canonical_names: set, map_data: dict) -> dict:
    if "joints" in item:
        return item["joints"]
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
    from config_loader import resolve_inputs

    eval_cfg = config.get("evaluation", {})
    if not eval_cfg["enabled"]:
        print("[Evaluation] Disabled by config: evaluation.enabled=false")
        return

    paths = config["paths"]
    inputs = resolve_inputs(config)
    map_data = load_keypoints3d_map(paths["keypoints3d_map"])
    joint_names = [k["name"] for k in map_data["keypoints"]]
    canonical_names = set(joint_names)
    priority1_names = map_data.get("priority1", [])
    priority2_names = map_data.get("priority2", [])
    joint_set = str(eval_cfg.get("joint_set", "priority1")).lower()
    if joint_set == "coco17":
        # Harmony4D 3D keypoints are COCO-17; face points come from "<cam>_face" in the outputs.
        priority1_names = list(COCO17_NAMES)
    elif joint_set != "priority1":
        raise ValueError(f"evaluation.joint_set must be 'priority1' or 'coco17', got {joint_set!r}")
    skeleton = map_data.get("skeleton", [])

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
    pck_threshold_mm = float(metrics_cfg.get("pck_threshold_mm", 150.0))
    enabled_metrics = [name for name, enabled in metric_enabled.items() if enabled]
    mble_enabled = bool(metrics_cfg.get("mble", False))
    accel_enabled = bool(metrics_cfg.get("accel", False))
    if not enabled_metrics and not mble_enabled and not accel_enabled:
        raise ValueError("At least one evaluation metric must be enabled")
    if mble_enabled and not skeleton:
        raise ValueError("MBLE requires skeleton entries in the 3D keypoint map")

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
    
    target_person = inputs.get("target_person", "aria01")
    gt_cam1_data = _load_ground_truth(cam1_gt_path, truth_dir=truth_dir, target_person=target_person)
    gt_cam2_data = _load_ground_truth(cam2_gt_path, truth_dir=truth_dir, target_person=target_person)
    gt_cam_data = {"camera1": gt_cam1_data, "camera2": gt_cam2_data}

    # EXT=1: move the world-frame GT into each camera frame with the real COLMAP extrinsics, so
    # MPJPE/PCK need no fitted rotation (root-centred only) and PA-MPJPE sees camera-frame GT.
    use_ext = ext_enabled()
    ext_info = {}
    if use_ext:
        ext_cfg = eval_cfg.get("extrinsics") or {}
        if config.get("pose", {}).get("use_world_coords", False):
            raise ValueError("EXT=1 compares against camera-frame GT, but pose.use_world_coords=true "
                             "produces WHAM world-frame predictions; set it to false.")
        # Source per camera: the camera's own YAML `extrinsic` block (inputs.<cam>_extrinsic, set by
        # the runners) wins; otherwise colmap/workplace images.txt (+ scale.npy).
        workplace = None
        for cam, idx in (("camera1", "1"), ("camera2", "2")):
            cam_name = (ext_cfg.get(f"{cam}_name")
                        or camera_name_from_path(inputs.get(f"cam{idx}_pkl"), inputs.get(f"{cam}_video")))
            block = inputs.get(f"{cam}_extrinsic")
            if block:
                R_ext, t_ext = extrinsic_from_config(block, cam_name or cam)
                source = "yaml"
            else:
                if not cam_name:
                    raise ValueError(f"EXT=1: cannot infer the COLMAP camera name of {cam}; "
                                     f"set evaluation.extrinsics.{cam}_name (e.g. cam03)")
                if workplace is None:
                    workplace = find_colmap_workplace(truth_dir, inputs.get("cam1_pkl"),
                                                      inputs.get("camera1_video"),
                                                      explicit=ext_cfg.get("colmap_dir"))
                R_ext, t_ext = load_world_to_camera(str(workplace), cam_name)
                source = str(workplace)
            gt_cam_data[cam] = transform_truth_map(gt_cam_data[cam], R_ext, t_ext)
            ext_info[cam] = {"camera": cam_name, "source": source}
            print(f"[Evaluation] EXT=1: {cam} <- {cam_name} extrinsics from {source} "
                  f"(|t|={np.linalg.norm(t_ext):.2f} m)")
        if mpjpe_alignment_requested(eval_cfg) == "sequence":
            print("[Evaluation] EXT=1: evaluation.mpjpe_alignment=sequence ignored "
                  "(MPJPE uses the real extrinsics, root-centred, no fitted rotation)")
    gt_cam1_data, gt_cam2_data = gt_cam_data["camera1"], gt_cam_data["camera2"]
    with (out_dir / "evaluation_protocol.json").open("w", encoding="utf-8") as f:
        json.dump({
            "ext": use_ext,
            "truth_frame": "camera (real extrinsics)" if use_ext else "world (raw GT)",
            "mpjpe_alignment": "root-centred, real extrinsics" if use_ext
                               else mpjpe_alignment_requested(eval_cfg),
            "colmap_cameras": ext_info,
        }, f, indent=2)
    # #region agent log
    _agent_dbg("H4", "evaluation_pipeline/executor.py:run_evaluation", "evaluation GT config", {
        "truth_dir": str(truth_dir), "target_person": target_person,
        "n_gt_cam1": len(gt_cam1_data), "n_gt_cam2": len(gt_cam2_data),
        "priority1": list(priority1_names),
    })
    # #endregion

    gt_camera_keys = {
        "camera1": _camera_key_from_video_path(inputs.get("camera1_video") or inputs.get("cam1_pkl")),
        "camera2": _camera_key_from_video_path(inputs.get("camera2_video") or inputs.get("cam2_pkl")),
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
    mble_results = {cam: {} for cam in CAMERAS}
    sequences = {cam: {} for cam in CAMERAS}
    source_frame_ids = {cam: {} for cam in CAMERAS}
    evaluated_frames = []

    mpjpe_alignment = mpjpe_alignment_requested(eval_cfg)
    if use_ext:
        mpjpe_alignment = "extrinsics"
    mpjpe_cache = {cam: {} for cam in CAMERAS}

    id_mask_cfg = eval_cfg.get("identity_mask")
    identity_excluded = {
        "camera1": _identity_excluded_ids(id_mask_cfg, cam1_gt_stem, target_person),
        "camera2": _identity_excluded_ids(id_mask_cfg, cam2_gt_stem, target_person),
    }
    identity_skipped = 0

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
        if cam1_frame_id in identity_excluded["camera1"] or cam2_frame_id in identity_excluded["camera2"]:
            identity_skipped += 1
            continue

        evaluated_frames.append(frame)

        fused_metadata_path = Path(paths.get("fused_output_dir", "")) / "metadata" / f"fused_data_{frame}.json"
        fused_meta = _load_json(fused_metadata_path) if fused_metadata_path.exists() else {}
        vis_by_cam = {
            "camera1": fused_meta.get("vis1", {}),
            "camera2": fused_meta.get("vis2", {}),
        }

        for cam in CAMERAS:
            gt_frame_id = cam1_frame_id if cam == "camera1" else cam2_frame_id
            truth_item = gt_cam_data[cam][gt_frame_id]
            truth_joints = _parse_new_gt(truth_item, canonical_names, map_data)
            vis_map = vis_by_cam.get(cam, {})

            if accel_enabled:
                source_frame_ids[cam][frame] = gt_frame_id
                sequences[cam][frame] = {"truth": truth_joints}

            for metric in results:
                if frame not in results[metric][cam]:
                    results[metric][cam][frame] = {}

            preds_by_mod = {}
            for mod in module_names:
                mod_path = module_frame_maps[mod][frame]
                mod_data = _load_json(mod_path)
                if cam not in mod_data:
                    raise ValueError(f"Missing {cam} in {mod_path}")
                pj = {k: np.array(v, dtype=float) for k, v in mod_data[cam].items()}
                # Optional COCO face points (nose/eyes/ears) stored outside the 21-joint skeleton.
                for k, v in (mod_data.get(f"{cam}_face") or {}).items():
                    pj.setdefault(k, np.array(v, dtype=float))
                preds_by_mod[mod] = pj
            # Score every module on the same joints (e.g. face points present in all modules).
            common_keys = set(truth_joints.keys()).intersection(*(set(p) for p in preds_by_mod.values()))

            for mod in module_names:
                pred_joints = preds_by_mod[mod]

                if mble_enabled:
                    mean_mble, bone_details = _compute_mble(pred_joints, truth_joints, skeleton)
                    mble_results[cam].setdefault(frame, {})[mod] = {
                        "mean_mm": mean_mble,
                        "details": bone_details,
                    }
                if accel_enabled:
                    sequences[cam][frame][mod] = pred_joints
                
                # Filter priority names to keys present in the truth and in every module's prediction
                available_keys = common_keys
                valid_priority1 = [k for k in priority1_names if k in available_keys]
                valid_priority2 = [k for k in priority2_names if k in available_keys]
                
                if not valid_priority1:
                    print(f"Warning: No overlapping priority1 keys for {mod} {cam} frame {frame}")

                occ_keys = [k for k in valid_priority1 if not vis_map.get(k, True)]
                vis_keys = [k for k in valid_priority1 if vis_map.get(k, True)]

                # compute metrics
                # MPJPE
                if metric_enabled["MPJPE"]:
                    results["MPJPE"][cam][frame].setdefault(mod, {})
                    mean_p1, dict_p1 = _compute_mpjpe(pred_joints, truth_joints, valid_priority1, rotate=not use_ext) if valid_priority1 else (0.0, {})
                    results["MPJPE"][cam][frame][mod]["priority1_mm"] = mean_p1
                    results["MPJPE"][cam][frame][mod]["priority1_details"] = dict_p1
                    
                    mean_p2, dict_p2 = _compute_mpjpe(pred_joints, truth_joints, valid_priority2, rotate=not use_ext) if valid_priority2 else (0.0, {})
                    results["MPJPE"][cam][frame][mod]["priority2_mm"] = mean_p2
                    results["MPJPE"][cam][frame][mod]["priority2_details"] = dict_p2

                    occ_errs = [dict_p1[k] for k in occ_keys if k in dict_p1]
                    vis_errs = [dict_p1[k] for k in vis_keys if k in dict_p1]
                    results["MPJPE"][cam][frame][mod]["occ_mm"] = float(np.mean(occ_errs)) if occ_errs else float("nan")
                    results["MPJPE"][cam][frame][mod]["vis_mm"] = float(np.mean(vis_errs)) if vis_errs else float("nan")
                    if mpjpe_alignment == "sequence" and valid_priority1:
                        mpjpe_cache[cam].setdefault(mod, {})[frame] = (
                            pred_joints, truth_joints, valid_priority1, valid_priority2, occ_keys, vis_keys)

                if metric_enabled["PA-MPJPE"]:
                    results["PA-MPJPE"][cam][frame].setdefault(mod, {})
                    mean_p1, dict_p1 = _compute_pa_mpjpe(pred_joints, truth_joints, valid_priority1) if valid_priority1 else (0.0, {})
                    results["PA-MPJPE"][cam][frame][mod]["priority1_mm"] = mean_p1
                    results["PA-MPJPE"][cam][frame][mod]["priority1_details"] = dict_p1
                    
                    mean_p2, dict_p2 = _compute_pa_mpjpe(pred_joints, truth_joints, valid_priority2) if valid_priority2 else (0.0, {})
                    results["PA-MPJPE"][cam][frame][mod]["priority2_mm"] = mean_p2
                    results["PA-MPJPE"][cam][frame][mod]["priority2_details"] = dict_p2

                    occ_errs = [dict_p1[k] for k in occ_keys if k in dict_p1]
                    vis_errs = [dict_p1[k] for k in vis_keys if k in dict_p1]
                    results["PA-MPJPE"][cam][frame][mod]["occ_mm"] = float(np.mean(occ_errs)) if occ_errs else float("nan")
                    results["PA-MPJPE"][cam][frame][mod]["vis_mm"] = float(np.mean(vis_errs)) if vis_errs else float("nan")

                if metric_enabled["PCK"]:
                    results["PCK"][cam][frame].setdefault(mod, {})
                    mean_p1, dict_p1 = _compute_pck(pred_joints, truth_joints, valid_priority1, pck_threshold_mm, rotate=not use_ext) if valid_priority1 else (0.0, {})
                    results["PCK"][cam][frame][mod]["priority1_mm"] = mean_p1
                    results["PCK"][cam][frame][mod]["priority1_details"] = dict_p1
                    
                    mean_p2, dict_p2 = _compute_pck(pred_joints, truth_joints, valid_priority2, pck_threshold_mm, rotate=not use_ext) if valid_priority2 else (0.0, {})
                    results["PCK"][cam][frame][mod]["priority2_mm"] = mean_p2
                    results["PCK"][cam][frame][mod]["priority2_details"] = dict_p2

                    occ_errs = [dict_p1[k] for k in occ_keys if k in dict_p1]
                    vis_errs = [dict_p1[k] for k in vis_keys if k in dict_p1]
                    results["PCK"][cam][frame][mod]["occ_mm"] = float(np.mean(occ_errs)) if occ_errs else float("nan")
                    results["PCK"][cam][frame][mod]["vis_mm"] = float(np.mean(vis_errs)) if vis_errs else float("nan")

    colmap_path = Path(inputs["colmap_images_txt"]) if inputs.get("colmap_images_txt") else _find_colmap_images_txt(
        truth_dir, (inputs.get("cam1_pkl"), inputs.get("cam2_pkl"))
    )
    extrinsics_source = None
    colmap_rotations = {}
    if colmap_path and colmap_path.is_file():
        colmap_rotations = _load_colmap_cam_to_world_rotations(colmap_path)
        extrinsics_source = str(colmap_path)
    if not colmap_rotations:
        colmap_rotations, bundled_seq = _load_bundled_harmony4d_rotations(
            (str(truth_dir), inputs.get("cam1_pkl"), inputs.get("cam2_pkl"))
        )
        if bundled_seq:
            extrinsics_source = f"configs/harmony4d_extrinsics.json:{bundled_seq}"
    cam_rotations = {}
    for cam in CAMERAS:
        pkl_key, vid_key = ("cam1_pkl", "camera1_video") if cam == "camera1" else ("cam2_pkl", "camera2_video")
        cid = _cam_num_from_text(inputs.get(pkl_key))
        if cid is None:
            cid = _cam_num_from_text(inputs.get(vid_key))
        if cid is not None and cid in colmap_rotations:
            cam_rotations[cam] = colmap_rotations[cid]
    if cam_rotations and not use_ext:  # EXT=1 never uses these rotations (truth is camera-frame)
        print(f"[Evaluation] Using Harmony4D COLMAP extrinsics (0% GT) from {extrinsics_source} for {sorted(cam_rotations)}")

    if metric_enabled["MPJPE"] and mpjpe_alignment == "sequence":
        # Prefer true COLMAP camera extrinsics (0% GT). Only if images.txt is missing for a camera,
        # fall back to one shared rotation estimated on baseline `posed` (never re-fit per module).
        for cam in CAMERAS:
            R = cam_rotations.get(cam)
            if R is None:
                ref_mod = "posed" if "posed" in mpjpe_cache[cam] else next(iter(mpjpe_cache[cam]), None)
                if ref_mod is None:
                    continue
                R = _sequence_rotation([(p, t, k1) for p, t, k1, *_ in mpjpe_cache[cam][ref_mod].values()])
                cam_rotations[cam] = R
                print(f"[Evaluation] WARNING: no COLMAP images.txt extrinsics for {cam}; falling back to shared posed R")
            for mod, frames_data in mpjpe_cache[cam].items():
                for frame, (p, t, k1, k2, occ_keys, vis_keys) in frames_data.items():
                    entry = results["MPJPE"][cam][frame][mod]
                    m1, d1 = _compute_mpjpe_fixed_rotation(p, t, k1, R)
                    entry["priority1_mm"], entry["priority1_details"] = m1, d1
                    if k2:
                        entry["priority2_mm"], entry["priority2_details"] = _compute_mpjpe_fixed_rotation(p, t, k2, R)
                    occ_errs = [d1[k] for k in occ_keys if k in d1]
                    vis_errs = [d1[k] for k in vis_keys if k in d1]
                    entry["occ_mm"] = float(np.mean(occ_errs)) if occ_errs else float("nan")
                    entry["vis_mm"] = float(np.mean(vis_errs)) if vis_errs else float("nan")

    module_output_names = {
        mod: EVALUATION_OUTPUT_MODULE_NAMES.get(mod, mod)
        for mod in module_names
    }
    # #region agent log
    if metric_enabled.get("PA-MPJPE") and "posed" in module_names and "fused" in module_names:
        for cam in CAMERAS:
            posed_vals = [results["PA-MPJPE"][cam][fr]["posed"]["priority1_mm"] for fr in evaluated_frames if fr in results["PA-MPJPE"][cam]]
            fused_vals = [results["PA-MPJPE"][cam][fr]["fused"]["priority1_mm"] for fr in evaluated_frames if fr in results["PA-MPJPE"][cam]]
            if not posed_vals or not fused_vals:
                continue
            b = float(np.mean(posed_vals)); m = float(np.mean(fused_vals))
            _agent_dbg("H2", "evaluation_pipeline/executor.py:run_evaluation", "PA-MPJPE posed vs fused", {
                "cam": cam, "n_frames": len(posed_vals), "posed_mean_mm": b, "fused_mean_mm": m,
                "abs_gain_mm": b - m, "pct_gain": ((b - m) * 100.0 / b) if b else None,
                "truth_dir": str(truth_dir), "target_person": target_person,
            })
    # #endregion

    accel_results = {cam: {} for cam in CAMERAS}
    if accel_enabled:
        for cam in CAMERAS:
            # EXT=1: truth was moved into the camera frame with the real extrinsics, like the
            # prediction, so the accelerations are compared as they are (no camera->world rotation).
            R_cam = None if use_ext else cam_rotations.get(cam)
            if R_cam is None and not use_ext and "posed" in module_names:
                ref_frames = [
                    (sequences[cam][f]["posed"], sequences[cam][f]["truth"],
                     [k for k in joint_names if k in sequences[cam][f]["posed"] and k in sequences[cam][f]["truth"]])
                    for f in evaluated_frames
                    if f in sequences[cam] and "posed" in sequences[cam][f]
                    and "left_hip" in sequences[cam][f]["posed"] and "right_hip" in sequences[cam][f]["posed"]
                    and "left_hip" in sequences[cam][f]["truth"] and "right_hip" in sequences[cam][f]["truth"]
                ]
                if ref_frames and ref_frames[0][2]:
                    R_cam = _sequence_rotation(ref_frames)
            for previous_frame, frame, next_frame in zip(evaluated_frames, evaluated_frames[1:], evaluated_frames[2:]):
                if source_frame_ids[cam][frame] - source_frame_ids[cam][previous_frame] != 1:
                    continue
                if source_frame_ids[cam][next_frame] - source_frame_ids[cam][frame] != 1:
                    continue

                truth_triplet = tuple(sequences[cam][f]["truth"] for f in (previous_frame, frame, next_frame))
                for mod in module_names:
                    pred_triplet = tuple(sequences[cam][f][mod] for f in (previous_frame, frame, next_frame))
                    keys = [
                        key for key in joint_names
                        if all(key in joints for joints in (*pred_triplet, *truth_triplet))
                    ]
                    mean_accel, joint_details = _compute_acceleration_error(
                        pred_triplet, truth_triplet, keys, rotation=R_cam
                    )
                    accel_results[cam].setdefault(frame, {})[mod] = {
                        "mean_mm_frame2": mean_accel,
                        "details": joint_details,
                    }

    for metric in enabled_metrics:
        unit = "pct" if metric == "PCK" else "mm"
        header = ["Frame", "Evaluated_Camera", "Ground_Truth_Camera"]
        for mod in module_names:
            out_name = module_output_names[mod]
            header.append(f"{out_name}_priority1_{unit}")
            header.append(f"{out_name}_priority2_{unit}")
            header.append(f"{out_name}_occ_{unit}")
            header.append(f"{out_name}_vis_{unit}")
            for joint in priority1_names:
                header.append(f"{out_name}_{joint}_{unit}")

        for cam in CAMERAS:
            filename = f"{metric}_{CAMERA_FILE_NAMES[cam]}.csv"
            out_file = out_dir / filename
            with out_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

                avg_sums = {h: 0.0 for h in header[3:]}
                avg_counts = {h: 0 for h in header[3:]}

                for frame in evaluated_frames:
                    row = [frame, cam, gt_camera_keys[cam]]
                    for mod in module_names:
                        out_name = module_output_names[mod]
                        
                        v1 = results[metric][cam][frame][mod]["priority1_mm"]
                        v2 = results[metric][cam][frame][mod]["priority2_mm"]
                        row.append(f"{v1:.2f}")
                        row.append(f"{v2:.2f}")
                        avg_sums[f"{out_name}_priority1_{unit}"] += v1
                        avg_sums[f"{out_name}_priority2_{unit}"] += v2
                        avg_counts[f"{out_name}_priority1_{unit}"] += 1
                        avg_counts[f"{out_name}_priority2_{unit}"] += 1

                        v_occ = results[metric][cam][frame][mod].get("occ_mm", float("nan"))
                        v_vis = results[metric][cam][frame][mod].get("vis_mm", float("nan"))

                        if np.isnan(v_occ):
                            row.append("N/A")
                        else:
                            row.append(f"{v_occ:.2f}")
                            avg_sums[f"{out_name}_occ_{unit}"] += v_occ
                            avg_counts[f"{out_name}_occ_{unit}"] += 1

                        if np.isnan(v_vis):
                            row.append("N/A")
                        else:
                            row.append(f"{v_vis:.2f}")
                            avg_sums[f"{out_name}_vis_{unit}"] += v_vis
                            avg_counts[f"{out_name}_vis_{unit}"] += 1
                        
                        p1_details = results[metric][cam][frame][mod]["priority1_details"]
                        for joint in priority1_names:
                            value = p1_details.get(joint)
                            if value is None:
                                row.append("N/A")
                            else:
                                row.append(f"{value:.2f}")
                                avg_sums[f"{out_name}_{joint}_{unit}"] += value
                                avg_counts[f"{out_name}_{joint}_{unit}"] += 1

                    writer.writerow(row)

                n_frames = len(evaluated_frames)
                if n_frames > 0:
                    avg_row = ["AVERAGE", cam, gt_camera_keys[cam]]
                    for h in header[3:]:
                        c = avg_counts[h]
                        avg_row.append(f"{(avg_sums[h]/c):.2f}" if c > 0 else "N/A")
                    writer.writerow(avg_row)

    if mble_enabled:
        header = [
            "Frame", "Evaluated_Camera", "Ground_Truth_Camera", "Module", "Bone",
            "Pred_Length_mm", "GT_Length_mm", "Bone_Error_mm", "Frame_MBLE_mm",
        ]
        for cam in CAMERAS:
            out_file = out_dir / f"MBLE_{CAMERA_FILE_NAMES[cam]}.csv"
            with out_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for frame in evaluated_frames:
                    for mod in module_names:
                        result = mble_results[cam][frame][mod]
                        for bone, values in result["details"].items():
                            writer.writerow([
                                frame, cam, gt_camera_keys[cam], module_output_names[mod], bone,
                                f'{values["pred_length_mm"]:.2f}',
                                f'{values["truth_length_mm"]:.2f}',
                                f'{values["error_mm"]:.2f}',
                                f'{result["mean_mm"]:.2f}',
                            ])

    if accel_enabled:
        header = [
            "Frame", "Evaluated_Camera", "Ground_Truth_Camera", "Module", "Joint",
            "Pred_Accel_mm_frame2", "GT_Accel_mm_frame2", "Accel_Error_mm_frame2", "Frame_Accel_Error_mm_frame2",
        ]
        for cam in CAMERAS:
            out_file = out_dir / f"Accel_{CAMERA_FILE_NAMES[cam]}.csv"
            with out_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for frame in evaluated_frames:
                    for mod in module_names:
                        result = accel_results[cam].get(frame, {}).get(mod)
                        if result is None:
                            continue
                        for joint, values in result["details"].items():
                            writer.writerow([
                                frame, cam, gt_camera_keys[cam], module_output_names[mod], joint,
                                f'{values["pred_accel_mm_frame2"]:.2f}',
                                f'{values["truth_accel_mm_frame2"]:.2f}',
                                f'{values["error_mm_frame2"]:.2f}',
                                f'{result["mean_mm_frame2"]:.2f}',
                            ])

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

    if id_mask_cfg and id_mask_cfg.get("path"):
        print(f"[Evaluation] Identity mask: excluded {identity_skipped} frame(s) "
              f"(margin > {float(id_mask_cfg.get('margin_mm', 10.0))} mm)")
    print(f"[Evaluation] Done. Output: {out_dir}")

"""## 12. Visualization phase"""

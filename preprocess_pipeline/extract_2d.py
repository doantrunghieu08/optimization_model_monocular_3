from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from json_io import read_json
from config_loader import resolve_inputs
from keypoints_map import load_keypoints2d_map
from json_io import write_json
from compat import load_joblib_compat, resolve_person_payload
from config_loader import resolve_preprocess_output_dir


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _load_tracking_payload(pkl_path: Path, target_person=None) -> Optional[dict]:
    if not pkl_path.exists():
        print(f"[Preprocess] 2D skip: PKL not found: {pkl_path}")
        return None

    person = resolve_person_payload(
        load_joblib_compat(pkl_path),
        pkl_path=pkl_path,
        target_person=target_person,
    )
    if person is not None:
        tracking = person.get("tracking_results_for_reproj")
        if isinstance(tracking, Mapping):
            return dict(tracking)

    tracking_files = sorted(pkl_path.parent.glob("tracking_results*.pth"))
    if tracking_files:
        pth_path = tracking_files[0]
        t_raw = load_joblib_compat(pth_path)
        t = resolve_person_payload(t_raw, pkl_path=pkl_path, target_person=target_person)
        if isinstance(t, Mapping) and "keypoints" in t:
            return dict(t)

    print(f"[Preprocess] 2D skip: tracking results missing in {pkl_path}")
    return None


def _load_keypoints2d_map(map_path: Path) -> dict[str, object]:
    return load_keypoints2d_map(str(map_path))


_FALLBACK_COCO_INDICES = {17: 15, 18: 15, 20: 16, 21: 16, 101: 9, 122: 10}


def _resolve_keypoint_spec(spec, source_keypoints: np.ndarray) -> list[float]:
    n_kp = len(source_keypoints)
    if isinstance(spec, int):
        idx = _FALLBACK_COCO_INDICES.get(spec, spec)
        if idx < n_kp:
            return source_keypoints[idx, :3].tolist()
        return [0.0, 0.0, 0.0]
    if isinstance(spec, Mapping) and "average" in spec:
        pts = [source_keypoints[_FALLBACK_COCO_INDICES.get(int(idx), int(idx)), :3] 
               for idx in spec["average"] 
               if _FALLBACK_COCO_INDICES.get(int(idx), int(idx)) < n_kp]
        return np.mean(pts, axis=0).tolist() if pts else [0.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0]


def _build_2d_camera_payload(tracking: dict, keypoints2d_map: dict[str, object]) -> tuple[dict, Optional[list]]:
    if "keypoints" not in tracking or "frame_id" not in tracking:
        missing = [k for k in ("frame_id", "keypoints") if k not in tracking]
        raise KeyError(f"Missing tracking key(s): {missing}")

    frame_ids = _to_numpy(tracking["frame_id"]).astype(int).reshape(-1)
    keypoints = _to_numpy(tracking["keypoints"]).astype(float)
    if keypoints.ndim != 3 or keypoints.shape[2] < 3:
        raise ValueError(f"Expected keypoints shape (frames, joints, 3+), got {keypoints.shape}")
    if keypoints.shape[0] != frame_ids.shape[0]:
        raise ValueError(f"frame_id/keypoints frame mismatch: {frame_ids.shape[0]} vs {keypoints.shape[0]}")

    by_frame = {}
    for row_idx, frame_id in enumerate(frame_ids):
        source_keypoints = keypoints[row_idx]
        resolved_keypoints: dict[str, list[float]] = {}
        for joint_name, spec in keypoints2d_map.items():
            resolved_keypoints[joint_name] = _resolve_keypoint_spec(spec, source_keypoints)

        if len(resolved_keypoints) != 21:
            raise ValueError(f"Expected 21 keys, got {len(resolved_keypoints)}")

        by_frame[str(int(frame_id))] = resolved_keypoints

    payload = {
        "frame_ids": frame_ids.tolist(),
        "keypoints": by_frame,
    }
    return payload, None


def export_tracking_2d_to_camera_profiles(config: dict) -> None:
    output_dir = Path(resolve_preprocess_output_dir(config))
    paths = config.get("paths", {})
    inputs = resolve_inputs(config)
    pkl_by_cam = {
        "cam1": Path(inputs.get("cam1_pkl", "")),
        "cam2": Path(inputs.get("cam2_pkl", "")),
    }
    keypoints2d_map = _load_keypoints2d_map(Path(paths["keypoints2d_map"]))

    for cam_id, pkl_path in pkl_by_cam.items():
        profile_path = output_dir / f"data_{cam_id}.json"
        if not profile_path.exists():
            print(f"[Preprocess] 2D skip: camera profile not found: {profile_path}")
            continue

        tracking = _load_tracking_payload(pkl_path, target_person=inputs.get("target_person"))
        if tracking is None:
            continue

        try:
            keypoints_2d, _ = _build_2d_camera_payload(tracking, keypoints2d_map)
        except Exception as exc:
            print(f"[Preprocess] 2D skip: {pkl_path.name}: {exc}")
            continue

        profile = read_json(profile_path)
        profile[f"2D_camera_{cam_id}"] = keypoints_2d
        write_json(profile_path, profile)
        print(f"[Preprocess] Exported 2D keypoints to {profile_path.name}")

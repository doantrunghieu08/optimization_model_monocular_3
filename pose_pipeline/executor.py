import copy
from pathlib import Path
import joblib
import numpy as np
from tqdm import tqdm  # Thêm thư viện tqdm tạo progress bar

from pose_pipeline.smpl_runner import create_smpl_model
from config_loader import resolve_inputs
from pose_pipeline.smpl_runner import get_3d_joints_for_frame
from json_io import write_json
from compat import load_joblib_compat, resolve_person_payload
from preprocess_pipeline.calib import resolve_selected_offset_from_camera_profile


POSE_OUTPUT_SUBDIRS = ("keypoints3d", "metadata")


def _extract_person_payload(data):
    return resolve_person_payload(data)


def load_pkl_data(file_path, target_person=None):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("PKL file not found: {}".format(file_path))
    person_data = resolve_person_payload(load_joblib_compat(file_path), pkl_path=file_path, target_person=target_person)
    if person_data is None:
        raise ValueError("Cannot read person payload from {}".format(file_path))
    for key in ("pose", "trans", "betas"):
        if key not in person_data:
            raise KeyError("Missing key {!r} in {}. Available keys: {}".format(key, file_path, list(person_data.keys())))
    return person_data


def slice_person_frames(person_data, start, frame_count):
    start = int(start)
    frame_count = int(frame_count)
    total_frames = int(np.asarray(person_data["pose"]).shape[0])
    end = min(total_frames, start + frame_count)
    out = copy.deepcopy(dict(person_data))
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_frames:
            out[key] = value[start:end].copy()
    return out


def _shared_betas(*person_data: dict) -> np.ndarray:
    betas = []
    for person in person_data:
        values = np.asarray(person["betas"], dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"Expected non-empty betas with shape (frames, dims), got {values.shape}")
        betas.append(values)
    if len({values.shape[1] for values in betas}) != 1:
        raise ValueError("Camera beta dimensions do not match")
    return np.median(np.concatenate(betas, axis=0), axis=0).astype(np.float32)


def _resolve_tracking_frame_ids(person_data):
    tracking = person_data.get("tracking_results_for_reproj")
    if not isinstance(tracking, dict) or "frame_id" not in tracking:
        return None

    frame_ids = np.asarray(tracking["frame_id"]).astype(int).reshape(-1)
    total_frames = int(np.asarray(person_data["pose"]).shape[0])
    if frame_ids.shape[0] != total_frames:
        print(
            "[Pose] Tracking frame_id length mismatch with pose frames: "
            f"{frame_ids.shape[0]} vs {total_frames}. Falling back to sequential indices."
        )
        return None
    return frame_ids.tolist()


def _clean_pose_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob("pose_data_*.json"):
        old_json.unlink(missing_ok=True)
    for subdir in POSE_OUTPUT_SUBDIRS:
        target_dir = output_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for old_json in target_dir.glob("pose_data_*.json"):
            old_json.unlink(missing_ok=True)


def run_pose_export(config: dict) -> None:
    paths = config["paths"]
    inputs = resolve_inputs(config)
    pose_cfg = config.get("pose_export", {})

    if not pose_cfg.get("enabled", True):
        print("[Pose] Disabled by config: pose_export.enabled=false")
        return

    cam1_file = Path(inputs["cam1_pkl"])
    cam2_file = Path(inputs["cam2_pkl"])
    smpl_model_path = Path(paths["smpl_model"])
    output_dir = Path(paths["pose_output_dir"])

    if config.get("runtime", {}).get("clean_output", True):
        _clean_pose_output(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("[Pose] Loading SMPL model...")
    model = create_smpl_model(smpl_model_path)

    target_person = inputs.get("target_person")
    print(f"[Pose] Loading camera 1: {cam1_file} (target_person={target_person})")
    cam1_data = load_pkl_data(cam1_file, target_person=target_person)
    print(f"[Pose] Loading camera 2: {cam2_file} (target_person={target_person})")
    cam2_data = load_pkl_data(cam2_file, target_person=target_person)
    cam1_tracking_frame_ids = _resolve_tracking_frame_ids(cam1_data)
    cam2_tracking_frame_ids = _resolve_tracking_frame_ids(cam2_data)

    offset, _ = resolve_selected_offset_from_camera_profile(config)
    cam1_start = max(0, -offset)
    cam2_start = max(0, offset)
    synced_total_frames = max(0, min(len(cam1_data["pose"]) - cam1_start, len(cam2_data["pose"]) - cam2_start))

    min_frames = synced_total_frames
    sync_result = {
        "offset": offset,
        "left_start": cam1_start,
        "right_start": cam2_start,
        "frame_count": min_frames,
        "source_pkls": {
            "camera1": str(cam1_file.resolve()),
            "camera2": str(cam2_file.resolve()),
        },
    }

    print(f"[Pose] Cam1 frames: {len(cam1_data['pose'])}")
    print(f"[Pose] Cam2 frames: {len(cam2_data['pose'])}")
    print(f"[Pose] Exporting full synced range: {min_frames} frames")
    print(f"[Pose] Using offset={offset}")
    print(f"[Pose] Exporting {min_frames} pose JSON files...")

    j_regressor_path = paths.get("j_regressor_3d", "optimization_model_monocular/models/J_regressor_body25_plus_palm27.npy")

    cam1_export_data = slice_person_frames(cam1_data, cam1_start, min_frames)
    cam2_export_data = slice_person_frames(cam2_data, cam2_start, min_frames)
    if min_frames:
        # Keep the master baseline identical while giving both views one body shape.
        shared_betas = _shared_betas(cam1_export_data)
        for camera_data in (cam1_export_data, cam2_export_data):
            camera_data["betas"] = np.repeat(shared_betas[None, :], min_frames, axis=0)

    use_world = bool(pose_cfg.get("use_world_coords", False))

    # Sử dụng tqdm bọc quanh range() để tạo progress bar
    for i in tqdm(range(min_frames), desc="[Pose] Exporting JSONs", unit="frame"):
        out_frame_id = i + 1
        keypoints3d_data = {
            "camera1": get_3d_joints_for_frame(model, cam1_export_data, i, j_regressor_path, paths["keypoints3d_map"], use_world=use_world),
            "camera2": get_3d_joints_for_frame(model, cam2_export_data, i, j_regressor_path, paths["keypoints3d_map"], use_world=use_world),
        }
        metadata_data = {
            "metadata": {
                "camera_sync": sync_result,
                "source_frame_indices": {
                    "camera1": int(cam1_tracking_frame_ids[cam1_start + i]) if cam1_tracking_frame_ids is not None else cam1_start + i,
                    "camera2": int(cam2_tracking_frame_ids[cam2_start + i]) if cam2_tracking_frame_ids is not None else cam2_start + i,
                },
            }
        }
        keypoints_path = output_dir / "keypoints3d" / f"pose_data_{out_frame_id}.json"
        metadata_path = output_dir / "metadata" / f"pose_data_{out_frame_id}.json"
        write_json(keypoints_path, keypoints3d_data)
        write_json(metadata_path, metadata_data)

    print(f"[Pose] Done. Output: {output_dir}")

"""## 8. Fusion phase"""

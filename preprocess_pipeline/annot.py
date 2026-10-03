import scipy.io
import json
import os
from pathlib import Path
import numpy as np


# ================= CONFIG =================

INPUT_SEQUENCES = [
    {
        "name": "s1_seq1",
        "image_sequence_dir": r"S1/Seq1/imageSequence",
    },
    {
        "name": "s1_seq2",
        "image_sequence_dir": r"S1/Seq2/imageSequence",
    },
    {
        "name": "s3_seq1",
        "image_sequence_dir": r"S3/Seq1/imageSequence",
    },
    {
        "name": "s3_seq2",
        "image_sequence_dir": r"S3/Seq2/imageSequence",
    },
]

OUTPUT_ROOT = "frames_json"

CAMERA_IDS = list(range(9))  # camera0 -> camera8

FPS = 25.0

JOINT_NAMES = [
    "spine3", "spine4", "spine2", "spine", "pelvis",
    "neck", "head", "head_top",
    "left_clavicle", "left_shoulder", "left_elbow", "left_wrist", "left_hand",
    "right_clavicle", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "left_hip", "left_knee", "left_ankle", "left_foot", "left_toe",
    "right_hip", "right_knee", "right_ankle", "right_foot", "right_toe",
]

# ==========================================


def mm_to_m(val: float) -> float:
    return float(val) / 1000.0


def resolve_annot_path(image_sequence_dir: str | Path) -> Path:

    image_sequence_dir = Path(image_sequence_dir)

    candidates = [
        image_sequence_dir / "annot.mat",
        image_sequence_dir.parent / "annot.mat",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Không tìm thấy annot.mat tại:\n"
        + "\n".join(str(p) for p in candidates)
    )


def get_camera_matrix(annot3: np.ndarray, cam_id: int) -> np.ndarray:
    """
    Lấy dữ liệu annot3 của 1 camera.

    scipy.io.loadmat thường đọc MATLAB cell array thành object array.
    Tùy file, shape có thể là:
      - (1, num_cams)
      - (num_cams, 1)
      - flat object array

    Output chuẩn: numpy array shape (frames, joints * 3)
    """
    if annot3.dtype == object:
        if annot3.ndim == 2:
            if annot3.shape[0] == 1:
                cam_data = annot3[0, cam_id]
            elif annot3.shape[1] == 1:
                cam_data = annot3[cam_id, 0]
            else:
                cam_data = annot3.flat[cam_id]
        else:
            cam_data = annot3.flat[cam_id]
    else:
        # Fallback nếu annot3 không phải cell array.
        # Ít gặp hơn với MPI-INF-3DHP annot.mat.
        cam_data = annot3[cam_id]

    cam_data = np.asarray(cam_data)

    if cam_data.ndim != 2:
        raise ValueError(
            f"Camera {cam_id}: dữ liệu annot3 phải là ma trận 2D, "
            f"nhưng nhận được shape {cam_data.shape}"
        )

    # Trường hợp bị đọc ngược shape: (84, frames) thay vì (frames, 84)
    if cam_data.shape[0] == len(JOINT_NAMES) * 3 and cam_data.shape[1] != len(JOINT_NAMES) * 3:
        cam_data = cam_data.T

    cols = cam_data.shape[1]
    expected_cols = len(JOINT_NAMES) * 3

    if cols != expected_cols:
        raise ValueError(
            f"Camera {cam_id}: số cột không khớp 28 joints * 3.\n"
            f"Expected: {expected_cols}, got: {cols}, shape={cam_data.shape}"
        )

    return cam_data


def matrix_to_pose3(cam_matrix: np.ndarray) -> np.ndarray:
    """
    Convert từ shape:
      (frames, 84)
    sang:
      (frames, 28, 3)
    """
    frames = cam_matrix.shape[0]
    joints = len(JOINT_NAMES)
    return cam_matrix.reshape(frames, joints, 3)


def keypoints_to_dict(coords_28x3: np.ndarray) -> dict:
    """
    Convert keypoints từ mm sang meters.

    Output:
    {
      "spine3": [x, y, z],
      ...
    }
    """
    result = {}

    for i, name in enumerate(JOINT_NAMES):
        x = mm_to_m(coords_28x3[i, 0])
        y = mm_to_m(coords_28x3[i, 1])
        z = mm_to_m(coords_28x3[i, 2])
        result[name] = [x, y, z]

    return result


def build_frame_json(frame_id: int, pose_by_camera: dict[int, np.ndarray]) -> dict:
    """
    Tạo JSON cho một frame.
    Mỗi frame chứa đủ camera0 -> camera8.
    """
    frame_data = {
        "frame_id": frame_id,
        "frame_name": f"frame_{frame_id:06d}",
        "timestamp_sec": frame_id / FPS,
        "coordinate_system": "absolute_camera_space_m",
        "skeleton": "MPI-INF-3DHP_28joints_VNect",
        "units": "meters",
    }

    for cam_id in CAMERA_IDS:
        frame_data[f"camera{cam_id}"] = keypoints_to_dict(
            pose_by_camera[cam_id][frame_id]
        )

    return frame_data


def export_sequence(seq_name: str, image_sequence_dir: str | Path):
    annot_path = resolve_annot_path(image_sequence_dir)
    output_dir = Path(OUTPUT_ROOT) / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[SEQUENCE] {seq_name}")
    print(f"[LOAD] {annot_path}")

    data = scipy.io.loadmat(annot_path)
    annot3 = data["annot3"]

    print(f"[INFO] annot3 shape: {annot3.shape}, dtype={annot3.dtype}")

    pose_by_camera: dict[int, np.ndarray] = {}
    frame_counts: dict[int, int] = {}

    for cam_id in CAMERA_IDS:
        cam_matrix = get_camera_matrix(annot3, cam_id)
        pose3 = matrix_to_pose3(cam_matrix)

        pose_by_camera[cam_id] = pose3
        frame_counts[cam_id] = pose3.shape[0]

        print(f"  camera{cam_id}: raw={cam_matrix.shape}, pose={pose3.shape}")

    min_frames = min(frame_counts.values())
    max_frames = max(frame_counts.values())

    print()
    print(f"[SYNC] min_frames = {min_frames}")
    print(f"[SYNC] max_frames = {max_frames}")

    if min_frames != max_frames:
        print("[SYNC] Có camera thừa frame cuối. Sẽ cắt toàn bộ theo min_frames.")
        for cam_id, n in frame_counts.items():
            if n != min_frames:
                print(f"  camera{cam_id}: {n} -> {min_frames}")

    print(f"[EXPORT] {min_frames} frames -> {output_dir}")

    for frame_id in range(min_frames):
        frame_json = build_frame_json(frame_id, pose_by_camera)

        out_path = output_dir / f"ground_truth_{frame_id}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(frame_json, f, ensure_ascii=False, indent=2)

        if frame_id % 500 == 0 or frame_id == min_frames - 1:
            print(f"  [{frame_id + 1:5d}/{min_frames}] {frame_json['frame_name']}")

    print(f"[DONE] saved to: {output_dir}")


def main():
    for seq in INPUT_SEQUENCES:
        export_sequence(
            seq_name=seq["name"],
            image_sequence_dir=seq["image_sequence_dir"],
        )

    print()
    print("[ALL DONE]")


if __name__ == "__main__":
    main()
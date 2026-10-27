from pathlib import Path
import joblib
import numpy as np
import cv2
from json_io import read_json
from config_loader import resolve_inputs, resolve_preprocess_output_dir
from json_io import write_json
from compat import load_joblib_compat, resolve_person_payload


def _extract_person_payload(wham_data):
    return resolve_person_payload(wham_data)


def _load_wham_camera_payload(pkl_path: Path, target_person=None) -> dict:
    if not pkl_path.exists():
        raise FileNotFoundError(f"WHAM PKL not found: {pkl_path}")
    person = resolve_person_payload(load_joblib_compat(pkl_path), pkl_path=pkl_path, target_person=target_person)
    if person is None:
        raise ValueError(f"Cannot extract person payload from WHAM PKL: {pkl_path}")
    return person


def _mean_shape_over_frames(shape_value) -> list[list[float]]:
    shape = np.asarray(shape_value, dtype=np.float32)
    if shape.ndim == 1:
        shape = shape[None, :]
    if shape.ndim != 2:
        raise ValueError(f"Expected betas/shape to have ndim 1 or 2, got {shape.ndim}")
    return shape.mean(axis=0, keepdims=True).astype(np.float32).tolist()


def _estimate_intrinsics(video_path: Path) -> list[list[float]]:
    calib_path = video_path.parent / "calib.txt"
    if calib_path.exists():
        values = [float(value) for value in calib_path.read_text(encoding="utf-8").split()]
        if len(values) != 4:
            raise ValueError(f"Expected fx fy cx cy in {calib_path}, got {len(values)} values")
        fx, fy, cx, cy = values
        if min(fx, fy) <= 0:
            raise ValueError(f"Invalid focal length in {calib_path}")
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    w, h = 0.0, 0.0
    if video_path.is_file() and video_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(str(video_path))
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    if w <= 0 or h <= 0:
        img_dir = video_path if video_path.is_dir() else (video_path.parent / "images" if (video_path.parent / "images").exists() else video_path.parent)
        if img_dir.exists():
            imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if imgs:
                img = cv2.imread(str(imgs[0]))
                if img is not None:
                    h, w = float(img.shape[0]), float(img.shape[1])

    if w <= 0 or h <= 0:
        w, h = 3840.0, 2160.0

    fx = fy = float((w * w + h * h) ** 0.5)
    cx = float(w / 2.0)
    cy = float(h / 2.0)
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def export_camera_jsons(config: dict, offset: int) -> None:
    output_dir = Path(resolve_preprocess_output_dir(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(config)
    source_by_id = {
        "cam1": str(Path(inputs.get("cam1_pkl", ""))),
        "cam2": str(Path(inputs.get("cam2_pkl", ""))),
    }
    video_by_id = {
        "cam1": Path(inputs.get("camera1_video", "input/video_1.mp4")),
        "cam2": Path(inputs.get("camera2_video", "input/video_2.mp4")),
    }
    pkl_by_id = {
        "cam1": Path(inputs.get("cam1_pkl", "")),
        "cam2": Path(inputs.get("cam2_pkl", "")),
    }

    for cam_id in ["cam1", "cam2"]:
        video_path = video_by_id.get(cam_id)
        pkl_path = pkl_by_id.get(cam_id)
        if video_path is None or (not video_path.exists() and not (video_path.parent / "images").exists() and not video_path.parent.exists()):
            print(f"[Preprocess] Video and image directory not found for {cam_id}: {video_path}")
            continue

        wham_payload = None
        if pkl_path is not None and pkl_path.exists():
            try:
                wham_payload = _load_wham_camera_payload(pkl_path, target_person=inputs.get("target_person"))
            except Exception as exc:
                print(f"[Preprocess] PKL skip for {cam_id}: {exc}")

        calib_txt = video_path.parent / "calib.txt" if video_path else Path("calib.txt")
        payload = {
            "camera_id": cam_id,
            "source_pkl": source_by_id.get(cam_id, ""),
            "offset": int(offset),
            "intrinsics_source": "calib.txt" if calib_txt.exists() else "intri_esti",
            "intrinsics_estimation": _estimate_intrinsics(video_path if video_path else Path("dummy")),
        }

        if wham_payload is not None and "betas" in wham_payload:
            payload["betas"] = _mean_shape_over_frames(wham_payload["betas"])

        write_json(output_dir / f"data_{cam_id}.json", payload)


def load_camera_profile(config: dict, cam_id: str) -> dict:
    output_dir = Path(resolve_preprocess_output_dir(config))
    path = output_dir / f"data_{cam_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Camera profile not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid camera profile format: {path}")
    return data


def resolve_selected_offset_from_camera_profile(config: dict) -> tuple[int, Path]:
    cam_profile = load_camera_profile(config, "cam1")
    if "offset" in cam_profile:
        return int(cam_profile["offset"]), Path("data_cam1.json")
    raise KeyError("Missing 'offset' in camera profile data_cam1.json")


def resolve_selected_intrinsics(config: dict, cam_id: str) -> list[list[float]]:
    profile = load_camera_profile(config, cam_id)
    return profile["intrinsics_estimation"]

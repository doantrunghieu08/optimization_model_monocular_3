from functools import lru_cache
import os
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


@lru_cache(maxsize=64)
def _probe_video_wh(abs_path: str) -> tuple[float, float]:
    # ponytail: cache W/H per worker so symlinked Drive mp4s aren't reopened every pair
    cap = cv2.VideoCapture(abs_path)
    w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


DEFAULT_IMAGE_SIZE = (3840.0, 2160.0)  # Harmony4D / EgoHumans exo GoPro videos


def _read_calib_txt(calib_path: Path) -> list[list[float]]:
    values = [float(value) for value in calib_path.read_text(encoding="utf-8").split()]
    if len(values) != 4:
        raise ValueError(f"Expected fx fy cx cy in {calib_path}, got {len(values)} values")
    fx, fy, cx, cy = values
    if min(fx, fy) <= 0:
        raise ValueError(f"Invalid focal length in {calib_path}")
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _camera_dirs(video_path, pkl_path) -> list[Path]:
    """Folders that may hold calib.txt / extracted frames: next to the video, else the camera
    folder of the pkl (exo/camXX/<x>_wham/<pkl> -> exo/camXX/images, exo/camXX)."""
    dirs = [Path(video_path).parent] if video_path is not None else []
    if pkl_path:
        cam_dir = Path(pkl_path).parent.parent
        dirs += [d for d in (cam_dir / "images", cam_dir) if d not in dirs]
    return dirs


def resolve_intrinsics(video_path=None, pkl_path=None, image_size=None) -> tuple[list[list[float]], str]:
    """(K, source). The video is OPTIONAL: it is only probed for its frame size (w, h).
    Order: calib.txt (next to the video, or in the pkl's camera folder) > image_size from the
    config > the video's frame size > an extracted frame > DEFAULT_IMAGE_SIZE.
    Guessed K: f = sqrt(w^2 + h^2), principal point at the image centre."""
    dirs = _camera_dirs(video_path, pkl_path)
    for d in dirs:
        if (d / "calib.txt").is_file():
            return _read_calib_txt(d / "calib.txt"), "calib.txt"

    w = h = 0.0
    source = "intri_esti"
    if image_size:
        w, h = (float(x) for x in image_size)
        source = "intri_esti(image_size)"
    elif video_path is not None:
        vp = Path(video_path)
        if vp.is_file() and vp.suffix.lower() in (".mp4", ".avi", ".mov"):
            w, h = _probe_video_wh(os.path.abspath(str(vp)))
    if w <= 0 or h <= 0:
        cands = [Path(video_path)] if video_path is not None and Path(video_path).is_dir() else []
        cands += [d / "images" for d in dirs] + dirs
        for img_dir in cands:
            imgs = (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) if img_dir.is_dir() else []
            img = cv2.imread(str(imgs[0])) if imgs else None
            if img is not None:
                h, w = float(img.shape[0]), float(img.shape[1])
                break
    if w <= 0 or h <= 0:
        w, h = DEFAULT_IMAGE_SIZE
        source = "intri_esti(default_3840x2160)"

    f = float((w * w + h * h) ** 0.5)
    return [[f, 0.0, float(w / 2.0)], [0.0, f, float(h / 2.0)], [0.0, 0.0, 1.0]], source


def _estimate_intrinsics(video_path) -> list[list[float]]:
    """Backward-compatible wrapper (video-based)."""
    return resolve_intrinsics(video_path=video_path)[0]


def export_camera_jsons(config: dict, offset: int) -> None:
    output_dir = Path(resolve_preprocess_output_dir(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(config)
    for cam_id, idx in (("cam1", "1"), ("cam2", "2")):
        pkl_value = inputs.get(f"cam{idx}_pkl") or ""
        pkl_path = Path(pkl_value) if pkl_value else None
        video_value = inputs.get(f"camera{idx}_video")
        video_path = Path(video_value) if video_value else None
        has_video = video_path is not None and video_path.is_file()

        wham_payload = None
        if pkl_path is not None and pkl_path.exists():
            try:
                wham_payload = _load_wham_camera_payload(pkl_path, target_person=inputs.get("target_person"))
            except Exception as exc:
                print(f"[Preprocess] PKL skip for {cam_id}: {exc}")

        K, source = resolve_intrinsics(video_path, pkl_path, inputs.get(f"camera{idx}_image_size"))
        if not has_video:
            print(f"[Preprocess] {cam_id}: no video -> K from {source} (f={K[0][0]:.1f}, c=({K[0][2]:.0f},{K[1][2]:.0f}))")
        payload = {
            "camera_id": cam_id,
            "source_pkl": str(pkl_path) if pkl_path else "",
            "offset": int(offset),
            "intrinsics_source": "calib.txt" if source == "calib.txt" else "intri_esti",
            "intrinsics_detail": source,
            "intrinsics_estimation": K,
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

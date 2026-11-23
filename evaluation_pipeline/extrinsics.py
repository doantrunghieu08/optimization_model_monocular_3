"""Real camera extrinsics from a Harmony4D/EgoHumans COLMAP reconstruction (opt-in via EXT=1).

Harmony4D ``processed_data/poses3d`` are in the *world* (Aria) frame, in metres.
``colmap/workplace`` holds:

* ``images.txt``  -- per image ``QW QX QY QZ TX TY TZ CAMERA_ID NAME`` (NAME = ``camXX/<frame>.jpg``),
  i.e. the COLMAP world->camera pose  X_cam = R(q) X_colmap + t   (OpenCV axes, COLMAP units);
* ``scale.npy``   -- 4x4 similarity  X_world = S X_colmap  (s * R_s | t_s).

Hence  X_cam[m] = s * (R(q) S^-1 X_world + t)  -- a rigid map in metres, verified by projecting the
3D GT through ``cameras.txt`` (OPENCV_FISHEYE) onto ``processed_data/poses2d`` (a few px per joint).
Exo cameras are static: every frame of one camera is registered, the median pose is used.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def ext_enabled() -> bool:
    """True when the EXT environment variable asks for real-extrinsic evaluation."""
    return str(os.environ.get("EXT", "")).strip().lower() in TRUTHY


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def camera_name_from_path(*paths) -> Optional[str]:
    """'cam03' from e.g. .../exo/cam03/output_wham/wham_output_cam03.pkl (last match wins)."""
    for p in paths:
        if not p:
            continue
        found = re.findall(r"cam(\d+)", str(p))
        if found:
            return f"cam{int(found[-1]):02d}"
    return None


def find_colmap_workplace(*starts, explicit=None) -> Path:
    """Locate ``colmap/workplace`` (with images.txt): explicit path, else walk up from each start."""
    if explicit:
        p = Path(explicit)
        for cand in (p, p / "workplace", p / "colmap" / "workplace"):
            if (cand / "images.txt").is_file():
                return cand
        raise FileNotFoundError(f"EXT=1: no images.txt under evaluation.extrinsics.colmap_dir={explicit}")
    tried = []
    for s in starts:
        if not s:
            continue
        p = Path(s).resolve()
        for parent in [p, *p.parents][:6]:
            cand = parent / "colmap" / "workplace"
            tried.append(str(cand))
            if (cand / "images.txt").is_file():
                return cand
    raise FileNotFoundError(
        "EXT=1 needs the COLMAP calibration of this sequence (colmap/workplace/images.txt + scale.npy), "
        "but none was found. Put the sequence's colmap/ folder next to processed_data/ or set "
        f"evaluation.extrinsics.colmap_dir. Searched: {tried[:6]}")


@lru_cache(maxsize=8)
def _read_images_txt(path: str) -> dict:
    """camera name -> list of 7-vectors (qw qx qy qz tx ty tz). Skips the POINTS2D lines."""
    poses: dict = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        expect_header = True
        for line in f:
            if line.startswith("#"):
                continue
            if expect_header:
                parts = line.split()
                if len(parts) >= 10:
                    name = parts[9]
                    cam = name.split("/")[0]
                    poses.setdefault(cam, []).append(np.array(parts[1:8], dtype=float))
            expect_header = not expect_header
    return poses


@lru_cache(maxsize=64)
def load_world_to_camera(workplace: str, cam_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with X_cam = R @ X_world + t, both in metres (OpenCV camera axes)."""
    wp = Path(workplace)
    poses = _read_images_txt(str(wp / "images.txt")).get(cam_name)
    if not poses:
        raise KeyError(f"EXT=1: camera '{cam_name}' is not registered in {wp / 'images.txt'}")
    arr = np.stack(poses)
    q = arr[:, :4] * np.sign(arr[:, :1] + 1e-12)  # q and -q are the same rotation
    q = np.median(q, axis=0)
    t = np.median(arr[:, 4:7], axis=0)
    R_cc = _qvec2rotmat(q)  # colmap world -> camera

    scale_path = wp / "scale.npy"
    if scale_path.is_file():
        S = np.asarray(np.load(scale_path, allow_pickle=True), dtype=float)
        if S.shape != (4, 4):
            raise ValueError(f"Expected 4x4 similarity in {scale_path}, got {S.shape}")
        s = float(np.cbrt(np.linalg.det(S[:3, :3])))
        R_s, t_s = S[:3, :3] / s, S[:3, 3]
        # X_colmap = R_s^T (X_world - t_s) / s ;  X_cam[m] = s * (R_cc X_colmap + t)
        R = R_cc @ R_s.T
        t_m = s * t - R @ t_s
    else:
        print(f"[Evaluation] EXT: {scale_path} missing -> assuming COLMAP world == GT world (metres)")
        R, t_m = R_cc, t
    return R, t_m


def extrinsic_from_config(block: dict, cam_name: str = "?") -> tuple[np.ndarray, np.ndarray]:
    """(R, t) in metres from a YAML camera ``extrinsic`` block (X_cam = R X_world + t).

    Accepted keys: ``R`` (3x3) or ``qvec_wxyz``; ``t`` or ``C`` (camera centre, t = -R C);
    ``world`` must be "gt" (the frame of processed_data/poses3d); ``unit`` "m" (default) or "mm".
    Redundant entries (R + qvec, t + C) are cross-checked so a mistyped block fails loudly."""
    if not isinstance(block, dict):
        raise ValueError(f"EXT=1: extrinsic of {cam_name} must be a mapping, got {type(block).__name__}")
    world = str(block.get("world", "gt")).strip().lower()
    if world != "gt":
        raise ValueError(f"EXT=1: extrinsic of {cam_name} has world='{world}'; only 'gt' "
                         "(the GT poses3d frame) can be compared with the GT")
    unit = str(block.get("unit", "m")).strip().lower()
    if unit not in ("m", "mm"):
        raise ValueError(f"EXT=1: extrinsic of {cam_name} has unit='{unit}' (expected m or mm)")
    scale = 1e-3 if unit == "mm" else 1.0

    R = np.asarray(block["R"], dtype=float) if "R" in block else None
    if "qvec_wxyz" in block:
        Rq = _qvec2rotmat(np.asarray(block["qvec_wxyz"], dtype=float))
        if R is not None and np.abs(R - Rq).max() > 1e-4:
            raise ValueError(f"EXT=1: extrinsic of {cam_name}: R and qvec_wxyz disagree")
        R = Rq if R is None else R
    if R is None or R.shape != (3, 3):
        raise ValueError(f"EXT=1: extrinsic of {cam_name} needs R (3x3) or qvec_wxyz")
    if np.abs(R @ R.T - np.eye(3)).max() > 1e-4 or np.linalg.det(R) < 0:
        raise ValueError(f"EXT=1: extrinsic of {cam_name}: R is not a rotation")

    t = np.asarray(block["t"], dtype=float) * scale if "t" in block else None
    if "C" in block:
        tC = -R @ (np.asarray(block["C"], dtype=float) * scale)
        if t is not None and np.linalg.norm(t - tC) > 1e-3:
            raise ValueError(f"EXT=1: extrinsic of {cam_name}: t and C disagree (t != -R C)")
        t = tC if t is None else t
    if t is None or t.shape != (3,):
        raise ValueError(f"EXT=1: extrinsic of {cam_name} needs t (3,) or C (3,)")
    return R, t


def transform_truth_map(gt_map: dict, R: np.ndarray, t: np.ndarray) -> dict:
    """Return a copy of a GT frame map with every joint moved into the camera frame."""
    out = {}
    for fid, item in gt_map.items():
        new_item = dict(item)
        if "joints" in item:
            new_item["joints"] = {k: R @ np.asarray(v, dtype=float) + t for k, v in item["joints"].items()}
        elif "pose3d" in item:
            # MPI-style item (mm): keep units, transform positions.
            p = np.asarray(item["pose3d"], dtype=float)
            new_item["pose3d"] = (p[:, :3] @ R.T + t * 1000.0).tolist()
        out[fid] = new_item
    return out

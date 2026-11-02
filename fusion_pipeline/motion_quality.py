"""Zero-GT, two-view motion-aware temporal smoothing."""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from scipy.signal import savgol_filter


def synchronized_segment_ids(paths, pose_metadata_dir=None):
    """Build a monotonic timeline that splits at output OR WHAM source gaps.

    Metadata is read from pose_data_*.json, never from evaluation/GT.
    """
    paths = [Path(x) for x in paths]
    frame_ids = [int(path.stem.split("_")[-1]) for path in paths]
    if not paths:
        return []
    sources = []
    metadata_dir = Path(pose_metadata_dir) if pose_metadata_dir else None
    for idx in frame_ids:
        source = {}
        if metadata_dir is not None:
            metadata_path = metadata_dir / f"pose_data_{idx}.json"
            if metadata_path.exists():
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    source = payload.get("metadata", {}).get("source_frame_indices", {})
                except (ValueError, OSError, AttributeError):
                    pass
        sources.append(source if isinstance(source, dict) else {})
    timeline = [0]
    for i in range(1, len(paths)):
        continuous = frame_ids[i] - frame_ids[i - 1] == 1
        for camera in ("camera1", "camera2"):
            prev, current = sources[i - 1].get(camera), sources[i].get(camera)
            if prev is not None and current is not None and current - prev != 1:
                continuous = False
        timeline.append(timeline[-1] + (1 if continuous else 2))
    return timeline


def motion_aware_smooth(coords1, coords2, quality1, quality2,
                        frame_ids=None, window_length=5, polyorder=2,
                        max_strength=0.95, good_quality_floor=0.15,
                        motion_scale_m=0.08):
    """Smooth reliable observations less, and protect genuine shared fast motion.

    Operates on synchronized T x J x 3 tracks without crossing discontinuous
    frame IDs. An isolated jump in only one view remains eligible for smoothing.
    """
    p1 = np.asarray(coords1, dtype=float)
    p2 = np.asarray(coords2, dtype=float)
    if p1.shape != p2.shape or p1.ndim != 3 or p1.shape[-1] != 3:
        raise ValueError("Expected two matching [time, joint, xyz] tracks")
    t, n, _ = p1.shape
    q1 = np.broadcast_to(np.asarray(quality1, dtype=float), (t, n))
    q2 = np.broadcast_to(np.asarray(quality2, dtype=float), (t, n))
    if not (np.isfinite(p1).all() and np.isfinite(p2).all()) or t < 3:
        return p1.copy(), p2.copy()
    ids = list(range(t)) if frame_ids is None else list(frame_ids)
    if len(ids) != t:
        raise ValueError("Frame IDs must match sequence length")
    out1, out2 = p1.copy(), p2.copy()
    starts = [0] + [i for i in range(1, t) if ids[i] - ids[i - 1] != 1]
    ends = starts[1:] + [t]
    for start, end in zip(starts, ends):
        length = end - start
        window = min(int(window_length), length)
        if window % 2 == 0:
            window -= 1
        if window < 3 or window <= polyorder:
            continue
        a, b = p1[start:end], p2[start:end]
        sa = savgol_filter(a, window_length=window, polyorder=polyorder, axis=0)
        sb = savgol_filter(b, window_length=window, polyorder=polyorder, axis=0)
        speed_a = np.linalg.norm(a[1:] - a[:-1], axis=-1)
        speed_b = np.linalg.norm(b[1:] - b[:-1], axis=-1)
        shared = np.minimum(speed_a, speed_b)
        shared_motion = np.zeros((length, n), dtype=float)
        shared_motion[1:] = np.maximum(shared_motion[1:], shared)
        shared_motion[:-1] = np.maximum(shared_motion[:-1], shared)
        motion_factor = 1.0 / (
            1.0 + (shared_motion / max(float(motion_scale_m), 1e-6)) ** 2
        )
        for pose, candidate, quality, dest in (
            (a, sa, q1[start:end], out1),
            (b, sb, q2[start:end], out2),
        ):
            q = np.clip(np.nan_to_num(quality, nan=0.0), 0.0, 1.0)
            strength = np.clip(max_strength, 0.0, 1.0) * (
                1.0 - (1.0 - np.clip(good_quality_floor, 0.0, 1.0)) * q
            ) * motion_factor
            dest[start:end] = pose + strength[..., None] * (candidate - pose)
    return out1, out2

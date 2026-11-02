"""Zero-GT bone-length quality and reference estimation for each camera/view.

Sequence medians and bilateral consistency are *signals*, not ground truth.
Do not derive references from evaluation annotations or holdout metrics.
"""
from __future__ import annotations

import math
import numpy as np
from fusion_pipeline.correction import PARENT_JOINTS


def _length(frame, camera, child, parent):
    pose = frame.get(camera, {})
    if child not in pose or parent not in pose:
        return None
    try:
        a = np.asarray(pose[child], dtype=float)
        b = np.asarray(pose[parent], dtype=float)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        value = float(np.linalg.norm(a - b))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 1e-5 else None


def estimate_bone_quality(frames, *, min_samples=3, tolerance_ratio=0.08,
                          residual_scale_ratio=0.18, bilateral_scale_ratio=0.25,
                          bilateral_weight=0.45):
    """Return (frame joint quality, view-specific robust median segment lengths).

    Each bone is calibrated within *the same camera* to tolerate view-specific
    scale differences. A bilateral ratio supplements temporal length consistency
    to identify stable but asymmetric reconstruction errors.

    If fewer than min_samples usable measurements exist, no learned reference
    is used for that bone. In such cases quality defaults to 1.0; this means
    *unknown*, not proof of a good joint.
    """
    names = tuple(PARENT_JOINTS)
    quality = [
        {"camera1": {n: 1.0 for n in names},
         "camera2": {n: 1.0 for n in names}}
        for _ in frames
    ]
    references = {"camera1": {}, "camera2": {}}
    if not frames:
        return quality, references

    min_samples = max(3, int(min_samples))
    tolerance_ratio = max(0.0, float(tolerance_ratio))
    residual_scale_ratio = max(1e-6, float(residual_scale_ratio))
    bilateral_scale_ratio = max(1e-6, float(bilateral_scale_ratio))
    bilateral_weight = float(np.clip(bilateral_weight, 0.0, 1.0))

    for camera in references:
        for child, parent in PARENT_JOINTS.items():
            valid = [
                v for f in frames
                if (v := _length(f, camera, child, parent)) is not None
            ]
            if len(valid) >= min_samples:
                references[camera][child] = float(np.median(valid))

    # Precompute bilateral correspondences for comparable left/right segments.
    for frame_idx, frame in enumerate(frames):
        for camera in references:
            for child, parent in PARENT_JOINTS.items():
                observed = _length(frame, camera, child, parent)
                expected = references[camera].get(child)
                if observed is None:
                    quality[frame_idx][camera][child] = 0.0
                    continue
                length_quality = 1.0
                if expected is not None and expected > 1e-5:
                    relative = abs(observed - expected) / expected
                    excess = max(0.0, relative - tolerance_ratio)
                    length_quality = 1.0 / (1.0 + (excess / residual_scale_ratio) ** 2)

                symmetry_quality = 1.0
                if child.startswith("left_"):
                    twin = "right_" + child[5:]
                elif child.startswith("right_"):
                    twin = "left_" + child[6:]
                else:
                    twin = None
                if twin is not None and twin in PARENT_JOINTS:
                    opposite = _length(frame, camera, twin, PARENT_JOINTS[twin])
                    if opposite is not None:
                        relative_difference = abs(observed - opposite) / max(
                            0.5 * (observed + opposite), 1e-6
                        )
                        symmetry_quality = 1.0 / (
                            1.0 + (relative_difference / bilateral_scale_ratio) ** 2
                        )
                quality[frame_idx][camera][child] = float(np.clip(
                    length_quality * (1.0 - bilateral_weight + bilateral_weight * symmetry_quality),
                    0.0, 1.0,
                ))
    return quality, references

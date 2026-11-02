"""Conservative, GT-free selection of slave joints for the master pose.

This is a quality *heuristic*, not a calibrated probability of 3D correctness.
It deliberately abstains on ambiguous, jointly visible joints. Decisions are
recorded per frame so downstream stages cannot silently overwrite KEEP joints.
"""
from collections import defaultdict
from math import exp
import numpy as np

from fusion_pipeline.correction import PARENT_JOINTS, transform_joint_kinematic
from fusion_pipeline.config import HINGE_TRIPLETS, DEFAULT_MIN_COS, DEFAULT_MAX_COS

# COCO-17 observations only: hands/feet in this project can be copied from
# wrists/ankles, and are not independent corroborating measurements.
DIRECT_2D_JOINTS = {
    "head", "neck", "pelvis", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
}
ACCEPTED = {"BLEND", "REPLACE"}


def reference_bone_lengths(master_frames):
    """Robust subject-specific reference; never use per-frame target lengths."""
    observed = defaultdict(list)
    for frame in master_frames:
        for child, parent in PARENT_JOINTS.items():
            if child not in frame or parent not in frame:
                continue
            v = np.asarray(frame[child], dtype=float) - np.asarray(frame[parent], dtype=float)
            length = float(np.linalg.norm(v))
            if np.isfinite(length) and 0.025 < length < 1.5:
                observed[child].append(length)
    return {child: float(np.median(lengths)) for child, lengths in observed.items() if lengths}


def _observation_2d(points, joint, keypoints2d, intrinsics):
    """Returns independent 2D confidence/reprojection agreement, or None."""
    if joint not in DIRECT_2D_JOINTS or not keypoints2d:
        return None
    kp = keypoints2d.get(joint)
    if kp is None:
        return None
    kp = np.asarray(kp, dtype=float).reshape(-1)
    if kp.size < 3 or not np.isfinite(kp[:3]).all():
        return None
    conf = float(np.clip(kp[2], 0.0, 1.0))
    if intrinsics is None:
        return conf
    K = np.asarray(intrinsics, dtype=float)
    xyz = np.asarray(points[joint], dtype=float)
    if K.shape != (3, 3) or xyz.shape != (3,) or xyz[2] <= 1e-6:
        return None
    uvw = K @ xyz
    uv = uvw[:2] / uvw[2]
    # Normalize the residual by focal length, not by absolute image size.
    px_scale = max(20.0, 0.025 * float(max(K[0, 0], K[1, 1])))
    residual = float(np.linalg.norm(uv - kp[:2]) / px_scale)
    return float(conf * exp(-0.5 * min(residual * residual, 100.0)))


def _bone_quality(points, joint, lengths):
    parent = PARENT_JOINTS.get(joint)
    target = lengths.get(joint)
    if parent not in points or target is None or joint not in points:
        return 0.5
    observed = float(np.linalg.norm(
        np.asarray(points[joint], dtype=float) - np.asarray(points[parent], dtype=float)
    ))
    return float(exp(-min(abs(observed / max(target, 1e-6) - 1.0), 5.0) / 0.15))


def _temporal_quality(points, joint, prev, prev_prev):
    if not prev or not prev_prev or joint not in prev or joint not in prev_prev:
        return 0.5
    # Acceleration is evidence of jitter, NOT a ground-truth motion model.
    pos = np.asarray(points[joint], dtype=float)
    pred = 2 * np.asarray(prev[joint], dtype=float) - np.asarray(prev_prev[joint], dtype=float)
    err = float(np.linalg.norm(pos - pred))
    return float(exp(-min(err, 5.0) / 0.10))


def _joint_quality(points, joint, visible, keypoints2d, K, lengths, prev, prev_prev):
    quality_2d = _observation_2d(points, joint, keypoints2d, K)
    return (
        0.40 * (1.0 if visible else 0.05)
        + 0.30 * (quality_2d if quality_2d is not None else 0.5)
        + 0.15 * _bone_quality(points, joint, lengths)
        + 0.15 * _temporal_quality(points, joint, prev, prev_prev)
    ), quality_2d


def _angle_violations(points, affected):
    invalid = 0
    for a, b, c in HINGE_TRIPLETS:
        if not ({a, b, c} & affected) or not all(j in points for j in (a, b, c)):
            continue
        u = np.asarray(points[a], dtype=float) - np.asarray(points[b], dtype=float)
        v = np.asarray(points[c], dtype=float) - np.asarray(points[b], dtype=float)
        den = float(np.linalg.norm(u) * np.linalg.norm(v))
        if den < 1e-8:
            invalid += 1
            continue
        cosine = float(np.dot(u, v) / den)
        if not DEFAULT_MIN_COS <= cosine <= DEFAULT_MAX_COS:
            invalid += 1
    return invalid


def _safe_local_geometry(current, candidate, joint, lengths, max_bone_error):
    """Reject proposals that worsen already-poor local anatomy or violate hinges."""
    changed = dict(current)
    changed[joint] = candidate
    neighbors = {joint}
    parent = PARENT_JOINTS.get(joint)
    if parent:
        neighbors.add(parent)
    for child, par in PARENT_JOINTS.items():
        if par == joint:
            neighbors.add(child)
    for child, par in PARENT_JOINTS.items():
        if child not in neighbors and par not in neighbors:
            continue
        if child not in current or par not in current or child not in changed or par not in changed:
            continue
        reference = lengths.get(child)
        if reference is None or reference < 0.025:
            continue
        old_length = float(np.linalg.norm(np.asarray(current[child]) - np.asarray(current[par])))
        new_length = float(np.linalg.norm(np.asarray(changed[child]) - np.asarray(changed[par])))
        old_err = abs(old_length - reference) / reference
        new_err = abs(new_length - reference) / reference
        if not np.isfinite(new_err) or new_err > max(max_bone_error, old_err + 0.015):
            return False
    return _angle_violations(changed, {joint}) <= _angle_violations(current, {joint})


def select_slave_joints(
    master, slave, t21, vis_master, vis_slave, h_master, h_slave,
    keypoints2d_master=None, keypoints2d_slave=None,
    K_master=None, K_slave=None, lengths=None,
    prev_master=None, prev_prev_master=None, prev_slave=None, prev_prev_slave=None,
    config=None,
):
    """Select only better-supported slave proposals; return pose and audit metadata."""
    cfg = config or {}
    lengths = lengths or {}
    master = {j: np.asarray(v, dtype=float).copy() for j, v in master.items()}
    slave = {j: np.asarray(v, dtype=float).copy() for j, v in slave.items()}
    result = {j: v.copy() for j, v in master.items()}
    decisions = {j: "KEEP" for j in master}
    explanations = {}
    max_disp_visible = float(cfg.get("max_displacement_visible", 0.12))
    max_disp_occluded = float(cfg.get("max_displacement_occluded", 0.35))
    visible_margin = float(cfg.get("quality_margin_visible", 0.10))
    occluded_margin = float(cfg.get("quality_margin_occluded", 0.06))
    blend = float(np.clip(cfg.get("blend_strength", 0.35), 0.0, 0.5))
    max_bone_error = float(cfg.get("max_bone_relative_error", 0.15))

    if t21 is None:
        return {k: v.tolist() for k, v in result.items()}, decisions, explanations

    depth = {}
    for name in master:
        p, n = PARENT_JOINTS.get(name), 0
        while p in PARENT_JOINTS and n < len(PARENT_JOINTS):
            n += 1
            p = PARENT_JOINTS[p]
        depth[name] = n

    for name in sorted(master, key=lambda j: (depth[j], j)):
        if name in {"pelvis", "neck"} or name not in slave:
            continue
        vm = bool(vis_master.get(name, True))
        vs = bool(vis_slave.get(name, True))
        if not vs:
            explanations[name] = "slave_occluded"
            continue

        qm, pixel_m = _joint_quality(
            master, name, vm, keypoints2d_master, K_master, lengths,
            prev_master, prev_prev_master
        )
        qs, pixel_s = _joint_quality(
            slave, name, vs, keypoints2d_slave, K_slave, lengths,
            prev_slave, prev_prev_slave
        )
        # In the absence of an independent quality signal, distance-based
        # belief must not be sufficient to overwrite a visible master.
        gain = qs - qm
        required = visible_margin if vm else occluded_margin
        if gain <= required:
            explanations[name] = "quality_ambiguous"
            continue
        if vm and pixel_s is None and pixel_m is None:
            geom_gain = (
                _bone_quality(slave, name, lengths) - _bone_quality(master, name, lengths)
                + _temporal_quality(slave, name, prev_slave, prev_prev_slave)
                - _temporal_quality(master, name, prev_master, prev_prev_master)
            )
            if geom_gain <= 0.15:
                explanations[name] = "no_independent_visible_advantage"
                continue
        if pixel_s is not None and pixel_s < float(cfg.get("minimum_slave_2d_support", 0.15)):
            explanations[name] = "weak_slave_2d"
            continue
        if float(h_slave.get(name, 0.0)) < float(cfg.get("minimum_slave_belief", 0.10)):
            explanations[name] = "weak_slave_belief"
            continue

        proposal = transform_joint_kinematic(name, slave, result, t21)
        if not np.isfinite(proposal).all():
            explanations[name] = "invalid_proposal"
            continue
        displacement = float(np.linalg.norm(proposal - master[name]))
        limit = max_disp_visible if vm else max_disp_occluded
        if displacement > limit:
            explanations[name] = "large_displacement"
            continue
        parent = PARENT_JOINTS.get(name)
        reference = lengths.get(name)
        if parent in result and reference and reference > 0:
            direction = proposal - result[parent]
            mag = float(np.linalg.norm(direction))
            if mag < 1e-8:
                explanations[name] = "degenerate_bone"
                continue
            proposal = result[parent] + reference * direction / mag

        # Hard replacement is reserved for a genuinely occluded master and
        # strong supporting evidence; other positives are softly corrected.
        mode = "REPLACE" if (not vm and gain >= 0.25 and (pixel_s is None or pixel_s >= 0.4)) else "BLEND"
        strength = 1.0 if mode == "REPLACE" else blend
        candidate = master[name] + strength * (proposal - master[name])
        if float(np.linalg.norm(candidate - master[name])) > limit:
            explanations[name] = "projected_displacement"
            continue
        if not _safe_local_geometry(result, candidate, name, lengths, max_bone_error):
            explanations[name] = "unsafe_local_geometry"
            continue
        result[name] = candidate
        decisions[name] = mode
        explanations[name] = {
            "reason": "slave_advantage",
            "quality_gain": round(float(gain), 6),
            "master_quality": round(float(qm), 6),
            "slave_quality": round(float(qs), 6),
            "displacement_m": round(float(displacement), 6),
        }
    return {k: v.tolist() for k, v in result.items()}, decisions, explanations

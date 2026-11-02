"""Zero-GT adaptive, per-joint/per-frame cross-view pose transfer.

This module does not use camera identities, labels, or ground truth. Reliability
is based on observation confidence, visibility, relative temporal acceleration,
and cross-view geometric agreement.
"""
from __future__ import annotations

import math

import numpy as np

from fusion_pipeline.config import HINGE_TRIPLETS, NON_REPLACEABLE_ANCHORS
from fusion_pipeline.correction import (
    KINEMATIC_DEPTH,
    PARENT_JOINTS,
    apply_similarity,
    transform_joint_kinematic,
)


def temporal_joint_quality(
    frames, source_indices=None, continuity=(1.0, 1.0),
    jitter_ratio=1.35, jitter_floor_m=0.025, jitter_scale_m=0.08,
):
    """Per-joint, per-camera quality without interpreting shared motion as noise.

    Skip nonconsecutive source-frame triples to avoid treating missing tracking
    frames as physically implausible acceleration.
    """
    if not frames:
        return []
    names = sorted(set(frames[0].get("camera1", {})) & set(frames[0].get("camera2", {})))
    continuity = [float(np.clip(v, 0.25, 1.0)) for v in continuity]
    result = [
        {cam: {n: continuity[c] for n in names}
         for c, cam in enumerate(("camera1", "camera2"))}
        for _ in frames
    ]
    if len(frames) < 3 or not names:
        return result

    for t in range(1, len(frames) - 1):
        if source_indices is not None and any(
            source_indices[t][cam] - source_indices[t - 1][cam] != 1
            or source_indices[t + 1][cam] - source_indices[t][cam] != 1
            for cam in ("camera1", "camera2")
        ):
            continue
        for name in names:
            try:
                acceleration = []
                for cam in ("camera1", "camera2"):
                    prev, mid, nex = (
                        np.asarray(frames[k][cam][name], dtype=float)
                        for k in (t - 1, t, t + 1)
                    )
                    acceleration.append(float(np.linalg.norm(nex - 2 * mid + prev)))
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(x) for x in acceleration):
                continue
            for c, cam in enumerate(("camera1", "camera2")):
                excess = max(
                    0.0,
                    acceleration[c] - max(jitter_floor_m, jitter_ratio * acceleration[1 - c]),
                )
                result[t][cam][name] = continuity[c] * math.exp(
                    -excess / max(float(jitter_scale_m), 1e-6)
                )
    return result


def _angle_is_valid_locally(points, changed_name, baseline, min_angle_deg=30., max_angle_deg=175.):
    """Only gate hinges containing the edited joint; ignore pre-existing faults.

    If the pose already violates a constraint, reject an update only if the
    candidate increases that violation.
    """
    min_cos = math.cos(math.radians(max_angle_deg))
    max_cos = math.cos(math.radians(min_angle_deg))

    def violation(pose, a, b, c):
        u = np.asarray(pose[a], dtype=float) - np.asarray(pose[b], dtype=float)
        v = np.asarray(pose[c], dtype=float) - np.asarray(pose[b], dtype=float)
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        if denom < 1e-8:
            return 0.0
        cosine = float(np.dot(u, v) / denom)
        return max(0.0, min_cos - cosine, cosine - max_cos)

    for a, b, c in HINGE_TRIPLETS:
        if changed_name not in (a, b, c) or any(n not in points for n in (a, b, c)):
            continue
        candidate_violation = violation(points, a, b, c)
        if candidate_violation > 1e-6 and candidate_violation > violation(
            baseline, a, b, c
        ) + 1e-6:
            return False
    return True


def _project_bone(candidate, name, out, baseline, reference_lengths=None):
    parent = PARENT_JOINTS.get(name)
    if parent not in out or parent not in baseline or name not in baseline:
        return candidate
    center = np.asarray(out[parent], dtype=float)
    reference_lengths = reference_lengths or {}
    ref_length = reference_lengths.get(name)
    if ref_length is None or not np.isfinite(ref_length) or ref_length <= 1e-6:
        ref_length = float(np.linalg.norm(
            np.asarray(baseline[name], dtype=float) - np.asarray(baseline[parent], dtype=float)
        ))
    ref_length = float(ref_length)
    vec = candidate - center
    length = float(np.linalg.norm(vec))
    if ref_length < 1e-6 or length < 1e-8:
        return candidate
    return center + ref_length * vec / length


def _reliability(confidence, visible, quality):
    return (
        float(np.clip(confidence, 0.0, 1.0))
        * float(np.clip(quality, 0.0, 1.0))
        * (1.0 if visible else 0.1)
    )


def _transfer_weight(source_score, target_score, source_visible, target_visible,
                     disagreement, transform_reliability, cfg):
    if not source_visible or source_score < float(cfg.get("min_source_reliability", 0.12)):
        return 0.0
    if not np.isfinite(disagreement):
        return 0.0
    both_visible = bool(source_visible and target_visible)
    scale = float(
        cfg.get("consensus_scale_m", 0.18)
        if both_visible else cfg.get("occluded_scale_m", 0.55)
    )
    agreement = 1.0 / (1.0 + (disagreement / max(scale, 1e-6)) ** 2)
    share = source_score / max(source_score + target_score, 1e-6)
    cap = float(
        cfg.get("max_visible_transfer", 0.55)
        if both_visible else cfg.get("max_occluded_transfer", 0.90)
    )
    return float(np.clip(share * agreement * transform_reliability, 0.0, cap))


def apply_adaptive_joint_fusion(
    cam1, cam2, names, t12, t21, h1, h2, vis1, vis2,
    joint_quality=None, config=None, bone_quality=None, bone_references=None,
    anchor_names=None, master_protected: bool = False,
):
    """Fuse both camera poses and return per-joint diagnostics for optimization."""
    cfg = config or {}
    quality = joint_quality or {}
    q1 = quality.get("camera1", {})
    q2 = quality.get("camera2", {})
    bone_quality = bone_quality or {}
    bone_references = bone_references or {}
    b1 = bone_quality.get("camera1", {})
    b2 = bone_quality.get("camera2", {})
    out1 = {n: np.asarray(v, dtype=float).copy() for n, v in cam1.items()}
    out2 = {n: np.asarray(v, dtype=float).copy() for n, v in cam2.items()}
    transfers = {
        "camera1": {n: 0.0 for n in names},
        "camera2": {n: 0.0 for n in names},
    }
    stats = {
        "candidate_joints": 0,
        "accepted_camera1": 0,
        "accepted_camera2": 0,
        "rejected_local_angle": 0,
        "mean_torso_residual_m": None,
        "transform_reliability": 0.0,
        "rejected_alignment": 0,
        "rejected_occluded_bone": 0,
        "rejected_visible_risk": 0,
        "mean_visible_risk_multiplier": None,
        "rejected_kinematic_projection": 0,
        "rejected_nonfinite": 0,
    }
    if (t12 is None or t21 is None or
            (anchor_names is not None and len(anchor_names) < int(cfg.get("min_alignment_anchors", 3)))):
        stats["rejected_alignment"] = len(names)
        return out1, out2, {
            "joint_transfer": transfers, "adaptive_stats": stats,
        }

    torso = ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
    residuals = [
        float(np.linalg.norm(
            apply_similarity(np.asarray(cam1[n], dtype=float), t12)
            - np.asarray(cam2[n], dtype=float)
        ))
        for n in torso if n in cam1 and n in cam2
    ]
    residual = float(np.median(residuals)) if residuals else 0.0
    transform_scale = max(float(cfg.get("transform_scale_m", 0.20)), 1e-6)
    transform_reliability = 1.0 / (1.0 + (residual / transform_scale) ** 2)
    stats["mean_torso_residual_m"] = float(np.mean(residuals)) if residuals else None
    stats["transform_reliability"] = float(transform_reliability)
    min_alignment_quality = float(cfg.get("min_occluded_alignment_reliability", 0.35))
    min_source_bone_quality = float(cfg.get("min_occluded_source_bone_quality", 0.35))
    max_projection_ratio = max(float(cfg.get("max_kinematic_projection_ratio", 0.55)), 0.0)
    risk_enabled = bool(cfg.get("risk_aware_enabled", True))
    visible_risk_enabled = risk_enabled and bool(cfg.get("visible_risk_enabled", True))
    visible_risk_factors = []

    for name in sorted(names, key=lambda n: KINEMATIC_DEPTH.get(n, 99)):
        if name in NON_REPLACEABLE_ANCHORS or name not in cam1 or name not in cam2:
            continue
        stats["candidate_joints"] += 1
        visible1 = bool(vis1.get(name, True))
        visible2 = bool(vis2.get(name, True))
        score1 = _reliability(h1.get(name, 0.0), visible1, q1.get(name, 1.0))
        score2 = _reliability(h2.get(name, 0.0), visible2, q2.get(name, 1.0))
        projected2to1 = transform_joint_kinematic(name, cam2, out1, t21)
        projected1to2 = transform_joint_kinematic(name, cam1, out2, t12)
        diff1 = float(np.linalg.norm(np.asarray(cam1[name], dtype=float) - projected2to1))
        diff2 = float(np.linalg.norm(np.asarray(cam2[name], dtype=float) - projected1to2))
        for key, output, source, source_score, target_score, src_visible, tgt_visible, disagreement, source_bone in (
            ("camera1", out1, projected2to1, score2, score1, visible2, visible1, diff1, b2.get(name, 1.0)),
            ("camera2", out2, projected1to2, score1, score2, visible1, visible2, diff2, b1.get(name, 1.0)),
        ):
            source_bone = float(np.clip(source_bone, 0.0, 1.0))
            if not src_visible:
                continue
            if key == "camera1" and master_protected and tgt_visible:
                continue
            if risk_enabled and not tgt_visible and source_bone < min_source_bone_quality:
                stats["rejected_occluded_bone"] += 1
                continue
            if risk_enabled and not tgt_visible and transform_reliability < min_alignment_quality:
                stats["rejected_alignment"] += 1
                continue
            weight = _transfer_weight(
                source_score, target_score, src_visible, tgt_visible,
                disagreement, transform_reliability, cfg,
            )
            if risk_enabled and not tgt_visible:
                # Occluded rescue: independent source bone and alignment evidence.
                weight = min(
                    weight,
                    float(cfg.get("max_occluded_transfer", 0.90))
                    * math.sqrt(source_bone)
                    * math.sqrt(transform_reliability),
                )
                weight *= source_bone
            elif visible_risk_enabled and tgt_visible:
                # M4 v2: visible target is protected per joint/frame, never by
                # globally locking the master camera.
                protect_score = float(cfg.get("visible_protect_quality", 0.65))
                gap_ratio = float(cfg.get("visible_max_source_deficit", 0.12))
                outlier_m = float(cfg.get("visible_hard_conflict_m", 0.16))
                if (target_score >= protect_score
                        and source_score < target_score * (1.0 - gap_ratio)
                        and disagreement >= outlier_m):
                    stats["rejected_visible_risk"] += 1
                    continue
                excess = max(
                    0.0, disagreement - float(cfg.get("visible_safe_disagreement_m", 0.04))
                )
                risk_scale = max(float(cfg.get("visible_risk_scale_m", 0.10)), 1e-6)
                target_advantage = max(0.0, target_score - source_score)
                penalty = 1.0 + target_advantage / max(
                    float(cfg.get("visible_quality_gap_scale", 0.15)), 1e-6
                )
                risk_factor = 1.0 / (1.0 + (excess / risk_scale) ** 2 * penalty)
                weight *= risk_factor * math.sqrt(source_bone)
                visible_risk_factors.append(risk_factor)
            if weight <= 1e-7:
                continue
            if not np.isfinite(source).all():
                stats["rejected_nonfinite"] += 1
                continue
            baseline = cam1 if key == "camera1" else cam2
            current = np.asarray(baseline[name], dtype=float)
            proposed = (1.0 - weight) * current + weight * source
            if not np.isfinite(proposed).all():
                stats["rejected_nonfinite"] += 1
                continue
            # M5: project using stable per-camera sequence references, never
            # automatically treat a corrupted current-frame bone as correct.
            ref_lengths = bone_references.get(key, {})
            projected = _project_bone(proposed, name, output, baseline, ref_lengths)
            parent = PARENT_JOINTS.get(name)
            nominal_length = ref_lengths.get(name)
            if nominal_length is None and parent in baseline and name in baseline:
                nominal_length = float(np.linalg.norm(
                    np.asarray(baseline[name], dtype=float)
                    - np.asarray(baseline[parent], dtype=float)
                ))
            correction = float(np.linalg.norm(projected - proposed))
            if (nominal_length is not None and nominal_length > 1e-6 and
                    correction / nominal_length > max_projection_ratio):
                stats["rejected_kinematic_projection"] += 1
                continue
            trial = dict(output)
            trial[name] = projected
            # Compare against the actual current partial pose, not the raw
            # baseline; parent joints may already have been corrected.
            if not _angle_is_valid_locally(trial, name, output):
                stats["rejected_local_angle"] += 1
                continue
            output[name] = projected
            transfers[key][name] = float(weight)
            stats[f"accepted_{key}"] += 1
    if visible_risk_factors:
        stats["mean_visible_risk_multiplier"] = float(np.mean(visible_risk_factors))
    return out1, out2, {
        "joint_transfer": transfers, "adaptive_stats": stats,
    }

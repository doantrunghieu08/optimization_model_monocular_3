"""AOTR-3D: explicit no-2D experimental surrogate.

This is NOT two-view 2D observability. It uses per-bone orientations
from two independently estimated WHAM 3D poses, rigid torso alignment,
mesh visibility, and temporal curvature to build an optimization-only
master-asymmetric merit function. No labels, learned reliability, or
synthetic 2D observations are used. Improvements in this surrogate do
not imply improvements in 3D MPJPE.
"""
import numpy as np

from fusion_pipeline.correction import (
    apply_similarity, estimate_bidirectional_similarity,
)
from fusion_pipeline.aotr import (
    _ordered_bones, _trust_region_1d, _unit, TORSO,
    _median_cross_view_bone_angle, _fallback_transform_from_prev,
)


def _bone_dir(pose, child, parent):
    if not isinstance(pose, dict) or child not in pose or parent not in pose:
        return None
    vec = np.asarray(pose[child], dtype=float) - np.asarray(pose[parent], dtype=float)
    if vec.shape != (3,) or not np.isfinite(vec).all():
        return None
    return _unit(vec)


def _bone_jitter(current, prev, prev_prev, child, parent):
    """Local orientation angular acceleration; 0 if history unavailable.

    A separate flag indicates whether history actually exists, so missing
    temporal information is not silently interpreted as perfect motion.
    """
    u0 = _bone_dir(current, child, parent)
    u1 = _bone_dir(prev, child, parent)
    u2 = _bone_dir(prev_prev, child, parent)
    if any(u is None for u in (u0, u1, u2)):
        return None
    return float(np.linalg.norm(u0 - 2.0 * u1 + u2))


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _cap_alpha(uM, uS, alpha, max_deg):
    """Largest alpha' <= alpha whose blended direction is within max_deg of uM."""
    def dev(a):
        u = _unit((1.0-a)*uM + a*uS)
        return 0.0 if u is None else _angle_deg(u, uM)
    if dev(alpha) <= max_deg:
        return alpha
    lo, hi = 0.0, alpha
    for _ in range(40):
        mid = 0.5*(lo+hi)
        if dev(mid) <= max_deg:
            lo = mid
        else:
            hi = mid
    return lo


def _as_pose(pose):
    if not isinstance(pose, dict):
        return None
    try:
        out = {j: np.asarray(v, dtype=float) for j, v in pose.items()}
    except (TypeError, ValueError):
        return None
    if not out or not all(p.shape == (3,) and np.isfinite(p).all() for p in out.values()):
        return None
    return out


def optimize_aotr_3d(
    master, slave, vis_master=None, vis_slave=None, config=None,
    prev_master=None, prev_slave=None,
    prev_prev_master=None, prev_prev_slave=None,
    prev_transform=None,
):
    """Optimize only bone orientation blending in Master coordinates.

    The torso transform is inferred from WHAM estimates and must not be
    mistaken for calibrated camera extrinsics. All correction scores are
    internal 3D surrogates, not direct measurements of accuracy.
    """
    cfg = dict(config or {})
    M, S = _as_pose(master), _as_pose(slave)
    raw = {j: list(v) for j, v in master.items()}
    details = {
        "status": "fallback", "reason": "", "evidence_mode": "3d_surrogate",
        "alignment_median_m": None, "alignment_q75_m": None,
        "observations": 0, "accepted_bones": [], "evaluations": 0,
        "cost_before": None, "cost_after": None,
        "joint_selection": {j: "KEEP" for j in master},
        "temporal_evidence_bones": 0,
    }

    def fail(reason):
        details["reason"] = reason
        return raw, details

    if M is None or S is None or M.keys() != S.keys():
        return fail("invalid_3d_input")
    v1, v2 = vis_master or {}, vis_slave or {}
    anchor = [
        j for j in TORSO if j in M and
        v1.get(j, True) and v2.get(j, True)
    ]
    t12 = t21 = None
    align_fail_reason = None
    used_temporal_alignment = False
    if len(anchor) < 3:
        align_fail_reason = "insufficient_torso_anchors"
    else:
        basis = np.vstack([M[j] for j in anchor])
        if np.linalg.matrix_rank(basis - basis.mean(0), tol=1e-5) < 2:
            align_fail_reason = "degenerate_torso_alignment"
        else:
            t12, t21, inliers = estimate_bidirectional_similarity(
                M, S, anchor,
                threshold=float(cfg.get("ransac_threshold_m", 0.06)),
                max_combos=int(cfg.get("ransac_max_combos", 100)),
            )
            if len(inliers) < 3:
                align_fail_reason = "insufficient_alignment_inliers"
            else:
                residual = [float(np.linalg.norm(apply_similarity(M[j], t12) - S[j]))
                            for j in anchor]
                median = float(np.median(residual))
                q75 = float(np.percentile(residual, 75))
                details["alignment_median_m"] = median
                details["alignment_q75_m"] = q75
                if median > float(cfg.get("alignment_median_max_m", 0.12)) or q75 > float(cfg.get("alignment_q75_max_m", 0.18)):
                    align_fail_reason = "unreliable_alignment"

    if align_fail_reason is not None:
        t12 = t21 = None
        if bool(cfg.get("temporal_alignment_fallback", False)):
            t12, t21 = _fallback_transform_from_prev(M, S, prev_transform, v1, v2)
        if t12 is None:
            return fail(align_fail_reason)
        used_temporal_alignment = True
        details["alignment_source"] = "temporal_fallback"
    else:
        details["alignment_source"] = "torso_ransac"
        details["_transform"] = (t12, t21)

    bones = _ordered_bones(M)
    prior_M, prior_S = _as_pose(prev_master), _as_pose(prev_slave)
    older_M, older_S = _as_pose(prev_prev_master), _as_pose(prev_prev_slave)
    rotation = t21[1]
    max_cv_angle = cfg.get("max_cross_view_bone_angle_deg")
    if max_cv_angle is not None:
        med_ang = _median_cross_view_bone_angle(M, S, rotation, bones, v1, v2)
        details["median_cross_view_bone_angle_deg"] = med_ang
        if med_ang > float(max_cv_angle):
            return fail("cross_view_pose_mismatch")
    alpha_by_bone = {}
    directions = {}
    weights = {}
    accepted = []
    total_before, total_after, evaluations = 0.0, 0.0, 0
    jitter_count = 0
    floor = float(cfg.get("three_d_consensus_weight", 0.12))
    advantage_scale = float(cfg.get("three_d_slave_advantage_gain", 3.0))
    master_weight = float(cfg.get("three_d_master_prior_weight", 0.75)) * (1.5 if used_temporal_alignment else 1.0)
    temporal_weight = float(cfg.get("three_d_temporal_weight", 0.15))
    jitter_smoothing = max(1e-5, float(cfg.get("three_d_jitter_epsilon", 0.06)))
    min_gain = max(0.0, float(cfg.get("minimum_cost_reduction", 1e-7)))
    # Master-protection options (ablation). Defaults reproduce the original behaviour.
    # They only restrict bones whose Master joint is visible; occluded Master bones keep
    # the original rescue path.
    floor_needs_advantage = bool(cfg.get("three_d_floor_requires_advantage", False))
    max_disagree = cfg.get("three_d_max_disagreement_deg")
    max_correct = cfg.get("three_d_max_correction_deg")
    gated_bones = 0

    for child, parent in bones:
        uM = _bone_dir(M, child, parent)
        uS_cam = _bone_dir(S, child, parent)
        if uM is None or uS_cam is None or not v2.get(child, True):
            continue
        uS = _unit(rotation @ uS_cam)
        if uS is None or float(uM @ uS) < float(cfg.get("min_bone_direction_cos", -0.8)):
            continue
        if float(np.linalg.norm(uM-uS)) < 1e-6:
            continue
        master_visible = bool(v1.get(child, True))
        if (max_disagree is not None and master_visible
                and _angle_deg(uM, uS) > float(max_disagree)):
            # Large disagreement with no independent evidence: undecidable, keep Master.
            gated_bones += 1
            continue

        jm = _bone_jitter(M, prior_M, older_M, child, parent)
        js = _bone_jitter(S, prior_S, older_S, child, parent)
        has_temporal = jm is not None and js is not None
        if has_temporal:
            jitter_count += 1
            qm = 1.0/(jitter_smoothing+jm)
            qs = 1.0/(jitter_smoothing+js)
        else:
            qm, qs = 1.0, 1.0
        if not v1.get(child, True):
            qm *= float(cfg.get("three_d_occluded_master_reliability", 0.15))
        # Require actual evidence to favour Slave for large movements.
        # Scale consensus floor by relative Slave quality so toxic/degraded Slaves (qs << qm)
        # cannot inject orientation errors into a clean, visible Master.
        advantage = max(0.0, (qs - qm) / max(qs + qm, 1e-8))
        rel_slave = qs / max(qs + qm, 1e-8)
        floor_term = floor * rel_slave
        if floor_needs_advantage and master_visible and advantage <= 0.0:
            floor_term = 0.0
        wS = max(0.0, floor_term + advantage_scale * advantage)
        wM = max(0.0, master_weight)
        # Local Master orientation is regularized more heavily when it is
        # observed and temporally stable.
        if has_temporal and v1.get(child, True) and jm <= js:
            wM *= 1.0 + (js-jm)/(jitter_smoothing+js)
        up = _bone_dir(prior_M, child, parent)
        upp = _bone_dir(older_M, child, parent)
        predicted = _unit(2*up-upp) if up is not None and upp is not None else None

        def cost(alpha):
            u = _unit((1.0-alpha)*uM + alpha*uS)
            if u is None:
                return 1e9
            loss = wS*float(np.sum((u-uS)**2))
            loss += wM*float(np.sum((u-uM)**2))
            if predicted is not None:
                loss += temporal_weight*float(np.sum((u-predicted)**2))
            return loss

        alpha, best, n_eval = _trust_region_1d(
            cost, int(cfg.get("trust_max_iter", 12)),
            float(cfg.get("trust_initial_radius", 0.25)),
        )
        if max_correct is not None and master_visible and alpha > 0.0:
            alpha = _cap_alpha(uM, uS, alpha, float(max_correct))
            best = float(cost(alpha))
        before = float(cost(0.0))
        total_before += before
        evaluations += n_eval
        gain = before-best
        if alpha > 1e-5 and gain > min_gain:
            alpha_by_bone[child] = alpha
            total_after += float(best)
            accepted.append({
                "joint": child, "alpha": round(float(alpha), 6),
                "cost_gain": round(float(gain), 10),
                "reliability_advantage": round(float(advantage), 6),
                "has_temporal_evidence": has_temporal,
            })
        else:
            total_after += before
        directions[child] = (uM, uS)
        weights[child] = float(np.linalg.norm(M[child]-M[parent]))

    details["evaluations"] = evaluations
    details["temporal_evidence_bones"] = jitter_count
    details["disagreement_gated_bones"] = gated_bones
    details["cost_before"] = total_before
    details["cost_after"] = total_after
    details["accepted_bones"] = accepted
    if not accepted:
        details["status"] = "unchanged"
        details["reason"] = "no_supported_3d_improvement"
        return raw, details

    pose = {j: p.copy() for j, p in M.items()}
    for child, parent in bones:
        if child in directions and child in alpha_by_bone:
            uM, uS = directions[child]
            a = alpha_by_bone[child]
            new_dir = _unit((1.0-a)*uM+a*uS)
            local = weights[child] * new_dir
        else:
            local = M[child]-M[parent]
        pose[child] = pose[parent] + local
    for j in M:
        if float(np.linalg.norm(pose[j]-M[j])) <= 1e-10:
            pose[j] = M[j].copy()
    for j in pose:
        if j in alpha_by_bone:
            details["joint_selection"][j] = "CORRECT"
        elif float(np.linalg.norm(pose[j]-M[j])) > 1e-7:
            details["joint_selection"][j] = "PROPAGATE"
    details["status"] = "corrected"
    details["reason"] = "3d_surrogate_improved"
    return {j: p.tolist() for j, p in pose.items()}, details

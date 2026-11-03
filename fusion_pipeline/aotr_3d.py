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
from fusion_pipeline.aotr import _ordered_bones, _trust_region_1d, _unit, TORSO


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
    if len(anchor) < 3:
        return fail("insufficient_torso_anchors")
    basis = np.vstack([M[j] for j in anchor])
    if np.linalg.matrix_rank(basis - basis.mean(0), tol=1e-5) < 2:
        return fail("degenerate_torso_alignment")

    t12, t21, inliers = estimate_bidirectional_similarity(
        M, S, anchor,
        threshold=float(cfg.get("ransac_threshold_m", 0.06)),
        max_combos=int(cfg.get("ransac_max_combos", 100)),
    )
    if len(inliers) < 3:
        return fail("insufficient_alignment_inliers")
    residual = [float(np.linalg.norm(apply_similarity(M[j], t12) - S[j]))
                for j in anchor]
    median = float(np.median(residual))
    q75 = float(np.percentile(residual, 75))
    details["alignment_median_m"] = median
    details["alignment_q75_m"] = q75
    if median > float(cfg.get("alignment_median_max_m", 0.12)) or q75 > float(cfg.get("alignment_q75_max_m", 0.18)):
        return fail("unreliable_alignment")

    bones = _ordered_bones(M)
    prior_M, prior_S = _as_pose(prev_master), _as_pose(prev_slave)
    older_M, older_S = _as_pose(prev_prev_master), _as_pose(prev_prev_slave)
    rotation = t21[1]
    alpha_by_bone = {}
    directions = {}
    weights = {}
    accepted = []
    total_before, total_after, evaluations = 0.0, 0.0, 0
    jitter_count = 0
    floor = float(cfg.get("three_d_consensus_weight", 0.12))
    advantage_scale = float(cfg.get("three_d_slave_advantage_gain", 3.0))
    master_weight = float(cfg.get("three_d_master_prior_weight", 0.75))
    temporal_weight = float(cfg.get("three_d_temporal_weight", 0.15))
    jitter_smoothing = max(1e-5, float(cfg.get("three_d_jitter_epsilon", 0.06)))
    min_gain = max(0.0, float(cfg.get("minimum_cost_reduction", 1e-7)))

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
        # A weak consensus floor allows measuring small corrections when
        # both views appear equally reliable; it is not a 3D certification.
        advantage = max(0.0, (qs-qm) / max(qs+qm, 1e-8))
        wS = max(0.0, floor + advantage_scale*advantage)
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

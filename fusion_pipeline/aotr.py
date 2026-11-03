"""AOTR: training-free, master-asymmetric, observability-aware trust-region fusion.

Experimental. Ground truth is NEVER read here. The second view supplies a
candidate *bone orientation*, not an unconditional 3D replacement.
The measured decrease is in a geometric surrogate, NOT guaranteed MPJPE.
"""
from collections import defaultdict
import numpy as np

from fusion_pipeline.correction import (
    PARENT_JOINTS, apply_similarity, estimate_bidirectional_similarity,
)

# Do not treat copied wrist->hand or ankle->foot COCO aliases as new 2D evidence.
DIRECT_2D = {
    "head", "neck", "pelvis", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
}
TORSO = ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _unit(vec):
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-9 else None


def _intrinsics(K):
    if K is None:
        return None
    a = np.asarray(K, dtype=float)
    if a.shape != (3, 3) or not np.isfinite(a).all() or a[0, 0] <= 0 or a[1, 1] <= 0:
        return None
    return a


def _observations(kps, K, names, min_confidence):
    if K is None or not isinstance(kps, dict):
        return {}
    result = {}
    for joint in names:
        if joint not in DIRECT_2D:
            continue
        kp = kps.get(joint)
        if kp is None:
            continue
        arr = np.asarray(kp, dtype=float).reshape(-1)
        if arr.size < 3 or not np.isfinite(arr[:3]).all():
            continue
        confidence = float(np.clip(arr[2], 0.0, 1.0))
        if confidence >= min_confidence:
            result[joint] = (arr[:2], confidence)
    return result


def _project(point, K):
    p = np.asarray(point, dtype=float)
    if not np.isfinite(p).all() or p[2] <= 1e-5:
        return None
    uvw = K @ p
    uv = uvw[:2] / uvw[2]
    if not np.isfinite(uv).all():
        return None
    return uv


def _jacobian(point, K):
    p = np.asarray(point, dtype=float)
    x, y, z = p
    if z <= 1e-5:
        return None
    f = float(np.sqrt(K[0, 0] * K[1, 1]))
    # Normalized image error, measured in roughly angular units.
    return np.array([
        [K[0, 0] / (f * z), 0., -K[0, 0] * x / (f * z*z)],
        [0., K[1, 1] / (f * z), -K[1, 1] * y / (f * z*z)],
    ])


def _huber_norm(r, delta):
    norm = float(np.linalg.norm(r))
    return 0.5 * norm * norm if norm <= delta else delta * (norm - 0.5 * delta)


def _ordered_bones(master):
    children = defaultdict(list)
    for child, parent in PARENT_JOINTS.items():
        if child in master and parent in master:
            children[parent].append(child)
    order = []
    seen = set()

    def visit(parent):
        for child in children.get(parent, []):
            if child not in seen:
                seen.add(child)
                order.append((child, parent))
                visit(child)
    for root in ("pelvis", "neck"):
        visit(root)
    # Missing roots must not cause an infinite traversal or silent data loss.
    for child, parent in PARENT_JOINTS.items():
        if child not in seen and child in master and parent in master:
            seen.add(child)
            order.append((child, parent))
    return order


def _trust_region_1d(cost, max_iter, initial_radius):
    """Bounded 1D trust-region using true/predicted merit reduction.

    Multiple starts prevent a local optimum near KEEP from automatically
    vetoing an otherwise beneficial full correction.
    """
    baseline = float(cost(0.0))
    best_x, best_cost = 0.0, baseline
    evaluations = 1
    for start in (0.0, 0.5, 1.0):
        x = start
        y = float(cost(x))
        evaluations += 1
        radius = float(np.clip(initial_radius, 0.05, 0.5))
        for _ in range(max_iter):
            h = min(0.02, radius * 0.3)
            left = max(0.0, x - h)
            right = min(1.0, x + h)
            if right - left < 1e-7:
                break
            yl, yr = float(cost(left)), float(cost(right))
            evaluations += 2
            # Derivatives from a local quadratic model fitted at three points.
            if left < x < right:
                g = (yr - yl) / (right - left)
                curvature = 2.0 * (((yr - y)/(right-x)) - ((y-yl)/(x-left))) / (right-left)
            else:
                g = (yr - yl) / (right - left)
                curvature = 0.0
            if curvature > 1e-12:
                step = -g / curvature
            else:
                step = -radius if g > 0 else radius
            step = float(np.clip(step, -radius, radius))
            proposal = float(np.clip(x + step, 0.0, 1.0))
            step = proposal - x
            if abs(step) < 1e-6:
                break
            trial = float(cost(proposal))
            evaluations += 1
            predicted = -g*step - 0.5*curvature*step*step
            actual = y-trial
            rho = actual / predicted if predicted > 1e-13 else -np.inf
            if actual > 1e-12 and rho > 0.1:
                x, y = proposal, trial
            if rho < 0.25:
                radius *= 0.5
            elif rho > 0.75 and abs(step) >= 0.8*radius:
                radius = min(1.0, 2.0*radius)
            if radius < 1e-4:
                break
        if y < best_cost:
            best_x, best_cost = x, y
    return best_x, best_cost, evaluations


def optimize_aotr(
    master, slave, observations_master=None, observations_slave=None,
    K_master=None, K_slave=None, vis_master=None, vis_slave=None,
    config=None,
):
    """Return (master-frame fused pose, diagnostics).

    FAIL CLOSED on invalid alignment or missing independent 2D support.
    Only the master pose is written; slave remains unchanged.
    """
    cfg = dict(config or {})
    M = {j: np.asarray(v, dtype=float) for j, v in master.items()}
    S = {j: np.asarray(v, dtype=float) for j, v in slave.items()}
    output_master = {j: list(np.asarray(v, dtype=float)) for j, v in master.items()}
    details = {
        "status": "fallback", "reason": "", "alignment_median_m": None,
        "alignment_q75_m": None, "accepted_bones": [], "joint_selection": {
            j: "KEEP" for j in M
        }, "cost_before": None, "cost_after": None,
        "evaluations": 0, "observations": 0,
    }

    def fallback(reason):
        details["reason"] = reason
        return output_master, details

    if M.keys() != S.keys() or not all(np.isfinite(x).all() and x.shape == (3,) for x in list(M.values()) + list(S.values())):
        return fallback("invalid_input")
    K1, K2 = _intrinsics(K_master), _intrinsics(K_slave)
    if K1 is None or K2 is None:
        return fallback("missing_intrinsics")
    min_conf = float(cfg.get("min_2d_confidence", 0.35))
    obs1 = _observations(observations_master, K1, M, min_conf)
    obs2 = _observations(observations_slave, K2, M, min_conf)
    valid_vis1 = vis_master or {}
    valid_vis2 = vis_slave or {}
    obs1 = {j: o for j, o in obs1.items() if valid_vis1.get(j, True)}
    obs2 = {j: o for j, o in obs2.items() if valid_vis2.get(j, True)}
    common_obs = set(obs1) & set(obs2)
    details["observations"] = len(common_obs)
    if len(common_obs) < int(cfg.get("min_correspondences", 4)):
        return fallback("insufficient_2d_correspondences")

    anchors = [j for j in TORSO if j in M and valid_vis1.get(j, True) and valid_vis2.get(j, True)]
    if len(anchors) < 3:
        return fallback("insufficient_torso_anchors")
    centered = np.array([M[j] for j in anchors])
    if np.linalg.matrix_rank(centered - centered.mean(axis=0), tol=1e-5) < 2:
        return fallback("degenerate_torso_alignment")
    t12, t21, inliers = estimate_bidirectional_similarity(
        M, S, anchors, threshold=float(cfg.get("ransac_threshold_m", 0.06)),
        max_combos=int(cfg.get("ransac_max_combos", 100)),
    )
    if len(inliers) < 3:
        return fallback("insufficient_alignment_inliers")
    errors = [float(np.linalg.norm(apply_similarity(M[j], t12) - S[j])) for j in anchors]
    median, q75 = float(np.median(errors)), float(np.percentile(errors, 75))
    details.update(alignment_median_m=median, alignment_q75_m=q75)
    if median > float(cfg.get("alignment_median_max_m", 0.12)) or q75 > float(cfg.get("alignment_q75_max_m", 0.18)):
        return fallback("unreliable_alignment")

    bones = _ordered_bones(M)
    R21 = t21[1]
    directions = {}
    lengths = {}
    for child, parent in bones:
        vm = M[child] - M[parent]
        vs = R21 @ (S[child] - S[parent])
        um, us = _unit(vm), _unit(vs)
        if um is None or us is None or float(um @ us) < float(cfg.get("min_bone_direction_cos", -0.80)):
            continue
        lengths[child] = float(np.linalg.norm(vm))
        directions[child] = (um, us)
    if not directions:
        return fallback("no_valid_bone_candidates")

    # Sensitivity includes both views in the master frame.
    R12 = t12[1]
    eigen_ratio = float(cfg.get("observability_ratio", 0.10))
    null_projectors = {}
    for joint in M:
        rows = []
        for obs, point, K, rot in (
            (obs1, M[joint], K1, np.eye(3)),
            (obs2, apply_similarity(M[joint], t12), K2, R12),
        ):
            if joint not in obs:
                continue
            J = _jacobian(point, K)
            if J is not None:
                rows.append(np.sqrt(obs[joint][1]) * J @ rot)
        if not rows:
            null_projectors[joint] = np.eye(3)
            continue
        A = np.vstack(rows)
        _, sv, vt = np.linalg.svd(A, full_matrices=True)
        cutoff = max(1e-8, eigen_ratio * float(sv[0])) if len(sv) else 1e-8
        rank = int(np.count_nonzero(sv > cutoff))
        null_basis = vt[rank:, :]
        null_projectors[joint] = null_basis.T @ null_basis

    prior_weak = float(cfg.get("observed_prior_weight", 1e-4))
    prior_strong = float(cfg.get("unobserved_prior_weight", 3e-3))
    huber_delta = float(cfg.get("huber_delta_normalized", 0.008))
    min_gain = float(cfg.get("minimum_cost_reduction", 1e-7))
    def reconstruct(alphas):
        pose = {j: M[j].copy() for j in M}
        for child, parent in bones:
            if child not in directions:
                # No candidate: preserve local master direction, but move with its parent.
                local = M[child] - M[parent]
            else:
                um, us = directions[child]
                alpha = float(alphas.get(child, 0.0))
                v = _unit((1.0-alpha)*um + alpha*us)
                if v is None:
                    v = um
                local = lengths[child] * v
            pose[child] = pose[parent] + local
        return pose

    def energy(alphas):
        pose = reconstruct(alphas)
        value = 0.0
        # Keep the full 2D objective for downstream joints affected by
        # a parent rotation; fixed bone lengths are enforced by FK.
        for obs, K, transform in (
            (obs1, K1, None), (obs2, K2, t12),
        ):
            f = float(np.sqrt(K[0, 0] * K[1, 1]))
            for j, (uv, conf) in obs.items():
                p = pose[j] if transform is None else apply_similarity(pose[j], transform)
                projected = _project(p, K)
                if projected is None:
                    return 1e9
                value += conf * _huber_norm((projected-uv)/f, huber_delta)
        for j, pos in pose.items():
            if j == "pelvis":
                continue
            delta = pos - M[j]
            N = null_projectors[j]
            n = N @ delta
            o = delta - n
            value += prior_strong*float(n @ n) + prior_weak*float(o @ o)
        return value

    alphas = {}
    initial_cost = float(energy(alphas))
    if not np.isfinite(initial_cost) or initial_cost >= 1e8:
        return fallback("invalid_baseline_projection")
    current_cost = initial_cost
    max_iter = int(cfg.get("trust_max_iter", 12))
    radius = float(cfg.get("trust_initial_radius", 0.25))
    accepted = []
    evaluations = 1
    for child, parent in bones:
        if child not in directions:
            continue
        # Require independent Slave 2D. If Master is visibly occluded, the
        # Slave view can still rescue the bone (with a null-space penalty).
        # Do not treat a missing Master observation as occlusion by itself.
        if child not in obs2 or (child not in obs1 and valid_vis1.get(child, True)):
            continue
        def probe(alpha):
            proposal = dict(alphas)
            proposal[child] = alpha
            return energy(proposal)
        alpha, best, n_eval = _trust_region_1d(probe, max_iter, radius)
        evaluations += n_eval
        gain = current_cost - best
        if alpha > 1e-5 and gain > min_gain:
            alphas[child] = alpha
            current_cost = float(energy(alphas))
            evaluations += 1
            accepted.append({"joint": child, "alpha": round(alpha, 6),
                             "cost_gain": round(gain, 10)})
    pose = reconstruct(alphas) if accepted else M
    # Preserve exact Master bits for joints with no meaningful displacement.
    # FK arithmetic alone can introduce 1e-16 numerical differences in KEEP.
    for j in M:
        if float(np.linalg.norm(pose[j] - M[j])) <= 1e-10:
            pose[j] = M[j].copy()
    # Report inherited motion explicitly: KEEP never secretly changes.
    for j, p in pose.items():
        moved = float(np.linalg.norm(p - M[j])) > 1e-7
        if j in alphas:
            details["joint_selection"][j] = "CORRECT"
        elif moved:
            details["joint_selection"][j] = "PROPAGATE"
    details["accepted_bones"] = accepted
    details["evaluations"] = evaluations
    details["cost_before"] = initial_cost
    details["cost_after"] = float(energy(alphas)) if accepted else initial_cost
    details["status"] = "corrected" if accepted else "unchanged"
    details["reason"] = "objective_improved" if accepted else "no_supported_improvement"
    # The raw master is returned exactly when no bone has been corrected.
    return ({j: pose[j].tolist() for j in M} if accepted else output_master), details

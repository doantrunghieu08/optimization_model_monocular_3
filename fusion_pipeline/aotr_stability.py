"""AOTR offline stability pass (no GT, no learned component).

We reduce acceleration of the *correction residual* F-M, not the actual
motion. Thus real fast motion in Master is preserved. Joint coordinates are
reconstructed with forward kinematics after temporal scaling; optional
anthropometric length calibration is independently configurable.

Not a guarantee for GT acceleration/MBLE/PA-MPJPE. Report each separately.
"""
import numpy as np
from scipy.optimize import minimize

from fusion_pipeline.aotr import _ordered_bones, _unit


def _xyz(frames):
    if not frames:
        return None, None
    names = list(frames[0])
    if any(set(f) != set(names) for f in frames):
        raise ValueError("Inconsistent joint sets in AOTR sequence")
    array = np.array([[frame[j] for j in names] for frame in frames], dtype=float)
    if array.shape != (len(frames), len(names), 3) or not np.isfinite(array).all():
        raise ValueError("Invalid AOTR 3D sequence")
    return names, array


def _residual_acceleration(residual):
    if len(residual) < 3:
        return 0.0
    accel = residual[2:] - 2.0 * residual[1:-1] + residual[:-2]
    return float(np.sqrt(np.mean(np.sum(accel ** 2, axis=2))))


def _temporal_scales(raw_delta, cfg):
    """Bounded quadratic minimization over one correction scale per frame.

    For s_t in [s_min, 1], minimize
       w_shape * sum ||D_t||^2 (s_t-1)^2
     + w_acc * sum ||Delta2(s_t D_t)||^2
     + w_vel * sum ||Delta1(s_t D_t)||^2.
    Frames where D_t == 0 remain zero; no new Slave correction is introduced.
    """
    n = len(raw_delta)
    active = np.sum(raw_delta ** 2, axis=(1, 2)) > 1e-14
    if n < 3 or not np.any(active):
        return np.ones(n), {"status": "skipped", "reason": "too_short_or_unchanged"}
    w_shape = max(1e-9, float(cfg.get("shape_weight", 1.0)))
    w_acc = max(0.0, float(cfg.get("residual_accel_weight", 10.0)))
    w_vel = max(0.0, float(cfg.get("residual_velocity_weight", 0.5)))
    low = float(np.clip(cfg.get("minimum_retention", 0.20), 0.0, 1.0))
    norm2 = np.sum(raw_delta ** 2, axis=(1, 2))

    def objective(s):
        residual = s[:, None, None] * raw_delta
        dev = s - 1.0
        loss = w_shape * np.dot(norm2, dev * dev)
        grad = 2.0 * w_shape * norm2 * dev
        if w_acc:
            diff = residual[2:] - 2 * residual[1:-1] + residual[:-2]
            loss += w_acc * float(np.sum(diff * diff))
            grad[2:] += 2 * w_acc * np.sum(diff * raw_delta[2:], axis=(1, 2))
            grad[1:-1] -= 4 * w_acc * np.sum(diff * raw_delta[1:-1], axis=(1, 2))
            grad[:-2] += 2 * w_acc * np.sum(diff * raw_delta[:-2], axis=(1, 2))
        if w_vel:
            diff = residual[1:] - residual[:-1]
            loss += w_vel * float(np.sum(diff * diff))
            grad[1:] += 2 * w_vel * np.sum(diff * raw_delta[1:], axis=(1, 2))
            grad[:-1] -= 2 * w_vel * np.sum(diff * raw_delta[:-1], axis=(1, 2))
        return float(loss), grad

    bounds = [(low, 1.0) if a else (0.0, 0.0) for a in active]
    x0 = np.where(active, 1.0, 0.0)
    before = objective(x0)[0]
    if w_acc == 0.0 and w_vel == 0.0:
        return x0, {"status": "unchanged", "cost_before": before, "cost_after": before}
    result = minimize(objective, x0, jac=True, bounds=bounds,
                      method="L-BFGS-B",
                      options={"maxiter": int(cfg.get("max_iter", 160)),
                               "ftol": 1e-11, "gtol": 1e-10, "maxls": 30})
    s = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])
    after = objective(s)[0]
    if not np.isfinite(after) or after > before + 1e-10:
        return x0, {"status": "rejected", "cost_before": before,
                    "cost_after": before}
    return s, {"status": "optimized" if np.any(s[active] < 0.999999) else "unchanged",
               "cost_before": before, "cost_after": after,
               "optimizer_success": bool(result.success)}


def _anthropometric_targets(master_frames, cfg):
    """Sequence-stable lengths, inferred without GT from each Master's limb total.

    Only humerus/forearm and femur/shank are modified. Other Master lengths
    stay as-is (including shoulders, wrists->hands and toe/foot segments).
    """
    lengths = {}
    if not cfg.get("enabled", False):
        return lengths
    upper_frac = float(cfg.get("arm_upper_fraction", 0.54))
    thigh_frac = float(cfg.get("leg_thigh_fraction", 0.53))
    if not (0.0 < upper_frac < 1.0 and 0.0 < thigh_frac < 1.0):
        raise ValueError("Anatomical length ratios must be in (0,1)")
    for side in ("left", "right"):
        arms = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        legs = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
        for (root, mid, tip), ratio in ((arms, upper_frac), (legs, thigh_frac)):
            totals = []
            for frame in master_frames:
                if not all(j in frame for j in (root, mid, tip)):
                    continue
                a, b, c = (np.asarray(frame[j], dtype=float) for j in (root, mid, tip))
                upper, lower = float(np.linalg.norm(b-a)), float(np.linalg.norm(c-b))
                if upper > 1e-5 and lower > 1e-5:
                    totals.append(upper+lower)
            if totals:
                total = float(np.median(totals))
                lengths[(mid, root)] = total*ratio
                lengths[(tip, mid)] = total*(1.0-ratio)
    return lengths


def _fk_reconstruct(master, fused, scale, lengths):
    """Interpolate local bone orientation, then project onto prescribed length.

    scale=0 preserves local orientation of Master; scale=1 preserves initial
    AOTR local orientation. Pelvis and neck stay at the Master positions.
    """
    out = {j: np.asarray(p, dtype=float).copy() for j, p in master.items()}
    for child, parent in _ordered_bones(master):
        m_child, m_parent = np.asarray(master[child]), np.asarray(master[parent])
        f_child, f_parent = np.asarray(fused[child]), np.asarray(fused[parent])
        uM, uF = _unit(m_child-m_parent), _unit(f_child-f_parent)
        if uM is None:
            continue
        if uF is None:
            uF = uM
        direction = _unit((1.0-scale)*uM + scale*uF)
        if direction is None:
            direction = uM
        target = lengths.get((child, parent), float(np.linalg.norm(m_child-m_parent)))
        out[child] = out[parent] + target * direction
    return out


def postprocess_aotr_3d(master_frames, fused_frames, config=None, editable_mask=None):
    """Return (frames, diagnostics) with no access to GT.

    The optional bone calibration is an *additional hypothesis*, not an
    AOTR slave transfer. Therefore it is reported separately.
    """
    cfg = dict(config or {})
    temporal = dict(cfg.get("temporal_stability", {}))
    bone = dict(cfg.get("anatomical_calibration", {}))
    names, M = _xyz(master_frames)
    other_names, F = _xyz(fused_frames)
    if names is None:
        return [], {"status": "empty"}
    if names != other_names:
        raise ValueError("AOTR Master/Fused joint ordering mismatch")
    editable = (
        np.ones(len(F), dtype=bool)
        if editable_mask is None else np.asarray(editable_mask, dtype=bool)
    )
    if editable.shape != (len(F),):
        raise ValueError("editable_mask must have one boolean per frame")
    if np.any(~editable & (np.sum((F-M)**2, axis=(1,2)) > 1e-14)):
        raise ValueError("Noneditable AOTR fallback frames contain Slave corrections")

    before = _residual_acceleration(F-M)
    if temporal.get("enabled", True):
        scales, optimization = _temporal_scales(F-M, temporal)
    else:
        scales = np.ones(len(F))
        optimization = {"status": "disabled"}

    lengths = _anthropometric_targets(master_frames, bone)
    if not lengths and (not temporal.get("enabled", True) or
                        np.allclose(scales, 1.0, atol=1e-7)):
        return fused_frames, {
            "status": "unchanged", "temporal": optimization,
            "initial_residual_accel_m_frame2": before,
            "final_residual_accel_m_frame2": before,
            "anatomically_calibrated_bones": 0,
            "mean_temporal_retention": 1.0,
        }

    output = []
    for i, master in enumerate(master_frames):
        if not editable[i]:
            output.append({j: list(master[j]) for j in names})
            continue
        reconstructed = _fk_reconstruct(master, fused_frames[i], scales[i], lengths)
        # Preserve exact Master coordinates if no FK/length change occurred.
        for joint in names:
            if float(np.linalg.norm(reconstructed[joint]-np.asarray(master[joint]))) <= 1e-10:
                reconstructed[joint] = np.asarray(master[joint]).copy()
        output.append({j: reconstructed[j].tolist() for j in names})
    _, stabilized = _xyz(output)
    active = np.sum((F-M)**2, axis=(1, 2)) > 1e-14
    return output, {
        "status": "applied",
        "temporal": optimization,
        "initial_residual_accel_m_frame2": before,
        "final_residual_accel_m_frame2": _residual_acceleration(stabilized-M),
        "anatomically_calibrated_bones": len(lengths),
        "frozen_fallback_frames": int(np.count_nonzero(~editable)),
        "mean_temporal_retention": float(np.mean(scales[active])) if np.any(active) else 1.0,
        "minimum_temporal_retention": float(np.min(scales[active])) if np.any(active) else 1.0,
    }

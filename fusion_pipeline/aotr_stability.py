"""AOTR offline stability pass (no GT, no learned component).

We reduce acceleration of the *correction residual* F-M, not the actual
motion. Thus real fast motion in Master is preserved. Joint coordinates are
reconstructed with forward kinematics after temporal scaling; optional
anthropometric length calibration is independently configurable.

Not a guarantee for GT acceleration/MBLE/PA-MPJPE. Report each separately.
"""
import numpy as np
from scipy.optimize import minimize
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

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


def _estimate_master_height(master_frames):
    """Estimate natural vertical standing height of Master skeleton along kinematic chain."""
    if not master_frames:
        return 0.0
    torso_lens = []
    for f in master_frames:
        if "neck" in f and "pelvis" in f:
            torso_lens.append(float(np.linalg.norm(np.asarray(f["neck"]) - np.asarray(f["pelvis"]))))
        elif "neck" in f and "left_hip" in f and "right_hip" in f:
            hip_mid = 0.5 * (np.asarray(f["left_hip"]) + np.asarray(f["right_hip"]))
            torso_lens.append(float(np.linalg.norm(np.asarray(f["neck"]) - hip_mid)))
        elif "neck" in f and "left_hip" in f:
            torso_lens.append(float(np.linalg.norm(np.asarray(f["neck"]) - np.asarray(f["left_hip"]))))
    med_torso = float(np.median(torso_lens)) if torso_lens else 0.5

    leg_lens = []
    for f in master_frames:
        for side in ("left", "right"):
            h, k, a = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
            if h in f and k in f and a in f:
                leg_lens.append(float(np.linalg.norm(np.asarray(f[k]) - np.asarray(f[h])) +
                                      np.linalg.norm(np.asarray(f[a]) - np.asarray(f[k]))))
    med_leg = float(np.median(leg_lens)) if leg_lens else 0.8

    head_lens = []
    for f in master_frames:
        if "head" in f and "neck" in f:
            head_lens.append(float(np.linalg.norm(np.asarray(f["head"]) - np.asarray(f["neck"]))))
        elif "nose" in f and "neck" in f:
            head_lens.append(float(np.linalg.norm(np.asarray(f["nose"]) - np.asarray(f["neck"]))))
    med_head = float(np.median(head_lens)) if head_lens else 0.15 * med_torso

    foot_lens = []
    for f in master_frames:
        for side in ("left", "right"):
            a, ft = f"{side}_ankle", f"{side}_foot"
            if a in f and ft in f:
                foot_lens.append(float(np.linalg.norm(np.asarray(f[ft]) - np.asarray(f[a]))))
            elif a in f and f"{side}_toe" in f:
                foot_lens.append(float(np.linalg.norm(np.asarray(f[f"{side}_toe"]) - np.asarray(f[a]))))
    med_foot = float(np.median(foot_lens)) if foot_lens else 0.06 * med_leg

    return max(0.5, med_head + med_torso + med_leg + med_foot)


def _anthropometric_targets(master_frames, cfg):
    """Sequence-stable lengths, inferred from known height or Master's limb total.

    Only humerus/forearm and femur/shank are modified. Other Master lengths
    stay as-is (including shoulders, wrists->hands and toe/foot segments).
    When arm_upper_fraction or leg_thigh_fraction is omitted, proportions are
    auto-derived from the subject's own Master sequence (generalized, zero-finetuning).
    """
    lengths = {}
    if not cfg.get("enabled", False) or not master_frames:
        return lengths

    upper_frac = cfg.get("arm_upper_fraction")
    thigh_frac = cfg.get("leg_thigh_fraction")
    if upper_frac is not None:
        upper_frac = float(upper_frac)
        if not (0.0 < upper_frac < 1.0):
            raise ValueError("Anatomical length ratios must be in (0,1)")
    if thigh_frac is not None:
        thigh_frac = float(thigh_frac)
        if not (0.0 < thigh_frac < 1.0):
            raise ValueError("Anatomical length ratios must be in (0,1)")

    known_height = cfg.get("person_height_m")
    if known_height is not None:
        try:
            known_height = float(known_height)
        except (TypeError, ValueError):
            known_height = None

    scale = 1.0
    if known_height is not None and known_height > 0:
        h_m = _estimate_master_height(master_frames)
        if h_m > 0:
            scale = known_height / h_m

    arm_h_frac = cfg.get("arm_height_fraction")
    leg_h_frac = cfg.get("leg_height_fraction")

    for side in ("left", "right"):
        arms = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        legs = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
        for (root, mid, tip), ratio, h_frac in ((arms, upper_frac, arm_h_frac), (legs, thigh_frac, leg_h_frac)):
            uppers, lowers = [], []
            for frame in master_frames:
                if not all(j in frame for j in (root, mid, tip)):
                    continue
                a, b, c = (np.asarray(frame[j], dtype=float) for j in (root, mid, tip))
                upper, lower = float(np.linalg.norm(b-a)), float(np.linalg.norm(c-b))
                if upper > 1e-5 and lower > 1e-5:
                    uppers.append(upper)
                    lowers.append(lower)
            if not uppers or not lowers:
                continue

            med_u = float(np.median(uppers))
            med_l = float(np.median(lowers))
            measured_total = med_u + med_l

            if ratio is not None:
                tgt_u, tgt_l = ratio, 1.0 - ratio
            else:
                # Generalized: auto-derived from subject's own Master sequence medians
                tgt_u = med_u / measured_total
                tgt_l = med_l / measured_total

            if h_frac is not None:
                total = known_height * float(h_frac)
            elif known_height is not None and known_height > 0:
                total = measured_total * scale
            else:
                total = measured_total

            lengths[(mid, root)] = total * tgt_u
            lengths[(tip, mid)] = total * tgt_l
    return lengths


def _fk_reconstruct(master, fused, scale, lengths):
    """Interpolate local bone orientation, then project onto prescribed length.

    scale=0 preserves local orientation of Master; scale=1 preserves initial
    AOTR local orientation. Pelvis and neck stay at the Master positions.

    When child was not corrected by AOTR (fused[child] == master[child]),
    we use the master bone direction only — blending with uF would be
    meaningless because uF is not an independent observation, just the
    master vector shifted by the corrected parent. This avoids spurious
    accel errors from FK propagation on blind/occluded joints.
    # ponytail: use_master_dir_if_uncorrected; lift if stability needs downstream blend.
    """
    out = {j: np.asarray(p, dtype=float).copy() for j, p in master.items()}
    for child, parent in _ordered_bones(master):
        m_child, m_parent = np.asarray(master[child]), np.asarray(master[parent])
        f_child, f_parent = np.asarray(fused[child]), np.asarray(fused[parent])
        uM = _unit(m_child - m_parent)
        if uM is None:
            continue
        # Child not directly corrected: use master direction to avoid spurious accel
        # from parent displacement feeding into the blend direction.
        if np.linalg.norm(f_child - m_child) <= 1e-7:
            direction = uM
        else:
            uF = _unit(f_child - f_parent)
            if uF is None:
                uF = uM
            direction = _unit((1.0 - scale) * uM + scale * uF)
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

    bones = _ordered_bones(master_frames[0])
    n = len(master_frames)
    w_len = int(temporal.get("smooth_window", 5))
    if w_len % 2 == 0:
        w_len -= 1

    reconstructed_frames = []
    for i, master in enumerate(master_frames):
        if not editable[i]:
            reconstructed_frames.append({j: np.asarray(master[j], dtype=float).copy() for j in names})
            continue
        rec = _fk_reconstruct(master, fused_frames[i], scales[i], lengths)
        for joint in names:
            if float(np.linalg.norm(rec[joint] - np.asarray(master[joint]))) <= 1e-10:
                rec[joint] = np.asarray(master[joint]).copy()
        reconstructed_frames.append(rec)

    # Trajectory smoothing + FK bone projection eliminates camera jitter without flattening motion
    if savgol_filter is not None and (temporal.get("smooth_trajectories", False) or temporal.get("smooth_directions", False)) and n >= w_len and w_len >= 3:
        name_to_idx = {name: i for i, name in enumerate(names)}
        curr_arr = np.array([[reconstructed_frames[t][j] for j in names] for t in range(n)], dtype=float)
        sm_arr = savgol_filter(curr_arr, window_length=w_len, polyorder=2, axis=0)
        proj_arr = sm_arr.copy()
        for t in range(n):
            if not editable[t]:
                continue
            for child, parent in bones:
                ci, pi = name_to_idx[child], name_to_idx[parent]
                mc = np.asarray(master_frames[t][child], dtype=float)
                mp = np.asarray(master_frames[t][parent], dtype=float)
                tgt = lengths.get((child, parent), float(np.linalg.norm(mc - mp)))
                diff = sm_arr[t, ci] - proj_arr[t, pi]
                d_norm = float(np.linalg.norm(diff))
                u = diff / d_norm if d_norm > 1e-6 else np.array([0.0, -1.0, 0.0])
                proj_arr[t, ci] = proj_arr[t, pi] + tgt * u
        for t in range(n):
            if editable[t]:
                reconstructed_frames[t] = {names[j]: proj_arr[t, j] for j in range(len(names))}

    output = []
    for t in range(n):
        rec = reconstructed_frames[t]
        for joint in names:
            if float(np.linalg.norm(rec[joint] - np.asarray(master_frames[t][joint]))) <= 1e-10:
                rec[joint] = np.asarray(master_frames[t][joint]).copy()
        output.append({j: rec[j].tolist() for j in names})
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

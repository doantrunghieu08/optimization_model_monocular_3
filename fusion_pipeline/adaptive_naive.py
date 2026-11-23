"""Adaptive naive two-view fusion (fusion.method=adaptive_naive).

Control for AOTR. It uses the same inputs and the same pipeline steps as AOTR, but
replaces the optimisation (2D reprojection objective, observability prior and
trust-region solver) with a closed-form per-bone weight:

1. Align the Slave to the Master with AOTR's rigid torso RANSAC: same anchors, same
   thresholds, same gates (the ``fusion.aotr`` alignment keys). A frame that fails the
   gate keeps the raw Master.
2. Score each bone in each view:

       q = c^p * exp(-j / tau)

   * ``c`` is the 2D detector confidence of the bone. It is the minimum over the bone's
     endpoints that have a direct 2D keypoint (AOTR's DIRECT_2D). When neither endpoint
     has one, or the frame has no 2D, ``missing_confidence`` is used.
   * ``j`` is the temporal jitter of the bone direction in that view: the norm of the
     second difference of the unit vector, averaged over +-``jitter_half_window``
     frames. It is divided by the sequence median of that bone over both views, so
     tau = 1 means "typical jitter".
3. Compute the Slave weight ``w = q_S / (q_M + q_S)``; it is 0.5 when both scores are 0,
   i.e. a plain average. Blend the directions:

       u = normalize((1 - w) u_M + w R21 u_S)

   A bone whose aligned directions disagree beyond ``min_bone_direction_cos`` keeps the
   Master direction, as in AOTR.
4. Rebuild the skeleton by forward kinematics with the Master bone lengths. Pelvis and
   neck stay at the Master positions.
5. Apply the same sequence-stability pass as AOTR (fusion_pipeline.aotr_stability).

No GT and no learning. ``confidence_power`` (p) and ``jitter_scale`` (tau) are the only
free numbers. With p = 0 and a very large tau, every weight is 0.5 (fixed-alpha naive).
Only camera1 (Master) is modified.
"""
from collections import Counter

import numpy as np

from fusion_pipeline.aotr import DIRECT_2D, TORSO, _ordered_bones, _unit
from fusion_pipeline.correction import apply_similarity, estimate_bidirectional_similarity

DEFAULTS = {
    "confidence_power": 1.0,
    "jitter_scale": 1.0,
    "jitter_half_window": 3,
    "missing_confidence": 0.5,
    # Alignment gates: overwritten by fusion.aotr in the executor so both methods match.
    "ransac_threshold_m": 0.06,
    "ransac_max_combos": 100,
    "alignment_median_max_m": 0.12,
    "alignment_q75_max_m": 0.18,
    "min_bone_direction_cos": -0.80,
}


def _as_pose(frame):
    return {k: np.asarray(v, dtype=float) for k, v in frame.items()}


def _score(value):
    """2D score from [x, y, c] (or a bare number); None when absent or not finite."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    c = float(arr[2]) if arr.size >= 3 else float(arr[0])
    return float(np.clip(c, 0.0, 1.0)) if np.isfinite(c) else None


def bone_confidence(kps, child, parent, missing):
    """Minimum 2D score over the bone endpoints that have direct 2D evidence."""
    if not isinstance(kps, dict):
        return float(missing)
    scores = [s for s in (_score(kps.get(j)) for j in (child, parent) if j in DIRECT_2D) if s is not None]
    return min(scores) if scores else float(missing)


def bone_jitter(frames, bones, half_window):
    """{child: array[n]} of locally averaged second differences of the unit bone direction."""
    n = len(frames)
    out = {}
    for child, parent in bones:
        u = np.zeros((n, 3))
        ok = np.zeros(n, dtype=bool)
        for t, f in enumerate(frames):
            v = _unit(f[child] - f[parent])
            if v is not None:
                u[t], ok[t] = v, True
        raw = np.full(n, np.nan)
        for t in range(1, n - 1):
            if ok[t - 1] and ok[t] and ok[t + 1]:
                raw[t] = float(np.linalg.norm(u[t + 1] - 2.0 * u[t] + u[t - 1]))
        smooth = np.full(n, np.nan)
        for t in range(n):
            window = raw[max(0, t - half_window): t + half_window + 1]
            window = window[np.isfinite(window)]
            if window.size:
                smooth[t] = float(window.mean())
        out[child] = smooth
    return out


def slave_weights(conf_m, conf_s, jit_m, jit_s, power, tau):
    """Per-bone Slave weight in [0, 1] from confidences and normalised jitter."""
    def quality(c, j):
        return (max(c, 0.0) ** power) * np.exp(-max(j, 0.0) / tau)
    qm, qs = quality(conf_m, jit_m), quality(conf_s, jit_s)
    total = qm + qs
    return 0.5 if total <= 1e-12 else float(qs / total)


def align(master, slave, cfg, vis_master=None, vis_slave=None):
    """AOTR's rigid torso alignment and gate. Returns (R21 or None, reason)."""
    vm, vs = vis_master or {}, vis_slave or {}
    anchors = [j for j in TORSO if j in master and j in slave and vm.get(j, True) and vs.get(j, True)]
    if len(anchors) < 3:
        return None, "insufficient_torso_anchors"
    pts = np.array([master[j] for j in anchors])
    if np.linalg.matrix_rank(pts - pts.mean(axis=0), tol=1e-5) < 2:
        return None, "degenerate_torso_alignment"
    t12, t21, inliers = estimate_bidirectional_similarity(
        master, slave, anchors, threshold=float(cfg["ransac_threshold_m"]),
        max_combos=int(cfg["ransac_max_combos"]))
    if len(inliers) < 3:
        return None, "insufficient_alignment_inliers"
    errors = [float(np.linalg.norm(apply_similarity(master[j], t12) - slave[j])) for j in anchors]
    if (float(np.median(errors)) > float(cfg["alignment_median_max_m"])
            or float(np.percentile(errors, 75)) > float(cfg["alignment_q75_max_m"])):
        return None, "unreliable_alignment"
    return np.asarray(t21[1], dtype=float), "aligned"


def blend_frame(master, slave, R21, weights, min_cos):
    """FK rebuild with Master lengths; weights = {child: Slave weight}. Returns (pose, rejected)."""
    pose = {k: v.copy() for k, v in master.items()}
    rejected = 0
    for child, parent in _ordered_bones(master):
        vm = master[child] - master[parent]
        um = _unit(vm)
        us = _unit(R21 @ (slave[child] - slave[parent])) if child in slave and parent in slave else None
        if um is None or us is None or float(um @ us) < min_cos:
            rejected += um is not None and us is not None
            local = vm
        else:
            w = weights.get(child, 0.0)
            v = _unit((1.0 - w) * um + w * us)
            local = float(np.linalg.norm(vm)) * (um if v is None else v)
        pose[child] = pose[parent] + local
    return pose, rejected


def fuse_adaptive_naive(master_frames, slave_frames, config=None, conf_master=None, conf_slave=None,
                        vis_master=None, vis_slave=None):
    """Return (camera1 frames as lists, diagnostics).

    master_frames/slave_frames: per-frame {joint: xyz} in each camera's frame.
    conf_*: per-frame {joint: [x, y, score]} (or None); vis_*: per-frame {joint: bool} (or None).
    config: fusion.adaptive_naive merged with the AOTR alignment keys and
    ``sequence_stability`` (AOTR's block).
    """
    cfg = {**DEFAULTS, **(config or {})}
    n = len(master_frames)
    if n == 0 or len(slave_frames) != n:
        raise ValueError("adaptive_naive fusion needs equally long, non-empty Master/Slave sequences")
    power, tau = float(cfg["confidence_power"]), float(cfg["jitter_scale"])
    if power < 0 or tau <= 0:
        raise ValueError(f"adaptive_naive needs confidence_power >= 0 and jitter_scale > 0, got {power}, {tau}")
    half = int(cfg["jitter_half_window"])
    missing = float(cfg["missing_confidence"])
    conf_master = conf_master or [None] * n
    conf_slave = conf_slave or [None] * n
    vis_master = vis_master or [None] * n
    vis_slave = vis_slave or [None] * n

    M = [_as_pose(f) for f in master_frames]
    S = [_as_pose(f) for f in slave_frames]
    bones = _ordered_bones(M[0])
    jm, js = bone_jitter(M, bones, half), bone_jitter(S, bones, half)
    scale = {}
    for child, _ in bones:
        pooled = np.concatenate([jm[child], js[child]])
        pooled = pooled[np.isfinite(pooled)]
        med = float(np.median(pooled)) if pooled.size else 0.0
        scale[child] = med if med > 1e-12 else None

    def norm_jitter(series, child, t):
        v = series[child][t]
        return 0.0 if scale[child] is None or not np.isfinite(v) else float(v / scale[child])

    fused, editable, reasons = [], [], Counter()
    weights_used, rejected = [], 0
    for t in range(n):
        R21, reason = align(M[t], S[t], cfg, vis_master[t], vis_slave[t])
        reasons[reason] += 1
        if R21 is None:
            fused.append({k: v.copy() for k, v in M[t].items()})
            editable.append(False)
            continue
        weights = {}
        for child, parent in bones:
            cm = bone_confidence(conf_master[t], child, parent, missing)
            cs = bone_confidence(conf_slave[t], child, parent, missing)
            weights[child] = slave_weights(cm, cs, norm_jitter(jm, child, t), norm_jitter(js, child, t), power, tau)
        pose, rej = blend_frame(M[t], S[t], R21, weights, float(cfg["min_bone_direction_cos"]))
        rejected += rej
        weights_used.extend(weights.values())
        fused.append(pose)
        editable.append(True)

    as_lists = [{k: v.tolist() for k, v in p.items()} for p in fused]
    w = np.asarray(weights_used) if weights_used else np.array([np.nan])
    diag = {
        "mode": "adaptive_naive", "frames": n,
        "aligned_frames": int(sum(editable)), "fallback_frames": int(n - sum(editable)),
        "reason_counts": dict(reasons),
        "confidence_power": power, "jitter_scale": tau, "jitter_half_window": half,
        "missing_confidence": missing,
        "slave_weight_mean": float(np.nanmean(w)), "slave_weight_p10": float(np.nanpercentile(w, 10)),
        "slave_weight_p90": float(np.nanpercentile(w, 90)),
        "rejected_bone_frames": int(rejected),
        "frames_without_2d_master": int(sum(not isinstance(c, dict) for c in conf_master)),
        "frames_without_2d_slave": int(sum(not isinstance(c, dict) for c in conf_slave)),
        "sequence_stability": {"status": "disabled"},
    }
    stab_cfg = dict(cfg.get("sequence_stability") or {})
    if stab_cfg.get("enabled", False):
        from fusion_pipeline.aotr_stability import postprocess_aotr_3d
        master_lists = [{k: v.tolist() for k, v in p.items()} for p in M]
        as_lists, stab_diag = postprocess_aotr_3d(master_lists, as_lists, config=stab_cfg, editable_mask=editable)
        diag["sequence_stability"] = stab_diag
    return as_lists, diag

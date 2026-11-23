"""Naive two-view fusion baseline (fusion.method=naive).

The simplest calibration-free late fusion, used as the lower bar that any proposed
fusion has to beat. Per frame, with no learning, no temporal model and no GT:

    1. centre both skeletons at their pelvis;
    2. rotate the Slave onto the Master with one rigid Kabsch rotation over all shared
       joints (no scale: both are metric WHAM skeletons);
    3. average joint positions: F = pelvis_M + w * (M - pelvis_M) + (1 - w) * R (S - pelvis_S).

w is a fixed Master weight (default 0.5 = plain average). Bone lengths are NOT preserved
(averaging positions shortens bent limbs), which is part of what the baseline shows.
Pelvis stays at the Master position, so root-centred metrics are comparable. Only
camera1 (Master) is modified.

With fusion.naive.use_aotr_stability=true the executor passes AOTR's sequence-stability
block, and the result goes through the same pass as AOTR (savgol + FK projection onto
the Master bone lengths, pelvis and neck at the Master). This removes the smoothing
advantage from comparisons with AOTR; bone lengths then equal the Master's.
"""
import numpy as np


def kabsch_rotation(src, dst):
    """Rotation R (3x3, det +1) minimising sum ||R @ src_i - dst_i||^2; rows are points."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ D @ U.T


def fuse_frame(master, slave, master_weight=0.5):
    """Fuse one frame. master/slave: {joint: xyz}. Returns {joint: np.ndarray} for Master joints."""
    M = {k: np.asarray(v, dtype=float) for k, v in master.items()}
    S = {k: np.asarray(v, dtype=float) for k, v in slave.items()}
    if "pelvis" not in M or "pelvis" not in S:
        raise ValueError("naive fusion requires a 'pelvis' joint in both views")
    shared = [k for k in M if k in S]
    if len(shared) < 3:
        raise ValueError(f"naive fusion needs >= 3 shared joints, got {len(shared)}")
    cm, cs = M["pelvis"], S["pelvis"]
    src = np.stack([S[k] - cs for k in shared])
    dst = np.stack([M[k] - cm for k in shared])
    R = kabsch_rotation(src, dst)
    w = float(master_weight)
    out = {}
    for k, m in M.items():
        if k in S:
            out[k] = cm + w * (m - cm) + (1.0 - w) * (R @ (S[k] - cs))
        else:
            out[k] = m.copy()
    out["pelvis"] = cm.copy()
    return out


def fuse_naive(master_frames, slave_frames, config=None):
    """Return (camera1 frames as lists, diagnostics). Inputs: per-frame {joint: xyz}."""
    config = config or {}
    if len(master_frames) != len(slave_frames) or not master_frames:
        raise ValueError("naive fusion needs equally long, non-empty Master/Slave sequences")
    w = float(config.get("master_weight", 0.5))
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"fusion.naive.master_weight must be in [0, 1], got {w}")
    fused = [fuse_frame(m, s, w) for m, s in zip(master_frames, slave_frames)]
    diag = {"mode": "naive", "frames": len(fused), "master_weight": w,
            "sequence_stability": {"status": "disabled"}}
    as_lists = [{k: v.tolist() for k, v in p.items()} for p in fused]
    stab_cfg = dict(config.get("sequence_stability") or {})
    if stab_cfg.get("enabled", False):
        from fusion_pipeline.aotr_stability import postprocess_aotr_3d
        master_lists = [{k: [float(x) for x in v] for k, v in m.items()} for m in master_frames]
        as_lists, stab_diag = postprocess_aotr_3d(
            master_lists, as_lists, config=stab_cfg, editable_mask=[True] * len(as_lists))
        diag["sequence_stability"] = stab_diag
    return as_lists, diag

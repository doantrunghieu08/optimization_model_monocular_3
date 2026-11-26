"""Part-wise symmetric two-view fusion (calibration-free, GT-free).

For every bone, the Master and Slave directions are blended and the skeleton is rebuilt
by forward kinematics with the Master bone lengths (so MBLE equals the Master/baseline):

    u = normalize(w * u_M + (1 - w) * R_part @ u_S)

* R_part maps the Slave's torso part onto the Master's, with no extrinsics:
  R_part = B_M @ B_S^T, where B is an orthonormal body frame built from
  - lower part: hip axis + spine (pelvis->neck), used for hips/legs/feet;
  - upper part: shoulder axis + spine, used for head/shoulders/arms/hands.
  A single whole-body rotation twists the upper body when one view is poor; anchoring
  each limb on the torso part it hangs from avoids that. R_part is smoothed over time.
* w is the Master weight per frame: a logistic function of the root-depth difference
  (depth_M - depth_S, camera frame). The farther view tends to be the more accurate
  one on the design split (frame AUC 0.825; 0.863 on held-out). The logistic parameters
  were fitted once on the design split and are frozen in the config.
* A sequence-stability pass (fusion_pipeline.aotr_stability) then shrinks the
  correction residual F - M where it would add acceleration.

Pelvis and neck stay at the Master positions. Only camera1 (Master) is modified.
"""
import json
from pathlib import Path
import numpy as np

from fusion_pipeline.adaptive_naive import bone_confidence, bone_jitter
from fusion_pipeline.aotr import _ordered_bones
from fusion_pipeline.detector import compute_bone_distortion_score

LOWER_PART = frozenset({
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_foot", "right_foot", "left_toe", "right_toe",
})
REQUIRED = ("pelvis", "neck", "left_hip", "right_hip", "left_shoulder", "right_shoulder")


def _as_pose(frame):
    return {k: np.asarray(v, dtype=float) for k, v in frame.items()}


def _orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def part_frame(pose, left, right):
    """Orthonormal frame (columns x, y, z): x = right->left joint, y = spine, z = x × y."""
    x = pose[left] - pose[right]
    x = x / np.linalg.norm(x)
    y = pose["neck"] - pose["pelvis"]
    y = y - x * (x @ y)
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def root_depth(pose):
    return float((0.5 * (pose["left_hip"] + pose["right_hip"]))[2])


LOWER_TORSO_JOINTS = ("pelvis", "neck", "left_hip", "right_hip")
UPPER_TORSO_JOINTS = ("pelvis", "neck", "left_shoulder", "right_shoulder")


def _depth_logits(master, slave, logit):
    d = np.array([root_depth(m) - root_depth(s) for m, s in zip(master, slave)], dtype=float)
    d = np.nan_to_num(d)
    return float(logit["a"]) + float(logit["b"]) * (d - float(logit["mu"])) / float(logit["sd"])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def master_weights(master, slave, logit):
    """Per-frame Master weight from the root-depth difference (farther view favoured)."""
    return _sigmoid(_depth_logits(master, slave, logit))


def _part_vis_weight(vis_m, vis_s, joints):
    if not vis_m and not vis_s:
        return 1.0
    vm = vis_m or {}
    vs = vis_s or {}
    both = sum(bool(vm.get(j, True)) and bool(vs.get(j, True)) for j in joints)
    return float(both) / float(len(joints))


PRIMARY_LIMB_JOINTS = frozenset({
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
})


def _median_bone_disagreement_deg(master_pose, slave_dirs, vis_m=None, vis_s=None):
    """Zero-GT cross-view pose disagreement: median angle (deg) across primary limb bones."""
    vm, vs = vis_m or {}, vis_s or {}
    vis_angles, all_angles = [], []
    for child, parent in _ordered_bones(master_pose):
        if child not in PRIMARY_LIMB_JOINTS:
            continue
        us = slave_dirs.get(child)
        if us is None:
            continue
        vec = master_pose[child] - master_pose[parent]
        n = float(np.linalg.norm(vec))
        if n <= 1e-9:
            continue
        ang = float(np.degrees(np.arccos(np.clip(float((vec / n) @ us), -1.0, 1.0))))
        all_angles.append(ang)
        if vm.get(child, True) and vs.get(child, True):
            vis_angles.append(ang)
    sample = vis_angles if len(vis_angles) >= 3 else all_angles
    return float(np.median(sample)) if sample else 0.0


def _frame_bone_jump_deg(prev_pose, curr_pose):
    if prev_pose is None:
        return 0.0
    angles = []
    for child, parent in _ordered_bones(curr_pose):
        v0 = prev_pose[child] - prev_pose[parent]
        v1 = curr_pose[child] - curr_pose[parent]
        n0, n1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
        if n0 > 1e-9 and n1 > 1e-9:
            angles.append(float(np.degrees(np.arccos(np.clip(float((v0 / n0) @ (v1 / n1)), -1.0, 1.0)))))
    return float(np.median(angles)) if angles else 0.0


def slave_directions(master, slave, half_window=4, vis_master=None, vis_slave=None, frame_weights=None):
    """Slave bone unit vectors in the Master frame, one dict per frame."""
    n_frames = len(master)
    rot = {"lower": [], "upper": []}
    part_w = {"lower": np.ones(n_frames, dtype=float), "upper": np.ones(n_frames, dtype=float)}
    for t, (m, s) in enumerate(zip(master, slave)):
        rot["lower"].append(part_frame(m, "left_hip", "right_hip") @ part_frame(s, "left_hip", "right_hip").T)
        rot["upper"].append(part_frame(m, "left_shoulder", "right_shoulder")
                            @ part_frame(s, "left_shoulder", "right_shoulder").T)
        base_fw = 1.0 if frame_weights is None else float(frame_weights[t])
        vm = vis_master[t] if vis_master is not None else None
        vs = vis_slave[t] if vis_slave is not None else None
        part_w["lower"][t] = base_fw * _part_vis_weight(vm, vs, LOWER_TORSO_JOINTS)
        part_w["upper"][t] = base_fw * _part_vis_weight(vm, vs, UPPER_TORSO_JOINTS)

    # ponytail: precompute nearest clean frame fallback when an entire local window is occluded
    clean_idx = {k: np.where(part_w[k] > 0.5)[0] for k in ("lower", "upper")}
    bones = _ordered_bones(master[0])
    out = []
    for t, s in enumerate(slave):
        lo, hi = max(0, t - half_window), min(n_frames, t + half_window + 1)
        R = {}
        for k in ("lower", "upper"):
            weights = part_w[k][lo:hi]
            w_sum = float(np.sum(weights))
            if w_sum > 1e-6:
                R_avg = sum(w_i * R_i for w_i, R_i in zip(weights, rot[k][lo:hi])) / w_sum
                R[k] = _orthonormalize(R_avg)
            elif len(clean_idx[k]) > 0:
                nearest = int(clean_idx[k][np.argmin(np.abs(clean_idx[k] - t))])
                R[k] = _orthonormalize(rot[k][nearest])
            else:
                R[k] = _orthonormalize(np.mean(rot[k][lo:hi], axis=0))
        dirs = {}
        for child, parent in bones:
            v = s[child] - s[parent]
            n = np.linalg.norm(v)
            if n > 1e-9:
                dirs[child] = R["lower" if child in LOWER_PART else "upper"] @ (v / n)
        out.append(dirs)
    return out


def blend_frame(master_pose, slave_dirs, w, bone_weights=None):
    """Blend bone directions (w = Master weight, or per-bone dict) and rebuild with Master bone lengths."""
    out = {k: v.copy() for k, v in master_pose.items()}
    for child, parent in _ordered_bones(master_pose):
        um = master_pose[child] - master_pose[parent]
        length = np.linalg.norm(um)
        if length < 1e-9:
            out[child] = out[parent] + (master_pose[child] - master_pose[parent])
            continue
        um = um / length
        us = slave_dirs.get(child)
        wb = float(bone_weights[child]) if (bone_weights is not None and child in bone_weights) else float(w)
        u = um if us is None else wb * um + (1.0 - wb) * us
        nu = np.linalg.norm(u)
        u = um if nu < 1e-9 else u / nu
        out[child] = out[parent] + length * u
    return out


def _geodesic_deg(R1, R2):
    tr = float(np.trace(R1.T @ R2))
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


def _torso_frame(pose):
    return _orthonormalize(0.5 * (
        part_frame(pose, "left_hip", "right_hip")
        + part_frame(pose, "left_shoulder", "right_shoulder")
    ))


def estimate_static_camera_rotation(master, slave, vis_master=None, vis_slave=None,
                                    frame_valid=None, inlier_deg=25.0):
    """Zero-GT robust SO(3) medoid + inlier mean for ONE static camera rotation R_cam (Slave -> Master)."""
    cands, weights = [], []
    for t, (m, s) in enumerate(zip(master, slave)):
        if frame_valid is not None and frame_valid[t] <= 0.0:
            continue
        vm = vis_master[t] if vis_master is not None else None
        vs = vis_slave[t] if vis_slave is not None else None
        wl = _part_vis_weight(vm, vs, LOWER_TORSO_JOINTS)
        wu = _part_vis_weight(vm, vs, UPPER_TORSO_JOINTS)
        if wl >= 0.5:
            cands.append(part_frame(m, "left_hip", "right_hip") @ part_frame(s, "left_hip", "right_hip").T)
            weights.append(wl)
        if wu >= 0.5:
            cands.append(part_frame(m, "left_shoulder", "right_shoulder") @ part_frame(s, "left_shoulder", "right_shoulder").T)
            weights.append(wu)
    if not cands:
        for m, s in zip(master, slave):
            cands.append(_torso_frame(m) @ _torso_frame(s).T)
            weights.append(1.0)
    step = max(1, len(cands) // 60)
    best_R, best_med = cands[0], float("inf")
    for idx in range(0, len(cands), step):
        R0 = cands[idx]
        med = float(np.median([_geodesic_deg(R0, R) for R in cands]))
        if med < best_med:
            best_med, best_R = med, R0
    inliers = [(w, R) for w, R in zip(weights, cands) if _geodesic_deg(best_R, R) <= float(inlier_deg)]
    if not inliers:
        return _orthonormalize(best_R), best_med
    w_sum = sum(w for w, _ in inliers)
    return _orthonormalize(sum(w * R for w, R in inliers) / max(w_sum, 1e-9)), best_med


def _apply_static_global_orientation(fused_lists, M, S, z_depth, vis_master=None, vis_slave=None,
                                     frame_valid=None, smooth_hw=2):
    """Decoupled global-orientation fusion for static cameras.

    Applies ONE rigid rotation dR(t) per frame about pelvis so PA-MPJPE and MBLE are
    mathematically invariant while sequence-aligned MPJPE gains from static R_cam.
    """
    R_cam, med_deg = estimate_static_camera_rotation(M, S, vis_master, vis_slave, frame_valid=frame_valid)
    # ponytail: fail-closed when cross-view R_cam is noisy (e.g. broken Master on MMA/grappling)
    if med_deg > 10.0:
        return fused_lists, med_deg
    bm = [_torso_frame(m) for m in M]
    bs = [R_cam @ _torso_frame(s) for s in S]
    torso_j = ("pelvis", "neck", "left_hip", "right_hip", "left_shoulder", "right_shoulder")

    targets = []
    for t, zt in enumerate(z_depth):
        vm = vis_master[t] if vis_master is not None else {}
        vs = vis_slave[t] if vis_slave is not None else {}
        occ_m = sum(not vm.get(j, True) for j in torso_j)
        occ_s = sum(not vs.get(j, True) for j in torso_j)
        zt_eff = float(zt) + 0.6 * (occ_s - occ_m)
        ang = _geodesic_deg(bm[t], bs[t])
        if ang > 45.0:
            jm = _geodesic_deg(bm[t - 1], bm[t]) if t > 0 else 0.0
            js = _geodesic_deg(bs[t - 1], bs[t]) if t > 0 else 0.0
            if jm > max(15.0, 2.0 * js):
                zt_eff = min(zt_eff, 0.0) - 3.0
            elif js > max(15.0, 2.0 * jm):
                zt_eff = max(zt_eff, 0.0) + 3.0
        w_ori = float(_sigmoid(zt_eff))
        if ang > 140.0:
            B_tgt = bm[t] if w_ori >= 0.5 else bs[t]
        else:
            B_tgt = _orthonormalize(w_ori * bm[t] + (1.0 - w_ori) * bs[t])
        targets.append(B_tgt)

    # Smooth target orientation locally (skip averaging across >45 deg discontinuities)
    n = len(targets)
    out = []
    for t, pose_dict in enumerate(fused_lists):
        if smooth_hw > 0 and n > 2:
            lo, hi = max(0, t - smooth_hw), min(n, t + smooth_hw + 1)
            local = [targets[k] for k in range(lo, hi) if _geodesic_deg(targets[t], targets[k]) <= 45.0]
            B_smooth = _orthonormalize(np.mean(local, axis=0)) if local else targets[t]
        else:
            B_smooth = targets[t]
        pose_arr = {k: np.asarray(v, dtype=float) for k, v in pose_dict.items()}
        # Rotate from the fused skeleton's ACTUAL torso frame to B_smooth (avoids double-rotating
        # when windowed R_part already shifted hips/shoulders toward Slave).
        dR = B_smooth @ _torso_frame(pose_arr).T
        root = pose_arr["pelvis"]
        out.append({k: (root + dR @ (v - root)).tolist() for k, v in pose_arr.items()})
    return out, med_deg


FEATURE_NAMES = (
    "d_depth", "d_occ_joint", "d_occ_torso", "d_conf2d",
    "d_bone_jit", "d_cam_accel", "d_bone_distort", "d_facing",
)


def _facing_abs_cos(pose):
    """|cos(angle)| between torso normal and camera optical ray -> 1 = frontal, 0 = side profile."""
    x = pose["left_shoulder"] - pose["right_shoulder"]
    y = pose["neck"] - pose["pelvis"]
    normal = np.cross(x, y)
    nn = float(np.linalg.norm(normal))
    ray = 0.5 * (pose["left_hip"] + pose["right_hip"])
    rn = float(np.linalg.norm(ray))
    if nn <= 1e-9 or rn <= 1e-9:
        return 0.5
    return float(abs((normal / nn) @ (ray / rn)))


def _seq_median_accel_mm(poses):
    if len(poses) < 3:
        return 0.0
    keys = sorted(poses[0])
    arr = np.array([[p[k] for k in keys] for p in poses], dtype=float)
    acc = np.linalg.norm(arr[2:] - 2.0 * arr[1:-1] + arr[:-2], axis=-1)
    return float(np.median(acc[5:] if len(acc) > 8 else acc) * 1000.0)


def _depth_outlier_flags(z_m_arr, z_s_arr, ratio=1.5, min_m=10.0):
    # ponytail: shared bbox-collapse depth explosion guard (z >= 10m & >1.5x other/median)
    z_m_med = float(np.median(z_m_arr[np.isfinite(z_m_arr)])) if np.any(np.isfinite(z_m_arr)) else 5.0
    z_s_med = float(np.median(z_s_arr[np.isfinite(z_s_arr)])) if np.any(np.isfinite(z_s_arr)) else 5.0
    s_exp = (z_s_arr >= min_m) & ((z_s_arr > ratio * np.maximum(z_m_arr, 1e-3)) | (z_s_arr > ratio * max(z_s_med, 1e-3)))
    m_exp = (z_m_arr >= min_m) & ((z_m_arr > ratio * np.maximum(z_s_arr, 1e-3)) | (z_m_arr > ratio * max(z_m_med, 1e-3)))
    return m_exp & ~s_exp, s_exp & ~m_exp


def extract_pair_features(M, S, vis_master=None, vis_slave=None, conf_master=None, conf_slave=None):
    """Return (features[T, B, 8], bone_children) with antisymmetric features phi(M, S) = -phi(S, M)."""
    n = len(M)
    bones = _ordered_bones(M[0])
    b_children = [c for c, _ in bones]
    z_m_arr = np.array([root_depth(m) for m in M], dtype=float)
    z_s_arr = np.array([root_depth(s) for s in S], dtype=float)
    d_depth = np.clip(np.nan_to_num(z_m_arr - z_s_arr), -3.0, 3.0)
    m_exp, s_exp = _depth_outlier_flags(z_m_arr, z_s_arr)
    d_depth = np.where(s_exp, 3.0, np.where(m_exp, -3.0, d_depth))
    jm, js = bone_jitter(M, bones, 3), bone_jitter(S, bones, 3)
    j_med = {}
    for c in b_children:
        pooled = np.concatenate([jm[c], js[c]])
        pooled = pooled[np.isfinite(pooled)]
        j_med[c] = float(np.median(pooled)) if pooled.size and float(np.median(pooled)) > 1e-12 else 1.0
    acc_m, acc_s = _seq_median_accel_mm(M), _seq_median_accel_mm(S)
    d_cam_acc = float(np.clip(np.log((acc_s + 1.0) / (acc_m + 1.0)), -3.0, 3.0))

    feats = np.zeros((n, len(bones), len(FEATURE_NAMES)), dtype=np.float32)
    for t in range(n):
        vm = (vis_master[t] if vis_master else None) or {}
        vs = (vis_slave[t] if vis_slave else None) or {}
        cm_t = conf_master[t] if conf_master else None
        cs_t = conf_slave[t] if conf_slave else None
        occ_m = float(sum(not vm.get(j, True) for j in M[t])) / max(len(M[t]), 1)
        occ_s = float(sum(not vs.get(j, True) for j in S[t])) / max(len(S[t]), 1)
        d_occ_torso = occ_s - occ_m
        d_distort = float(np.clip(compute_bone_distortion_score(S[t]) - compute_bone_distortion_score(M[t]), -1.0, 1.0))
        d_face = _facing_abs_cos(M[t]) - _facing_abs_cos(S[t])
        for bi, (child, parent) in enumerate(bones):
            d_occ_j = float(bool(vm.get(child, True))) - float(bool(vs.get(child, True)))
            c_m = bone_confidence(cm_t, child, parent, 0.5)
            c_s = bone_confidence(cs_t, child, parent, 0.5)
            d_c2d = float(np.log((c_m + 0.05) / (c_s + 0.05)))
            nj_m = float(jm[child][t] / j_med[child]) if np.isfinite(jm[child][t]) else 0.0
            nj_s = float(js[child][t] / j_med[child]) if np.isfinite(js[child][t]) else 0.0
            d_jit = float(np.clip(np.log((nj_s + 0.1) / (nj_m + 0.1)), -3.0, 3.0))
            feats[t, bi] = (d_depth[t], d_occ_j, d_occ_torso, d_c2d, d_jit, d_cam_acc, d_distort, d_face)
    return feats, b_children


def eval_weight_model(feats, wm_cfg):
    """Evaluate learned antisymmetric weight model -> w_Master[T, B] in (0, 1)."""
    if isinstance(wm_cfg.get("path"), str) and wm_cfg["path"].strip():
        p = Path(wm_cfg["path"])
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[1] / p
        wm_cfg = {**json.loads(p.read_text(encoding="utf-8")), **wm_cfg}
    mean = np.asarray(wm_cfg.get("mean", [0.0] * feats.shape[-1]), dtype=float)
    std = np.maximum(np.asarray(wm_cfg.get("std", [1.0] * feats.shape[-1]), dtype=float), 1e-6)
    # Center-only or symmetric scale: to preserve phi(M,S) = -phi(S,M), mean must be 0!
    x = (feats - mean) / std
    x = np.where(np.asarray(wm_cfg.get("active_features", [True] * feats.shape[-1])), x, 0.0)
    kind = str(wm_cfg.get("type", "linear")).lower()
    if kind == "sym_mlp":
        W1 = np.asarray(wm_cfg["W1"], dtype=float)
        b1 = np.asarray(wm_cfg["b1"], dtype=float)
        W2 = np.asarray(wm_cfg["W2"], dtype=float).reshape(-1)
        logits = 0.5 * (np.tanh(x @ W1 + b1) @ W2 - np.tanh((-x) @ W1 + b1) @ W2)
    else:
        w_vec = np.asarray(wm_cfg["weights"], dtype=float).reshape(-1)
        logits = x @ w_vec
    return _sigmoid(logits)


def fuse_partwise(master_frames, slave_frames, config, vis_master=None, vis_slave=None,
                  conf_master=None, conf_slave=None):
    """Return (camera1 frames as lists, diagnostics). Inputs: per-frame {joint: xyz}."""
    if len(master_frames) != len(slave_frames) or not master_frames:
        raise ValueError("partwise fusion needs equally long, non-empty Master/Slave sequences")
    M = [_as_pose(f) for f in master_frames]
    S = [_as_pose(f) for f in slave_frames]
    missing = [j for j in REQUIRED if j not in M[0] or j not in S[0]]
    if missing:
        raise ValueError(f"partwise fusion requires joints {missing}")
    z_m_arr = np.array([root_depth(m) for m in M], dtype=float)
    z_s_arr = np.array([root_depth(s) for s in S], dtype=float)
    z_depth = _depth_logits(M, S, config["depth_logit"])
    w = _sigmoid(z_depth)
    hw = int(config.get("rotation_half_window", 4))
    mp_cfg = dict(config.get("multi_person", {}) or {})
    mp_enabled = bool(mp_cfg.get("enabled", False))

    swap_gated = 0
    depth_outlier_frames = 0
    occ_master_bones = 0
    occ_slave_bones = 0
    disagreements = []

    wm_cfg = config.get("weight_model")
    use_wm = isinstance(wm_cfg, dict) and bool(wm_cfg.get("enabled", True))
    w_mat = None
    if use_wm:
        feats, b_children = extract_pair_features(M, S, vis_master, vis_slave, conf_master, conf_slave)
        w_mat = eval_weight_model(feats, wm_cfg)
    # ponytail: always guard against occlusion-induced bbox-collapse depth explosion (not gated on mp_enabled)
    depth_outlier_ratio = float(mp_cfg.get("depth_outlier_ratio", 1.5))
    depth_outlier_min_m = float(mp_cfg.get("depth_outlier_min_m", 10.0))
    depth_outlier_bias = float(mp_cfg.get("depth_outlier_logit_bias", 3.0))
    if mp_enabled:
        max_depth_diff = float(mp_cfg.get("max_depth_diff_m", 3.0))
        d_clamped = np.clip(np.nan_to_num(z_m_arr - z_s_arr), -max_depth_diff, max_depth_diff)
        logit = config["depth_logit"]
        z_depth = float(logit["a"]) + float(logit["b"]) * (d_clamped - float(logit["mu"])) / float(logit["sd"])
    m_exp, s_exp = _depth_outlier_flags(z_m_arr, z_s_arr, depth_outlier_ratio, depth_outlier_min_m)
    for t in range(len(M)):
        if s_exp[t]:
            z_depth[t] = max(z_depth[t], 0.0) + depth_outlier_bias
            depth_outlier_frames += 1
            if use_wm:
                w_mat[t, :] = _sigmoid(z_depth[t])
        elif m_exp[t]:
            z_depth[t] = min(z_depth[t], 0.0) - depth_outlier_bias
            depth_outlier_frames += 1
            if use_wm:
                w_mat[t, :] = _sigmoid(z_depth[t])
    w = np.mean(w_mat, axis=1) if use_wm else _sigmoid(z_depth)
    if mp_enabled:
        use_vis_rot = bool(mp_cfg.get("visibility_weighted_rotation", True))
        dirs_init = slave_directions(
            M, S, hw,
            vis_master=vis_master if use_vis_rot else None,
            vis_slave=vis_slave if use_vis_rot else None,
        )
        anti_swap = bool(mp_cfg.get("anti_swap_gate", True))
        max_ang = float(mp_cfg.get("max_median_bone_angle_deg", 50.0))
        frame_valid = np.ones(len(M), dtype=float)
        frame_gate_keep = [False] * len(M)
        for t in range(len(M)):
            vm = vis_master[t] if vis_master is not None else {}
            vs = vis_slave[t] if vis_slave is not None else {}
            ang = _median_bone_disagreement_deg(M[t], dirs_init[t], vm, vs)
            disagreements.append(ang)
            if anti_swap and ang > max_ang:
                frame_valid[t] = 0.0
                jump_m = _frame_bone_jump_deg(M[t - 1] if t > 0 else None, M[t])
                jump_s = _frame_bone_jump_deg(S[t - 1] if t > 0 else None, S[t])
                occ_m_ratio = float(sum(not vm.get(j, True) for j in M[t])) / max(len(M[t]), 1)
                # Keep Master untouched unless Master itself clearly broke/jumped under heavy occlusion
                if not (occ_m_ratio >= 0.35 and jump_m > max(15.0, 2.0 * jump_s)):
                    frame_gate_keep[t] = True
                    swap_gated += 1
        # Recompute R_part excluding suspected swap frames so they cannot contaminate neighbors
        dirs = slave_directions(
            M, S, hw,
            vis_master=vis_master if use_vis_rot else None,
            vis_slave=vis_slave if use_vis_rot else None,
            frame_weights=frame_valid if anti_swap else None,
        ) if anti_swap and np.any(frame_valid == 0.0) else dirs_init

        m_occ_bias = float(mp_cfg.get("master_occluded_logit_bias", 2.0))
        s_occ_bias = float(mp_cfg.get("slave_occluded_logit_bias", 2.5))
        both_occ_bias = float(mp_cfg.get("both_occluded_logit_bias", 0.5))
        bones = _ordered_bones(M[0])
        fused = []
        for t, (m, d, zt, wt) in enumerate(zip(M, dirs, z_depth, w)):
            if frame_gate_keep[t]:
                fused.append({k: v.copy() for k, v in m.items()})
                continue
            vm = vis_master[t] if vis_master is not None else None
            vs = vis_slave[t] if vis_slave is not None else None
            if vm is None and vs is None and not use_wm:
                fused.append(blend_frame(m, d, wt))
                continue
            vm, vs = vm or {}, vs or {}
            bw = {}
            for bi, (child, _) in enumerate(bones):
                base_w = float(w_mat[t, bi]) if use_wm else float(wt)
                z_bone = float(np.log(np.clip(base_w, 1e-9, 1 - 1e-9) /
                                      np.clip(1 - base_w, 1e-9, 1))) if use_wm else float(zt)
                m_vis = bool(vm.get(child, True))
                s_vis = bool(vs.get(child, True))
                if not m_vis and s_vis:
                    # Occluded Master + visible Slave: never let depth prior override visibility
                    bw[child] = float(_sigmoid(min(z_bone, 0.0) - m_occ_bias))
                    occ_master_bones += 1
                elif m_vis and not s_vis:
                    # Visible Master + occluded Slave: protect Master regardless of depth
                    bw[child] = float(_sigmoid(max(z_bone, 0.0) + s_occ_bias))
                    occ_slave_bones += 1
                elif not m_vis and not s_vis:
                    bw[child] = float(_sigmoid(z_bone + both_occ_bias))
                else:
                    bw[child] = base_w
            fused.append(blend_frame(m, d, wt, bone_weights=bw))
    elif use_wm:
        dirs = slave_directions(M, S, hw)
        w = np.mean(w_mat, axis=1)
        fused = [blend_frame(m, d, w[t], bone_weights=dict(zip(b_children, w_mat[t])))
                 for t, (m, d) in enumerate(zip(M, dirs))]
    else:
        dirs = slave_directions(M, S, hw)
        fused = [blend_frame(m, d, wt) for m, d, wt in zip(M, dirs, w)]

    diag = {
        "mode": "partwise",
        "frames": len(fused),
        "master_weight_mean": float(np.mean(w)),
        "master_weight_min": float(np.min(w)),
        "master_weight_max": float(np.max(w)),
        "weight_model_enabled": use_wm,
        "multi_person_enabled": mp_enabled,
        "swap_gated_frames": swap_gated,
        "depth_outlier_frames": depth_outlier_frames,
        "occluded_master_bones": occ_master_bones,
        "occluded_slave_bones": occ_slave_bones,
        "median_bone_angle_deg_mean": float(np.mean(disagreements)) if disagreements else None,
        "sequence_stability": {"status": "disabled"},
    }
    as_lists = [{k: v.tolist() for k, v in p.items()} for p in fused]
    stab_cfg = dict(config.get("sequence_stability", {}))
    if stab_cfg.get("enabled", True):
        from fusion_pipeline.aotr_stability import postprocess_aotr_3d
        master_lists = [{k: v.tolist() for k, v in p.items()} for p in M]
        as_lists, stab_diag = postprocess_aotr_3d(
            master_lists, as_lists, config=stab_cfg, editable_mask=[True] * len(as_lists))
        diag["sequence_stability"] = stab_diag
    if mp_enabled and bool(mp_cfg.get("static_camera_orientation", False)):
        as_lists, rcam_med = _apply_static_global_orientation(
            as_lists, M, S, z_depth,
            vis_master=vis_master, vis_slave=vis_slave,
            frame_valid=frame_valid if anti_swap else None,
            smooth_hw=int(mp_cfg.get("orientation_smooth_half_window", 2)),
        )
        diag["static_camera_orientation"] = {"enabled": True, "r_cam_median_deg": round(rcam_med, 2)}
    return as_lists, diag

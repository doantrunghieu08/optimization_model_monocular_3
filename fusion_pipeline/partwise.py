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
import numpy as np

from fusion_pipeline.aotr import _ordered_bones

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


def master_weights(master, slave, logit):
    """Per-frame Master weight from the root-depth difference (farther view favoured)."""
    d = np.array([root_depth(m) - root_depth(s) for m, s in zip(master, slave)])
    d = np.nan_to_num(d)
    z = float(logit["a"]) + float(logit["b"]) * (d - float(logit["mu"])) / float(logit["sd"])
    return 1.0 / (1.0 + np.exp(-z))


def slave_directions(master, slave, half_window=4):
    """Slave bone unit vectors in the Master frame, one dict per frame."""
    rot = {"lower": [], "upper": []}
    for m, s in zip(master, slave):
        rot["lower"].append(part_frame(m, "left_hip", "right_hip") @ part_frame(s, "left_hip", "right_hip").T)
        rot["upper"].append(part_frame(m, "left_shoulder", "right_shoulder")
                            @ part_frame(s, "left_shoulder", "right_shoulder").T)
    bones = _ordered_bones(master[0])
    out = []
    for t, s in enumerate(slave):
        lo, hi = max(0, t - half_window), t + half_window + 1
        R = {k: _orthonormalize(np.mean(v[lo:hi], axis=0)) for k, v in rot.items()}
        dirs = {}
        for child, parent in bones:
            v = s[child] - s[parent]
            n = np.linalg.norm(v)
            if n > 1e-9:
                dirs[child] = R["lower" if child in LOWER_PART else "upper"] @ (v / n)
        out.append(dirs)
    return out


def blend_frame(master_pose, slave_dirs, w):
    """Blend bone directions (w = Master weight) and rebuild with Master bone lengths."""
    out = {k: v.copy() for k, v in master_pose.items()}
    for child, parent in _ordered_bones(master_pose):
        um = master_pose[child] - master_pose[parent]
        length = np.linalg.norm(um)
        if length < 1e-9:
            out[child] = out[parent] + (master_pose[child] - master_pose[parent])
            continue
        um = um / length
        us = slave_dirs.get(child)
        u = um if us is None else w * um + (1.0 - w) * us
        nu = np.linalg.norm(u)
        u = um if nu < 1e-9 else u / nu
        out[child] = out[parent] + length * u
    return out


def fuse_partwise(master_frames, slave_frames, config):
    """Return (camera1 frames as lists, diagnostics). Inputs: per-frame {joint: xyz}."""
    if len(master_frames) != len(slave_frames) or not master_frames:
        raise ValueError("partwise fusion needs equally long, non-empty Master/Slave sequences")
    M = [_as_pose(f) for f in master_frames]
    S = [_as_pose(f) for f in slave_frames]
    missing = [j for j in REQUIRED if j not in M[0] or j not in S[0]]
    if missing:
        raise ValueError(f"partwise fusion requires joints {missing}")
    w = master_weights(M, S, config["depth_logit"])
    dirs = slave_directions(M, S, int(config.get("rotation_half_window", 4)))
    fused = [blend_frame(m, d, wt) for m, d, wt in zip(M, dirs, w)]
    diag = {"mode": "partwise", "frames": len(fused),
            "master_weight_mean": float(np.mean(w)), "master_weight_min": float(np.min(w)),
            "master_weight_max": float(np.max(w)), "sequence_stability": {"status": "disabled"}}
    as_lists = [{k: v.tolist() for k, v in p.items()} for p in fused]
    stab_cfg = dict(config.get("sequence_stability", {}))
    if stab_cfg.get("enabled", True):
        from fusion_pipeline.aotr_stability import postprocess_aotr_3d
        master_lists = [{k: v.tolist() for k, v in p.items()} for p in M]
        as_lists, stab_diag = postprocess_aotr_3d(
            master_lists, as_lists, config=stab_cfg, editable_mask=[True] * len(as_lists))
        diag["sequence_stability"] = stab_diag
    return as_lists, diag

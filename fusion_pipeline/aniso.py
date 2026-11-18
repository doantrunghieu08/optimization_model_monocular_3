"""Calibration-free anisotropic two-view 3D fusion (``fusion.method: aniso``).

Each camera's monocular estimate is treated as a Gaussian measurement of one shared
root-relative skeleton and the two are fused by maximum likelihood (inverse covariance):

    Sigma_i(j,t) = (a_i^2 (I - d d^T) + b_i^2 d d^T) / w_i(j,t),      d = P_i / ||P_i||
    X*           = (L1 + L2')^-1 (L1 X1 + L2' R_t X2),                L = Sigma^-1

* d (camera ray) needs no intrinsics: joints are already in each camera frame.
* R_t (camera2 -> camera1) is estimated per frame from the body itself (robust weighted
  Kabsch, Cauchy IRLS): the monocular global orientation drifts frame to frame, so a single
  sequence-level rotation does not fit.
* a_i, b_i (lateral / depth noise) are self-calibrated by maximum likelihood on the
  cross-view residuals; w_i is the Stage-1 belief (2D confidence x occlusion factor).
* Optional closed-form 2nd-order temporal smoothing of the absolute trajectories, then the
  same per-frame bone-length projection as the legacy path.

No camera parameters and no annotation data are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solveh_banded

from fusion_pipeline.config import DEFAULT_OCCLUDED_FACTOR

MIN_BELIEF = 0.05
EPS = 1e-9


def kabsch(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Rotation R minimising sum_k w_k ||A_k - R B_k||^2 for (N,3) point sets."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    if w.sum() < EPS:
        w = np.ones_like(w)
    H = (B * w[:, None]).T @ A
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    return Vt.T @ D @ U.T


def robust_rotation(A: np.ndarray, B: np.ndarray, w0: np.ndarray, iters: int = 5) -> np.ndarray:
    """Kabsch with Cauchy IRLS reweighting so mispredicted joints lose influence."""
    w = np.asarray(w0, dtype=float).copy()
    R = np.eye(3)
    for _ in range(max(1, iters)):
        R = kabsch(A, B, w)
        r = np.linalg.norm(A - B @ R.T, axis=1)
        c = 1.4826 * np.median(r) + 1e-6
        w = w0 / (1.0 + (r / c) ** 2)
    return R


def per_frame_rotations(X1, X2, W1, W2, iters: int = 5, window: int = 0) -> np.ndarray:
    """camera2 -> camera1 rotation for every frame (optionally pooled over +-window frames)."""
    T = X1.shape[0]
    Rs = np.empty((T, 3, 3))
    for t in range(T):
        lo, hi = max(0, t - window), min(T, t + window + 1)
        Rs[t] = robust_rotation(
            X1[lo:hi].reshape(-1, 3), X2[lo:hi].reshape(-1, 3),
            np.minimum(W1[lo:hi], W2[lo:hi]).reshape(-1), iters,
        )
    return Rs


def _ray_projectors(D: np.ndarray) -> np.ndarray:
    return D[..., :, None] * D[..., None, :]


def information(D: np.ndarray, W: np.ndarray, a: float, b: float) -> np.ndarray:
    """w * Sigma^-1 with Sigma = a^2 (I - dd^T) + b^2 dd^T (closed-form inverse)."""
    P = _ray_projectors(D)
    return W[..., None, None] * ((np.eye(3) - P) / a ** 2 + P / b ** 2)


def self_calibrate_noise(X1, X2R, D1, D2R, W1, W2, max_residual: float = 0.5, iters: int = 200):
    """ML estimate of (a1, b1, a2, b2) from r = X1 - R X2 ~ N(0, Sigma1/w1 + Sigma2'/w2)."""
    import torch

    r = X1 - X2R
    keep = np.isfinite(r).all(-1) & (np.linalg.norm(r, axis=-1) < max_residual)
    if keep.sum() < 10:
        return 0.03, 0.03, 0.03, 0.03
    t = lambda x: torch.tensor(x[keep], dtype=torch.float64)  # noqa: E731
    r_t, d1, d2, w1, w2 = t(r), t(D1), t(D2R), t(W1), t(W2)
    I = torch.eye(3, dtype=torch.float64)
    P1 = d1[:, :, None] * d1[:, None, :]
    P2 = d2[:, :, None] * d2[:, None, :]
    logp = torch.log(torch.tensor([0.03, 0.06, 0.03, 0.06], dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.LBFGS([logp], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        a1, b1, a2, b2 = torch.exp(logp)
        S = (a1 ** 2 * (I - P1) + b1 ** 2 * P1) / w1[:, None, None] + (a2 ** 2 * (I - P2) + b2 ** 2 * P2) / w2[:, None, None]
        nll = 0.5 * (torch.logdet(S) + (r_t[:, None, :] @ torch.linalg.solve(S, r_t[:, :, None])).squeeze()).mean()
        nll.backward()
        return nll

    opt.step(closure)
    vals = torch.exp(logp).detach().numpy()
    if not np.all(np.isfinite(vals)):
        return 0.03, 0.03, 0.03, 0.03
    return tuple(float(np.clip(v, 1e-3, 1.0)) for v in vals)


def fuse(X1, X2R, D1, D2R, W1, W2, a1, b1, a2, b2) -> np.ndarray:
    L1 = information(D1, W1, a1, b1)
    L2 = information(D2R, W2, a2, b2)
    return np.linalg.solve(L1 + L2, (L1 @ X1[..., None] + L2 @ X2R[..., None]))[..., 0]


def smooth_trajectory(X: np.ndarray, lam: float) -> np.ndarray:
    """argmin_y ||y - X||^2 + lam ||D2 y||^2 along axis 0 (pentadiagonal banded solve)."""
    T = X.shape[0]
    if lam <= 0 or T < 3:
        return X
    # Diagonals of D2^T D2 for the second-difference operator.
    main = np.full(T, 6.0)
    main[[0, -1]] = 1.0
    main[[1, -2]] = 5.0
    off1 = np.full(T - 1, -4.0)
    off1[[0, -1]] = -2.0
    off2 = np.ones(T - 2)
    if T == 3:
        main = np.array([1.0, 4.0, 1.0])
        off1 = np.array([-2.0, -2.0])
    ab = np.zeros((3, T))  # upper form for solveh_banded
    ab[2] = 1.0 + lam * main
    ab[1, 1:] = lam * off1
    ab[0, 2:] = lam * off2
    return solveh_banded(ab, X.reshape(T, -1)).reshape(X.shape)


def _frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _beliefs(metadata: dict, names: list[str], cam: str, vis_key: str) -> np.ndarray:
    jc = metadata.get("joint_confidence", {}).get(cam, {}) or {}
    vis = metadata.get(vis_key, {}) or {}
    return np.array([
        float(jc.get(n, 1.0)) * (1.0 if vis.get(n, True) else DEFAULT_OCCLUDED_FACTOR) for n in names
    ])


CHI2_3_MEDIAN = 2.366  # median of chi-square with 3 dof


def per_joint_noise(X1, X2R, rel_floor=0.05):
    """Identifiable per-camera, per-joint noise variance (trace, m^2), zero-GT.

    Cross-view residual r = X1 - R X2 gives the SUM v1 + v2 per joint. Each camera's own
    second difference has E||D2 x_i||^2 = motion + 6 v_i, and the true motion term is common to
    both views (root-relative, rotation-invariant norm), so the DIFFERENCE v1 - v2 is identified
    by the jitter difference. Medians make both estimates robust.
    """
    r2 = np.sum((X1 - X2R) ** 2, axis=-1)                       # (T,J)
    S = 3.0 * np.median(r2, axis=0) / CHI2_3_MEDIAN             # v1 + v2
    a1 = 3.0 * np.median(np.sum(np.diff(X1, 2, axis=0) ** 2, -1), axis=0) / CHI2_3_MEDIAN
    a2 = 3.0 * np.median(np.sum(np.diff(X2R, 2, axis=0) ** 2, -1), axis=0) / CHI2_3_MEDIAN
    d = (a1 - a2) / 6.0                                         # v1 - v2
    floor = rel_floor * np.maximum(S, EPS)
    v1 = np.clip(0.5 * (S + d), floor, None)
    v2 = np.clip(0.5 * (S - d), floor, None)
    return v1, v2


def gcv_lambda(Y: np.ndarray, grid=None) -> float:
    """Smoothing weight minimising generalized cross-validation for argmin ||y-Y||^2 + lam ||D2 y||^2."""
    T = Y.shape[0]
    if T < 5:
        return 0.0
    grid = np.logspace(-3, 3, 61) if grid is None else grid
    if T <= 4000:
        Dm = np.diff(np.eye(T), 2, axis=0)
        mu, Q = np.linalg.eigh(Dm.T @ Dm)
        Z2 = np.sum((Q.T @ Y.reshape(T, -1)) ** 2, axis=1)
    else:  # cosine-basis approximation for very long sequences
        k = np.arange(T)
        mu = (2.0 - 2.0 * np.cos(np.pi * k / T)) ** 2
        from scipy.fft import dct
        Z2 = np.sum(dct(Y.reshape(T, -1), axis=0, norm="ortho") ** 2, axis=1)
    mu = np.clip(mu, 0.0, None)
    best, best_lam = np.inf, 0.0
    for lam in grid:
        g = lam * mu / (1.0 + lam * mu)                         # eigenvalues of I - H
        score = T * np.sum(g ** 2 * Z2) / max(np.sum(g), EPS) ** 2
        if score < best:
            best, best_lam = score, float(lam)
    return best_lam


def shared_bone_lengths(F: np.ndarray, names: list[str], chains) -> dict:
    """One length per bone, constant in time and shared by both views (median of the fused skeleton)."""
    out = {}
    for c, p in chains:
        if c in names and p in names:
            out[(c, p)] = float(np.median(np.linalg.norm(F[:, names.index(c)] - F[:, names.index(p)], axis=-1)))
    return out


def project_bone_lengths(points: dict, lengths: dict) -> None:
    """Same projection as the legacy per-frame bone lock, but onto fixed target lengths."""
    for (c, p), l_ref in lengths.items():
        if c not in points or p not in points or l_ref < 1e-4:
            continue
        pc, pp = np.asarray(points[c], float), np.asarray(points[p], float)
        v = pc - pp
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            points[c] = [round(float(x), 6) for x in (pp + v * (l_ref / n))]


def fuse_sequence(P1, P2, W1, W2, names, rotation_window=0, irls_iters=5, max_residual=0.5, smoothing_lambda=0.0,
                  noise_model="global"):
    """Core algorithm on arrays. P*: (T,J,3) camera-frame joints, W*: (T,J) beliefs."""
    W1 = np.clip(W1, MIN_BELIEF, 1.0)
    W2 = np.clip(W2, MIN_BELIEF, 1.0)
    li, ri = names.index("left_hip"), names.index("right_hip")
    r1 = 0.5 * (P1[:, li] + P1[:, ri])
    r2 = 0.5 * (P2[:, li] + P2[:, ri])
    X1, X2 = P1 - r1[:, None], P2 - r2[:, None]
    D1 = P1 / np.maximum(np.linalg.norm(P1, axis=-1, keepdims=True), EPS)
    D2 = P2 / np.maximum(np.linalg.norm(P2, axis=-1, keepdims=True), EPS)

    Rs = per_frame_rotations(X1, X2, W1, W2, iters=irls_iters, window=rotation_window)
    X2R = np.einsum("tab,tjb->tja", Rs, X2)
    D2R = np.einsum("tab,tjb->tja", Rs, D2)
    sig = self_calibrate_noise(X1, X2R, D1, D2R, W1, W2, max_residual=max_residual)
    extra = {}
    if noise_model == "per_joint":
        # Scale each (camera, joint) ellipsoid so its trace equals the identified variance v_ij.
        v1, v2 = per_joint_noise(X1, X2R)
        t1 = 2 * sig[0] ** 2 + sig[1] ** 2
        t2 = 2 * sig[2] ** 2 + sig[3] ** 2
        W1 = W1 * (t1 / v1)[None, :]
        W2 = W2 * (t2 / v2)[None, :]
        extra["per_joint_std_mm"] = {"camera1": dict(zip(names, np.round(np.sqrt(v1 / 3) * 1e3, 1).tolist())),
                                     "camera2": dict(zip(names, np.round(np.sqrt(v2 / 3) * 1e3, 1).tolist()))}
    F = fuse(X1, X2R, D1, D2R, W1, W2, *sig)

    A1 = F + r1[:, None]
    A2 = np.einsum("tba,tjb->tja", Rs, F) + r2[:, None]
    if isinstance(smoothing_lambda, str) and smoothing_lambda.lower() == "gcv":
        smoothing_lambda = gcv_lambda(A1)
    smoothing_lambda = float(smoothing_lambda)
    Y1 = smooth_trajectory(A1, smoothing_lambda)
    Y2 = smooth_trajectory(A2, smoothing_lambda)
    info = {**extra,
        "noise_model": noise_model,
        "sigma_mm": {"camera1_lateral": sig[0] * 1e3, "camera1_depth": sig[1] * 1e3,
                     "camera2_lateral": sig[2] * 1e3, "camera2_depth": sig[3] * 1e3},
        "median_view_angle_deg": float(np.degrees(np.median(np.arccos(np.clip((D1 * D2R).sum(-1), -1, 1))))),
        "smoothing_lambda": float(smoothing_lambda),
        "rotation_window": int(rotation_window),
    }
    info["_fused_root_relative"] = F
    return Y1, Y2, info


def run_aniso_fusion(keypoints_dir: Path, metadata_dir: Path, pose_dir: Path, aniso_cfg: dict | None = None,
                     enforce_bone_lengths: bool = True) -> dict:
    """Overwrite fused keypoints with the anisotropic fusion of the raw monocular poses.

    Uses the Stage-1 metadata (``joint_confidence``, ``vis1``, ``vis2``) as beliefs.
    """
    from fusion_pipeline.executor import _enforce_bone_lengths

    cfg = aniso_cfg or {}
    pose_files = sorted(Path(pose_dir).glob("pose_data_*.json"), key=_frame_index)
    if len(pose_files) < 3:
        print("[Fusion][aniso] Fewer than 3 frames; skipped.")
        return {}
    poses = [json.loads(p.read_text(encoding="utf-8")) for p in pose_files]
    names = [n for n in poses[0]["camera1"] if n in poses[0]["camera2"]]
    P1 = np.array([[f["camera1"][n] for n in names] for f in poses], dtype=float)
    P2 = np.array([[f["camera2"][n] for n in names] for f in poses], dtype=float)
    W1 = np.ones(P1.shape[:2])
    W2 = np.ones(P2.shape[:2])
    for t, p in enumerate(pose_files):
        mp = Path(metadata_dir) / f"fused_data_{_frame_index(p)}.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            W1[t] = _beliefs(m, names, "camera1", "vis1")
            W2[t] = _beliefs(m, names, "camera2", "vis2")

    Y1, Y2, info = fuse_sequence(
        P1, P2, W1, W2, names,
        rotation_window=int(cfg.get("rotation_window", 0)),
        irls_iters=int(cfg.get("irls_iters", 5)),
        max_residual=float(cfg.get("max_residual_m", 0.5)),
        smoothing_lambda=cfg.get("smoothing_lambda", 0.5),
        noise_model=str(cfg.get("noise_model", "global")),
    )
    F = info.pop("_fused_root_relative")
    bone_mode = str(cfg.get("bone_lengths", "mono"))
    from fusion_pipeline.executor import KINEMATIC_CHAINS
    shared = shared_bone_lengths(F, names, KINEMATIC_CHAINS) if bone_mode == "shared" else None
    info["bone_lengths"] = bone_mode

    for t, p in enumerate(pose_files):
        fid = _frame_index(p)
        out = {"camera1": {n: [round(float(x), 6) for x in Y1[t, j]] for j, n in enumerate(names)},
               "camera2": {n: [round(float(x), 6) for x in Y2[t, j]] for j, n in enumerate(names)}}
        if enforce_bone_lengths:
            for cam in ("camera1", "camera2"):
                if shared is not None:
                    project_bone_lengths(out[cam], shared)
                else:
                    _enforce_bone_lengths(out[cam], poses[t][cam])
        (Path(keypoints_dir) / f"fused_data_{fid}.json").write_text(json.dumps(out), encoding="utf-8")
        mp = Path(metadata_dir) / f"fused_data_{fid}.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            m["fusion_method"] = "aniso"
            mp.write_text(json.dumps(m), encoding="utf-8")
    print(f"[Fusion][aniso] Fused {len(pose_files)} frames: {info}")
    return info

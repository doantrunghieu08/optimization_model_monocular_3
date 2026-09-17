from itertools import combinations

import numpy as np
from fusion_pipeline.detector import as_xyz



def _to_arrays(cam1, cam2, names):
    src = np.array([as_xyz(cam1[name]) for name in names], dtype=float)
    dst = np.array([as_xyz(cam2[name]) for name in names], dtype=float)
    return src, dst


def estimate_umeyama(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 3 or src.shape[1] != 3:
        raise ValueError("Similarity estimation requires matching (N, 3) arrays with N >= 3")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Similarity estimation requires finite points")

    n, m = src.shape
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    if np.linalg.matrix_rank(src_c) < 2 or np.linalg.matrix_rank(dst_c) < 2:
        raise ValueError("Similarity estimation requires at least 3 non-collinear points")
    sigma = np.mean(np.sum(src_c ** 2, axis=1))
    h = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(h)
    s_mat = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(vt.T) < 0:
        s_mat[m - 1, m - 1] = -1
    r = u @ s_mat @ vt
    scale = 1.0 if sigma < 1e-12 else float(np.trace(np.diag(d) @ s_mat) / sigma)
    t = mu_d - scale * (r @ mu_s)
    return scale, r, t


def apply_similarity(point, transform):
    scale, r, t = transform
    return scale * (r @ as_xyz(point)) + t


def ransac_umeyama(cam1, cam2, names, threshold, max_combos, rng=None):
    if len(names) < 3:
        raise ValueError("At least 3 anchors are required for cross-camera similarity")
    src_all, dst_all = _to_arrays(cam1, cam2, names)
    n = len(names)
    c3 = n * (n - 1) * (n - 2) // 6
    if rng is None:
        rng = np.random.default_rng(42)
    triplets = list(combinations(range(n), 3)) if c3 <= max_combos else [tuple(rng.choice(n, 3, replace=False)) for _ in range(max_combos)]
    best_inliers = []
    for tri in triplets:
        tri = list(tri)
        try:
            tf = estimate_umeyama(src_all[tri], dst_all[tri])
        except (ValueError, np.linalg.LinAlgError):
            continue
        pred = np.array([apply_similarity(src_all[i], tf) for i in range(n)])
        err = np.linalg.norm(pred - dst_all, axis=1)
        inliers = np.where(err < threshold)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if len(best_inliers) < 3:
        raise ValueError("RANSAC could not find at least 3 inliers for cross-camera similarity")
    inlier_names = [names[i] for i in best_inliers]
    tf_refined = estimate_umeyama(src_all[best_inliers], dst_all[best_inliers])
    return tf_refined, inlier_names


def estimate_bidirectional_similarity(cam1, cam2, candidate_names, threshold, max_combos):
    t12, anchor_names = ransac_umeyama(cam1, cam2, candidate_names, threshold=threshold, max_combos=max_combos)
    if len(anchor_names) >= 3:
        src_21 = np.array([cam2[n] for n in anchor_names], dtype=float)
        dst_21 = np.array([cam1[n] for n in anchor_names], dtype=float)
        t21 = estimate_umeyama(src_21, dst_21)
    else:
        t21 = (1.0, np.eye(3), np.zeros(3))
    return t12, t21, anchor_names


def _pose_root(pose):
    if not pose:
        return np.zeros(3)
    if "neck" in pose and "left_hip" in pose and "right_hip" in pose:
        return (as_xyz(pose["neck"]) + as_xyz(pose["left_hip"]) + as_xyz(pose["right_hip"])) / 3.0
    if "left_hip" in pose and "right_hip" in pose:
        return (as_xyz(pose["left_hip"]) + as_xyz(pose["right_hip"])) / 2.0
    coords = [as_xyz(v) for v in pose.values()]
    return np.mean(coords, axis=0) if coords else np.zeros(3)



def _estimate_origin_similarity(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 3 or src.shape[1] != 3:
        raise ValueError("Origin similarity estimation requires matching (N, 3) arrays with N >= 3")
    h = dst.T @ src
    u, singular, vt = np.linalg.svd(h)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt.T) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    denominator = float(np.sum(src ** 2))
    if denominator < 1e-12:
        raise ValueError("Origin similarity estimation requires non-zero points")
    scale = float(np.trace(np.diag(singular) @ sign) / denominator)
    return scale, rotation, np.zeros(3)


def estimate_sequence_root_similarity(frames, candidate_names, threshold):
    src, dst = [], []
    for frame in frames:
        cam1, cam2 = frame["camera1"], frame["camera2"]
        root1, root2 = _pose_root(cam1), _pose_root(cam2)
        for name in candidate_names:
            if name in cam1 and name in cam2:
                src.append(as_xyz(cam1[name]) - root1)
                dst.append(as_xyz(cam2[name]) - root2)
    if len(src) < 3:
        raise ValueError("Sequence alignment requires at least 3 joint observations")

    src = np.asarray(src)
    dst = np.asarray(dst)
    mask = np.ones(len(src), dtype=bool)
    for _ in range(5):
        transform = _estimate_origin_similarity(src[mask], dst[mask])
        residual = np.linalg.norm(
            np.asarray([apply_similarity(point, transform) for point in src]) - dst,
            axis=1,
        )
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        cutoff = max(float(threshold), median + 2.5 * 1.4826 * mad)
        new_mask = residual <= cutoff
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    t12 = _estimate_origin_similarity(src[mask], dst[mask])
    t21 = _estimate_origin_similarity(dst[mask], src[mask])
    return t12, t21, {"observations": len(src), "inliers": int(mask.sum())}


def apply_root_relative(point, source_pose, target_pose, transform):
    relative = as_xyz(point) - _pose_root(source_pose)
    return _pose_root(target_pose) + apply_similarity(relative, transform)


def _correction_alpha(name, source_confidence, target_confidence, blend_mode):
    if blend_mode == "hard":
        return 1.0
    if blend_mode != "confidence":
        raise ValueError("fusion.correction.blend_mode must be hard or confidence")
    source = max(0.0, float(source_confidence.get(name, 0.0)))
    target = max(0.0, float(target_confidence.get(name, 0.0)))
    return source / max(source + target, 1e-12)


def apply_confidence_corrections(
    cam1,
    cam2,
    k1_set,
    k2_set,
    t12,
    t21,
    max_displacement,
    h1=None,
    h2=None,
    blend_mode="hard",
    root_relative=False,
    return_applied=False,
):
    cam1_corr = dict(cam1)
    cam2_corr = dict(cam2)
    applied_k1 = set()
    applied_k2 = set()
    h1 = h1 or {}
    h2 = h2 or {}
    for name in k1_set:
        candidate = apply_root_relative(cam1[name], cam1, cam2, t12) if root_relative else apply_similarity(cam1[name], t12)
        if np.linalg.norm(candidate - as_xyz(cam2[name])) <= max_displacement:
            alpha = _correction_alpha(name, h1, h2, blend_mode)
            cam2_corr[name] = alpha * candidate + (1.0 - alpha) * as_xyz(cam2[name])
            if alpha > 0.0:
                applied_k1.add(name)
    for name in k2_set:
        candidate = apply_root_relative(cam2[name], cam2, cam1, t21) if root_relative else apply_similarity(cam2[name], t21)
        if np.linalg.norm(candidate - as_xyz(cam1[name])) <= max_displacement:
            alpha = _correction_alpha(name, h2, h1, blend_mode)
            cam1_corr[name] = alpha * candidate + (1.0 - alpha) * as_xyz(cam1[name])
            if alpha > 0.0:
                applied_k2.add(name)
    if return_applied:
        return cam1_corr, cam2_corr, applied_k1, applied_k2
    return cam1_corr, cam2_corr


def apply_rotation_mismatch_corrections(cam1_corr, cam2_corr, cam1, cam2, m_set, k1_set, k2_set, h1, h2, t12, t21, root_relative=False):
    cam1_fixed = dict(cam1_corr)
    cam2_fixed = dict(cam2_corr)
    for name in m_set:
        if name in k1_set or name in k2_set:
            continue
        if h1.get(name, 0.5) > h2.get(name, 0.5):
            cam2_fixed[name] = apply_root_relative(cam1[name], cam1, cam2, t12) if root_relative else apply_similarity(cam1[name], t12)
        else:
            cam1_fixed[name] = apply_root_relative(cam2[name], cam2, cam1, t21) if root_relative else apply_similarity(cam2[name], t21)
    return cam1_fixed, cam2_fixed

from itertools import combinations

import numpy as np
from fusion_pipeline.detector import as_xyz



def _to_arrays(cam1, cam2, names):
    src = np.array([as_xyz(cam1[name]) for name in names], dtype=float)
    dst = np.array([as_xyz(cam2[name]) for name in names], dtype=float)
    return src, dst


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes exact isometric rigid transform (s=1.0, R, t) mapping src to dst.
    Enforces scale=1.0 to preserve metric bone lengths and SMPL proportions.
    """
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    h = dst_c.T @ src_c
    u, _, vt = np.linalg.svd(h)
    s_mat = np.eye(src.shape[1])
    if np.linalg.det(u @ vt) < 0:
        s_mat[-1, -1] = -1
    r = u @ s_mat @ vt
    t = mu_d - r @ mu_s
    return 1.0, r, t


def invert_similarity(transform: tuple[float, np.ndarray, np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes exact mathematical inverse of a rigid similarity transform.
    Guarantees R_21 = R_12^T and perfect roundtrip symmetry without scale drift.
    """
    scale, r, t = transform
    inv_scale = 1.0 / max(scale, 1e-12)
    inv_r = r.T
    inv_t = -inv_scale * (inv_r @ t)
    return inv_scale, inv_r, inv_t


def estimate_umeyama(src, dst):
    n, m = src.shape
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
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


PARENT_JOINTS = {
    "head": "neck",
    "left_shoulder": "neck",
    "right_shoulder": "neck",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "left_toe": "left_ankle",
    "left_foot": "left_ankle",
    "right_toe": "right_ankle",
    "right_foot": "right_ankle",
}

KINEMATIC_DEPTH = {
    "pelvis": 0, "neck": 1, "head": 2,
    "left_shoulder": 2, "right_shoulder": 2,
    "left_elbow": 3, "right_elbow": 3,
    "left_wrist": 4, "right_wrist": 4,
    "left_hand": 5, "right_hand": 5,
    "left_hip": 1, "right_hip": 1,
    "left_knee": 2, "right_knee": 2,
    "left_ankle": 3, "right_ankle": 3,
    "left_foot": 4, "right_foot": 4,
    "left_toe": 5, "right_toe": 5,
}


def transform_joint_kinematic(name, src_cam, tgt_cam, transform):
    """
    Transforms joint 'name' from src_cam into tgt_cam frame.
    For limb/extremity joints, rotates bone vector around parent joint in tgt_cam,
    eliminating the 1-meter lever-arm error of rigid torso rotation.
    """
    scale, r, t = transform
    parent = PARENT_JOINTS.get(name)
    if parent and parent in src_cam and parent in tgt_cam:
        v_src = as_xyz(src_cam[name]) - as_xyz(src_cam[parent])
        return as_xyz(tgt_cam[parent]) + scale * (r @ v_src)
    return apply_similarity(src_cam[name], transform)


def ransac_umeyama(cam1, cam2, names, threshold, max_combos, rng=None):
    if len(names) < 3:
        return (1.0, np.eye(3), np.zeros(3)), list(names)
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
            tf = estimate_rigid_transform(src_all[tri], dst_all[tri])
        except np.linalg.LinAlgError:
            continue
        pred = np.array([apply_similarity(src_all[i], tf) for i in range(n)])
        err = np.linalg.norm(pred - dst_all, axis=1)
        inliers = np.where(err < threshold)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if not best_inliers:
        best_inliers = list(range(n))
    inlier_names = [names[i] for i in best_inliers]
    tf_refined = estimate_rigid_transform(src_all[best_inliers], dst_all[best_inliers])
    return tf_refined, inlier_names


def estimate_bidirectional_similarity(cam1, cam2, candidate_names, threshold, max_combos):
    torso_candidates = [n for n in candidate_names if n in {"pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"}]
    # ponytail: fallback to stable non-torso anchors when torso joints are occluded in hug/contact scenes
    if len(torso_candidates) < 3:
        fallback = [n for n in candidate_names if n in {"neck", "pelvis", "left_knee", "right_knee", "head", "left_elbow", "right_elbow"}]
        torso_candidates = fallback if len(fallback) >= 3 else candidate_names
    use_names = torso_candidates
    t12, anchor_names = ransac_umeyama(cam1, cam2, use_names, threshold=threshold, max_combos=max_combos)
    if len(anchor_names) >= 3:
        src_12 = np.array([cam1[n] for n in anchor_names], dtype=float)
        dst_12 = np.array([cam2[n] for n in anchor_names], dtype=float)
        t12 = estimate_rigid_transform(src_12, dst_12)
        t21 = invert_similarity(t12)
    else:
        t12 = (1.0, np.eye(3), np.zeros(3))
        t21 = (1.0, np.eye(3), np.zeros(3))
    return t12, t21, anchor_names


from fusion_pipeline.config import (
    NON_REPLACEABLE_ANCHORS,
    DEFAULT_MASTER_MARGIN,
    DEFAULT_MASTER_MAX_DISPLACEMENT,
    DEFAULT_MASTER_BLEND,
    HINGE_TRIPLETS,
    DEFAULT_MIN_COS,
    DEFAULT_MAX_COS,
)


EXTREMITIES = {
    "left_wrist", "right_wrist", "left_hand", "right_hand",
    "left_ankle", "right_ankle", "left_toe", "right_toe", "left_foot", "right_foot"
}

KINEMATIC_CHAINS = (
    ("head", "neck"),
    ("left_elbow", "left_shoulder"),
    ("right_elbow", "right_shoulder"),
    ("left_wrist", "left_elbow"),
    ("right_wrist", "right_elbow"),
    ("left_hand", "left_wrist"),
    ("right_hand", "right_wrist"),
    ("left_knee", "left_hip"),
    ("right_knee", "right_hip"),
    ("left_ankle", "left_knee"),
    ("right_ankle", "right_knee"),
    ("left_toe", "left_ankle"),
    ("left_foot", "left_ankle"),
    ("right_toe", "right_ankle"),
    ("right_foot", "right_ankle"),
)


def check_bone_lengths_valid(points: dict, ref_points: dict, max_ratio_error: float = 0.20) -> bool:
    # ponytail: 3D physical decider - rejects any candidate if bone length deviates > 20% from baseline
    for child, parent in KINEMATIC_CHAINS:
        if child not in points or parent not in points or child not in ref_points or parent not in ref_points:
            continue
        r_c = as_xyz(ref_points[child])
        r_p = as_xyz(ref_points[parent])
        l_ref = float(np.linalg.norm(r_c - r_p))
        if l_ref < 0.05:
            continue
        p_c = as_xyz(points[child])
        p_p = as_xyz(points[parent])
        l_cur = float(np.linalg.norm(p_c - p_p))
        if abs(l_cur - l_ref) / l_ref > max_ratio_error:
            return False
    return True


def check_joint_angles_valid(points: dict, min_cos: float = DEFAULT_MIN_COS, max_cos: float = DEFAULT_MAX_COS) -> bool:
    # ponytail: fast anatomical angle sanity check on hinge joints (elbows, knees)
    for a, b, c in HINGE_TRIPLETS:
        if a not in points or b not in points or c not in points:
            continue
        p_a = as_xyz(points[a])
        p_b = as_xyz(points[b])
        p_c = as_xyz(points[c])
        u = p_a - p_b
        v = p_c - p_b
        norm_u = float(np.linalg.norm(u))
        norm_v = float(np.linalg.norm(v))
        if norm_u < 1e-4 or norm_v < 1e-4:
            continue
        cos_val = float(np.dot(u, v) / (norm_u * norm_v))
        if not (min_cos <= cos_val <= max_cos):
            return False
    return True


def apply_confidence_corrections(
    cam1,
    cam2,
    k1_set,
    k2_set,
    t12,
    t21,
    h1=None,
    h2=None,
    blend=True,
    uncertainty_gating=True,
    max_displacement=0.35,
    vis1=None,
    vis2=None,
    master_margin=DEFAULT_MASTER_MARGIN,
):
    cam1_corr = dict(cam1)
    cam2_corr = dict(cam2)
    sorted_k1 = sorted(k1_set, key=lambda n: KINEMATIC_DEPTH.get(n, 99))
    sorted_k2 = sorted(k2_set, key=lambda n: KINEMATIC_DEPTH.get(n, 99))

    for name in sorted_k1:
        p_from_1 = transform_joint_kinematic(name, cam1, cam2_corr, t12)
        is_target_occluded = vis2 is not None and not vis2.get(name, True)
        is_extremity = name in EXTREMITIES
        limit = 0.75 if is_target_occluded else (0.05 if is_extremity else min(float(max_displacement), 0.15))
        if uncertainty_gating and max_displacement is not None:
            disp = float(np.linalg.norm(p_from_1 - as_xyz(cam2[name])))
            if disp > limit:
                continue
        if blend and h1 and h2:
            w1 = float(h1.get(name, 1.0))
            w2 = float(h2.get(name, 0.5))
            alpha = w1 / max(w1 + w2, 1e-6)
            if not is_target_occluded:
                alpha = min(alpha, 0.35 if is_extremity else 0.5)
            cand = alpha * p_from_1 + (1.0 - alpha) * as_xyz(cam2[name])
        else:
            cand = p_from_1
        test_pose = dict(cam2_corr)
        test_pose[name] = cand
        if check_joint_angles_valid(test_pose) and check_bone_lengths_valid(test_pose, cam2):
            cam2_corr[name] = cand

    for name in sorted_k2:
        is_target_occluded = vis1 is not None and not vis1.get(name, True)
        w1 = float(h1.get(name, 0.5)) if h1 else 0.5
        w2 = float(h2.get(name, 1.0)) if h2 else 1.0
        # ponytail: occluded joints skip margin check — any slave advantage is enough
        if not is_target_occluded and w2 <= w1 + master_margin:
            continue

        p_from_2 = transform_joint_kinematic(name, cam2, cam1_corr, t21)
        is_extremity = name in EXTREMITIES
        limit = 0.75 if is_target_occluded else (0.05 if is_extremity else min(float(max_displacement), 0.15))
        if uncertainty_gating and max_displacement is not None:
            disp = float(np.linalg.norm(p_from_2 - as_xyz(cam1[name])))
            if disp > limit:
                continue
        if blend and h1 and h2:
            alpha = w2 / max(w1 + w2, 1e-6)
            alpha = min(alpha, DEFAULT_MASTER_BLEND if is_target_occluded else 0.35)
            cand = alpha * p_from_2 + (1.0 - alpha) * as_xyz(cam1[name])
        else:
            alpha = DEFAULT_MASTER_BLEND if is_target_occluded else 0.35
            cand = alpha * p_from_2 + (1.0 - alpha) * as_xyz(cam1[name])

        test_pose = dict(cam1_corr)
        test_pose[name] = cand
        if check_joint_angles_valid(test_pose) and check_bone_lengths_valid(test_pose, cam1):
            cam1_corr[name] = cand
    return cam1_corr, cam2_corr


def apply_rotation_mismatch_corrections(
    cam1_corr,
    cam2_corr,
    cam1,
    cam2,
    m_set,
    k1_set,
    k2_set,
    h1,
    h2,
    t12,
    t21,
    blend=True,
    uncertainty_gating=True,
    max_displacement=0.35,
    vis1=None,
    vis2=None,
    master_margin=DEFAULT_MASTER_MARGIN,
):
    cam1_fixed = dict(cam1_corr)
    cam2_fixed = dict(cam2_corr)
    sorted_m = sorted(m_set, key=lambda n: KINEMATIC_DEPTH.get(n, 99))
    for name in sorted_m:
        if name in k1_set or name in k2_set:
            continue
        w1 = float(h1.get(name, 0.5)) if h1 else 0.5
        w2 = float(h2.get(name, 0.5)) if h2 else 0.5
        is_extremity = name in EXTREMITIES
        if w1 >= w2:
            p_trans = transform_joint_kinematic(name, cam1, cam2_fixed, t12)
            is_target_occluded = vis2 is not None and not vis2.get(name, True)
            limit = 0.75 if is_target_occluded else (0.05 if is_extremity else min(float(max_displacement), 0.15))
            if uncertainty_gating and max_displacement is not None:
                if float(np.linalg.norm(p_trans - as_xyz(cam2[name]))) > limit:
                    continue
            if blend:
                alpha = w1 / max(w1 + w2, 1e-6)
                if not is_target_occluded:
                    alpha = min(alpha, 0.35 if is_extremity else 0.5)
                cand = alpha * p_trans + (1.0 - alpha) * as_xyz(cam2[name])
            else:
                cand = p_trans
            test_pose = dict(cam2_fixed)
            test_pose[name] = cand
            if check_joint_angles_valid(test_pose) and check_bone_lengths_valid(test_pose, cam2):
                cam2_fixed[name] = cand
        else:
            is_target_occluded = vis1 is not None and not vis1.get(name, True)
            if w2 <= w1 + master_margin:
                continue

            p_trans = transform_joint_kinematic(name, cam2, cam1_fixed, t21)
            limit = DEFAULT_MASTER_MAX_DISPLACEMENT if max_displacement is None else min(float(max_displacement), DEFAULT_MASTER_MAX_DISPLACEMENT)
            if not is_target_occluded and is_extremity:
                limit = min(limit, 0.05)
            if uncertainty_gating and max_displacement is not None:
                if float(np.linalg.norm(p_trans - as_xyz(cam1[name]))) > limit:
                    continue
            if blend:
                alpha = w2 / max(w1 + w2, 1e-6)
                alpha = min(alpha, DEFAULT_MASTER_BLEND if is_target_occluded else 0.35)
                cand = alpha * p_trans + (1.0 - alpha) * as_xyz(cam1[name])
            else:
                alpha = DEFAULT_MASTER_BLEND if is_target_occluded else 0.35
                cand = alpha * p_trans + (1.0 - alpha) * as_xyz(cam1[name])
            test_pose = dict(cam1_fixed)
            test_pose[name] = cand
            if check_joint_angles_valid(test_pose) and check_bone_lengths_valid(test_pose, cam1):
                cam1_fixed[name] = cand
    return cam1_fixed, cam2_fixed


def apply_multiview_geometric_fusion(
    cam1,
    cam2,
    l_list,
    t12,
    t21,
    h1=None,
    h2=None,
    vis1=None,
    vis2=None,
    max_displacement=0.35,
    consensus_threshold=0.08,
    consensus_blend_strength=0.25,
    master_margin=0.0,
):
    cam1_fused = dict(cam1)
    cam2_fused = dict(cam2)
    if t12 is None or t21 is None:
        return cam1_fused, cam2_fused
    if np.allclose(t21[1], np.eye(3)) and np.allclose(t21[2], np.zeros(3)):
        return cam1_fused, cam2_fused

    # ponytail: gate on transform quality - skip fusion if residual is too large
    # (occurs when cameras are far/angled enough that torso-only RANSAC diverges)
    _TRANSFORM_QUALITY_THRESHOLD = 0.15  # metres
    torso_check = [n for n in ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip") if n in cam1 and n in cam2]
    if torso_check:
        residuals = [float(np.linalg.norm(apply_similarity(np.asarray(cam1[n], dtype=float), t12) - np.asarray(cam2[n], dtype=float))) for n in torso_check]
        if np.mean(residuals) > _TRANSFORM_QUALITY_THRESHOLD:
            return cam1_fused, cam2_fused

    core_anchors = NON_REPLACEABLE_ANCHORS | {"pelvis", "neck"}
    max_disp = 0.35 if max_displacement is None else float(max_displacement)
    cons_thresh = 0.08 if consensus_threshold is None else float(consensus_threshold)
    blend_strength = float(np.clip(consensus_blend_strength, 0.0, 1.0))

    sorted_l = sorted(l_list, key=lambda n: KINEMATIC_DEPTH.get(n, 99))
    for name in sorted_l:
        if name in core_anchors:
            continue
        v1 = vis1.get(name, True) if vis1 else True
        v2 = vis2.get(name, True) if vis2 else True
        if not (v1 or v2):
            continue
        p1 = as_xyz(cam1[name])
        p2 = as_xyz(cam2[name])
        p2_to_1 = transform_joint_kinematic(name, cam2, cam1_fused, t21)
        p1_to_2 = transform_joint_kinematic(name, cam1, cam2_fused, t12)

        disp1 = float(np.linalg.norm(p1 - p2_to_1))
        disp2 = float(np.linalg.norm(p2 - p1_to_2))

        w1 = float(h1.get(name, 0.5)) if (h1 and v1) else 0.0
        w2 = float(h2.get(name, 0.5)) if (h2 and v2) else 0.0
        tot = max(w1 + w2, 1e-6)

        is_extremity = name in EXTREMITIES
        effective_cons1 = min(cons_thresh, 0.08) if is_extremity else cons_thresh
        effective_cons2 = min(cons_thresh, 0.08) if is_extremity else cons_thresh

        limit1 = min(max_disp, DEFAULT_MASTER_MAX_DISPLACEMENT)
        limit2 = 0.75 if not v2 else max_disp

        protect_visible = master_margin > 0.0
        jointly_visible_consensus = (
            v1 and v2 and (not protect_visible or w2 > w1 + master_margin)
            and disp1 <= effective_cons1
        )
        # ponytail: occluded joints don't need the full margin fence; 5% slave lead is enough
        occluded_rescue = not v1 and v2 and w2 > w1 + 0.05 and disp1 <= limit1
        if jointly_visible_consensus or occluded_rescue:
            transfer = min(blend_strength * w2 / tot, DEFAULT_MASTER_BLEND)
            cand = (1.0 - transfer) * p1 + transfer * p2_to_1
            test_pose = dict(cam1_fused)
            test_pose[name] = cand
            if check_joint_angles_valid(test_pose) and check_bone_lengths_valid(test_pose, cam1):
                cam1_fused[name] = cand

        # prevent negative transfer on camera 2
        if not v2 and v1 and disp2 <= limit2:
            cand = p1_to_2
            test_pose = dict(cam2_fused)
            test_pose[name] = cand
            if check_joint_angles_valid(test_pose) and check_bone_lengths_valid(test_pose, cam2):
                cam2_fused[name] = cand
        elif v1 and v2 and (not protect_visible or w1 > w2 + master_margin) and disp2 <= effective_cons2:
            transfer = blend_strength * w1 / tot
            cand = (1.0 - transfer) * p2 + transfer * p1_to_2
            test_pose = dict(cam2_fused)
            test_pose[name] = cand
            if check_joint_angles_valid(test_pose) and check_bone_lengths_valid(test_pose, cam2):
                cam2_fused[name] = cand
    return cam1_fused, cam2_fused


def enforce_bone_lengths(points: dict, ref_points: dict) -> None:
    """
    Locks bone lengths to reference pose using forward kinematic ray projection.
    Preserves joint vector orientation while forcing bone lengths to exactly match ref_points.
    """
    for child, parent in KINEMATIC_CHAINS:
        if child not in points or parent not in points or child not in ref_points or parent not in ref_points:
            continue
        p_c = np.asarray(points[child], dtype=float)
        p_p = np.asarray(points[parent], dtype=float)
        r_c = np.asarray(ref_points[child], dtype=float)
        r_p = np.asarray(ref_points[parent], dtype=float)
        l_ref = float(np.linalg.norm(r_c - r_p))
        if l_ref < 1e-4:
            continue
        v = p_c - p_p
        dist = float(np.linalg.norm(v))
        if dist > 1e-6:
            points[child] = [round(float(x), 6) for x in (p_p + v * (l_ref / dist))]


def apply_belief_joint_replacement(
    cam1: dict,
    cam2: dict,
    t12: tuple,
    t21: tuple,
    h1: dict | None = None,
    h2: dict | None = None,
    vis1: dict | None = None,
    vis2: dict | None = None,
    names: list[str] | None = None,
    max_displacement: float = 0.75,
) -> tuple[dict, dict]:
    """
    Ponytail 3D pose fusion:
    1. Transforms Cam 2 (slave) into Cam 1 (master) frame via kinematic bone rotation.
    2. Replaces joint in Cam 1 whenever belief of slave > master (H2 > H1).
    3. Symmetrically updates Cam 2 whenever H1 > H2.
    4. Enforces rigid bone lengths immediately to prevent stretching/deformations.
    """
    cam1_fused = {k: list(v) for k, v in cam1.items()}
    cam2_fused = {k: list(v) for k, v in cam2.items()}
    if t12 is None or t21 is None:
        return cam1_fused, cam2_fused

    if names is None:
        names = list(cam1.keys())

    core_anchors = {"pelvis", "neck"}
    sorted_names = sorted(names, key=lambda n: KINEMATIC_DEPTH.get(n, 99))

    for name in sorted_names:
        if name in core_anchors:
            continue

        w1 = float(h1.get(name, 0.5)) if h1 else 0.5
        w2 = float(h2.get(name, 0.5)) if h2 else 0.5

        # If slave (cam2) belief is higher, replace joint in master (cam1)
        if w2 > w1:
            p_slave_in_master = transform_joint_kinematic(name, cam2, cam1_fused, t21)
            disp = float(np.linalg.norm(p_slave_in_master - as_xyz(cam1[name])))
            if disp <= max_displacement:
                cam1_fused[name] = [round(float(x), 6) for x in p_slave_in_master]

        # Symmetrically, if master (cam1) belief is higher, update joint in slave (cam2)
        elif w1 > w2:
            p_master_in_slave = transform_joint_kinematic(name, cam1, cam2_fused, t12)
            disp = float(np.linalg.norm(p_master_in_slave - as_xyz(cam2[name])))
            if disp <= max_displacement:
                cam2_fused[name] = [round(float(x), 6) for x in p_master_in_slave]

    # Enforce rigid bone lengths immediately to eliminate deformation
    enforce_bone_lengths(cam1_fused, cam1)
    enforce_bone_lengths(cam2_fused, cam2)

    return cam1_fused, cam2_fused



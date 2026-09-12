import numpy as np
from fusion_pipeline.config import TORSO_PART_IDS
from fusion_pipeline.config import ROTATION_PARENT_JOINTS
from fusion_pipeline.config import NON_REPLACEABLE_ANCHORS
from fusion_pipeline.config import HARMONIC_EPSILON
from fusion_pipeline.config import ORIENTATION_EPSILON
from fusion_pipeline.config import RIGID_BONES_RATIO
from fusion_pipeline.config import OCCLUSION_CHECK_JOINTS
from fusion_pipeline.config import CONFIDENCE_DELTA_CAP
from fusion_pipeline import context

def as_xyz(point):
    arr = np.asarray(point, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Expected shape (3,), got {}".format(arr.shape))
    return arr


_as_xyz = as_xyz


def get_orientation_flag(joints, epsilon=ORIENTATION_EPSILON):
    required = ("right_shoulder", "left_shoulder", "right_hip", "left_hip")
    if not all(name in joints for name in required):
        return {name: 0 for name in joints}
    rs = as_xyz(joints["right_shoulder"])
    ls = as_xyz(joints["left_shoulder"])
    rh = as_xyz(joints["right_hip"])
    lh = as_xyz(joints["left_hip"])
    mid_shoulders = (rs + ls) / 2.0
    mid_hips = (rh + lh) / 2.0
    v_lr = rs - ls
    v_spine = mid_shoulders - mid_hips
    forward_vec = np.cross(v_lr, v_spine)
    norm = float(np.linalg.norm(forward_vec))
    if norm < 1e-8:
        return {name: 0 for name in joints}
    forward_vec /= norm
    flags = {}
    for name, pos in joints.items():
        parent = ROTATION_PARENT_JOINTS.get(name)
        if parent is None or parent not in joints:
            flags[name] = 0
            continue
        dot = float(np.dot(as_xyz(pos) - as_xyz(joints[parent]), forward_vec))
        flags[name] = 0 if abs(dot) < epsilon else (1 if dot > 0 else -1)
    return flags


def load_torso_faces(vertex_parts, faces):
    vertex_parts = np.asarray(vertex_parts)
    faces = np.asarray(faces, dtype=np.int64)
    if vertex_parts.shape != (6890,) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Invalid SMPL vertex parts or faces for occlusion ray-casting")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertex_parts)):
        raise ValueError("SMPL face index is outside the vertex-part range")
    torso_mask = np.isin(vertex_parts, list(TORSO_PART_IDS))
    torso_faces = faces[np.all(torso_mask[faces], axis=1)]
    if not len(torso_faces):
        raise ValueError("No torso faces found in SMPL skinning parts")
    print(f"[Fusion] Torso ray-casting mesh: {len(torso_faces)} triangles")
    return torso_faces


def _ray_hits_before_target(target, triangle_origins, edge1, edge2, margin):
    target_distance = float(np.linalg.norm(target))
    if target_distance <= margin:
        return False

    direction = target / target_distance
    direction_rows = np.broadcast_to(direction, edge2.shape)
    pvec = np.cross(direction_rows, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(determinant) > 1e-9
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]

    tvec = -triangle_origins
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", direction_rows, qvec) * inverse
    distance = np.einsum("ij,ij->i", edge2, qvec) * inverse
    hit = valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
    hit &= (distance > 1e-9) & (distance < target_distance - margin)
    return bool(np.any(hit))


def compute_visibility_from_mesh_vertices(joints, verts, torso_faces, occlusion_tau=0.05):
    # ponytail: torso-only covers the dominant self-occlusion; use part-aware full mesh if limb-on-limb cases matter.
    visibility = {name: True for name in joints.keys()}
    verts = np.asarray(verts, dtype=float)
    torso_faces = np.asarray(torso_faces, dtype=np.int64)
    if verts.shape != (6890, 3):
        raise ValueError(f"Expected SMPL vertices shape (6890, 3), got {verts.shape}")
    if occlusion_tau < 0:
        raise ValueError("fusion.occlusion.tau must be non-negative")

    triangles = verts[torso_faces]
    triangle_origins = triangles[:, 0]
    edge1 = triangles[:, 1] - triangle_origins
    edge2 = triangles[:, 2] - triangle_origins
    for name, pos in joints.items():
        if name not in OCCLUSION_CHECK_JOINTS:
            continue
        kp_3d = as_xyz(pos)
        if kp_3d[2] <= 0:
            visibility[name] = False
            continue
        visibility[name] = not _ray_hits_before_target(
            kp_3d, triangle_origins, edge1, edge2, float(occlusion_tau)
        )
    return visibility


def compute_harmonic_precision(
    cam1,
    cam2,
    joint_names,
    vis1,
    vis2,
    alpha,
    beta,
    epsilon=HARMONIC_EPSILON,
):
    neighbors = {}
    for child, parent in RIGID_BONES_RATIO.keys():
        neighbors.setdefault(child, []).append(parent)
        neighbors.setdefault(parent, []).append(child)
    #Dự phòng sửa hàm này: https://docs.google.com/document/d/1yWfUcBP3AAykBXCK-aihj92ZplWtqjaSpFuPn-7N-eg/edit?usp=sharing
    def calc_P(cam, vis):
        P = {}
        for name in joint_names:
            if name not in cam:
                P[name] = 0.0
                continue
            C = 1.0 if vis.get(name, True) else 0.0
            L = float(np.linalg.norm(as_xyz(cam[name])))
            P[name] = C / (1.0 + alpha * (L ** 2))
        return P

    def calc_H(P):
        H = {}
        for name in joint_names:
            p = P[name]
            nb = [P[n] for n in neighbors.get(name, []) if n in P]
            b = beta * (sum(nb) / len(nb)) if nb else p
            H[name] = (2.0 * b * p) / (b + p + epsilon)
        return H

    P1, P2 = calc_P(cam1, vis1), calc_P(cam2, vis2)
    H1, H2 = calc_H(P1), calc_H(P2)
    weights = {name: (H1[name] + H2[name]) / 2.0 for name in joint_names}
    # Cập nhật giá trị vào biến context lưu ngữ cảnh
    context.current_H1 = H1
    context.current_H2 = H2
    return weights, H1, H2


def _confidence_value(confidence_by_joint, name):
    if not confidence_by_joint or name not in confidence_by_joint:
        return None
    value = confidence_by_joint[name]
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) < 3:
            return None
        value = value[2]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return max(0.0, value)


def _harmonic_blend(base_confidence, external_confidence, epsilon=HARMONIC_EPSILON):
    if external_confidence is None:
        return float(base_confidence)
    base_confidence = max(0.0, float(base_confidence))
    external_confidence = max(0.0, float(external_confidence))
    return float((2.0 * base_confidence * external_confidence) / (base_confidence + external_confidence + epsilon))


def _blend_detector_confidences(joint_names, base_confidences, external_confidences):
    return {
        name: _harmonic_blend(base_confidences[name], _confidence_value(external_confidences, name))
        for name in joint_names
    }


def detect_cross_view_errors(
    cam1,
    cam2,
    names,
    vis1,
    vis2,
    alpha,
    beta,
    confidence2d1=None,
    confidence2d2=None,
):
    flags1 = get_orientation_flag(cam1)
    flags2 = get_orientation_flag(cam2)
    m_set = {
        n
        for n in names
        if (flags1.get(n, 0) == 1 and flags2.get(n, 0) == -1)
        or (flags1.get(n, 0) == -1 and flags2.get(n, 0) == 1)
    }

    _, H1_old, H2_old = compute_harmonic_precision(cam1, cam2, names, vis1, vis2, alpha=alpha, beta=beta)
    H1_all = _blend_detector_confidences(names, H1_old, confidence2d1)
    H2_all = _blend_detector_confidences(names, H2_old, confidence2d2)
    all_weights = {name: (H1_all[name] + H2_all[name]) / 2.0 for name in names}
    abs_diffs = [abs(H1_all[n] - H2_all[n]) for n in names]
    delta = min(float(np.percentile(abs_diffs, 75)) if abs_diffs else 0.0, CONFIDENCE_DELTA_CAP)
    k1_set = {n for n in names if H1_all[n] > H2_all[n] + delta}
    k2_set = {n for n in names if H2_all[n] > H1_all[n] + delta}
    k1_set.difference_update(NON_REPLACEABLE_ANCHORS)
    k2_set.difference_update(NON_REPLACEABLE_ANCHORS)

    l_list = [n for n in names if n not in (m_set | k1_set | k2_set)]
    return {
        "M": m_set,
        "K1": k1_set,
        "K2": k2_set,
        "L": l_list,
        "weights": all_weights,
        "H1": H1_all,
        "H2": H2_all,
        "flags1": flags1,
        "flags2": flags2,
    }


def _pairwise_joint_distance_stats(data):
    cam1, cam2 = data.get("camera1", {}), data.get("camera2", {})
    distances = []
    for j in sorted(set(cam1) & set(cam2)):
        try:
            d = np.linalg.norm(np.asarray(cam1[j])[:3] - np.asarray(cam2[j])[:3])
            distances.append(float(d))
        except Exception:
            pass
    if not distances:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.array(distances)
    return float(np.percentile(arr, 25)), float(np.percentile(arr, 75)), float(np.mean(arr)), float(np.median(arr))


def make_raw_judgement_fallback(data, index, error=None):
    stats = _pairwise_joint_distance_stats(data)
    common = sorted(set(data["camera1"]) & set(data["camera2"]))
    return {
        "M": [],
        "K1": [],
        "K2": [],
        "A_new": common,
        "F": [],
        "before_stats": stats,
        "after_stats": stats,
        "optimized": data,
        "fallback_reason": str(error),
        "joint_confidence": {"camera1": {j: 1.0 for j in common}, "camera2": {j: 1.0 for j in common}},
        "vis1": {j: True for j in common},
        "vis2": {j: True for j in common},
    }


if __name__ == "__main__":
    triangle = np.array([[[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]]])
    origin = triangle[:, 0]
    edge1, edge2 = triangle[:, 1] - origin, triangle[:, 2] - origin
    assert _ray_hits_before_target(np.array([0.0, 0.0, 2.0]), origin, edge1, edge2, 0.01)
    assert not _ray_hits_before_target(np.array([0.0, 0.0, 0.5]), origin, edge1, edge2, 0.01)
    print("Occlusion ray-casting self-check passed")

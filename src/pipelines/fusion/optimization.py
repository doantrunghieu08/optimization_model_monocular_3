import numpy as np
from scipy.optimize import minimize
from src.pipelines.fusion.config import BONE_LENGTH_MIN_SCALE
from src.pipelines.fusion.config import BONE_LENGTH_MAX_SCALE
from src.pipelines.fusion.detector import as_xyz
from src.pipelines.fusion.config import DEFAULT_OCCLUDED_FACTOR
from src.pipelines.fusion.config import RIGID_BONES_RATIO
from src.pipelines.fusion.config import HEIGHT
from src.pipelines.fusion.config import HUBER_DELTA
from src.pipelines.fusion.config import TORSO_HEIGHT_RATIO


def get_diff_f(f_name, anchors, cam1, cam2, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR):
    p1_f = as_xyz(cam1[f_name])
    p2_f = as_xyz(cam2[f_name])
    sum_wdiff, sum_w = 0.0, 0.0
    for a in anchors:
        ca1 = float(conf1.get(a, 1.0)) if conf1 else 1.0
        ca2 = float(conf2.get(a, 1.0)) if conf2 else 1.0
        va1 = 1.0 if vis1 is None or vis1.get(a, True) else float(occluded_factor)
        va2 = 1.0 if vis2 is None or vis2.get(a, True) else float(occluded_factor)
        w = ((ca1 + ca2) / 2.0) * (va1 * va2)
        d1 = np.linalg.norm(p1_f - as_xyz(cam1[a]))
        d2 = np.linalg.norm(p2_f - as_xyz(cam2[a]))
        sum_wdiff += w * abs(d1 - d2)
        sum_w += w
    return sum_wdiff / max(sum_w, 1e-12)


def calculate_stats(cam1, cam2, f_list, anchors, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR, f_weights=None, huber_delta=HUBER_DELTA, loss_type="huber"):
    if not f_list or not anchors:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = [get_diff_f(f, anchors, cam1, cam2, conf1=conf1, conf2=conf2, vis1=vis1, vis2=vis2, occluded_factor=occluded_factor) for f in f_list]
    arr_diffs = np.array(diffs, dtype=float)
    q1 = float(np.percentile(arr_diffs, 25))
    q3 = float(np.percentile(arr_diffs, 75))
    mean_val = float(np.mean(arr_diffs))
    median_val = float(np.median(arr_diffs))
    if loss_type == "huber":
        loss = np.where(arr_diffs <= huber_delta, 0.5 * arr_diffs ** 2, huber_delta * (arr_diffs - 0.5 * huber_delta))
    elif loss_type == "mse":
        loss = arr_diffs ** 2
    else:
        raise ValueError("loss_type must be huber or mse")
    if f_weights:
        loss *= np.array([f_weights[name] for name in f_list], dtype=float)
    return q1, q3, mean_val, median_val, float(np.mean(loss))


def _torso_height_estimate(cam_dict):
    needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(name in cam_dict for name in needed):
        return None
    mid_shoulder = (as_xyz(cam_dict["left_shoulder"]) + as_xyz(cam_dict["right_shoulder"])) / 2.0
    mid_hip = (as_xyz(cam_dict["left_hip"]) + as_xyz(cam_dict["right_hip"])) / 2.0
    torso = float(np.linalg.norm(mid_shoulder - mid_hip))
    if torso < 1e-6:
        return None
    return torso / TORSO_HEIGHT_RATIO


def compute_dynamic_scale(cam_dict, f_list, ratios):
    sum_len, sum_ratio = 0.0, 0.0
    for (c, p), r in ratios.items():
        if c not in f_list and p not in f_list and c in cam_dict and p in cam_dict:
            sum_len += float(np.linalg.norm(as_xyz(cam_dict[c]) - as_xyz(cam_dict[p])))
            sum_ratio += r
    if sum_ratio > 0:
        return sum_len / sum_ratio
    sum_len, sum_ratio = 0.0, 0.0
    for (c, p), r in ratios.items():
        if c in cam_dict and p in cam_dict:
            sum_len += float(np.linalg.norm(as_xyz(cam_dict[c]) - as_xyz(cam_dict[p])))
            sum_ratio += r
    if sum_ratio > 0:
        return sum_len / sum_ratio
    return _torso_height_estimate(cam_dict) or HEIGHT


def _pose_root(pose):
    if not pose:
        return np.zeros(3)
    if "neck" in pose and "left_hip" in pose and "right_hip" in pose:
        return (as_xyz(pose["neck"]) + as_xyz(pose["left_hip"]) + as_xyz(pose["right_hip"])) / 3.0
    if "left_hip" in pose and "right_hip" in pose:
        return (as_xyz(pose["left_hip"]) + as_xyz(pose["right_hip"])) / 2.0
    coords = [as_xyz(v) for v in pose.values()]
    return np.mean(coords, axis=0) if coords else np.zeros(3)



def optimize_f_points(data, anchors, f_list, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR, regularization=False, regularization_lambda=1.0, prev_data=None, prev_prev_data=None, temporal_lambda=1.0, accel_lambda=3.0, max_iter=1000, use_kinematic_constraints=True, loss_type="huber", t12=None, t21=None, cross_view_lambda=0.0, root_relative=False):
    cam1 = {k: as_xyz(v) for k, v in data["camera1"].items()}
    cam2 = {k: as_xyz(v) for k, v in data["camera2"].items()}
    f_weights = {}
    for name in f_list:
        c1 = float(conf1.get(name, 1.0)) if conf1 else 1.0
        c2 = float(conf2.get(name, 1.0)) if conf2 else 1.0
        v1 = 1.0 if vis1 is None or vis1.get(name, True) else float(occluded_factor)
        v2 = 1.0 if vis2 is None or vis2.get(name, True) else float(occluded_factor)
        f_weights[name] = ((c1 + c2) / 2.0) * (v1 * v2)

    num_f = len(f_list)
    if not f_list:
        return {"camera1": cam1, "camera2": cam2}, None
    if loss_type not in ("huber", "mse"):
        raise ValueError("loss_type must be huber or mse")

    f_index = {name: index for index, name in enumerate(f_list)}
    initial1 = np.asarray([cam1[name] for name in f_list], dtype=float)
    initial2 = np.asarray([cam2[name] for name in f_list], dtype=float)
    f_weight_values = np.asarray([f_weights[name] for name in f_list], dtype=float)
    conf1_values = np.asarray([float(conf1.get(name, 1.0)) if conf1 else 1.0 for name in f_list])
    conf2_values = np.asarray([float(conf2.get(name, 1.0)) if conf2 else 1.0 for name in f_list])

    def indexed_points(names, camera):
        indices = np.asarray([f_index.get(name, -1) for name in names], dtype=int)
        fixed = np.asarray([camera[name] for name in names], dtype=float).reshape(-1, 3)
        return indices, fixed

    def gather(points, indices, fixed):
        result = fixed.copy()
        dynamic = indices >= 0
        result[dynamic] = points[indices[dynamic]]
        return result

    anchor_indices, anchor1_fixed = indexed_points(anchors, cam1)
    _, anchor2_fixed = indexed_points(anchors, cam2)
    anchor_weights = np.asarray([
        ((float(conf1.get(name, 1.0)) if conf1 else 1.0) + (float(conf2.get(name, 1.0)) if conf2 else 1.0))
        / 2.0
        * (1.0 if vis1 is None or vis1.get(name, True) else float(occluded_factor))
        * (1.0 if vis2 is None or vis2.get(name, True) else float(occluded_factor))
        for name in anchors
    ])
    anchor_weight_sum = max(float(anchor_weights.sum()), 1e-12)
    root1, root2 = _pose_root(cam1), _pose_root(cam2)

    def motion_target(history, previous_history=None):
        targets1 = np.zeros_like(initial1)
        targets2 = np.zeros_like(initial2)
        masks1 = np.zeros(num_f, dtype=bool)
        masks2 = np.zeros(num_f, dtype=bool)
        if history is None:
            return targets1, targets2, masks1, masks2
        for camera_name, targets, masks, root in (
            ("camera1", targets1, masks1, root1), ("camera2", targets2, masks2, root2)
        ):
            previous = history.get(camera_name, {})
            older = previous_history.get(camera_name, {}) if previous_history is not None else None
            previous_root = _pose_root(previous)
            older_root = _pose_root(older) if older is not None else None
            for index, name in enumerate(f_list):
                if name not in previous or (older is not None and name not in older):
                    continue
                previous_relative = as_xyz(previous[name]) - previous_root
                targets[index] = (
                    2.0 * previous_relative - (as_xyz(older[name]) - older_root)
                    if older is not None else previous_relative
                ) + root
                masks[index] = True
        return targets1, targets2, masks1, masks2

    temporal1, temporal2, temporal_mask1, temporal_mask2 = motion_target(prev_data)
    accel1, accel2, accel_mask1, accel_mask2 = motion_target(prev_data, prev_prev_data)

    bone_specs = [
        (child, parent, ratio)
        for (child, parent), ratio in RIGID_BONES_RATIO.items()
        if child in cam1 and parent in cam1 and (child in f_index or parent in f_index)
    ]
    bone_children = [child for child, _, _ in bone_specs]
    bone_parents = [parent for _, parent, _ in bone_specs]
    child_indices, child1_fixed = indexed_points(bone_children, cam1)
    parent_indices, parent1_fixed = indexed_points(bone_parents, cam1)
    _, child2_fixed = indexed_points(bone_children, cam2)
    _, parent2_fixed = indexed_points(bone_parents, cam2)
    bone_ratios = np.asarray([ratio for _, _, ratio in bone_specs], dtype=float)
    target1 = bone_ratios * compute_dynamic_scale(cam1, f_list, RIGID_BONES_RATIO)
    target2 = bone_ratios * compute_dynamic_scale(cam2, f_list, RIGID_BONES_RATIO)
    lower1, upper1 = (BONE_LENGTH_MIN_SCALE * target1) ** 2, (BONE_LENGTH_MAX_SCALE * target1) ** 2
    lower2, upper2 = (BONE_LENGTH_MIN_SCALE * target2) ** 2, (BONE_LENGTH_MAX_SCALE * target2) ** 2

    def split_points(x):
        points = np.asarray(x, dtype=float).reshape(2, num_f, 3)
        return points[0], points[1]

    def bone_vectors(points, child_fixed, parent_fixed):
        return gather(points, child_indices, child_fixed) - gather(points, parent_indices, parent_fixed)

    def objective(x):
        points1, points2 = split_points(x)
        anchors1 = gather(points1, anchor_indices, anchor1_fixed)
        anchors2 = gather(points2, anchor_indices, anchor2_fixed)
        distances1 = np.linalg.norm(points1[:, None, :] - anchors1[None, :, :], axis=2)
        distances2 = np.linalg.norm(points2[:, None, :] - anchors2[None, :, :], axis=2)
        diffs = np.abs(distances1 - distances2) @ anchor_weights / anchor_weight_sum
        loss = (
            np.where(diffs <= HUBER_DELTA, 0.5 * diffs ** 2, HUBER_DELTA * (diffs - 0.5 * HUBER_DELTA))
            if loss_type == "huber" else diffs ** 2
        )
        value = float(np.mean(loss * f_weight_values))
        if regularization:
            value += regularization_lambda * float(
                np.sum(conf1_values[:, None] * (points1 - initial1) ** 2)
                + np.sum(conf2_values[:, None] * (points2 - initial2) ** 2)
            )
        if cross_view_lambda > 0.0 and t21 is not None:
            scale, rotation, translation = t21
            source = points2 - root2 if root_relative else points2
            points2_in_1 = scale * (source @ np.asarray(rotation).T) + np.asarray(translation)
            if root_relative:
                points2_in_1 += root1
            value += cross_view_lambda * float(np.sum(f_weight_values[:, None] * (points1 - points2_in_1) ** 2))
        if prev_data is not None:
            value += temporal_lambda * float(
                np.sum((points1[temporal_mask1] - temporal1[temporal_mask1]) ** 2)
                + np.sum((points2[temporal_mask2] - temporal2[temporal_mask2]) ** 2)
            )
        if prev_data is not None and prev_prev_data is not None:
            value += accel_lambda * float(
                np.sum((points1[accel_mask1] - accel1[accel_mask1]) ** 2)
                + np.sum((points2[accel_mask2] - accel2[accel_mask2]) ** 2)
            )
        if use_kinematic_constraints and bone_specs:
            lengths1 = np.linalg.norm(bone_vectors(points1, child1_fixed, parent1_fixed), axis=1)
            lengths2 = np.linalg.norm(bone_vectors(points2, child2_fixed, parent2_fixed), axis=1)
            value += 15.0 * float(
                np.sum(((lengths1 - target1) / np.maximum(target1, 1e-6)) ** 2)
                + np.sum(((lengths2 - target2) / np.maximum(target2, 1e-6)) ** 2)
            )
        return value

    def bone_constraints(x):
        points1, points2 = split_points(x)
        vectors1 = bone_vectors(points1, child1_fixed, parent1_fixed)
        vectors2 = bone_vectors(points2, child2_fixed, parent2_fixed)
        squared1 = np.einsum("ij,ij->i", vectors1, vectors1)
        squared2 = np.einsum("ij,ij->i", vectors2, vectors2)
        return np.concatenate((squared1 - lower1, upper1 - squared1, squared2 - lower2, upper2 - squared2))

    def bone_constraints_jac(x):
        points1, points2 = split_points(x)
        vectors = (
            bone_vectors(points1, child1_fixed, parent1_fixed),
            bone_vectors(points2, child2_fixed, parent2_fixed),
        )
        bone_count = len(bone_specs)
        jacobian = np.zeros((4 * bone_count, 6 * num_f), dtype=float)
        for camera_index, camera_vectors in enumerate(vectors):
            lower_offset = camera_index * 2 * bone_count
            variable_offset = camera_index * 3 * num_f
            for bone_index, vector in enumerate(camera_vectors):
                lower_row = lower_offset + bone_index
                upper_row = lower_offset + bone_count + bone_index
                for joint_index, sign in ((child_indices[bone_index], 1.0), (parent_indices[bone_index], -1.0)):
                    if joint_index >= 0:
                        column = variable_offset + 3 * joint_index
                        derivative = sign * 2.0 * vector
                        jacobian[lower_row, column:column + 3] += derivative
                        jacobian[upper_row, column:column + 3] -= derivative
        return jacobian

    constraints = (
        [{"type": "ineq", "fun": bone_constraints, "jac": bone_constraints_jac}]
        if use_kinematic_constraints and bone_specs else []
    )
    x0 = np.concatenate((initial1.ravel(), initial2.ravel()))
    res = minimize(objective, x0, constraints=constraints, method="SLSQP", options={"maxiter": max_iter})
    use_result = bool(res.success) and np.isfinite(res.x).all()
    if not use_result:
        print(f"[Optimization] SLSQP info ({res.message}); keeping pre-optimization pose")
    solution = res.x if use_result else x0

    p1_opt, p2_opt = dict(cam1), dict(cam2)
    for i, name in enumerate(f_list):
        p1_opt[name] = solution[i * 3:i * 3 + 3].tolist()
        p2_opt[name] = solution[(num_f + i) * 3:(num_f + i) * 3 + 3].tolist()
    return {"camera1": p1_opt, "camera2": p2_opt}, res

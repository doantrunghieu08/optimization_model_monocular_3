"""Carry the COCO face keypoints (nose/eyes/ears) through fusion.

The fusion skeleton has 21 joints without eyes/ears. Face points exported by the pose stage
("camera1_face") are attached rigidly to the neck->head bone: they are rotated by the
minimal rotation that maps the input neck->head direction onto the fused one and keep
their offset from the neck. Used for evaluation only; the fusion itself never reads them.
"""
import numpy as np


def _rotation_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def _head_frame(pose):
    """Orthonormal frame on neck->head with shoulder axis resolving yaw twist."""
    y = np.asarray(pose["head"], float) - np.asarray(pose["neck"], float)
    yn = float(np.linalg.norm(y))
    if yn < 1e-9 or "left_shoulder" not in pose or "right_shoulder" not in pose:
        return None
    y = y / yn
    x = np.asarray(pose["left_shoulder"], float) - np.asarray(pose["right_shoulder"], float)
    x = x - y * float(y @ x)
    xn = float(np.linalg.norm(x))
    if xn < 1e-9:
        return None
    x = x / xn
    return np.stack([x, y, np.cross(x, y)], axis=1)


def attach_face(input_pose, fused_pose, input_face):
    """Face points for fused_pose, rigidly following neck->head + shoulder yaw."""
    n_in, h_in = np.asarray(input_pose["neck"], float), np.asarray(input_pose["head"], float)
    n_out, h_out = np.asarray(fused_pose["neck"], float), np.asarray(fused_pose["head"], float)
    F_in, F_out = _head_frame(input_pose), _head_frame(fused_pose)
    if F_in is not None and F_out is not None:
        R = F_out @ F_in.T
    elif np.linalg.norm(h_in - n_in) < 1e-9 or np.linalg.norm(h_out - n_out) < 1e-9:
        R = np.eye(3)
    else:
        R = _rotation_between(h_in - n_in, h_out - n_out)
    return {k: (n_out + R @ (np.asarray(p, float) - n_in)).tolist() for k, p in input_face.items()}

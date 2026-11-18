"""Harmony4D-style scoring: COCO-17 joints, sequence-level MPJPE alignment, face points."""
import unittest

import numpy as np

from evaluation_pipeline.executor import (
    COCO17_NAMES, _compute_mpjpe, _compute_mpjpe_fixed_rotation, _sequence_rotation,
)
from fusion_pipeline.face import attach_face


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def pose(seed):
    rng = np.random.default_rng(seed)
    p = {k: rng.normal(size=3) * 0.3 for k in COCO17_NAMES}
    return p


class SequenceAlignmentTests(unittest.TestCase):
    def test_constant_camera_rotation_is_fully_removed(self):
        R = rot_y(37)
        frames = []
        for s in range(5):
            t = pose(s)
            p = {k: R.T @ v + np.array([1.0, 2.0, 3.0]) for k, v in t.items()}  # pred in a rotated camera frame
            frames.append((p, t, COCO17_NAMES))
        Rs = _sequence_rotation(frames)
        for p, t, keys in frames:
            self.assertLess(_compute_mpjpe_fixed_rotation(p, t, keys, Rs)[0], 1e-6)

    def test_per_frame_orientation_error_is_counted(self):
        frames = []
        for s, deg in enumerate([0, 20, -20, 10]):
            t = pose(s)
            p = {k: rot_y(deg) @ v for k, v in t.items()}  # orientation drifts frame to frame
            frames.append((p, t, COCO17_NAMES))
        Rs = _sequence_rotation(frames)
        seq = np.mean([_compute_mpjpe_fixed_rotation(p, t, k, Rs)[0] for p, t, k in frames])
        per_frame = np.mean([_compute_mpjpe(p, t, k)[0] for p, t, k in frames])
        self.assertLess(per_frame, 1e-6)        # per-frame rotation hides the drift
        self.assertGreater(seq, 10.0)           # sequence alignment reports it (mm)


class FaceAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.inp = {"neck": np.array([0.0, 1.4, 3.0]), "head": np.array([0.0, 1.6, 3.0])}
        self.face = {"nose": [0.0, 1.62, 2.9], "left_ear": [0.07, 1.6, 3.0]}

    def test_unchanged_head_keeps_face(self):
        out = attach_face(self.inp, self.inp, self.face)
        for k in self.face:
            self.assertTrue(np.allclose(out[k], self.face[k]))

    def test_face_follows_rotated_head_bone_and_keeps_distance(self):
        fused = {"neck": self.inp["neck"], "head": self.inp["neck"] + rot_y(0) @ np.array([0.0, 0.0, -0.2])}
        out = attach_face(self.inp, fused, self.face)
        for k, p in self.face.items():
            self.assertAlmostEqual(np.linalg.norm(np.subtract(out[k], fused["neck"])),
                                   np.linalg.norm(np.subtract(p, self.inp["neck"])), places=9)
        # head bone now points along -z, so the nose moved forward (more negative z)
        self.assertLess(out["nose"][2], self.face["nose"][2] - 0.1)


if __name__ == "__main__":
    unittest.main()

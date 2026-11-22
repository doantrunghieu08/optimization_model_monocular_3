"""Harmony4D-style scoring: COCO-17 joints, sequence-level MPJPE alignment, face points."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation_pipeline.executor import (
    COCO17_NAMES, _compute_acceleration_error, _compute_mpjpe, _compute_mpjpe_fixed_rotation,
    _find_colmap_images_txt, _load_bundled_harmony4d_rotations, _load_colmap_cam_to_world_rotations,
    _qvec_to_rotmat, _sequence_rotation,
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

    def test_face_follows_pure_yaw_rotation_about_vertical_neck_head_axis(self):
        inp = {
            "neck": np.array([0.0, 1.4, 3.0]), "head": np.array([0.0, 1.6, 3.0]),
            "left_shoulder": np.array([-0.2, 1.35, 3.0]), "right_shoulder": np.array([0.2, 1.35, 3.0]),
        }
        R = rot_y(90)
        pivot = inp["neck"]
        fused = {k: pivot + R @ (v - pivot) for k, v in inp.items()}
        out = attach_face(inp, fused, self.face)
        for k, p in self.face.items():
            expected = pivot + R @ (np.asarray(p, float) - pivot)
            self.assertTrue(np.allclose(out[k], expected, atol=1e-9))

    def test_accel_error_respects_camera_to_world_rotation(self):
        R = rot_y(90)
        gt_seq = [{"pelvis": np.array([t * t * 0.01, 0.0, 0.0])} for t in range(3)]
        pred_seq = [{"pelvis": R.T @ g["pelvis"]} for g in gt_seq]
        err_unaligned, _ = _compute_acceleration_error(tuple(pred_seq), tuple(gt_seq), ["pelvis"])
        err_aligned, _ = _compute_acceleration_error(tuple(pred_seq), tuple(gt_seq), ["pelvis"], rotation=R)
        self.assertGreater(err_unaligned, 10.0)
        self.assertLess(err_aligned, 1e-6)

    def test_colmap_images_txt_gives_exact_gt_free_world_rotation(self):
        # 60 deg rotation around Y: q = (cos 30, 0, sin 30, 0)
        q_w2c = np.array([np.cos(np.radians(30)), 0.0, np.sin(np.radians(30)), 0.0])
        R_w2c = _qvec_to_rotmat(q_w2c)
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp) / "002_hugging"
            gt_dir = seq / "processed_data" / "poses3d"
            gt_dir.mkdir(parents=True)
            imgs = seq / "colmap" / "workplace" / "images.txt"
            imgs.parent.mkdir(parents=True)
            imgs.write_text(
                "# Image list with two lines of data per image:\n"
                f"1 {q_w2c[0]} {q_w2c[1]} {q_w2c[2]} {q_w2c[3]} 0.1 -0.2 3.0 1 cam02/00001.jpg\n"
                "100.0 200.0 -1 300.0 400.0 5\n"
                f"2 {q_w2c[0]} {q_w2c[1]} {q_w2c[2]} {q_w2c[3]} 0.1 -0.2 3.0 1 exo/cam02/00002.jpg\n"
                "\n",
                encoding="utf-8",
            )
            self.assertEqual(_find_colmap_images_txt(gt_dir), imgs)
            rots = _load_colmap_cam_to_world_rotations(imgs)
            self.assertIn(2, rots)
            self.assertTrue(np.allclose(rots[2], R_w2c.T, atol=1e-9))
            # Camera-frame pose X_cam = R_w2c @ X_world + t -> 0 mm MPJPE with 0% GT rotation fitting
            gt_pose = pose(42)
            cam_pose = {k: R_w2c @ v + np.array([0.1, -0.2, 3.0]) for k, v in gt_pose.items()}
            err, _ = _compute_mpjpe_fixed_rotation(cam_pose, gt_pose, COCO17_NAMES, rots[2])
            self.assertLess(err, 1e-6)

    def test_bundled_harmony4d_extrinsics_covers_sequences_with_valid_so3(self):
        rots, seq = _load_bundled_harmony4d_rotations(("inputs/drive/002_hugging/processed_data/poses3d",))
        self.assertEqual(seq, "002_hugging")
        self.assertIn(6, rots)
        R = rots[6]
        self.assertTrue(np.allclose(R.T @ R, np.eye(3), atol=1e-6))
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

"""EXT=1 evaluation: COLMAP extrinsics (+ scale.npy similarity) move world GT into the camera frame."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evaluation_pipeline.executor import _compute_mpjpe, _compute_pa_mpjpe
from evaluation_pipeline.extrinsics import (_qvec2rotmat, camera_name_from_path, ext_enabled, extrinsic_from_config,
                                            find_colmap_workplace, load_world_to_camera,
                                            transform_truth_map)


def _rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotmat2qvec(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


class ExtrinsicsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        seq = Path(self.tmp.name) / "001_seq"
        self.truth_dir = seq / "processed_data" / "poses3d"
        self.truth_dir.mkdir(parents=True)
        self.wp = seq / "colmap" / "workplace"
        self.wp.mkdir(parents=True)
        # colmap world -> camera
        self.R_cc = _rot("x", 30) @ _rot("z", 20)
        self.t_cc = np.array([0.3, -0.2, 3.0])
        q = _rotmat2qvec(self.R_cc)
        np.testing.assert_allclose(_qvec2rotmat(q), self.R_cc, atol=1e-9)
        rows = ["# header", "# header"]
        for i, fr in enumerate(["00001", "00101", "00201"]):
            rows.append(" ".join(map(str, [i + 1, *q, *self.t_cc, 3, f"cam03/{fr}.jpg"])))
            rows.append("1.0 2.0 -1")
        rows.append(" ".join(map(str, [9, 1, 0, 0, 0, 0, 0, 0, 1, "mobile/00001.jpg"])))
        rows.append("")
        (self.wp / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        # colmap -> world similarity
        self.s, self.R_s, self.t_s = 0.9, _rot("z", -40), np.array([1.0, 2.0, 0.5])
        S = np.eye(4)
        S[:3, :3] = self.s * self.R_s
        S[:3, 3] = self.t_s
        np.save(self.wp / "scale.npy", S)

    def tearDown(self):
        self.tmp.cleanup()

    def test_world_to_camera_matches_definition(self):
        X_colmap = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 1.0]])
        X_world = (self.s * X_colmap @ self.R_s.T) + self.t_s
        expected = self.s * (X_colmap @ self.R_cc.T + self.t_cc)  # camera frame in metres
        R, t = load_world_to_camera(str(self.wp), "cam03")
        np.testing.assert_allclose(X_world @ R.T + t, expected, atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)

    def test_unknown_camera_and_lookup(self):
        self.assertEqual(find_colmap_workplace(self.truth_dir), self.wp.resolve())
        with self.assertRaises(KeyError):
            load_world_to_camera(str(self.wp), "cam99")
        with self.assertRaises(FileNotFoundError):
            find_colmap_workplace(Path(self.tmp.name) / "elsewhere")

    def test_camera_name_and_env(self):
        self.assertEqual(camera_name_from_path("in/005/exo/cam03/output_wham/wham_output_cam03.pkl"), "cam03")
        self.assertEqual(camera_name_from_path(None, "x/harmony4D_005_sword3_cam4_ffmpeg.mp4"), "cam04")
        for value, expected in [("1", True), ("TRUE", True), ("yes", True), ("0", False), ("", False)]:
            with mock.patch.dict(os.environ, {"EXT": value}):
                self.assertEqual(ext_enabled(), expected)

    def test_camera_frame_mpjpe_needs_no_rotation(self):
        R, t = load_world_to_camera(str(self.wp), "cam03")
        names = ["left_hip", "right_hip", "left_knee", "right_knee", "nose"]
        rng = np.random.default_rng(0)
        world = {k: rng.normal(size=3) for k in names}
        truth = transform_truth_map({0: {"joints": world}}, R, t)[0]["joints"]
        pred = {k: v + 5.0 for k, v in truth.items()}  # same pose, other root position
        self.assertAlmostEqual(_compute_mpjpe(pred, truth, names, rotate=False)[0], 0.0, places=6)
        # a pure rotation error is now counted (it was absorbed by the fitted rotation before)
        Rz = _rot("z", 10)
        rotated = {k: Rz @ v for k, v in truth.items()}
        self.assertGreater(_compute_mpjpe(rotated, truth, names, rotate=False)[0], 1.0)
        self.assertAlmostEqual(_compute_mpjpe(rotated, truth, names, rotate=True)[0], 0.0, places=6)
        # PA-MPJPE is invariant to the rigid world->camera map
        self.assertAlmostEqual(_compute_pa_mpjpe(rotated, truth, names)[0],
                               _compute_pa_mpjpe(rotated, world, names)[0], places=6)

    def test_yaml_extrinsic_block(self):
        R, t = load_world_to_camera(str(self.wp), "cam03")
        q = _rotmat2qvec(R)
        C = -R.T @ t
        full = {"world": "gt", "unit": "m", "R": R.tolist(), "t": t.tolist(), "C": C.tolist(),
                "qvec_wxyz": q.tolist()}
        for block in (full, {"R": R.tolist(), "t": t.tolist()}, {"qvec_wxyz": q.tolist(), "C": C.tolist()},
                      {"R": R.tolist(), "t": (t * 1000).tolist(), "unit": "mm"}):
            R2, t2 = extrinsic_from_config(block, "cam03")
            np.testing.assert_allclose(R2, R, atol=1e-9)
            np.testing.assert_allclose(t2, t, atol=1e-9)
        bad = [{**full, "world": "colmap"}, {**full, "t": (t + 0.1).tolist()},
               {**full, "qvec_wxyz": [1, 0, 0, 0]}, {"R": (2 * R).tolist(), "t": t.tolist()},
               {"R": R.tolist()}, {**full, "unit": "cm"}]
        for block in bad:
            with self.assertRaises(ValueError):
                extrinsic_from_config(block, "cam03")


if __name__ == "__main__":
    unittest.main()

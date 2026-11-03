"""Isolated synthetic regressions for experimental optimization-based AOTR.

No ground truth files or pretrained models are required: synthetic reference
positions are used ONLY to construct observations and assert correctness.
"""
import unittest
import numpy as np

from fusion_pipeline.aotr import optimize_aotr, _trust_region_1d


def _fixture():
    master = {
        "pelvis": [0.0, -0.05, 5.0],
        "neck": [0.0, 0.58, 5.0],
        "head": [0.0, 0.78, 5.0],
        "left_shoulder": [-0.30, 0.46, 5.0],
        "right_shoulder": [0.31, 0.47, 5.0],
        "left_elbow": [-0.52, 0.26, 5.0],
        "right_elbow": [0.54, 0.23, 5.0],
        "left_wrist": [-0.68, 0.03, 5.05],
        "right_wrist": [0.65, 0.02, 5.03],
        "left_hip": [-0.17, -0.14, 5.0],
        "right_hip": [0.17, -0.14, 5.0],
        "left_knee": [-0.18, -0.59, 5.01],
        "right_knee": [0.19, -0.58, 5.01],
        "left_ankle": [-0.20, -1.01, 5.05],
        "right_ankle": [0.20, -1.02, 5.04],
    }
    # Same bone length as Master, but different wrist orientation.
    wrist_master = np.asarray(master["left_wrist"], float)
    elbow = np.asarray(master["left_elbow"], float)
    length = np.linalg.norm(wrist_master-elbow)
    target_vector = np.asarray([0.20, -0.16, 0.17], float)
    target_vector = length * target_vector / np.linalg.norm(target_vector)
    truth = {j: np.asarray(v, float).copy() for j, v in master.items()}
    truth["left_wrist"] = elbow + target_vector
    cam_shift = np.array([0.75, 0.0, 0.0])
    slave = {j: (v+cam_shift).tolist() for j, v in truth.items()}
    K = np.array([[1200., 0., 320.], [0., 1200., 240.], [0., 0., 1.]])
    def observations(points):
        out = {}
        for j, p in points.items():
            uvw = K @ p
            out[j] = [float(uvw[0]/uvw[2]), float(uvw[1]/uvw[2]), 0.95]
        return out
    obs_m = observations(truth)
    obs_s = observations({j: v+cam_shift for j, v in truth.items()})
    cfg = {"minimum_cost_reduction": 1e-10, "trust_max_iter": 12}
    return master, slave, truth, K, obs_m, obs_s, cfg


class AOTRTests(unittest.TestCase):
    def test_trust_region_tracks_interior_optimum(self):
        a, v, count = _trust_region_1d(lambda x: (x-0.83)**2, 16, 0.25)
        self.assertAlmostEqual(a, 0.83, delta=0.04)
        self.assertLess(v, 0.002)
        self.assertGreater(count, 3)

    def test_good_slave_direction_can_receive_near_full_correction(self):
        M, S, T, K, y1, y2, cfg = _fixture()
        output, diag = optimize_aotr(M, S, y1, y2, K, K, config=cfg)
        self.assertEqual(diag["status"], "corrected")
        self.assertIn("left_wrist", [b["joint"] for b in diag["accepted_bones"]])
        self.assertGreater(diag["accepted_bones"][0]["alpha"], 0.5)
        old = np.linalg.norm(np.asarray(M["left_wrist"])-T["left_wrist"])
        new = np.linalg.norm(np.asarray(output["left_wrist"])-T["left_wrist"])
        self.assertLess(new, old * 0.5)
        self.assertLess(diag["cost_after"], diag["cost_before"])
        self.assertEqual(output["pelvis"], M["pelvis"])

    def test_bone_lengths_and_unchanged_master_are_preserved(self):
        M, S, T, K, y1, y2, cfg = _fixture()
        output, diag = optimize_aotr(M, S, y1, y2, K, K, config=cfg)
        for child, parent in (
            ("left_wrist", "left_elbow"),
            ("left_elbow", "left_shoulder"),
            ("left_knee", "left_hip"),
        ):
            before = np.linalg.norm(np.asarray(M[child])-np.asarray(M[parent]))
            after = np.linalg.norm(np.asarray(output[child])-np.asarray(output[parent]))
            self.assertAlmostEqual(before, after, places=8)
        self.assertEqual(output["right_wrist"], M["right_wrist"])
        self.assertEqual(diag["joint_selection"]["right_wrist"], "KEEP")

    def test_bad_slave_direction_does_not_override_correct_master(self):
        M, S, T, K, y1, y2, cfg = _fixture()
        M = {j: v.tolist() for j, v in T.items()}
        S["left_wrist"] = [-0.88 + 0.75, -0.21, 4.75]
        output, diag = optimize_aotr(M, S, y1, y2, K, K, config=cfg)
        self.assertEqual(diag["status"], "unchanged")
        self.assertEqual(output, M)

    def test_missing_observations_fail_closed(self):
        M, S, _, K, y1, y2, cfg = _fixture()
        out, d = optimize_aotr(M, S, y1, {}, K, K, config=cfg)
        self.assertEqual(d["reason"], "insufficient_2d_correspondences")
        self.assertEqual(out, M)
        out, d = optimize_aotr(M, S, y1, y2, None, K, config=cfg)
        self.assertEqual(d["reason"], "missing_intrinsics")
        self.assertEqual(out, M)

    def test_inconsistent_camera_alignment_fails_closed(self):
        M, S, _, K, y1, y2, cfg = _fixture()
        S["left_shoulder"] = [-2.0, 1.0, 9.0]
        S["right_shoulder"] = [2.0, 1.0, 9.0]
        S["neck"] = [0.0, 1.7, 8.0]
        cfg["alignment_median_max_m"] = 0.01
        cfg["alignment_q75_max_m"] = 0.015
        out, diag = optimize_aotr(M, S, y1, y2, K, K, config=cfg)
        self.assertEqual(diag["status"], "fallback")
        self.assertIn(diag["reason"], {"unreliable_alignment", "insufficient_alignment_inliers"})
        self.assertEqual(out, M)


if __name__ == "__main__":
    unittest.main()

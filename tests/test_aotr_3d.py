"""Synthetic regressions for explicit AOTR-3D fallback without keypoints2D."""
import unittest
import numpy as np
from fusion_pipeline.aotr import optimize_aotr
from fusion_pipeline.aotr_3d import optimize_aotr_3d


def fixture():
    master = {
        "pelvis": [0.0, -0.10, 4.5],
        "neck": [0.0, 0.56, 4.5],
        "left_shoulder": [-0.27, 0.45, 4.5],
        "right_shoulder": [0.27, 0.45, 4.5],
        "left_hip": [-0.15, -0.14, 4.5],
        "right_hip": [0.15, -0.14, 4.5],
        "left_elbow": [-0.48, 0.24, 4.5],
        "left_wrist": [-0.68, 0.03, 4.5],
        "right_elbow": [0.49, 0.24, 4.5],
        "right_wrist": [0.69, 0.03, 4.5],
    }
    slave = {j: (np.asarray(p)+np.array([0.8, 0.02, 0.0])).tolist()
             for j, p in master.items()}
    slave["left_wrist"] = (np.asarray(slave["left_elbow"]) +
                           np.array([0.16, -0.18, 0.17])).tolist()
    return master, slave


class AOTR3DTests(unittest.TestCase):
    def test_opt_in_works_without_intrinsics_or_tracking(self):
        master, slave = fixture()
        fused, diag = optimize_aotr(
            master, slave, K_master=None, K_slave=None,
            observations_master=None, observations_slave=None,
            config={"missing_2d_mode": "3d_surrogate",
                    "minimum_cost_reduction": 1e-10}
        )
        self.assertEqual(diag["evidence_mode"], "3d_surrogate")
        self.assertEqual(diag["status"], "corrected")
        self.assertEqual(diag["observations"], 0)
        self.assertGreater(len(diag["accepted_bones"]), 0)
        self.assertLess(diag["cost_after"], diag["cost_before"])
        self.assertNotEqual(fused["left_wrist"], master["left_wrist"])
        self.assertEqual(fused["pelvis"], master["pelvis"])

    def test_opt_out_retains_strict_fallback(self):
        master, slave = fixture()
        fused, diag = optimize_aotr(master, slave,
            config={"missing_2d_mode": "fail_closed"})
        self.assertEqual(diag["status"], "fallback")
        self.assertEqual(diag["reason"], "missing_intrinsics")
        self.assertEqual(fused, master)

    def test_bone_lengths_and_untouched_joints_are_preserved(self):
        master, slave = fixture()
        fused, diag = optimize_aotr_3d(master, slave)
        for child, parent in (
            ("left_wrist", "left_elbow"),
            ("left_elbow", "left_shoulder"),
            ("right_wrist", "right_elbow"),
        ):
            a = np.linalg.norm(np.asarray(master[child]) - np.asarray(master[parent]))
            b = np.linalg.norm(np.asarray(fused[child]) - np.asarray(fused[parent]))
            self.assertAlmostEqual(a, b, places=8)
        self.assertEqual(fused["right_wrist"], master["right_wrist"])
        self.assertEqual(diag["joint_selection"]["right_wrist"], "KEEP")

    def test_master_occlusion_allows_large_legitimate_directional_correction(self):
        master, slave = fixture()
        # Compare identical pose pair except master-visibility signal.
        _, seen = optimize_aotr_3d(master, slave)
        _, hidden = optimize_aotr_3d(
            master, slave, vis_master={"left_wrist": False}
        )
        def alpha(diag, joint):
            return next((b["alpha"] for b in diag["accepted_bones"]
                        if b["joint"] == joint), 0.)
        self.assertGreater(alpha(hidden, "left_wrist"), alpha(seen, "left_wrist"))
        self.assertGreater(alpha(hidden, "left_wrist"), 0.5)

    def test_bad_alignment_is_rejected(self):
        master, slave = fixture()
        for j in ("neck", "right_hip", "left_hip", "right_shoulder"):
            slave[j] = (np.asarray(slave[j]) +
                        np.array([0.9, 0.8, 0.45])).tolist()
        pose, diag = optimize_aotr_3d(
            master, slave,
            config={"alignment_median_max_m": 0.01,
                    "alignment_q75_max_m": 0.015}
        )
        self.assertEqual(diag["status"], "fallback")
        self.assertIn(diag["reason"], {
            "insufficient_alignment_inliers", "unreliable_alignment"
        })
        self.assertEqual(pose, master)

    def test_temporal_inputs_are_supported_without_gt(self):
        master, slave = fixture()
        out, diag = optimize_aotr_3d(
            master, slave,
            prev_master=master, prev_slave=slave,
            prev_prev_master=master, prev_prev_slave=slave,
        )
        self.assertGreater(diag["temporal_evidence_bones"], 0)
        self.assertTrue(all(np.isfinite(v).all() for v in map(np.asarray, out.values())))


if __name__ == "__main__":
    unittest.main()

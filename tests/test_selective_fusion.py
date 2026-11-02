"""Regression tests for conservative, single-direction selective fusion.

These tests use synthetic poses only; benchmark gains require Harmony4D inputs.
"""
import unittest
import numpy as np

from fusion_pipeline.selective import (
    select_slave_joints, reference_bone_lengths, _safe_local_geometry,
)


def _baseline():
    return {
        "pelvis": [0., 0., 3.],
        "neck": [0., .50, 3.],
        "head": [0., .70, 3.],
        "left_shoulder": [-.20, .50, 3.],
        "right_shoulder": [.20, .50, 3.],
        "left_elbow": [-.40, .43, 3.],
        "right_elbow": [.40, .43, 3.],
        "left_wrist": [-.50, .30, 3.],
        "right_wrist": [.50, .30, 3.],
        "left_hand": [-.58, .26, 3.],
        "right_hand": [.58, .26, 3.],
        "left_hip": [-.11, -.04, 3.],
        "right_hip": [.11, -.04, 3.],
        "left_knee": [-.12, -.43, 3.10],
        "right_knee": [.12, -.43, 3.10],
        "left_ankle": [-.12, -.81, 3.10],
        "right_ankle": [.12, -.81, 3.10],
        "left_toe": [-.12, -.84, 3.30],
        "right_toe": [.12, -.84, 3.30],
        "left_foot": [-.12, -.86, 3.22],
        "right_foot": [.12, -.86, 3.22],
    }


def _select(master, slave, hidden_master=(), hidden_slave=(), **kwargs):
    visible_m = {j: j not in hidden_master for j in master}
    visible_s = {j: j not in hidden_slave for j in slave}
    return select_slave_joints(
        master, slave, (1.0, np.eye(3), np.zeros(3)),
        visible_m, visible_s,
        {j: 0.95 for j in master}, {j: 0.95 for j in slave},
        **kwargs
    )


class SelectiveFusionTests(unittest.TestCase):
    def test_equal_visible_quality_keeps_master_exact(self):
        master = _baseline()
        slave = _baseline()
        slave["left_hand"] = [-.56, .28, 3.]
        fused, modes, details = _select(master, slave)
        self.assertTrue(all(mode == "KEEP" for mode in modes.values()))
        self.assertEqual(fused, master)

    def test_slave_occluded_never_transfers(self):
        master, slave = _baseline(), _baseline()
        slave["left_hand"] = [-.45, .15, 3.]
        fused, modes, details = _select(
            master, slave, hidden_master={"left_hand"}, hidden_slave={"left_hand"}
        )
        self.assertEqual(modes["left_hand"], "KEEP")
        self.assertEqual(fused["left_hand"], master["left_hand"])

    def test_occluded_master_receives_valid_slave_hand(self):
        master, slave = _baseline(), _baseline()
        slave["left_hand"] = [-.52, .20, 3.]
        fused, modes, _ = _select(
            master, slave, hidden_master={"left_hand"},
            lengths=reference_bone_lengths([master]),
        )
        self.assertIn(modes["left_hand"], {"BLEND", "REPLACE"})
        self.assertNotEqual(fused["left_hand"], master["left_hand"])
        self.assertEqual(fused["right_hand"], master["right_hand"])

    def test_occluded_master_rejects_large_displacement(self):
        master, slave = _baseline(), _baseline()
        slave["left_hand"] = [1.5, 0.2, 3.]
        fused, modes, details = _select(master, slave, hidden_master={"left_hand"})
        self.assertEqual(modes["left_hand"], "KEEP")
        self.assertEqual(details["left_hand"], "large_displacement")
        self.assertEqual(fused["left_hand"], master["left_hand"])

    def test_visible_only_blends_with_independent_2d_advantage(self):
        master, slave = _baseline(), _baseline()
        master["head"] = [.10, .70, 3.]
        K = np.diag([1000., 1000., 1.])
        kp = [0., 1000. * .7 / 3., 1.]
        fused, modes, _ = _select(
            master, slave,
            keypoints2d_master={"head": kp},
            keypoints2d_slave={"head": kp},
            K_master=K, K_slave=K,
        )
        self.assertEqual(modes["head"], "BLEND")
        self.assertLess(abs(fused["head"][0]), abs(master["head"][0]))
        self.assertEqual(fused["left_wrist"], master["left_wrist"])

    def test_invalid_bone_length_proposal_is_rejected(self):
        master = _baseline()
        ref = reference_bone_lengths([master])
        self.assertFalse(_safe_local_geometry(master, [10., 10., 3.], "left_hand", ref, .15))

    def test_reference_bones_use_temporal_median(self):
        good = _baseline()
        bad = _baseline()
        bad["left_hand"] = [-1., .26, 3.]
        lengths = reference_bone_lengths([good, good, bad])
        expected = np.linalg.norm(np.array(good["left_hand"]) - np.array(good["left_wrist"]))
        self.assertAlmostEqual(lengths["left_hand"], expected, places=8)

    def test_no_ground_truth_dependency(self):
        import inspect
        import fusion_pipeline.selective as selective
        self.assertNotIn("ground_truth", inspect.getsource(selective))
        self.assertNotIn("gt_dir", inspect.getsource(selective))

if __name__ == "__main__":
    unittest.main()

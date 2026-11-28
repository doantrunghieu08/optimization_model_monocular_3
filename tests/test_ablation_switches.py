"""Ablation switches: defaults are unchanged, each switch removes exactly one component."""
import unittest

import numpy as np

from fusion_pipeline.adaptive_naive import fuse_adaptive_naive
from fusion_pipeline.aotr import optimize_aotr
from fusion_pipeline.aotr_3d import optimize_aotr_3d
from test_aotr import _fixture
from test_partwise import max_diff, skeleton

NO_STAB = {"sequence_stability": {"enabled": False}}


class AdaptiveNaiveTorsoGateTests(unittest.TestCase):
    def _broken_sequence(self):
        master = [skeleton() for _ in range(4)]
        broken = skeleton()
        broken["left_hip"] = broken["left_hip"] + np.array([0.6, 0.4, 0.0])
        broken["right_shoulder"] = broken["right_shoulder"] + np.array([-0.5, -0.6, 0.3])
        broken["neck"] = broken["neck"] + np.array([0.4, -0.5, 0.5])
        slave = [skeleton(elbow_lift=0.2), broken, skeleton(elbow_lift=0.2), skeleton(elbow_lift=0.2)]
        return master, slave

    def test_default_keeps_the_gate(self):
        master, slave = self._broken_sequence()
        out, diag = fuse_adaptive_naive(master, slave, dict(NO_STAB))
        self.assertTrue(diag["torso_gate"])
        self.assertEqual(diag["fallback_frames"], 1)
        self.assertLess(max_diff(out[1], master[1]), 1e-12)

    def test_no_gate_fuses_every_frame(self):
        master, slave = self._broken_sequence()
        out, diag = fuse_adaptive_naive(master, slave, {**NO_STAB, "torso_gate": False})
        self.assertFalse(diag["torso_gate"])
        self.assertEqual(diag["fallback_frames"], 0)
        self.assertEqual(diag["reason_counts"], {"aligned_all_joints": 4})
        self.assertGreater(max_diff(out[1], master[1]), 1e-3)   # the broken frame is now fused

    def test_no_gate_on_identical_views_returns_master(self):
        master = [skeleton() for _ in range(3)]
        out, _ = fuse_adaptive_naive(master, [dict(m) for m in master], {**NO_STAB, "torso_gate": False})
        for o, m in zip(out, master):
            self.assertLess(max_diff(o, m), 1e-9)


class AotrFixedAlphaTests(unittest.TestCase):
    def test_default_is_optimized(self):
        M, S, _, K, y1, y2, cfg = _fixture()
        _, diag = optimize_aotr(M, S, y1, y2, K, K, config=cfg)
        alphas = {b["joint"]: b["alpha"] for b in diag["accepted_bones"]}
        self.assertGreater(alphas.get("left_wrist", 0.0), 0.5)

    def test_fixed_alpha_skips_search_and_acceptance(self):
        M, S, _, K, y1, y2, cfg = _fixture()
        _, diag = optimize_aotr(M, S, y1, y2, K, K, config={**cfg, "fixed_alpha": 0.5})
        self.assertTrue(diag["accepted_bones"])
        self.assertTrue(all(b["alpha"] == 0.5 for b in diag["accepted_bones"]))

    def test_fixed_alpha_zero_returns_master(self):
        M, S, _, K, y1, y2, cfg = _fixture()
        out, diag = optimize_aotr(M, S, y1, y2, K, K, config={**cfg, "fixed_alpha": 0.0})
        self.assertEqual(diag["accepted_bones"], [])
        self.assertLess(max_diff(out, M), 1e-12)

    def test_fixed_alpha_in_3d_surrogate(self):
        M, S, *_ = _fixture()
        _, diag = optimize_aotr_3d(M, S, config={"fixed_alpha": 0.5})
        self.assertTrue(diag["accepted_bones"])
        self.assertTrue(all(abs(b["alpha"] - 0.5) < 1e-9 or b["alpha"] < 0.5 for b in diag["accepted_bones"]))


if __name__ == "__main__":
    unittest.main()

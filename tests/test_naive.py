"""Naive two-view fusion baseline (fusion.method=naive) and its benchmark identity."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion_pipeline.aotr_benchmark import evaluation_protocol, report_csv_name, verify_aotr_execution
from fusion_pipeline.naive import fuse_frame, fuse_naive, kabsch_rotation
from test_partwise import max_diff, rotate, skeleton


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class KabschTests(unittest.TestCase):
    def test_recovers_known_rotation(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(12, 3))
        R_true = _rot("y", 63) @ _rot("x", -25)
        R = kabsch_rotation(src, src @ R_true.T)
        self.assertLess(np.abs(R - R_true).max(), 1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)

    def test_never_returns_reflection(self):
        src = np.random.default_rng(1).normal(size=(10, 3))
        mirrored = src * np.array([-1.0, 1.0, 1.0])
        self.assertAlmostEqual(np.linalg.det(kabsch_rotation(src, mirrored)), 1.0, places=9)


class NaiveFusionTests(unittest.TestCase):
    def test_identical_views_return_master(self):
        m = skeleton()
        self.assertLess(max_diff(fuse_frame(m, m), m), 1e-12)

    def test_rigidly_moved_slave_returns_master(self):
        m = skeleton()
        s = rotate(m, 70, shift=(1.0, 0.3, 2.0))
        self.assertLess(max_diff(fuse_frame(m, s), m), 1e-9)

    def test_average_is_midpoint_after_alignment(self):
        m = skeleton()
        s_local = skeleton(elbow_lift=0.2)
        s = rotate(s_local, -40, shift=(0.5, 0.0, 1.0))
        fused = fuse_frame(m, s, 0.5)
        # Back in the Master frame the Slave equals s_local up to the Kabsch fit; the left
        # wrist should land close to the midpoint of the two estimates.
        mid = 0.5 * (m["left_wrist"] + s_local["left_wrist"])
        self.assertLess(np.linalg.norm(fused["left_wrist"] - mid), 0.03)
        self.assertLess(np.linalg.norm(fused["pelvis"] - m["pelvis"]), 1e-12)

    def test_weight_one_is_master_weight_zero_is_aligned_slave(self):
        m = skeleton()
        s = rotate(skeleton(elbow_lift=0.2), 25, shift=(0.0, 0.0, 1.5))
        self.assertLess(max_diff(fuse_frame(m, s, 1.0), m), 1e-12)
        aligned = fuse_frame(m, s, 0.0)
        # Weight 0 gives the rigidly aligned Slave: its shape (pairwise distances) is the Slave's.
        d_aligned = np.linalg.norm(aligned["left_wrist"] - aligned["right_ankle"])
        d_slave = np.linalg.norm(s["left_wrist"] - s["right_ankle"])
        self.assertAlmostEqual(d_aligned, d_slave, places=9)

    def test_joints_missing_in_slave_keep_master(self):
        m = skeleton()
        s = {k: v for k, v in rotate(m, 30).items() if k != "left_hand"}
        out = fuse_frame(m, s)
        self.assertLess(np.linalg.norm(out["left_hand"] - m["left_hand"]), 1e-9)

    def test_sequence_interface_and_validation(self):
        seq = [skeleton() for _ in range(4)]
        out, diag = fuse_naive(seq, seq, {"master_weight": 0.5})
        self.assertEqual(diag, {"mode": "naive", "frames": 4, "master_weight": 0.5})
        self.assertIsInstance(out[0]["pelvis"], list)
        with self.assertRaises(ValueError):
            fuse_naive(seq, seq[:3], {})
        with self.assertRaises(ValueError):
            fuse_naive(seq, seq, {"master_weight": 1.5})
        no_pelvis = [{k: v for k, v in p.items() if k != "pelvis"} for p in seq]
        with self.assertRaises(ValueError):
            fuse_naive(no_pelvis, seq, {})


def _config(tmp, method, weight=0.5):
    return {
        "inputs": {"target_person": "aria01"},
        "paths": {"fused_output_dir": str(tmp)},
        "fusion": {"method": method, "aotr": {"enabled": True},
                   "partwise": {"depth_logit": {"a": 0, "b": 1, "mu": 0, "sd": 1}},
                   "naive": {"master_weight": weight}},
        "evaluation": {"joint_set": "coco17", "mpjpe_alignment": "sequence"},
    }


class NaiveBenchmarkIdentityTests(unittest.TestCase):
    def test_protocol_and_report_are_distinct_per_method_and_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = {m: evaluation_protocol(_config(tmp, m)) for m in ("aotr", "partwise", "naive")}
            self.assertEqual(len(set(p.values())), 3)
            self.assertIn("naive_kabsch_average_v1", p["naive"])
            self.assertNotEqual(p["naive"], evaluation_protocol(_config(tmp, "naive", 0.7)))
            self.assertTrue(report_csv_name(_config(tmp, "naive")).startswith("brute_force_report_naive_"))

    def test_verify_requires_matching_naive_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, "naive")
            summary = {"mode": "naive", "frames": 5, "sequence_stability": {"status": "disabled"}}
            (Path(tmp) / "aotr_diagnostics.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_aotr_execution(cfg)
            (Path(tmp) / "naive_diagnostics.json").write_text(json.dumps({"frames": 4}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_aotr_execution(cfg)
            (Path(tmp) / "naive_diagnostics.json").write_text(json.dumps({"frames": 5}), encoding="utf-8")
            self.assertEqual(verify_aotr_execution(cfg)["mode"], "naive")


if __name__ == "__main__":
    unittest.main()

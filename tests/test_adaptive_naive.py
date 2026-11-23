"""Adaptive naive fusion (fusion.method=adaptive_naive) and its benchmark identity."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion_pipeline.adaptive_naive import (
    align, blend_frame, bone_confidence, bone_jitter, fuse_adaptive_naive, slave_weights,
)
from fusion_pipeline.aotr import _ordered_bones
from fusion_pipeline.aotr_benchmark import evaluation_protocol, report_csv_name, verify_aotr_execution
from fusion_pipeline.naive import fuse_naive
from test_partwise import max_diff, rotate, skeleton

NO_STAB = {"sequence_stability": {"enabled": False}}
STAB = {"sequence_stability": {"enabled": True, "temporal_stability": {
    "enabled": True, "shape_weight": 2.5, "residual_accel_weight": 1.0, "residual_velocity_weight": 0.5,
    "minimum_retention": 0.8, "smooth_trajectories": True, "smooth_window": 7, "max_iter": 160},
    "anatomical_calibration": {"enabled": False}}}


def kps(pose, score):
    """2D record {joint: [x, y, score]} with one score, or a {joint: score} dict."""
    scores = score if isinstance(score, dict) else {j: score for j in pose}
    return {j: [0.0, 0.0, scores.get(j, 0.9)] for j in pose}


def lengths(pose):
    return {c: float(np.linalg.norm(np.asarray(pose[c]) - np.asarray(pose[p]))) for c, p in _ordered_bones(pose)}


class WeightTests(unittest.TestCase):
    def test_equal_quality_is_plain_average(self):
        self.assertAlmostEqual(slave_weights(0.8, 0.8, 1.0, 1.0, 1.0, 1.0), 0.5)

    def test_higher_slave_confidence_or_lower_jitter_raises_slave_weight(self):
        self.assertGreater(slave_weights(0.3, 0.9, 1.0, 1.0, 1.0, 1.0), 0.5)
        self.assertGreater(slave_weights(0.8, 0.8, 3.0, 0.5, 1.0, 1.0), 0.5)
        self.assertLess(slave_weights(0.9, 0.3, 1.0, 1.0, 1.0, 1.0), 0.5)

    def test_degenerate_settings_reduce_to_fixed_naive(self):
        # p = 0 ignores confidence; a huge tau ignores jitter.
        self.assertAlmostEqual(slave_weights(0.1, 0.9, 0.2, 5.0, 0.0, 1e9), 0.5, places=6)
        self.assertAlmostEqual(slave_weights(0.0, 0.0, 1.0, 1.0, 1.0, 1.0), 0.5)

    def test_bone_confidence_uses_direct_2d_endpoints_only(self):
        p = skeleton()
        k = kps(p, {"left_elbow": 0.4, "left_wrist": 0.9, "left_hand": 0.1})
        self.assertAlmostEqual(bone_confidence(k, "left_wrist", "left_elbow", 0.5), 0.4)
        # left_hand is a copied alias, not 2D evidence: only the wrist score counts.
        self.assertAlmostEqual(bone_confidence(k, "left_hand", "left_wrist", 0.5), 0.9)
        self.assertAlmostEqual(bone_confidence(None, "left_hand", "left_wrist", 0.5), 0.5)
        self.assertAlmostEqual(bone_confidence({"left_wrist": [1, 2, float("nan")]}, "left_hand", "left_wrist", 0.3), 0.3)


class JitterTests(unittest.TestCase):
    def test_static_bone_has_zero_jitter_and_shaking_bone_has_more(self):
        rng = np.random.default_rng(0)
        steady = [skeleton() for _ in range(12)]
        shaky = []
        for _ in range(12):
            p = skeleton()
            p["left_wrist"] = p["left_wrist"] + rng.normal(scale=0.05, size=3)
            shaky.append(p)
        bones = _ordered_bones(steady[0])
        js, jk = bone_jitter(steady, bones, 2), bone_jitter(shaky, bones, 2)
        self.assertLess(np.nanmax(js["left_wrist"]), 1e-12)
        self.assertGreater(np.nanmean(jk["left_wrist"]), 0.05)


class FusionTests(unittest.TestCase):
    def test_identical_views_return_master(self):
        seq = [skeleton() for _ in range(6)]
        out, diag = fuse_adaptive_naive(seq, seq, NO_STAB)
        self.assertLess(max(max_diff(o, m) for o, m in zip(out, seq)), 1e-9)
        self.assertEqual(diag["fallback_frames"], 0)

    def test_rigidly_moved_slave_returns_master(self):
        master = [skeleton() for _ in range(6)]
        slave = [rotate(p, 70, shift=(1.0, 0.3, 2.0)) for p in master]
        out, _ = fuse_adaptive_naive(master, slave, NO_STAB)
        self.assertLess(max(max_diff(o, m) for o, m in zip(out, master)), 1e-9)

    def test_master_bone_lengths_pelvis_and_neck_are_kept(self):
        master = [skeleton() for _ in range(6)]
        slave = [rotate(skeleton(elbow_lift=0.2), 30) for _ in range(6)]
        out, _ = fuse_adaptive_naive(master, slave, NO_STAB)
        for o, m in zip(out, master):
            for bone, length in lengths(m).items():
                self.assertAlmostEqual(lengths(o)[bone], length, places=9)
            for j in ("pelvis", "neck"):
                self.assertLess(np.linalg.norm(np.asarray(o[j]) - m[j]), 1e-12)

    def test_confident_slave_pulls_the_arm_towards_slave(self):
        master = [skeleton() for _ in range(6)]
        slave_local = skeleton(elbow_lift=0.2)
        slave = [rotate(slave_local, -40, shift=(0.5, 0.0, 1.0)) for _ in range(6)]
        arm = {"left_elbow": 0.2, "left_wrist": 0.2}
        lo = [kps(m, {**{j: 0.9 for j in m}, **arm}) for m in master]
        hi = [kps(s, 0.9) for s in slave]
        trust_slave, d1 = fuse_adaptive_naive(master, slave, NO_STAB, conf_master=lo, conf_slave=hi)
        trust_master, _ = fuse_adaptive_naive(master, slave, NO_STAB, conf_master=hi, conf_slave=lo)
        target = slave_local["left_wrist"]
        err_s = np.linalg.norm(np.asarray(trust_slave[0]["left_wrist"]) - target)
        err_m = np.linalg.norm(np.asarray(trust_master[0]["left_wrist"]) - target)
        self.assertLess(err_s, err_m)
        self.assertGreater(d1["slave_weight_p90"], 0.5)

    def test_failed_alignment_keeps_master_and_is_counted(self):
        master = [skeleton() for _ in range(4)]
        broken = skeleton()
        broken["left_hip"] = broken["left_hip"] + np.array([0.6, 0.4, 0.0])
        broken["right_shoulder"] = broken["right_shoulder"] + np.array([-0.5, -0.6, 0.3])
        broken["neck"] = broken["neck"] + np.array([0.4, -0.5, 0.5])
        slave = [skeleton(elbow_lift=0.2), broken, skeleton(elbow_lift=0.2), skeleton(elbow_lift=0.2)]
        R, reason = align(master[1], broken, {"ransac_threshold_m": 0.06, "ransac_max_combos": 100,
                                              "alignment_median_max_m": 0.12, "alignment_q75_max_m": 0.18})
        self.assertIsNone(R)
        out, diag = fuse_adaptive_naive(master, slave, NO_STAB)
        self.assertLess(max_diff(out[1], master[1]), 1e-12)
        self.assertEqual(diag["fallback_frames"], 1)
        self.assertEqual(diag["aligned_frames"], 3)

    def test_opposite_bone_directions_keep_master(self):
        m = skeleton()
        s = skeleton()
        s["left_wrist"] = s["left_elbow"] - (s["left_wrist"] - s["left_elbow"])
        s["left_hand"] = s["left_wrist"]
        out, rejected = blend_frame(m, s, np.eye(3), {c: 1.0 for c, _ in _ordered_bones(m)}, -0.8)
        self.assertEqual(rejected, 1)
        self.assertLess(np.linalg.norm(out["left_wrist"] - m["left_wrist"]), 1e-12)

    def test_stability_pass_runs_and_keeps_lengths(self):
        rng = np.random.default_rng(3)
        master = [skeleton() for _ in range(15)]
        slave = []
        for _ in range(15):
            p = skeleton(elbow_lift=0.15)
            p["left_wrist"] = p["left_wrist"] + rng.normal(scale=0.02, size=3)
            slave.append(rotate(p, 20))
        out, diag = fuse_adaptive_naive(master, slave, STAB)
        self.assertEqual(diag["sequence_stability"]["status"], "applied")
        for o, m in zip(out, master):
            for bone, length in lengths(m).items():
                self.assertAlmostEqual(lengths(o)[bone], length, places=6)

    def test_validation(self):
        seq = [skeleton() for _ in range(3)]
        with self.assertRaises(ValueError):
            fuse_adaptive_naive(seq, seq[:2], NO_STAB)
        with self.assertRaises(ValueError):
            fuse_adaptive_naive(seq, seq, {**NO_STAB, "jitter_scale": 0.0})

    def test_naive_can_use_the_aotr_stability_pass(self):
        # A static sequence has no residual acceleration and the pass returns "unchanged";
        # a jittery Slave makes it act (savgol + FK onto the Master lengths).
        rng = np.random.default_rng(5)
        master = [skeleton() for _ in range(15)]
        slave = []
        for _ in range(15):
            p = skeleton(elbow_lift=0.2)
            p["left_wrist"] = p["left_wrist"] + rng.normal(scale=0.03, size=3)
            slave.append(rotate(p, 25))
        out, diag = fuse_naive(master, slave, {"master_weight": 0.5, **STAB})
        self.assertEqual(diag["sequence_stability"]["status"], "applied")
        for o, m in zip(out, master):  # FK projection gives the Master bone lengths back
            for bone, length in lengths(m).items():
                self.assertAlmostEqual(lengths(o)[bone], length, places=6)


def _config(tmp, method, adaptive=None, naive=None):
    return {
        "inputs": {"target_person": "aria01"},
        "paths": {"fused_output_dir": str(tmp)},
        "fusion": {"method": method, "aotr": {"enabled": True, "sequence_stability": {"enabled": True}},
                   "naive": naive or {"master_weight": 0.5},
                   "adaptive_naive": adaptive or {"confidence_power": 1.0, "jitter_scale": 1.0}},
        "evaluation": {"joint_set": "coco17", "mpjpe_alignment": "sequence"},
    }


class BenchmarkIdentityTests(unittest.TestCase):
    def test_protocol_is_distinct_and_follows_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = evaluation_protocol(_config(tmp, "adaptive_naive"))
            self.assertIn("adaptive_naive_conf_jitter_v1", p)
            self.assertNotEqual(p, evaluation_protocol(_config(tmp, "aotr")))
            self.assertNotEqual(p, evaluation_protocol(_config(tmp, "adaptive_naive", {"confidence_power": 2.0})))
            self.assertTrue(report_csv_name(_config(tmp, "adaptive_naive")).startswith(
                "brute_force_report_adaptive_naive_"))

    def test_naive_stability_flag_changes_protocol_only_when_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = evaluation_protocol(_config(tmp, "naive"))
            off = evaluation_protocol(_config(tmp, "naive", naive={"master_weight": 0.5, "use_aotr_stability": False}))
            on = evaluation_protocol(_config(tmp, "naive", naive={"master_weight": 0.5, "use_aotr_stability": True}))
            self.assertEqual(plain, off)
            self.assertNotEqual(plain, on)

    def test_verify_requires_diagnostics_aligned_frames_and_stability(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, "adaptive_naive")
            summary = {"mode": "adaptive_naive", "frames": 5}
            (Path(tmp) / "aotr_diagnostics.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_aotr_execution(cfg)
            diag_path = Path(tmp) / "adaptive_naive_diagnostics.json"
            for bad in ({"frames": 4, "fallback_frames": 0, "sequence_stability": {"status": "applied"}},
                        {"frames": 5, "fallback_frames": 5, "sequence_stability": {"status": "applied"}},
                        {"frames": 5, "fallback_frames": 0, "sequence_stability": {"status": "disabled"}}):
                diag_path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    verify_aotr_execution(cfg)
            diag_path.write_text(json.dumps({"frames": 5, "fallback_frames": 1,
                                             "sequence_stability": {"status": "applied"}}), encoding="utf-8")
            self.assertEqual(verify_aotr_execution(cfg)["mode"], "adaptive_naive")


if __name__ == "__main__":
    unittest.main()

"""Part-wise symmetric fusion (fusion.method=partwise) and its benchmark identity."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion_pipeline.aotr import _ordered_bones
from fusion_pipeline.aotr_benchmark import evaluation_protocol, report_csv_name, verify_aotr_execution
from fusion_pipeline.partwise import blend_frame, fuse_partwise, master_weights, slave_directions

LOGIT = {"a": 0.0, "b": 1.7, "mu": 0.0, "sd": 1.4}
CFG = {"depth_logit": LOGIT, "rotation_half_window": 4, "sequence_stability": {"enabled": False}}


def skeleton(depth=4.0, elbow_lift=0.0):
    p = {
        "pelvis": [0.0, -0.10, 0.0], "neck": [0.0, 0.50, 0.0], "head": [0.0, 0.70, 0.0],
        "left_hip": [-0.12, -0.12, 0.0], "right_hip": [0.12, -0.12, 0.0],
        "left_knee": [-0.13, -0.55, 0.02], "right_knee": [0.13, -0.55, 0.02],
        "left_ankle": [-0.13, -0.95, 0.0], "right_ankle": [0.13, -0.95, 0.0],
        "left_foot": [-0.13, -1.00, 0.08], "right_foot": [0.13, -1.00, 0.08],
        "left_toe": [-0.13, -1.00, 0.14], "right_toe": [0.13, -1.00, 0.14],
        "left_shoulder": [-0.20, 0.45, 0.0], "right_shoulder": [0.20, 0.45, 0.0],
        "left_elbow": [-0.45, 0.30 + elbow_lift, 0.05], "right_elbow": [0.45, 0.30, 0.05],
        "left_wrist": [-0.60, 0.10 + elbow_lift, 0.15], "right_wrist": [0.60, 0.10, 0.15],
        "left_hand": [-0.65, 0.05 + elbow_lift, 0.18], "right_hand": [0.65, 0.05, 0.18],
    }
    return {k: np.asarray(v) + np.array([0.0, 0.0, depth]) for k, v in p.items()}


def rotate(pose, deg, shift=(0.0, 0.0, 0.0)):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c = pose["pelvis"]
    return {k: R @ (v - c) + c + np.asarray(shift) for k, v in pose.items()}


def max_diff(a, b):
    return max(float(np.linalg.norm(np.asarray(a[k]) - np.asarray(b[k]))) for k in a)


class PartwiseFusionTests(unittest.TestCase):
    def test_identical_views_return_master(self):
        seq = [skeleton() for _ in range(6)]
        out, _ = fuse_partwise(seq, seq, CFG)
        self.assertLess(max(max_diff(o, m) for o, m in zip(out, seq)), 1e-9)

    def test_rigidly_moved_slave_is_equivalent_to_master(self):
        master = [skeleton() for _ in range(6)]
        slave = [rotate(p, 70, shift=(1.0, 0.0, 2.0)) for p in master]
        dirs = slave_directions(master, slave, 4)
        for m, d in zip(master, dirs):
            self.assertLess(max_diff(blend_frame(m, d, 0.0), m), 1e-9)

    def test_master_bone_lengths_are_preserved(self):
        master = [skeleton() for _ in range(6)]
        slave = [rotate(skeleton(elbow_lift=0.15), 30) for _ in range(6)]
        out, _ = fuse_partwise(master, slave, CFG)
        for o, m in zip(out, master):
            o = {k: np.asarray(v) for k, v in o.items()}
            for child, parent in _ordered_bones(m):
                self.assertAlmostEqual(np.linalg.norm(o[child] - o[parent]),
                                       np.linalg.norm(m[child] - m[parent]), places=9)

    def test_slave_limb_pose_is_used_relative_to_its_torso(self):
        master = [skeleton() for _ in range(6)]
        slave = [rotate(skeleton(elbow_lift=0.15), 40) for _ in range(6)]
        dirs = slave_directions(master, slave, 4)
        fused = blend_frame(master[0], dirs[0], 0.0)  # w=0: take slave directions
        target = skeleton(elbow_lift=0.15)
        for child, parent in (("left_elbow", "left_shoulder"), ("left_wrist", "left_elbow")):
            u = fused[child] - fused[parent]
            v = target[child] - target[parent]
            # Direction comes from the slave (in Master frame); length stays Master's.
            self.assertLess(np.linalg.norm(u / np.linalg.norm(u) - v / np.linalg.norm(v)), 1e-9)

    def test_farther_view_gets_more_weight(self):
        near, far = [skeleton(depth=3.0)], [skeleton(depth=6.0)]
        self.assertGreater(master_weights(far, near, LOGIT)[0], 0.5)
        self.assertLess(master_weights(near, far, LOGIT)[0], 0.5)
        self.assertAlmostEqual(master_weights(near, near, LOGIT)[0], 0.5, places=9)

    def test_degenerate_bone_moves_with_parent(self):
        m = skeleton()
        m["left_hand"] = m["left_wrist"].copy()
        s = skeleton(elbow_lift=0.2)
        out = blend_frame(m, slave_directions([m], [s], 0)[0], 0.0)
        self.assertLess(np.linalg.norm(out["left_hand"] - out["left_wrist"]), 1e-12)

    def test_sequence_stability_runs_when_enabled(self):
        master = [skeleton() for _ in range(12)]
        slave = [rotate(skeleton(elbow_lift=0.1 * (i % 2)), 20) for i in range(12)]
        cfg = dict(CFG, sequence_stability={"enabled": True, "temporal_stability": {"enabled": True},
                                            "anatomical_calibration": {"enabled": False}})
        _, diag = fuse_partwise(master, slave, cfg)
        self.assertIn(diag["sequence_stability"]["status"], {"applied", "unchanged"})


class PartwiseBenchmarkIdentityTests(unittest.TestCase):
    def cfg(self, method):
        return {"fusion": {"method": method, "aotr": {"enabled": True}, "partwise": CFG},
                "inputs": {"target_person": "aria02"}}

    def test_protocol_and_report_distinguish_methods(self):
        a, p = self.cfg("aotr"), self.cfg("partwise")
        self.assertNotEqual(evaluation_protocol(a), evaluation_protocol(p))
        self.assertTrue(report_csv_name(p).startswith("brute_force_report_partwise_"))
        self.assertTrue(report_csv_name(a).startswith("brute_force_report_aotr_"))

    def test_identity_mask_changes_protocol(self):
        a = self.cfg("aotr")
        b = self.cfg("aotr")
        b["evaluation"] = {"identity_mask": {"path": "configs/x.json", "margin_mm": 10.0}}
        self.assertNotEqual(evaluation_protocol(a), evaluation_protocol(b))

    def test_partwise_verification_requires_its_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg("partwise")
            cfg["paths"] = {"fused_output_dir": tmp}
            Path(tmp, "aotr_diagnostics.json").write_text(json.dumps({"mode": "partwise", "frames": 5}))
            with self.assertRaises(RuntimeError):
                verify_aotr_execution(cfg)
            Path(tmp, "partwise_diagnostics.json").write_text(json.dumps(
                {"frames": 5, "sequence_stability": {"status": "applied"}}))
            self.assertEqual(verify_aotr_execution(cfg)["mode"], "partwise")


class PerformanceOptimizationTests(unittest.TestCase):
    def test_joblib_cache_invalidates_on_file_update(self):
        import joblib
        from compat import load_joblib_compat
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.pkl"
            joblib.dump({"v": 1}, p)
            self.assertEqual(load_joblib_compat(p)["v"], 1)
            joblib.dump({"v": 2, "extra": "changed_size"}, p)
            self.assertEqual(load_joblib_compat(p)["v"], 2)

    def test_offset_estimation_skips_smpl_when_frame_ids_present(self):
        from unittest.mock import patch
        import joblib
        from preprocess_pipeline.offset_paper import compute_offset_from_pkls
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2 = Path(tmp) / "c1.pkl", Path(tmp) / "c2.pkl"
            payload = {0: {"pose": np.zeros((4, 72)), "betas": np.zeros((4, 10)), "frame_ids": np.arange(4)}}
            joblib.dump(payload, p1)
            joblib.dump(payload, p2)
            with patch("preprocess_pipeline.offset_paper.smplx.create", side_effect=AssertionError("should skip SMPL")):
                self.assertEqual(compute_offset_from_pkls(p1, p2, "unused", "unused", "unused"), 0)

    def test_skip_aotr_opt_preserves_visibility_without_running_aotr(self):
        from unittest.mock import patch
        from fusion_pipeline.executor import run_phase3_pipeline
        from keypoints_map import load_keypoints3d_map
        names = [k["name"] for k in load_keypoints3d_map("configs/keypoints3D_map.yml")["keypoints"]]
        pose = {k: [float(i) * 0.1, 0.2, 3.0] for i, k in enumerate(names)}
        with patch("fusion_pipeline.executor.optimize_aotr", side_effect=AssertionError("should skip optimize_aotr")):
            res = run_phase3_pipeline(
                {"camera1": pose, "camera2": pose}, "configs/keypoints3D_map.yml",
                0.05, 0.05, 100, 0.001, 0.8,
                confidence2d_by_cam={"camera1": {"left_wrist": [0.0, 0.0, 0.1]}, "camera2": {}},
                aotr_cfg={"enabled": True}, skip_aotr_opt=True,
            )
        self.assertFalse(res["vis1"]["left_wrist"])
        self.assertTrue(res["vis2"]["left_wrist"])


if __name__ == "__main__":
    unittest.main()

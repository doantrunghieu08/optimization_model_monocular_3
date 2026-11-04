"""Offline AOTR residual-acceleration and limb-length calibration tests.

These are synthetic deterministic tests, NOT Harmony4D accuracy evidence.
"""
import unittest
import numpy as np

from fusion_pipeline.aotr_stability import (
    _residual_acceleration, _temporal_scales, postprocess_aotr_3d,
)


def _frames(n=17):
    raw, fused = [], []
    for t in range(n):
        # Real Master translation and articulation must be preserved.
        movement = np.array([0.007*t, 0.02*np.sin(t/5.0), 0.0])
        m = {
            "pelvis": [0.0, 0.0, 5.0],
            "neck": [0.0, 0.7, 5.0],
            "left_shoulder": [-0.25, 0.5, 5.0],
            "left_elbow": [-0.48, 0.29, 5.0],
            "left_wrist": [-0.68, 0.06, 5.0],
            "right_shoulder": [0.25, 0.5, 5.0],
            "right_elbow": [0.48, 0.29, 5.0],
            "right_wrist": [0.68, 0.06, 5.0],
            "left_hip": [-0.17, -0.11, 5.0],
            "left_knee": [-0.17, -0.65, 5.0],
            "left_ankle": [-0.17, -1.17, 5.0],
            "right_hip": [0.17, -0.11, 5.0],
            "right_knee": [0.17, -0.65, 5.0],
            "right_ankle": [0.17, -1.17, 5.0],
        }
        m = {j: (np.asarray(p)+movement).tolist() for j, p in m.items()}
        f = {j: list(p) for j,p in m.items()}
        # Wiggly but constant bone length AOTR orientation correction.
        e = np.asarray(m["left_elbow"])
        w = np.asarray(m["left_wrist"])
        length = np.linalg.norm(w-e)
        direction = w-e + np.array([0.11*(-1)**t, 0.0, 0.05])
        direction *= length/np.linalg.norm(direction)
        f["left_wrist"] = (e+direction).tolist()
        raw.append(m)
        fused.append(f)
    return raw, fused


def _stack(frames):
    joints = sorted(frames[0])
    return np.array([[f[j] for j in joints] for f in frames])


class AOTRStabilityTests(unittest.TestCase):
    def test_postprocessing_reduces_accel_of_correction_and_preserves_bones(self):
        raw, fused = _frames()
        before = _residual_acceleration(_stack(fused)-_stack(raw))
        output, summary = postprocess_aotr_3d(raw, fused, {
            "temporal_stability": {
                "enabled": True, "residual_accel_weight": 30.0,
                "residual_velocity_weight": 1.0,
                "minimum_retention": 0.05,
            },
            "anatomical_calibration": {"enabled": False},
        })
        after = _residual_acceleration(_stack(output)-_stack(raw))
        self.assertLess(after, before * 0.9)
        self.assertLess(summary["mean_temporal_retention"], 1.0)
        for m, f in zip(raw, output):
            for child, parent in [("left_wrist", "left_elbow"),
                                  ("right_wrist", "right_elbow"),
                                  ("right_elbow", "right_shoulder")]:
                a = np.linalg.norm(np.asarray(m[child])-np.asarray(m[parent]))
                b = np.linalg.norm(np.asarray(f[child])-np.asarray(f[parent]))
                self.assertAlmostEqual(a, b, places=8)
            self.assertEqual(m["pelvis"], f["pelvis"])
            self.assertEqual(m["right_wrist"], f["right_wrist"])

    def test_anatomical_limb_ratios_and_constant_lengths(self):
        raw, fused = _frames()
        for t, m in enumerate(raw):
            # Simulate per-frame length drift from WHAM.
            m["left_elbow"][0] -= 0.01 * np.sin(t)
            m["left_wrist"][1] -= 0.018 * np.cos(t)
        output, diagnostics = postprocess_aotr_3d(raw, fused, {
            "temporal_stability": {"enabled": False},
            "anatomical_calibration": {"enabled": True,
                                       "arm_upper_fraction": 0.54,
                                       "leg_thigh_fraction": 0.53},
        })
        self.assertEqual(diagnostics["anatomically_calibrated_bones"], 8)
        for side in ("left", "right"):
            for root, mid, tip, share in [
                (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", 0.54),
                (f"{side}_hip", f"{side}_knee", f"{side}_ankle", 0.53),
            ]:
                total_lengths, upper_lengths = [], []
                for pose in output:
                    a, b, c = (np.asarray(pose[j]) for j in (root, mid, tip))
                    upper, lower = np.linalg.norm(b-a), np.linalg.norm(c-b)
                    upper_lengths.append(upper)
                    total_lengths.append(upper+lower)
                    self.assertAlmostEqual(upper/(upper+lower), share, places=7)
                self.assertLess(float(np.std(total_lengths)), 1e-9)
                self.assertLess(float(np.std(upper_lengths)), 1e-9)

    def test_fallback_frames_remain_exact_raw_master(self):
        raw, fused = _frames()
        frozen = [False, True, False, True] + [True] * (len(raw)-4)
        # Unsafe frames already returned exact Master in the AOTR stage.
        for i, allowed in enumerate(frozen):
            if not allowed:
                fused[i] = {j: list(v) for j, v in raw[i].items()}
        output, diagnostics = postprocess_aotr_3d(raw, fused, {
            "temporal_stability": {"enabled": True},
            "anatomical_calibration": {"enabled": True}
        }, editable_mask=frozen)
        self.assertEqual(output[0], raw[0])
        self.assertEqual(output[2], raw[2])
        self.assertEqual(diagnostics["frozen_fallback_frames"], 2)

    def test_exact_noop_when_both_mechanisms_disabled(self):
        raw, fused = _frames()
        output, d = postprocess_aotr_3d(raw, fused, {
            "temporal_stability": {"enabled": False},
            "anatomical_calibration": {"enabled": False},
        })
        self.assertEqual(output, fused)
        self.assertEqual(d["status"], "unchanged")

    def test_no_new_slave_transfer_at_master_only_frames(self):
        raw, fused = _frames()
        for i in (0, 4, 8, 12, 16):
            fused[i] = dict(raw[i])
        delta = _stack(fused) - _stack(raw)
        s, _ = _temporal_scales(delta, {"minimum_retention": 0.2})
        for i in (0, 4, 8, 12, 16):
            self.assertEqual(s[i], 0.0)
        self.assertTrue(np.all(s >= 0.0))
        self.assertTrue(np.all(s <= 1.0))
        self.assertTrue(np.all(s[np.sum(delta**2,axis=(1,2))>1e-14] >= 0.2))

    def test_aotr_skips_legacy_smoothing_and_anatomy_rewrite(self):
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] /
                  "fusion_pipeline" / "executor.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        run_fusion = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_fusion"
        )
        def calls(node, function):
            return any(
                isinstance(n, ast.Call) and
                isinstance(n.func, ast.Name) and n.func.id == function
                for n in ast.walk(node)
            )
        smoothing = [
            n for n in ast.walk(run_fusion) if isinstance(n, ast.If)
            and any(calls(stmt, "_apply_temporal_smoothing") for stmt in n.body)
        ]
        self.assertEqual(len(smoothing), 1)
        self.assertIn("not aotr_enabled", ast.unparse(smoothing[0].test))
        # AOTR terminates after its own diagnostic emission; the unconditional
        # legacy _enforce_anatomical_limb_proportions loop comes later.
        branch = [
            n for n in ast.walk(run_fusion)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
            and n.test.id == "aotr_enabled" and
            any(isinstance(stmt, ast.Return) for stmt in n.body)
        ]
        self.assertTrue(branch)
        self.assertTrue(any(
            calls(stmt, "_enforce_anatomical_limb_proportions")
            for stmt in run_fusion.body
        ))

    def test_sequence_stability_also_runs_for_two_view_aotr(self):
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] /
                  "fusion_pipeline" / "executor.py").read_text(encoding="utf-8")
        run_fusion = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_fusion"
        )
        gates = [
            node for node in ast.walk(run_fusion)
            if isinstance(node, ast.If) and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "postprocess_aotr_3d"
                for child in ast.walk(node)
            )
        ]
        self.assertEqual(ast.unparse(gates[-1].test), "evidence")

    def test_anatomy_toggle_really_changes_lengths_before_evaluation(self):
        master, fused = _frames()
        baseline, _ = postprocess_aotr_3d(master, fused, {
            "temporal_stability": {"enabled": False},
            "anatomical_calibration": {"enabled": False},
        })
        modified, diagnostics = postprocess_aotr_3d(master, fused, {
            "temporal_stability": {"enabled": False},
            "anatomical_calibration": {"enabled": True,
                                       "arm_upper_fraction": 0.54,
                                       "leg_thigh_fraction": 0.53},
        })
        self.assertEqual(diagnostics["anatomically_calibrated_bones"], 8)
        def length(pose, child, parent):
            return float(np.linalg.norm(
                np.asarray(pose[child])-np.asarray(pose[parent])
            ))
        self.assertNotAlmostEqual(
            length(baseline[0], "left_elbow", "left_shoulder"),
            length(modified[0], "left_elbow", "left_shoulder"),
            places=4,
        )
        self.assertAlmostEqual(
            length(baseline[0], "left_elbow", "left_shoulder"),
            length(master[0], "left_elbow", "left_shoulder"),
            places=8,
        )

    def test_no_ground_truth_access(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] /
                  "fusion_pipeline" / "aotr_stability.py").read_text(encoding="utf-8")
        self.assertNotIn("evaluation_pipeline", source)
        self.assertNotIn("ground_truth_dir", source)
        self.assertNotIn("load_ground_truth", source)


if __name__ == "__main__":
    unittest.main()

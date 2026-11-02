"""Zero-GT Full Model v2: M4/M6 behavioral and M7 CPU smoke tests."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion_pipeline.adaptive_gate import apply_adaptive_joint_fusion
from fusion_pipeline.motion_quality import motion_aware_smooth, synchronized_segment_ids
from fusion_pipeline.optimization import optimize_sequence_gpu


class FullModelV2Tests(unittest.TestCase):
    def gate(self, target_quality, source_quality, distance, config=None):
        cam1 = {"left_elbow": [0., 0., 0.], "left_wrist": [1., 0., 0.]}
        cam2 = {"left_elbow": [0., 0., 0.], "left_wrist": [1., distance, 0.]}
        identity = (1.0, np.eye(3), np.zeros(3))
        return apply_adaptive_joint_fusion(
            cam1, cam2, list(cam1), identity, identity,
            {"left_wrist": .9}, {"left_wrist": .9},
            {"left_wrist": True}, {"left_wrist": True},
            joint_quality={
                "camera1": {"left_wrist": target_quality},
                "camera2": {"left_wrist": source_quality},
            },
            config=config or {},
        )

    def test_weak_visible_source_cannot_damage_strong_target(self):
        _, _, metadata = self.gate(1.0, .3, .25)
        self.assertEqual(metadata["joint_transfer"]["camera1"]["left_wrist"], 0)
        self.assertGreater(metadata["adaptive_stats"]["rejected_visible_risk"], 0)

    def test_equal_reliable_joint_still_contributes(self):
        a, _, metadata = self.gate(1.0, 1.0, .04)
        self.assertGreater(metadata["joint_transfer"]["camera1"]["left_wrist"], 0)
        self.assertGreater(a["left_wrist"][1], 0)

    def test_visible_risk_ablation_toggle_changes_output(self):
        _, _, enabled = self.gate(1.0, .3, .25)
        _, _, disabled = self.gate(1.0, .3, .25, {"visible_risk_enabled": False})
        self.assertEqual(enabled["joint_transfer"]["camera1"]["left_wrist"], 0)
        self.assertGreater(disabled["joint_transfer"]["camera1"]["left_wrist"], 0)

    def test_occluded_transfer_disabled_by_zero_cap(self):
        a = {"left_elbow": [0., 0., 0.], "left_wrist": [1., 0., 0.]}
        b = {"left_elbow": [0., 0., 0.], "left_wrist": [.9, .2, 0.]}
        identity = (1., np.eye(3), np.zeros(3))
        _, _, meta = apply_adaptive_joint_fusion(
            a, b, list(a), identity, identity,
            {"left_wrist": .8}, {"left_wrist": .95},
            {"left_wrist": False}, {"left_wrist": True},
            config={"max_occluded_transfer": 0.0},
        )
        self.assertEqual(meta["joint_transfer"]["camera1"]["left_wrist"], 0)

    def test_shared_fast_motion_is_preserved(self):
        raw = np.zeros((5, 1, 3))
        raw[:, 0, 0] = [0., .01, .5, .51, .52]
        still = np.zeros_like(raw)
        q = np.zeros((5, 1))
        shared, partner = motion_aware_smooth(
            raw, raw, q, q, motion_scale_m=.05,
        )
        isolated, _ = motion_aware_smooth(
            raw, still, q, q, motion_scale_m=.05,
        )
        self.assertLess(abs(shared[2, 0, 0] - raw[2, 0, 0]),
                        abs(isolated[2, 0, 0] - raw[2, 0, 0]))
        np.testing.assert_allclose(shared, partner)

    def test_no_smoothing_across_frame_gap(self):
        raw = np.zeros((6, 1, 3))
        raw[:, 0, 0] = [0., 0., 0., 10., 10., 10.]
        q = np.zeros((6, 1))
        smooth, _ = motion_aware_smooth(
            raw, raw, q, q, frame_ids=[0, 1, 2, 6, 7, 8],
        )
        np.testing.assert_allclose(smooth, raw, atol=1e-7)

    def test_wham_source_index_gap_is_segmented(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            sequence = [0, 1, 8, 9]
            for i, source_id in enumerate(sequence, 1):
                metadata = {"metadata": {"source_frame_indices": {
                    "camera1": source_id, "camera2": source_id,
                }}}
                (path / f"pose_data_{i}.json").write_text(json.dumps(metadata))
            files = [path / f"fused_data_{i}.json" for i in range(1, 5)]
            ids = synchronized_segment_ids(files, path)
            self.assertEqual(ids, [0, 1, 3, 4])

    def test_good_joint_smoothing_is_weaker(self):
        raw = np.zeros((5, 1, 3))
        raw[2, 0, 0] = .04
        still = np.zeros_like(raw)
        high, _ = motion_aware_smooth(raw, still, np.ones((5, 1)), np.ones((5, 1)))
        low, _ = motion_aware_smooth(raw, still, np.zeros((5, 1)), np.zeros((5, 1)))
        self.assertLess(abs(high[2, 0, 0] - raw[2, 0, 0]),
                        abs(low[2, 0, 0] - raw[2, 0, 0]))

    def test_m7_optimizer_cpu_with_observation_prior(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fused, meta, pose = root / "fused", root / "meta", root / "pose"
            for path in (fused, meta, pose):
                path.mkdir()
            for idx, x in enumerate((0.0, 0.5, 0.0), 1):
                name = f"fused_data_{idx}.json"
                joints = {"pelvis": [x, 0., 0.], "neck": [x, 1., 0.]}
                cameras = {"camera1": joints, "camera2": joints}
                (fused / name).write_text(json.dumps(cameras), encoding="utf-8")
                original = {
                    "camera1": {"pelvis": [0., 0., 0.], "neck": [0., 1., 0.]},
                    "camera2": {"pelvis": [0., 0., 0.], "neck": [0., 1., 0.]},
                }
                (pose / f"pose_data_{idx}.json").write_text(json.dumps(original), encoding="utf-8")
                info = {
                    "joint_confidence": {"camera1": {"pelvis": .99, "neck": .99},
                                         "camera2": {"pelvis": .99, "neck": .99}},
                    "joint_quality": {"camera1": {"pelvis": 1., "neck": 1.},
                                      "camera2": {"pelvis": 1., "neck": 1.}},
                    "joint_transfer": {"camera1": {}, "camera2": {}},
                }
                (meta / name).write_text(json.dumps(info), encoding="utf-8")
            cfg = {"fusion": {
                "adaptive_protection": {"enabled": True},
                "optimization": {"max_iter": 6, "regularization": True,
                                 "temporal_lambda": 2.0, "adaptive_objectives_enabled": True},
                "angle_limits": {"enabled": False},
            }}
            optimize_sequence_gpu(fused, meta, pose_dir=pose, config=cfg, device="cpu")
            result = json.loads((fused / "fused_data_2.json").read_text(encoding="utf-8"))
            self.assertTrue(np.isfinite(result["camera1"]["pelvis"]).all())


if __name__ == "__main__":
    unittest.main()

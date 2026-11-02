"""Regression checks for zero-GT joint/frame adaptive fusion."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion_pipeline.adaptive_gate import apply_adaptive_joint_fusion, temporal_joint_quality
from fusion_pipeline.optimization import optimize_sequence_gpu


class AdaptiveJointProtectionTests(unittest.TestCase):
    def gate(self, d=0.08, visible1=True, visible2=True, quality2=1.0):
        camera1 = {"left_elbow": [0.0, 0.0, 0.0], "left_wrist": [1.0, 0.0, 0.0]}
        camera2 = {"left_elbow": [0.0, 0.0, 0.0], "left_wrist": [1.0, d, 0.0]}
        identity = (1.0, np.eye(3), np.zeros(3))
        return apply_adaptive_joint_fusion(
            camera1, camera2, list(camera1), identity, identity,
            {"left_wrist": 0.9}, {"left_wrist": 0.9},
            {"left_wrist": visible1}, {"left_wrist": visible2},
            joint_quality={"camera2": {"left_wrist": quality2}},
        )

    def test_equal_belief_two_good_cameras_contribute(self):
        a, b, diagnostics = self.gate()
        self.assertGreater(a["left_wrist"][1], 0.0)
        self.assertLess(b["left_wrist"][1], 0.08)
        self.assertGreater(diagnostics["joint_transfer"]["camera1"]["left_wrist"], 0)
        self.assertAlmostEqual(np.linalg.norm(a["left_wrist"]), 1.0, places=5)

    def test_occluded_target_receives_more_transfer(self):
        _, _, visible = self.gate(visible1=True)
        _, _, occluded = self.gate(visible1=False)
        self.assertGreater(
            occluded["joint_transfer"]["camera1"]["left_wrist"],
            visible["joint_transfer"]["camera1"]["left_wrist"],
        )

    def test_geometry_softly_rejects_outlier(self):
        _, _, agree = self.gate(d=.08)
        _, _, disagree = self.gate(d=2.0)
        self.assertLess(
            disagree["joint_transfer"]["camera1"]["left_wrist"],
            .05 * agree["joint_transfer"]["camera1"]["left_wrist"],
        )

    def test_noisy_source_weight_decreases(self):
        _, _, reliable = self.gate(quality2=1)
        _, _, noisy = self.gate(quality2=.15)
        self.assertLess(
            noisy["joint_transfer"]["camera1"]["left_wrist"],
            reliable["joint_transfer"]["camera1"]["left_wrist"],
        )

    def test_temporal_quality_is_joint_and_frame_specific(self):
        frames = [
            {"camera1": {"left_elbow": [0, 0, 0], "left_wrist": [0, 0, 0]},
             "camera2": {"left_elbow": [0, 0, 0], "left_wrist": [x, 0, 0]}}
            for x in (0., 0., .3, 0., 0.)
        ]
        quality = temporal_joint_quality(frames)
        self.assertLess(quality[2]["camera2"]["left_wrist"], quality[2]["camera1"]["left_wrist"])
        self.assertEqual(quality[2]["camera2"]["left_elbow"], 1.0)
        indices = [
            {"camera1": i, "camera2": i} for i in (0, 1, 5, 6, 7)
        ]
        no_fake_acc = temporal_joint_quality(frames, source_indices=indices)
        self.assertEqual(no_fake_acc[2]["camera2"]["left_wrist"], 1.0)

    def test_gpu_high_confidence_joint_not_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fused_dir, meta_dir = root / "keypoints3d", root / "metadata"
            fused_dir.mkdir()
            meta_dir.mkdir()
            for idx, x in enumerate((0.0, 1.0, 0.0), 1):
                position = [x, 0.0, 0.0]
                joints = {"pelvis": position, "neck": [x, 1.0, 0.0]}
                pose = {"camera1": joints, "camera2": joints}
                metadata = {
                    "joint_confidence": {
                        "camera1": {"pelvis": 0.99, "neck": 0.99},
                        "camera2": {"pelvis": 0.99, "neck": 0.99},
                    },
                    "joint_quality": {
                        "camera1": {"pelvis": 1., "neck": 1.},
                        "camera2": {"pelvis": 1., "neck": 1.},
                    },
                    "joint_transfer": {
                        "camera1": {"pelvis": .5, "neck": .5},
                        "camera2": {"pelvis": .5, "neck": .5},
                    },
                }
                (fused_dir / f"fused_data_{idx}.json").write_text(json.dumps(pose), encoding="utf-8")
                (meta_dir / f"fused_data_{idx}.json").write_text(json.dumps(metadata), encoding="utf-8")
            config = {"fusion": {
                "adaptive_protection": {"enabled": True},
                "master_protection": {"enabled": True},
                "optimization": {"regularization": False, "max_iter": 12, "temporal_lambda": 2.0},
                "angle_limits": {"enabled": False},
            }}
            optimize_sequence_gpu(
                fused_dir, meta_dir, config=config, device="cpu", master_only=True
            )
            middle = json.loads((fused_dir / "fused_data_2.json").read_text(encoding="utf-8"))
            self.assertLess(middle["camera1"]["pelvis"][0], 1.0)
            self.assertLess(middle["camera2"]["pelvis"][0], 1.0)


if __name__ == "__main__":
    unittest.main()

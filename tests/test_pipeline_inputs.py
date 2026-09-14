import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from config_loader import load_config, validate_config
from pipeline import _evaluation_input_dirs


class EvaluationInputsTest(unittest.TestCase):
    def test_disabled_learnable_output_is_not_required(self):
        config = {
            "paths": {
                "pose_output_dir": "pose",
                "fused_output_dir": "fused",
                "learnable_output_dir": "learnable",
                "learnable_extra_output_dir": "learnable_extra",
            },
            "fusion": {"enabled": False},
            "learnable": {"enabled": False},
            "learnable_extra": {"enabled": True},
        }

        self.assertEqual(
            _evaluation_input_dirs(config),
            [
                Path("pose") / "keypoints3d",
                Path("learnable_extra") / "keypoints3d",
            ],
        )

    def test_learnable_requires_fusion(self):
        config = copy.deepcopy(load_config("configs/pipeline.yml"))
        config["fusion"]["enabled"] = False
        config["learnable"]["enabled"] = True

        with self.assertRaisesRegex(ValueError, "requires fusion"):
            validate_config(config)

    def test_fallback_ratio_is_bounded(self):
        config = copy.deepcopy(load_config("configs/pipeline.yml"))
        config["fusion"]["max_fallback_ratio"] = 1.1

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            validate_config(config)

    def test_regressive_fusion_stage_flags_are_booleans(self):
        config = copy.deepcopy(load_config("configs/pipeline.yml"))
        config["fusion"]["optimization"]["enabled"] = "false"

        with self.assertRaisesRegex(ValueError, "optimization.enabled must be a boolean"):
            validate_config(config)

    def test_ablation_environment_is_loaded(self):
        with patch.dict(os.environ, {
            "GLOBAL": "False",
            "LOCAL_METHOD": "optical_aware_belief",
            "KINEMATIC_CONSTRAINTS": "False",
        }, clear=True):
            config = load_config("configs/pipeline.yml")

        self.assertFalse(config["fusion"]["belief"]["global"])
        self.assertEqual(config["fusion"]["belief"]["local_method"], "optical_aware_belief")
        self.assertFalse(config["fusion"]["optimization"]["use_kinematic_constraints"])

    def test_enabled_fusion_output_is_required(self):
        config = {
            "paths": {
                "pose_output_dir": "pose",
                "fused_output_dir": "fused",
                "learnable_output_dir": "learnable",
                "learnable_extra_output_dir": "learnable_extra",
            },
            "fusion": {"enabled": True},
            "learnable": {"enabled": False},
            "learnable_extra": {"enabled": False},
        }

        self.assertEqual(
            _evaluation_input_dirs(config),
            [Path("pose") / "keypoints3d", Path("fused") / "keypoints3d"],
        )


if __name__ == "__main__":
    unittest.main()

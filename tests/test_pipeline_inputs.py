import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.config_loader import load_config, set_env_from_filename, validate_config
from src.pipelines.orchestrator import _evaluation_input_dirs


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

    def test_correction_controls_are_validated(self):
        config = copy.deepcopy(load_config("configs/pipeline.yml"))
        config["fusion"]["correction"]["confidence_delta_cap"] = -1
        with self.assertRaisesRegex(ValueError, "confidence_delta_cap"):
            validate_config(config)

        config = copy.deepcopy(load_config("configs/pipeline.yml"))
        config["fusion"]["correction"]["blend_mode"] = "soft"
        with self.assertRaisesRegex(ValueError, "blend_mode"):
            validate_config(config)

    def test_ablation_environment_is_loaded(self):
        with patch.dict(os.environ, {
            "GLOBAL": "False",
            "LOCAL_METHOD": "optical_aware_belief",
            "KINEMATIC_CONSTRAINTS": "False",
            "LOSS_TYPE": "mse",
        }, clear=True):
            config = load_config("configs/pipeline.yml")

        self.assertFalse(config["fusion"]["belief"]["global"])
        self.assertEqual(config["fusion"]["belief"]["local_method"], "optical_aware_belief")
        self.assertFalse(config["fusion"]["optimization"]["use_kinematic_constraints"])
        self.assertEqual(config["fusion"]["optimization"]["loss_type"], "mse")

    def test_ablation_notebook_names_select_distinct_configs(self):
        cases = {
            "ablation_naive_global_kinematic_huber_alpha1E_1_beta85E_2.ipynb":
                ("naive_distance_belief", True, True, "huber"),
            "ablation_optical_global_kinematic_mse_alpha1E_1_beta85E_2.ipynb":
                ("optical_aware_belief", True, True, "mse"),
            "ablation_optical_global_unconstrained_huber_alpha1E_1_beta85E_2.ipynb":
                ("optical_aware_belief", True, False, "huber"),
            "ablation_optical_local_kinematic_huber_alpha1E_1_beta85E_2.ipynb":
                ("optical_aware_belief", False, True, "huber"),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename), patch.dict(os.environ, {}, clear=True):
                set_env_from_filename(filename)
                config = load_config("configs/pipeline.yml")
                belief = config["fusion"]["belief"]
                optimization = config["fusion"]["optimization"]
                self.assertEqual((belief["alpha"], belief["beta"]), (0.1, 0.85))
                self.assertEqual(
                    (belief["local_method"], belief["global"],
                     optimization["use_kinematic_constraints"], optimization["loss_type"]),
                    expected,
                )

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

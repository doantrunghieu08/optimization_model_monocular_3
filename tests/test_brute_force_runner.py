import copy
import unittest

from brute_force_runner import _build_report_rows, _matches_active_config
from config_loader import load_config


class BruteForceResumeTest(unittest.TestCase):
    def test_resume_requires_the_same_full_config(self):
        config = load_config("configs/pipeline.yml")
        result = {
            "alpha": config["fusion"]["belief"]["alpha"],
            "beta": config["fusion"]["belief"]["beta"],
            "global_belief": str(config["fusion"]["belief"]["global"]),
            "local_method": config["fusion"]["belief"]["local_method"],
            "kinematic_constraints": str(config["fusion"]["optimization"]["use_kinematic_constraints"]),
            "loss_type": config["fusion"]["optimization"]["loss_type"],
            "optimization_enabled": str(config["fusion"]["optimization"]["enabled"]),
            "orientation_correction": str(config["fusion"]["correction"]["orientation_enabled"]),
            "learnable_enabled": str(config["learnable"]["enabled"]),
            "learnable_extra_enabled": str(config["learnable_extra"]["enabled"]),
        }

        self.assertTrue(_matches_active_config(result, config))
        changed = copy.deepcopy(config)
        changed["fusion"]["belief"]["local_method"] = "optical_aware_belief"
        self.assertFalse(_matches_active_config(result, changed))

        header, row = _build_report_rows({"segment": [{"master": "m", **result}]}, [])
        self.assertEqual(len(header), len(row))
        self.assertIn("Optimization Enabled", header)


if __name__ == "__main__":
    unittest.main()

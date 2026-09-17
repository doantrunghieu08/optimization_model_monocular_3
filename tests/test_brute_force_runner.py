import copy
import unittest

from brute_force_runner import _build_report_rows, _get_header_indices, _matches_active_config, _parse_history_row
from config_loader import load_config


class BruteForceResumeTest(unittest.TestCase):
    def test_history_row_keeps_report_identity(self):
        header = ["Set", "Segment", "Cam Master", "Cam Slave", "Alpha", "Beta", "MPJPE"]
        row = ["S1/Seq1", "seg_1", "video_5", "video_6", "0.1", "0.85", "N/A"]

        key, result = _parse_history_row(row, _get_header_indices(header))

        self.assertEqual(key, ("seg_1", "video_5", "video_6"))
        self.assertEqual(result["set"], "S1/Seq1")
        self.assertEqual(result["master"], "video_5")
        self.assertEqual(result["supplement"], "video_6")
        _build_report_rows({"seg_1": [result]}, [])

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
            "confidence_correction": str(config["fusion"]["correction"]["enabled"]),
            "orientation_correction": str(config["fusion"]["correction"]["orientation_enabled"]),
            "correction_blend": config["fusion"]["correction"]["blend_mode"],
            "alignment_mode": config["fusion"]["correction"]["alignment_mode"],
            "correction_selector": config["fusion"]["correction"]["selector"],
            "confidence_delta_cap": config["fusion"]["correction"]["confidence_delta_cap"],
            "learnable_enabled": str(config["learnable"]["enabled"]),
            "learnable_extra_enabled": str(config["learnable_extra"]["enabled"]),
        }

        self.assertTrue(_matches_active_config(result, config))
        changed = copy.deepcopy(config)
        changed["fusion"]["belief"]["local_method"] = "naive_distance_belief"
        self.assertFalse(_matches_active_config(result, changed))

        changed = copy.deepcopy(config)
        changed["fusion"]["correction"]["enabled"] = not config["fusion"]["correction"]["enabled"]
        self.assertFalse(_matches_active_config(result, changed))

        header, row = _build_report_rows({"segment": [{"master": "m", **result}]}, [])
        self.assertEqual(len(header), len(row))
        self.assertIn("Optimization Enabled", header)
        self.assertIn("Confidence Correction", header)


if __name__ == "__main__":
    unittest.main()

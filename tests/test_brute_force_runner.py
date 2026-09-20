import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from brute_force_runner import _build_report_rows, _get_header_indices, _matches_active_config, _parse_history_row, parse_visibility_mpjpe
from src.core.config_loader import load_config


class BruteForceResumeTest(unittest.TestCase):
    def test_visibility_mpjpe_splits_joint_frame_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            with open(root / "MPJPE_cam1.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Frame",
                    "fused_priority1_mm", "fused_neck_mm", "fused_left_wrist_mm",
                    "posed_neck_mm", "posed_left_wrist_mm",
                    "only_learnable_neck_mm", "only_learnable_left_wrist_mm",
                ])
                writer.writerow([1, 20, 10, 30, 12, 32, 8, 28])
                writer.writerow(["AVERAGE", 20, 10, 30, 12, 32, 8, 28])
            with open(metadata_dir / "fused_data_1.json", "w", encoding="utf-8") as f:
                json.dump({"vis1": {"neck": True, "left_wrist": False}}, f)

            occ, vis = parse_visibility_mpjpe(root / "MPJPE_cam1.csv", "fused", metadata_dir)
            baseline_occ, baseline_vis = parse_visibility_mpjpe(root / "MPJPE_cam1.csv", "posed", metadata_dir)
            learnable_occ, learnable_vis = parse_visibility_mpjpe(
                root / "MPJPE_cam1.csv", "only_learnable", metadata_dir
            )

        self.assertEqual(occ, 30.0)
        self.assertEqual(vis, 10.0)
        self.assertEqual((baseline_occ, baseline_vis), (32.0, 12.0))
        self.assertEqual((learnable_occ, learnable_vis), (28.0, 8.0))

    def test_history_row_keeps_report_identity(self):
        header = ["Set", "Segment", "Cam Master", "Cam Slave", "Alpha", "Beta", "MPJPE"]
        row = ["S1/Seq1", "seg_1", "video_5", "video_6", "0.1", "0.85", "N/A"]

        key, result = _parse_history_row(row, _get_header_indices(header))

        self.assertEqual(key, ("seg_1", "video_5", "video_6", "proposed"))
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
            "learnable_enabled": str(config["learnable"]["enabled"]),
            "learnable_extra_enabled": str(config["learnable_extra"]["enabled"]),
        }

        self.assertTrue(_matches_active_config(result, config))
        changed = copy.deepcopy(config)
        changed["fusion"]["belief"]["local_method"] = "naive_distance_belief"
        self.assertFalse(_matches_active_config(result, changed))

        header, row = _build_report_rows({"segment": [{"master": "m", **result}]})
        self.assertEqual(len(header), len(row))
        self.assertIn("Optimization Enabled", header)

        visibility_metrics = {
            "le_occ_mpjpe_master": 41.0,
            "le_vis_mpjpe_master": 21.0,
            "baseline_occ_mpjpe": 51.0,
            "baseline_vis_mpjpe": 31.0,
        }
        header, row = _build_report_rows({
            "segment": [{
                "master": "m", "supplement": "s", "fusion_method": "aligned_averaging",
                **result, **visibility_metrics,
            }]
        })
        key, parsed = _parse_history_row(row, _get_header_indices(header))
        self.assertEqual(key, ("segment", "m", "s", "aligned_averaging"))
        self.assertEqual(parsed["fusion_method"], "aligned_averaging")
        for metric, expected in visibility_metrics.items():
            self.assertEqual(parsed[metric], expected)


if __name__ == "__main__":
    unittest.main()

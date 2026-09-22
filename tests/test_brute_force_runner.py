import copy
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brute_force_runner import _active_config_signature, _build_report_rows, _get_header_indices, _get_sheet_data, _matches_active_config, _parse_history_row, _process_segment, generate_spreadsheet_report_safe, load_existing_csv_results, load_existing_spreadsheet_results, parse_visibility_mpjpe
from src.core.config_loader import load_config


class BruteForceResumeTest(unittest.TestCase):
    def test_google_sheet_resume_prefers_unfinished_worksheet(self):
        class Worksheet:
            def __init__(self, title, rows):
                self.title = title
                self.rows = rows

            def get_all_values(self):
                return self.rows

        class Spreadsheet:
            def worksheets(self):
                return [
                    Worksheet("run_in_progress", [["Segment", "Cam Master"], ["seg_1", "cam0"]]),
                    Worksheet("run_done", [["Segment", "Cam Master"], ["End", "End"]]),
                ]

        client = unittest.mock.Mock()
        client.open.return_value = Spreadsheet()
        with patch("brute_force_runner.get_gspread_client", return_value=client):
            header, rows, title, has_end = _get_sheet_data("report")

        self.assertEqual(header, ["Segment", "Cam Master"])
        self.assertEqual(rows, [["seg_1", "cam0"]])
        self.assertEqual(title, "run_in_progress")
        self.assertFalse(has_end)

    def test_google_sheet_resume_accepts_older_report_columns(self):
        sheet_data = (
            ["Segment", "Master", "Slave", "MPJPE (mm)"],
            [["seg_1", "cam0", "cam1", "12.5"]],
            "run_in_progress",
            False,
        )
        with patch("brute_force_runner._get_sheet_data", return_value=sheet_data):
            existing, title, has_end = load_existing_spreadsheet_results("report")

        self.assertIn(("seg_1", "cam0", "cam1", "proposed"), existing)
        self.assertEqual(title, "run_in_progress")
        self.assertFalse(has_end)

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
        method = "higher_belief_selection"
        method_config = copy.deepcopy(config)
        method_config["fusion"]["method"] = method
        result = {
            "config_signature": _active_config_signature(method_config),
            "fusion_method": method,
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
        changed = copy.deepcopy(config)
        changed["fusion"]["correction"]["selector"] = "belief"
        self.assertFalse(_matches_active_config(result, changed))

        header, row = _build_report_rows({"segment": [{"master": "m", **result}]})
        self.assertEqual(len(header), len(row))
        self.assertIn("Optimization Enabled", header)
        self.assertEqual(
            [column for column in (
                "Method", "All MPJPE", "Occ. MPJPE", "Vis. MPJPE",
                "PA-MPJPE", "MBLE", "Accel Error (mm/frame^2)",
                "Config Signature",
            ) if column not in header],
            [],
        )

        visibility_metrics = {
            "fusion_occ_mpjpe": 42.0,
            "fusion_vis_mpjpe": 22.0,
            "le_occ_mpjpe_master": 41.0,
            "le_vis_mpjpe_master": 21.0,
            "baseline_occ_mpjpe": 51.0,
            "baseline_vis_mpjpe": 31.0,
        }
        header, row = _build_report_rows({
            "segment": [{
                "master": "m", "supplement": "s", **result,
                "fusion_method": "aligned_averaging", **visibility_metrics,
            }]
        })
        key, parsed = _parse_history_row(row, _get_header_indices(header))
        self.assertEqual(key, ("segment", "m", "s", "aligned_averaging"))
        self.assertEqual(parsed["fusion_method"], "aligned_averaging")
        for metric, expected in visibility_metrics.items():
            self.assertEqual(parsed[metric], expected)

    def test_local_report_marks_completion(self):
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                generate_spreadsheet_report_safe(
                    {"seg_1": [{"master": "cam0", "supplement": "cam1", "mpjpe": 1.0}]},
                    "resume_test",
                    silent=True,
                    is_final=True,
                )
                _, has_end_marker = load_existing_csv_results(Path("output/reports/resume_test.csv"))
            finally:
                os.chdir(old_cwd)
        self.assertTrue(has_end_marker)

    def test_checkpoints_after_each_completed_pair(self):
        segment = {
            "name": "seg_1",
            "ground_truth_dir": "gt",
            "cameras": [{"id": "cam0"}, {"id": "cam1"}],
        }

        def result(cam_a, cam_b, *_args):
            return {
                "master": cam_a["id"], "supplement": cam_b["id"],
                "fusion_method": "proposed", "mpjpe": 1.0,
            }

        with patch("brute_force_runner._evaluate_camera_pair", side_effect=result), patch(
            "brute_force_runner.generate_spreadsheet_report_safe"
        ) as save:
            _process_segment(
                segment, {}, {"fusion": {}}, ["proposed"], Path("."),
                "report", "run", {}, 0, 2,
            )

        self.assertEqual(save.call_count, 2)


if __name__ == "__main__":
    unittest.main()

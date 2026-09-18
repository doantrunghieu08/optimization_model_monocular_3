import unittest
from pathlib import Path

import numpy as np

from src.pipelines.evaluation.executor import (
    _compute_pa_mpjpe,
    _compute_pck,
    _parse_new_gt,
    _validate_fusion_config,
    _validate_pose_sources,
)


class ProcrustesMetricTest(unittest.TestCase):
    def test_similarity_is_removed_but_reflection_is_not(self):
        keys = ["a", "b", "c", "d"]
        truth_array = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        truth = dict(zip(keys, truth_array))
        transformed = dict(zip(keys, 2.5 * truth_array @ rotation + 7.0))
        reflected = dict(zip(keys, truth_array * np.array([-1.0, 1.0, 1.0])))

        self.assertLess(_compute_pa_mpjpe(transformed, truth, keys)[0], 1e-9)
        self.assertGreater(_compute_pa_mpjpe(reflected, truth, keys)[0], 1.0)

    def test_ground_truth_requires_complete_finite_pose(self):
        with self.assertRaisesRegex(ValueError, r"shape \(28, 3\)"):
            _parse_new_gt({"pose3d": [[0, 0, 0]] * 27}, set(), {})

        invalid = np.zeros((28, 3))
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite GT"):
            _parse_new_gt({"pose3d": invalid}, set(), {})

    def test_collapsed_prediction_is_not_a_perfect_pa_mpjpe(self):
        keys = ["a", "b", "c"]
        pred = {key: np.zeros(3) for key in keys}
        truth = dict(zip(keys, np.eye(3)))

        self.assertGreater(_compute_pa_mpjpe(pred, truth, keys)[0], 0.0)

    def test_pck_reports_percentage_below_threshold_after_root_alignment(self):
        pred = {
            "left_hip": np.array([-0.1, 0.0, 0.0]),
            "right_hip": np.array([0.1, 0.0, 0.0]),
            "a": np.array([0.1, 0.0, 0.0]),
            "b": np.array([0.2, 0.0, 0.0]),
        }
        truth = {
            "left_hip": np.array([0.9, 0.0, 0.0]),
            "right_hip": np.array([1.1, 0.0, 0.0]),
            "a": np.array([1.2, 0.0, 0.0]),
            "b": np.array([1.5, 0.0, 0.0]),
        }

        mean, details = _compute_pck(pred, truth, ["a", "b"], threshold_mm=150.0)

        self.assertEqual(mean, 50.0)
        self.assertEqual(details, {"a": 100.0, "b": 0.0})

    def test_pose_sources_must_match_evaluation_inputs(self):
        expected = {"camera1": "video_0_seg_1", "camera2": "video_2_seg_1"}
        metadata = {"metadata": {"source_pkl_stems": expected}}

        _validate_pose_sources(metadata, expected, Path("pose_data_1.json"))
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            _validate_pose_sources({}, expected, Path("pose_data_1.json"))

    def test_fusion_config_must_match_evaluation_config(self):
        expected = {"enabled": True, "belief": {"alpha": 0.001, "beta": 0.8}}
        metadata = {"metadata": {"fusion_config": expected}}

        _validate_fusion_config(metadata, expected, Path("fused_data_1.json"))
        with self.assertRaisesRegex(ValueError, "Fusion config mismatch"):
            _validate_fusion_config(metadata, {**expected, "max_fallback_ratio": 0.0}, Path("fused_data_1.json"))


if __name__ == "__main__":
    unittest.main()

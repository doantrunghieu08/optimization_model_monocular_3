import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fusion_pipeline.executor import (
    _frame_confidence_from_profile,
    _load_pose_frame,
    run_phase3_pipeline,
    run_fusion,
)


def _config(root: Path, max_fallback_ratio=0.0):
    return {
        "runtime": {"clean_output": True},
        "paths": {
            "pose_output_dir": str(root / "pose"),
            "fused_output_dir": str(root / "fused"),
            "preprocess_output_dir": str(root / "preprocess"),
            "keypoints3d_map": "configs/keypoints3D_map.yml",
        },
        "fusion": {
            "enabled": True,
            "max_fallback_ratio": max_fallback_ratio,
            "belief": {"alpha": 0.001, "beta": 0.8},
            "occlusion": {"enabled": False, "tau": 0.01},
            "ransac": {"threshold": 0.05, "max_combos": 10},
            "correction": {"orientation_enabled": False, "reject_new_mismatches": True},
            "optimization": {
                "enabled": False,
                "regularization": True,
                "regularization_lambda": 1.0,
                "temporal_lambda": 2.0,
                "max_iter": 10,
            },
        },
    }


def _write_pose_frame(root: Path, frame: int):
    keypoints_dir = root / "pose" / "keypoints3d"
    metadata_dir = root / "pose" / "metadata"
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = {"camera1": {"joint": [0, 0, 1]}, "camera2": {"joint": [0, 0, 1]}}
    metadata = {
        "metadata": {
            "source_pkl_stems": {"camera1": "cam1", "camera2": "cam2"},
            "source_frame_indices": {"camera1": frame - 1, "camera2": frame + 4},
        }
    }
    (keypoints_dir / f"pose_data_{frame}.json").write_text(json.dumps(payload), encoding="utf-8")
    (metadata_dir / f"pose_data_{frame}.json").write_text(json.dumps(metadata), encoding="utf-8")


class FusionExecutorTest(unittest.TestCase):
    @patch("fusion_pipeline.executor._orientation_mismatches", side_effect=[{"right_elbow"}, set()])
    @patch("fusion_pipeline.executor.optimize_f_points")
    @patch("fusion_pipeline.executor.apply_rotation_mismatch_corrections")
    @patch("fusion_pipeline.executor.calculate_stats", return_value=(0, 0, 0, 0, 0))
    @patch("fusion_pipeline.executor.estimate_bidirectional_similarity", return_value=((1, None, None), (1, None, None), ["head", "neck", "pelvis"]))
    @patch("fusion_pipeline.executor.apply_confidence_corrections")
    @patch("fusion_pipeline.executor.detect_cross_view_errors")
    def test_disabled_regressive_stages_are_skipped(
        self, detect, confidence_correction, _similarity, stats, orientation_correction, optimizer, _mismatches
    ):
        names = [
            "head", "neck", "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist", "pelvis", "right_hip",
            "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "left_toe", "left_foot", "right_toe", "right_foot", "left_hand", "right_hand",
        ]
        camera = {name: [float(index), float(index % 3), 2.0] for index, name in enumerate(names)}
        detect.return_value = {
            "M": set(), "K1": set(), "K2": set(), "L": names,
            "weights": {name: 1.0 for name in names},
            "H1": {name: 1.0 for name in names}, "H2": {name: 1.0 for name in names},
        }
        corrected = {name: list(value) for name, value in camera.items()}
        corrected["right_elbow"] = [999.0, 999.0, 999.0]
        confidence_correction.return_value = (corrected, corrected)

        result = run_phase3_pipeline(
            {"camera1": camera, "camera2": camera},
            map_path="configs/keypoints3D_map.yml", occlusion_tau=0.01,
            regularization=True, regularization_lambda=1.0, temporal_lambda=2.0,
            max_iter=10, ransac_threshold=0.05, ransac_max_combos=10,
            belief_alpha=0.001, belief_beta=0.8,
            orientation_correction_enabled=False, optimization_enabled=False,
        )

        orientation_correction.assert_not_called()
        optimizer.assert_not_called()
        self.assertEqual(result["F_optimized"], [])
        self.assertEqual(stats.call_args_list[0].args[2], result["F"])
        self.assertEqual(result["rejected_new_mismatches"], ["right_elbow"])
        self.assertEqual(result["optimized"]["camera1"]["right_elbow"], camera["right_elbow"])

    def test_authoritative_source_index_does_not_fall_back_to_wrong_frame(self):
        profile = {"0": {"joint": [1, 2, 0.9]}}

        self.assertIsNone(_frame_confidence_from_profile(profile, source_idx=9, frame_idx=1))

    def test_pose_frame_requires_matching_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keypoints = root / "pose_data_1.json"
            keypoints.write_text(json.dumps({"camera1": {}, "camera2": {}}), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "Pose metadata not found"):
                _load_pose_frame(keypoints, root / "metadata")

            (root / "metadata").mkdir()
            (root / "metadata" / keypoints.name).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid pose metadata"):
                _load_pose_frame(keypoints, root / "metadata")

    @patch("fusion_pipeline.executor._load_2d_profiles", return_value={"camera1": None, "camera2": None})
    def test_empty_pose_directory_fails(self, _profiles):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pose" / "keypoints3d").mkdir(parents=True)
            (root / "pose" / "metadata").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "No pose JSON files"):
                run_fusion(_config(root))

    @patch("fusion_pipeline.executor._load_2d_profiles", return_value={"camera1": None, "camera2": None})
    def test_fallback_is_not_used_as_next_temporal_target_and_metadata_is_preserved(self, _profiles):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pose_frame(root, 1)
            _write_pose_frame(root, 2)
            previous_values = []

            def fake_pipeline(data_in, **kwargs):
                previous_values.append(kwargs["prev_optimized_data"])
                if len(previous_values) == 1:
                    raise ValueError("bad frame")
                return {"optimized": {"camera1": data_in["camera1"], "camera2": data_in["camera2"]}}

            with patch("fusion_pipeline.executor.run_phase3_pipeline", side_effect=fake_pipeline):
                run_fusion(_config(root, max_fallback_ratio=1.0))

            self.assertEqual(previous_values, [None, None])
            metadata = json.loads((root / "fused" / "metadata" / "fused_data_2.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["metadata"]["source_pkl_stems"]["camera1"], "cam1")
            self.assertEqual(metadata["metadata"]["fusion_config"]["belief"]["beta"], 0.8)

    @patch("fusion_pipeline.executor._load_2d_profiles", return_value={"camera1": None, "camera2": None})
    def test_rejected_fallback_run_writes_no_partial_output(self, _profiles):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pose_frame(root, 1)
            with patch("fusion_pipeline.executor.run_phase3_pipeline", side_effect=ValueError("bad frame")):
                with self.assertRaisesRegex(RuntimeError, "Fallback ratio"):
                    run_fusion(_config(root))

            self.assertEqual(list((root / "fused" / "keypoints3d").glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()

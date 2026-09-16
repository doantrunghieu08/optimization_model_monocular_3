import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fusion_pipeline.detector import compute_harmonic_precision
from fusion_pipeline.detector import detect_cross_view_errors
from fusion_pipeline.executor import (
    _frame_confidence_from_profile,
    _load_pose_frame,
    run_phase3_pipeline,
    run_fusion,
)
from fusion_pipeline.optimization import calculate_stats, optimize_f_points
from pose_pipeline.executor import _midpoint_body_pose


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
            "belief": {
                "alpha": 0.001,
                "beta": 0.8,
                "global": True,
                "local_method": "naive_distance_belief",
            },
            "occlusion": {"enabled": False, "tau": 0.01},
            "ransac": {"threshold": 0.05, "max_combos": 10},
            "correction": {"enabled": False, "orientation_enabled": False, "reject_new_mismatches": True},
            "optimization": {
                "enabled": False,
                "use_kinematic_constraints": True,
                "loss_type": "huber",
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
    def test_body_pose_midpoint_handles_axis_angle_wrap(self):
        pose1 = np.zeros((1, 72))
        pose2 = np.zeros((1, 72))
        pose1[0, 3:6] = [0.0, 0.0, np.deg2rad(170.0)]
        pose2[0, 3:6] = [0.0, 0.0, np.deg2rad(-170.0)]

        midpoint = _midpoint_body_pose(pose1, pose2)

        self.assertAlmostEqual(abs(midpoint[0, 2]), np.pi, places=6)
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
            confidence_correction_enabled=False,
            orientation_correction_enabled=False, optimization_enabled=False,
        )

        confidence_correction.assert_not_called()
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

    def test_belief_ablation_switches_method_and_scope(self):
        names = ["left_shoulder", "left_elbow"]
        camera = {"left_shoulder": [0.0, 0.0, 1.5], "left_elbow": [0.0, 0.0, 3.0]}
        visible = dict.fromkeys(names, True)

        _, naive_local, _ = compute_harmonic_precision(
            camera, camera, names, visible, visible, alpha=0.1, beta=0.8,
            global_belief=False, local_method="naive_distance_belief",
        )
        _, optical_local, _ = compute_harmonic_precision(
            camera, camera, names, visible, visible, alpha=0.1, beta=0.8,
            global_belief=False, local_method="optical_aware_belief",
        )
        _, naive_global, _ = compute_harmonic_precision(
            camera, camera, names, visible, visible, alpha=0.1, beta=0.8,
            global_belief=True, local_method="naive_distance_belief",
        )

        self.assertAlmostEqual(optical_local["left_shoulder"], 1.0)
        self.assertNotEqual(naive_local, optical_local)
        self.assertNotEqual(naive_local, naive_global)

    def test_global_belief_does_not_spread_occlusion_to_visible_neighbor(self):
        names = ["left_shoulder", "left_elbow"]
        camera = {name: [0.0, 0.0, 2.0] for name in names}
        visibility = {"left_shoulder": True, "left_elbow": False}

        _, local, _ = compute_harmonic_precision(
            camera, camera, names, visibility, visibility, alpha=0.1, beta=0.8,
            global_belief=False, local_method="naive_distance_belief",
        )
        _, global_, _ = compute_harmonic_precision(
            camera, camera, names, visibility, visibility, alpha=0.1, beta=0.8,
            global_belief=True, local_method="naive_distance_belief",
        )

        self.assertAlmostEqual(global_["left_shoulder"], local["left_shoulder"], delta=1e-6)
        self.assertEqual(global_["left_elbow"], 0.0)

    def test_confidence_delta_cap_controls_correction_coverage(self):
        names = ["left_elbow", "left_wrist", "right_elbow"]
        cam1 = {name: [0.0, 0.0, 2.0] for name in names}
        cam2 = {name: [0.0, 0.0, 2.0] for name in names}
        visible = dict.fromkeys(names, True)
        confidence1 = {name: [0.0, 0.0, value] for name, value in zip(names, (0.9, 0.8, 0.7))}
        confidence2 = {name: [0.0, 0.0, 0.6] for name in names}

        low = detect_cross_view_errors(
            cam1, cam2, names, visible, visible, alpha=0.0, beta=1.0,
            confidence2d1=confidence1, confidence2d2=confidence2,
            global_belief=False, confidence_delta_cap=0.01,
        )
        high = detect_cross_view_errors(
            cam1, cam2, names, visible, visible, alpha=0.0, beta=1.0,
            confidence2d1=confidence1, confidence2d2=confidence2,
            global_belief=False, confidence_delta_cap=1.0,
        )

        self.assertGreater(len(low["K1"]), len(high["K1"]))

    @patch("fusion_pipeline.optimization.minimize")
    def test_kinematic_ablation_controls_slsqp_constraints(self, minimize):
        minimize.return_value = SimpleNamespace(success=True, x=np.array([0, 0, 2, 0, 0, 2], dtype=float), message="")
        data = {
            "camera1": {"left_shoulder": [0, 0, 1], "left_elbow": [0, 0, 2]},
            "camera2": {"left_shoulder": [0, 0, 1], "left_elbow": [0, 0, 2]},
        }
        kwargs = dict(data=data, anchors=["left_shoulder"], f_list=["left_elbow"], max_iter=1)

        optimize_f_points(**kwargs, use_kinematic_constraints=False)
        self.assertEqual(minimize.call_args.kwargs["constraints"], [])
        optimize_f_points(**kwargs, use_kinematic_constraints=True)
        self.assertGreater(len(minimize.call_args.kwargs["constraints"]), 0)

    def test_loss_ablation_switches_huber_and_mse(self):
        cam1 = {"anchor": [0, 0, 0], "joint": [0, 0, 0]}
        cam2 = {"anchor": [0, 0, 0], "joint": [2, 0, 0]}

        huber = calculate_stats(cam1, cam2, ["joint"], ["anchor"], loss_type="huber")[-1]
        mse = calculate_stats(cam1, cam2, ["joint"], ["anchor"], loss_type="mse")[-1]

        self.assertGreater(mse, huber)

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

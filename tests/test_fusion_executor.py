import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.pipelines.fusion.detector import compute_harmonic_precision
from src.pipelines.fusion.detector import detect_cross_view_errors
from src.pipelines.fusion.executor import (
    _frame_belief_from_profile,
    _load_pose_frame,
    run_phase3_pipeline,
    run_fusion,
)
from src.pipelines.fusion.optimization import calculate_stats, optimize_f_points
from src.pipelines.pose.executor import _midpoint_body_pose


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
    @patch("src.pipelines.fusion.executor._orientation_mismatches", side_effect=[{"right_elbow"}, set()])
    @patch("src.pipelines.fusion.executor.optimize_f_points")
    @patch("src.pipelines.fusion.executor.calculate_stats", return_value=(0, 0, 0, 0, 0))
    @patch(
        "src.pipelines.fusion.executor.estimate_bidirectional_similarity",
        return_value=(
            (1, np.eye(3), np.zeros(3)),
            (1, np.eye(3), np.zeros(3)),
            ["head", "neck", "pelvis"],
        ),
    )
    @patch("src.pipelines.fusion.executor.detect_cross_view_errors")
    def test_aligned_skips_belief_correction_and_optimizer(
        self, detect, _similarity, stats, optimizer, _mismatches
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
        result = run_phase3_pipeline(
            {"camera1": camera, "camera2": camera},
            map_path="configs/keypoints3D_map.yml", occlusion_tau=0.01,
            regularization=True, regularization_lambda=1.0, temporal_lambda=2.0,
            max_iter=10, ransac_threshold=0.05, ransac_max_combos=10,
            belief_alpha=0.001, belief_beta=0.8,
            belief_correction_enabled=False,
            orientation_correction_enabled=False, optimization_enabled=False,
            fusion_method="aligned_averaging",
            precomputed_transforms=((1, np.eye(3), np.zeros(3)), (1, np.eye(3), np.zeros(3))),
        )

        detect.assert_not_called()
        optimizer.assert_not_called()
        self.assertEqual(result["F_optimized"], [])
        self.assertEqual(stats.call_args_list[0].args[2], result["F"])
        self.assertEqual(result["rejected_new_mismatches"], [])
        self.assertEqual(result["optimized"]["camera1"]["right_elbow"], camera["right_elbow"])

    def test_aligned_is_transform_then_mean_without_alpha_or_beta(self):
        names = [
            "head", "neck", "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist", "pelvis", "right_hip",
            "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "left_toe", "left_foot", "right_toe", "right_foot", "left_hand", "right_hand",
        ]
        cam1 = {name: [float(i), float(i % 3), 2.0] for i, name in enumerate(names)}
        cam2 = {name: [float(i) + 0.2, float((i + 1) % 4), 2.1] for i, name in enumerate(names)}
        identity = (1.0, np.eye(3), np.zeros(3))
        kwargs = dict(
            data_in={"camera1": cam1, "camera2": cam2},
            map_path="configs/keypoints3D_map.yml", occlusion_tau=0.01,
            regularization=True, regularization_lambda=1.0, temporal_lambda=2.0,
            max_iter=10, ransac_threshold=0.05, ransac_max_combos=10,
            belief_correction_enabled=True, optimization_enabled=False,
            precomputed_transforms=(identity, identity),
        )

        aligned = run_phase3_pipeline(**kwargs, fusion_method="aligned_averaging")

        for camera in ("camera1", "camera2"):
            for name in names:
                np.testing.assert_allclose(
                    aligned["optimized"][camera][name],
                    0.5 * (np.asarray(cam1[name]) + np.asarray(cam2[name])),
                )
        self.assertEqual(aligned["fusion_method"], "aligned_averaging")
        self.assertEqual(aligned["F_optimized"], [])

    @patch("src.pipelines.fusion.executor._orientation_mismatches", return_value=set())
    @patch("src.pipelines.fusion.executor.optimize_f_points")
    @patch("src.pipelines.fusion.executor.calculate_stats", return_value=(0, 0, 0, 0, 0))
    @patch("src.pipelines.fusion.executor.estimate_bidirectional_similarity")
    @patch("src.pipelines.fusion.executor.detect_cross_view_errors")
    def test_proposed_runs_belief_correction_and_configured_optimizer(
        self, detect, similarity, _stats, optimizer, _mismatches
    ):
        names = [
            "head", "neck", "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist", "pelvis", "right_hip",
            "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "left_toe", "left_foot", "right_toe", "right_foot", "left_hand", "right_hand",
        ]
        camera = {name: [float(index), float(index % 3), 2.0] for index, name in enumerate(names)}
        camera2 = {name: [value[0] + 2.0, value[1], value[2]] for name, value in camera.items()}
        detect.return_value = {
            "M": set(), "K1": set(), "K2": set(), "L": names,
            "weights": {name: 1.0 for name in names},
            "H1": {name: 1.0 for name in names}, "H2": {name: 1.0 for name in names},
        }
        identity = (1.0, np.eye(3), np.zeros(3))
        similarity.return_value = (identity, identity, ["head", "neck", "pelvis"])
        optimizer.return_value = ({"camera1": camera, "camera2": camera}, None)

        result = run_phase3_pipeline(
            {"camera1": camera, "camera2": camera2},
            map_path="configs/keypoints3D_map.yml", occlusion_tau=0.01,
            regularization=True, regularization_lambda=1.0, temporal_lambda=2.0,
            max_iter=10, ransac_threshold=0.05, ransac_max_combos=10,
            belief_alpha=0.001, belief_beta=0.8,
            belief_correction_enabled=True, optimization_enabled=True,
        )

        self.assertIn("right_elbow", result["F"])
        self.assertNotIn("right_elbow", result["A_new"])
        self.assertEqual(
            set(result["A_new"]),
            {"left_hip", "right_hip", "left_shoulder", "right_shoulder"},
        )
        self.assertEqual(len(result["F_optimized"]), 17)
        self.assertEqual(result["corrections_skipped"], [])
        self.assertTrue({"left_hip", "right_hip", "left_shoulder", "right_shoulder"} <= set(result["A_new"]))
        detect.assert_called_once()
        similarity.assert_called_once()
        optimizer.assert_called_once()
        self.assertEqual(set(optimizer.call_args.args[1]), set(result["A_new"]))
        self.assertEqual(optimizer.call_args.args[2], result["F"])
        np.testing.assert_allclose(
            optimizer.call_args.args[0]["camera1"]["right_elbow"],
            camera["right_elbow"],
        )
        self.assertTrue(optimizer.call_args.kwargs["use_kinematic_constraints"])
        self.assertEqual(optimizer.call_args.kwargs["loss_type"], "huber")
        self.assertTrue(optimizer.call_args.kwargs["regularization"])
        self.assertFalse(optimizer.call_args.kwargs["root_relative"])

    def test_authoritative_source_index_does_not_fall_back_to_wrong_frame(self):
        profile = {"0": {"joint": [1, 2, 0.9]}}

        self.assertIsNone(_frame_belief_from_profile(profile, source_idx=9, frame_idx=1))

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

    @patch("src.pipelines.fusion.executor._load_2d_profiles", return_value={"camera1": None, "camera2": None})
    def test_empty_pose_directory_fails(self, _profiles):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pose" / "keypoints3d").mkdir(parents=True)
            (root / "pose" / "metadata").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "No pose JSON files"):
                run_fusion(_config(root))

    @patch("src.pipelines.fusion.executor._load_2d_profiles", return_value={"camera1": None, "camera2": None})
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

            with patch("src.pipelines.fusion.executor.run_phase3_pipeline", side_effect=fake_pipeline):
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

    def test_zero_beta_preserves_local_belief(self):
        names = ["left_shoulder", "left_elbow"]
        camera = {
            "left_shoulder": [0.0, 0.0, 1.5],
            "left_elbow": [0.0, 0.0, 3.0],
        }
        visible = dict.fromkeys(names, True)

        _, local, _ = compute_harmonic_precision(
            camera, camera, names, visible, visible, alpha=0.1, beta=0.0,
            global_belief=False, local_method="naive_distance_belief",
        )
        _, global_, _ = compute_harmonic_precision(
            camera, camera, names, visible, visible, alpha=0.1, beta=0.0,
            global_belief=True, local_method="naive_distance_belief",
        )

        for name in names:
            self.assertAlmostEqual(global_[name], local[name], places=6)

    def test_belief_delta_cap_controls_correction_coverage(self):
        names = ["left_elbow", "left_wrist", "right_elbow"]
        cam1 = {name: [0.0, 0.0, 2.0] for name in names}
        cam2 = {name: [0.0, 0.0, 2.0] for name in names}
        visible = dict.fromkeys(names, True)
        belief1 = {name: [0.0, 0.0, value] for name, value in zip(names, (0.9, 0.8, 0.7))}
        belief2 = {name: [0.0, 0.0, 0.6] for name in names}

        low = detect_cross_view_errors(
            cam1, cam2, names, visible, visible, alpha=0.0, beta=1.0,
            belief2d1=belief1, belief2d2=belief2,
            global_belief=False, belief_delta_cap=0.01,
        )
        high = detect_cross_view_errors(
            cam1, cam2, names, visible, visible, alpha=0.0, beta=1.0,
            belief2d1=belief1, belief2d2=belief2,
            global_belief=False, belief_delta_cap=1.0,
        )

        self.assertGreater(len(low["K1"]), len(high["K1"]))

    @patch("src.pipelines.fusion.optimization.minimize")
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

    @patch("src.pipelines.fusion.optimization.minimize")
    def test_temporal_penalty_ignores_whole_body_translation(self, minimize):
        current = {
            "left_hip": [10.0, 0.0, 0.0],
            "right_hip": [12.0, 0.0, 0.0],
            "left_elbow": [11.0, 1.0, 0.0],
        }
        previous = {
            "left_hip": [0.0, 0.0, 0.0],
            "right_hip": [2.0, 0.0, 0.0],
            "left_elbow": [1.0, 1.0, 0.0],
        }

        def solve(fun, x0, **_kwargs):
            self.assertAlmostEqual(fun(x0), 0.0)
            return SimpleNamespace(success=True, x=x0, message="")

        minimize.side_effect = solve
        optimize_f_points(
            data={"camera1": current, "camera2": current},
            anchors=["left_hip", "right_hip"],
            f_list=["left_elbow"],
            prev_data={"camera1": previous, "camera2": previous},
            temporal_lambda=1.0,
            use_kinematic_constraints=False,
        )

    def test_loss_ablation_switches_huber_and_mse(self):
        cam1 = {"anchor": [0, 0, 0], "joint": [0, 0, 0]}
        cam2 = {"anchor": [0, 0, 0], "joint": [2, 0, 0]}

        huber = calculate_stats(cam1, cam2, ["joint"], ["anchor"], loss_type="huber")[-1]
        mse = calculate_stats(cam1, cam2, ["joint"], ["anchor"], loss_type="mse")[-1]

        self.assertGreater(mse, huber)

    @patch("src.pipelines.fusion.executor._load_2d_profiles", return_value={"camera1": None, "camera2": None})
    def test_rejected_fallback_run_writes_no_partial_output(self, _profiles):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pose_frame(root, 1)
            with patch("src.pipelines.fusion.executor.run_phase3_pipeline", side_effect=ValueError("bad frame")):
                with self.assertRaisesRegex(RuntimeError, "Fallback ratio"):
                    run_fusion(_config(root))

            self.assertEqual(list((root / "fused" / "keypoints3d").glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()

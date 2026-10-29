import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation_pipeline.executor import _compute_pa_mpjpe, _compute_pck
from brute_force_runner import EVALUATION_PROTOCOL, _get_header_indices, _parse_history_row, _ranking_key, _resolve_pair_mode
from fusion_pipeline.correction import apply_multiview_geometric_fusion
from fusion_pipeline.executor import _source_frame_indices, _restore_master_pose
from fusion_pipeline.executor import detect_adaptive_fusion_mode
from fusion_pipeline.detector import require_ray_casting_backend
from fusion_pipeline.optimization import optimize_sequence_gpu
from learnable_pipeline.executor import _slice_person
from learnable_pipeline.post_opt import post_optimize_smpl_sequence
from pipeline import _evaluation_input_dirs, _require_evaluation_inputs
from pose_pipeline.executor import _shared_betas
from preprocess_pipeline.calib import _estimate_intrinsics
from preprocess_pipeline.extract_2d import _load_tracking_payload
from preprocess_pipeline.offset_paper import load_pkl_data as load_offset_pkl_data


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.names = ["left_hip", "right_hip", "left_wrist", "right_ankle"]
        self.truth = {
            "left_hip": np.array([0.0, 0.0, 0.0]),
            "right_hip": np.array([1.0, 0.0, 0.0]),
            "left_wrist": np.array([0.2, 1.0, 0.3]),
            "right_ankle": np.array([0.7, 0.4, 1.2]),
        }

    def test_pa_mpjpe_rejects_reflection_and_degenerate_pose(self):
        angle = np.pi / 3.0
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotated = {name: point @ rotation + 2.0 for name, point in self.truth.items()}
        mean_error, _ = _compute_pa_mpjpe(rotated, self.truth, self.names)
        self.assertAlmostEqual(mean_error, 0.0, places=9)

        reflected = {name: point * np.array([-1.0, 1.0, 1.0]) for name, point in self.truth.items()}
        mean_error, _ = _compute_pa_mpjpe(reflected, self.truth, self.names)
        self.assertGreater(mean_error, 1.0)

        collapsed = {name: np.zeros(3) for name in self.names}
        with self.assertRaises(ValueError):
            _compute_pa_mpjpe(collapsed, self.truth, self.names)

    def test_pck_returns_percentage(self):
        score, details = _compute_pck(self.truth, self.truth, self.names, threshold_mm=150.0)
        self.assertEqual(score, 100.0)
        self.assertTrue(all(value == 100.0 for value in details.values()))


class SynchronizationTests(unittest.TestCase):
    def test_occlusion_backend_is_validated(self):
        if importlib.util.find_spec("rtree") is None:
            with self.assertRaisesRegex(RuntimeError, "rtree"):
                require_ray_casting_backend()
        else:
            require_ray_casting_backend()

    def test_fusion_uses_each_camera_source_frame(self):
        data = {"metadata": {"source_frame_indices": {"camera1": 7, "camera2": 3}}}
        self.assertEqual(_source_frame_indices(data, 1), {"camera1": 7, "camera2": 3})

    def test_learnable_slices_wham_frames_after_offset(self):
        person = {
            "pose": np.arange(30).reshape(5, 6),
            "trans": np.arange(15).reshape(5, 3),
            "betas": np.arange(50).reshape(5, 10),
            "gender": "neutral",
        }
        sliced = _slice_person(person, start=2, frame_count=2)
        np.testing.assert_array_equal(sliced["pose"], person["pose"][2:4])
        np.testing.assert_array_equal(sliced["trans"], person["trans"][2:4])
        self.assertEqual(sliced["gender"], "neutral")

    def test_master_betas_are_constant_across_cameras(self):
        shared = _shared_betas({"betas": np.array([[0.0, 0.0], [10.0, 10.0]])})
        np.testing.assert_array_equal(shared, np.array([5.0, 5.0], dtype=np.float32))

    def test_offset_estimation_uses_target_person_mapping(self):
        import joblib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "person_mapping.txt").write_text("0 = aria01\n1 = aria02", encoding="utf-8")
            pkl_path = root / "cam.pkl"
            joblib.dump({
                0: {"pose": np.zeros((2, 72)), "betas": np.zeros((2, 10)), "name": "wrong"},
                1: {"pose": np.ones((2, 72)), "betas": np.ones((2, 10)), "name": "target"},
            }, pkl_path)
            self.assertEqual(load_offset_pkl_data(pkl_path, target_person="aria02")["name"], "target")

    def test_named_tracking_file_uses_target_person_mapping(self):
        import joblib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "person_mapping.txt").write_text("0 = aria01\n1 = aria02", encoding="utf-8")
            pkl_path = root / "cam.pkl"
            joblib.dump({0: {"pose": np.zeros((1, 72))}}, pkl_path)
            joblib.dump({
                0: {"frame_id": np.array([0]), "keypoints": np.zeros((1, 17, 3)), "name": "wrong"},
                1: {"frame_id": np.array([0]), "keypoints": np.ones((1, 17, 3)), "name": "target"},
            }, root / "tracking_results_video.pth")
            self.assertEqual(_load_tracking_payload(pkl_path, target_person="aria02")["name"], "target")

    def test_calibration_file_is_preferred_over_video_heuristic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "calib.txt").write_text("1000 1001 320 240", encoding="utf-8")
            self.assertEqual(_estimate_intrinsics(root / "missing.mp4"), [[1000.0, 0.0, 320.0], [0.0, 1001.0, 240.0], [0.0, 0.0, 1.0]])


class FusionOptimizationTests(unittest.TestCase):
    JOINTS = ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip")

    def _run(self, regularization):
        cam1 = {name: [float(i), 0.0, 1.0] for i, name in enumerate(self.JOINTS)}
        cam2 = {name: [float(i) * 1.5, 0.0, 1.0] for i, name in enumerate(self.JOINTS)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            keypoints = root / "keypoints3d"
            metadata = root / "metadata"
            keypoints.mkdir()
            metadata.mkdir()
            for frame in range(1, 4):
                payload = {"camera1": cam1, "camera2": cam2}
                (keypoints / f"fused_data_{frame}.json").write_text(json.dumps(payload), encoding="utf-8")
            config = {"fusion": {"optimization": {
                "regularization": regularization,
                "regularization_lambda": 1000.0,
                "temporal_lambda": 0.0,
                "max_iter": 10,
            }}}
            optimize_sequence_gpu(keypoints, metadata, config=config, device="cpu")
            result = json.loads((keypoints / "fused_data_1.json").read_text(encoding="utf-8"))
            return np.linalg.norm(np.asarray(result["camera1"]["right_hip"]) - np.asarray(cam1["right_hip"]))

    def test_regularization_flag_changes_sequence_optimizer(self):
        self.assertLess(self._run(True), self._run(False))

    def test_fusion_package_has_no_learnable_backend(self):
        fusion_dir = Path(__file__).resolve().parents[1] / "fusion_pipeline"
        source = "\n".join(path.read_text(encoding="utf-8").lower() for path in fusion_dir.glob("*.py"))
        self.assertNotIn("learnable", source)
        self.assertNotIn("netbody", source)

    def test_fusion_and_pose_packages_have_no_ground_truth_access(self):
        root = Path(__file__).resolve().parents[1]
        for pkg in ("fusion_pipeline", "pose_pipeline"):
            source = "\n".join(path.read_text(encoding="utf-8").lower() for path in (root / pkg).glob("*.py"))
            self.assertNotIn("ground_truth", source)
            self.assertNotIn("calibrate_domain_offsets", source)

    def test_fusion_executor_uses_gated_consensus(self):
        executor = (Path(__file__).resolve().parents[1] / "fusion_pipeline" / "executor.py").read_text(encoding="utf-8")
        self.assertIn("apply_multiview_geometric_fusion(", executor)
        self.assertIn("detect_adaptive_fusion_mode(", executor)
        self.assertNotIn("apply_belief_joint_replacement", executor)
        self.assertNotIn("apply_pose_level_fusion", executor)

    def test_stale_report_protocol_is_not_reused(self):
        header = ["Set", "Segment", "Cam Master", "Cam Slave"]
        self.assertEqual(_parse_history_row(["S1", "seg1", "cam1", "cam2"], _get_header_indices(header)), (None, None))
        header = ["Set", "Segment", "Protocol", "Cam Master", "Cam Slave"]
        stale = ["S1", "seg1", "no_gt_v6_visible_confidence_transfer", "cam1", "cam2"]
        self.assertEqual(_parse_history_row(stale, _get_header_indices(header)), (None, None))
        self.assertEqual(EVALUATION_PROTOCOL, "no_gt_v8_target_sync_master_shape")

    def test_adaptive_detector_uses_aggressive_mode_only_for_reliable_slave(self):
        names = [f"joint_{idx}" for idx in range(8)]
        cam1 = {name: np.array([idx * 0.1, 0.0, 1.0]) for idx, name in enumerate(names)}
        transform = (1.0, np.eye(3), np.array([0.01, 0.0, 0.0]))
        cam2 = {name: point + transform[2] for name, point in cam1.items()}
        visible = {name: True for name in names}

        mode, _ = detect_adaptive_fusion_mode(
            cam1, cam2, transform,
            {name: 0.6 for name in names}, {name: 0.9 for name in names},
            visible, visible, set(),
        )
        self.assertEqual(mode, "aggressive")

        inconsistent = {name: point + np.array([0.3, 0.0, 0.0]) for name, point in cam2.items()}
        mode, _ = detect_adaptive_fusion_mode(
            cam1, inconsistent, transform,
            {name: 0.6 for name in names}, {name: 0.9 for name in names},
            visible, visible, set(),
        )
        self.assertEqual(mode, "conservative")

    def test_visible_consensus_requires_confidence_margin(self):
        cam1 = {"left_elbow": [0.0, 0.0, 0.0], "left_wrist": [1.0, 0.0, 0.0]}
        cam2 = {"left_elbow": [0.0, 0.0, 0.0], "left_wrist": [1.032, 0.0, 0.0]}
        transform = (1.0, np.eye(3), np.array([0.01, 0.0, 0.0]))
        fused1, fused2 = apply_multiview_geometric_fusion(
            cam1,
            cam2,
            ["left_wrist"],
            transform,
            transform,
            h1={"left_wrist": 1.0},
            h2={"left_wrist": 1.0},
            consensus_threshold=0.08,
            consensus_blend_strength=0.25,
            master_margin=0.15,
        )
        self.assertEqual(fused1["left_wrist"], cam1["left_wrist"])
        self.assertEqual(fused2["left_wrist"], cam2["left_wrist"])

        rescued1, protected2 = apply_multiview_geometric_fusion(
            cam1,
            cam2,
            ["left_wrist"],
            transform,
            transform,
            h1={"left_wrist": 0.6},
            h2={"left_wrist": 0.9},
            consensus_threshold=0.08,
            consensus_blend_strength=0.25,
            master_margin=0.15,
        )
        self.assertGreater(float(rescued1["left_wrist"][0]), 1.0)
        self.assertLess(float(rescued1["left_wrist"][0]), 1.032)
        self.assertEqual(protected2["left_wrist"], cam2["left_wrist"])

        aggressive1, aggressive2 = apply_multiview_geometric_fusion(
            cam1,
            cam2,
            ["left_wrist"],
            transform,
            transform,
            h1={"left_wrist": 1.0},
            h2={"left_wrist": 1.0},
            consensus_threshold=0.08,
            consensus_blend_strength=0.25,
            master_margin=0.0,
        )
        self.assertGreater(float(aggressive1["left_wrist"][0]), 1.0)
        self.assertLess(float(aggressive2["left_wrist"][0]), 1.032)

        far_cam2 = {"left_elbow": [0.0, 0.0, 0.0], "left_wrist": [1.2, 0.0, 0.0]}
        unchanged1, unchanged2 = apply_multiview_geometric_fusion(
            cam1,
            far_cam2,
            ["left_wrist"],
            transform,
            transform,
            h1={"left_wrist": 1.0},
            h2={"left_wrist": 1.0},
            consensus_threshold=0.08,
            consensus_blend_strength=0.25,
            master_margin=0.15,
        )
        self.assertEqual(unchanged1["left_wrist"], cam1["left_wrist"])
        self.assertEqual(unchanged2["left_wrist"], far_cam2["left_wrist"])

    def test_pair_mode_defaults_to_aggressive_and_supports_protected_overrides(self):
        config = {"default_mode": "aggressive", "pair_modes": {"cam13->cam21": "posed_fallback"}}
        self.assertEqual(_resolve_pair_mode(config, "cam13", "cam21"), "posed_fallback")
        self.assertEqual(_resolve_pair_mode(config, "cam21", "cam04"), "aggressive")

    def test_posed_fallback_restores_only_master_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fused_dir = root / "fused"
            pose_dir = root / "pose"
            fused_dir.mkdir()
            pose_dir.mkdir()
            (fused_dir / "fused_data_1.json").write_text(json.dumps({
                "camera1": {"joint": [9, 0, 0]}, "camera2": {"joint": [2, 0, 0]},
            }), encoding="utf-8")
            (pose_dir / "pose_data_1.json").write_text(json.dumps({
                "camera1": {"joint": [1, 0, 0]}, "camera2": {"joint": [8, 0, 0]},
            }), encoding="utf-8")

            _restore_master_pose(fused_dir, pose_dir)

            restored = json.loads((fused_dir / "fused_data_1.json").read_text(encoding="utf-8"))
            self.assertEqual(restored["camera1"]["joint"], [1, 0, 0])
            self.assertEqual(restored["camera2"]["joint"], [2, 0, 0])

    def test_confidence_corrections_protects_visible_extremities_from_negative_transfer(self):
        from fusion_pipeline.correction import apply_confidence_corrections
        cam1 = {"left_wrist": np.array([1.0, 0.0, 0.0])}
        cam2 = {"left_wrist": np.array([1.2, 0.0, 0.0])}  # 20cm away
        identity = (1.0, np.eye(3), np.zeros(3))
        # When cam1 can see left_wrist (vis1=True), 20cm displacement exceeds 5cm extremity limit -> ignored
        c1, _ = apply_confidence_corrections(
            cam1, cam2, k1_set=set(), k2_set={"left_wrist"},
            t12=identity, t21=identity,
            h1={"left_wrist": 0.2}, h2={"left_wrist": 0.9},
            vis1={"left_wrist": True}, vis2={"left_wrist": True}
        )
        np.testing.assert_array_equal(c1["left_wrist"], cam1["left_wrist"])

        # Occluded master accepts a geometrically plausible correction.
        cam2_occ = {"left_wrist": np.array([1.1, 0.0, 0.0])}
        c1_occ, _ = apply_confidence_corrections(
            cam1, cam2_occ, k1_set=set(), k2_set={"left_wrist"},
            t12=identity, t21=identity,
            h1={"left_wrist": 0.2}, h2={"left_wrist": 0.9},
            vis1={"left_wrist": False}, vis2={"left_wrist": True}
        )
        self.assertGreater(float(c1_occ["left_wrist"][0]), 1.0)

    def test_ranking_minimizes_final_pa_error_then_motion_error(self):
        low_final_error = {
            "pa_mpjpe": 30.0, "% delta_pa_mpjpe": -0.5,
            "accel": 18.0, "mble": 18.0,
        }
        larger_gain_but_worse_result = {
            "pa_mpjpe": 50.0, "% delta_pa_mpjpe": 2.0,
            "accel": 10.0, "mble": 10.0,
        }
        self.assertGreater(_ranking_key(low_final_error), _ranking_key(larger_gain_but_worse_result))

        same_pose_better_motion = dict(low_final_error, accel=10.0, mble=10.0)
        self.assertGreater(_ranking_key(same_pose_better_motion), _ranking_key(low_final_error))


class LearnablePostOptimizationTests(unittest.TestCase):
    def test_best_parameters_match_reported_loss(self):
        class FakeSmplLayer:
            def __call__(self, betas, global_orient, body_pose):
                tail = torch.zeros((global_orient.shape[0], 6889, 3), device=global_orient.device)
                return SimpleNamespace(vertices=torch.cat((global_orient[:, None, :], tail), dim=1))

        net = SimpleNamespace(human_model=SimpleNamespace(layer={"neutral": FakeSmplLayer()}))
        pose_init = np.zeros((1, 72), dtype=np.float32)
        trans_init = np.zeros((1, 3), dtype=np.float32)
        target_names = [f"joint_{i}" for i in range(21)]
        fusion_frames = [{name: [1.0, 0.0, 0.0] for name in target_names}]
        map_data = {"keypoints": [
            {"name": name, "regressor_index": i} for i, name in enumerate(target_names)
        ]}

        with tempfile.TemporaryDirectory() as temp:
            regressor_path = Path(temp) / "regressor.npy"
            regressor = np.zeros((21, 6890), dtype=np.float32)
            regressor[:, 0] = 1.0
            np.save(regressor_path, regressor)
            pose, trans, initial_loss, best_loss = post_optimize_smpl_sequence(
                pose_init,
                trans_init,
                np.zeros((1, 10), dtype=np.float32),
                fusion_frames,
                net,
                str(regressor_path),
                map_data,
                torch.device("cpu"),
                iters=3,
                lr=0.1,
            )

        predicted = pose[0, :3] + trans[0]
        recomputed = np.sum((predicted - np.array([1.0, 0.0, 0.0])) ** 2)
        recomputed += 0.01 * np.sum((pose - pose_init) ** 2)
        recomputed += 0.01 * np.sum((trans - trans_init) ** 2)
        self.assertAlmostEqual(best_loss, recomputed, places=6)
        self.assertLess(best_loss, initial_loss)


class EvaluationStageTests(unittest.TestCase):
    def test_disabled_learnable_is_not_required_and_sources_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pose = root / "pose"
            fused = root / "fused"
            for directory in (pose / "keypoints3d", pose / "metadata", fused / "keypoints3d"):
                directory.mkdir(parents=True)
            (pose / "keypoints3d" / "pose_data_1.json").write_text("{}", encoding="utf-8")
            (fused / "keypoints3d" / "fused_data_1.json").write_text("{}", encoding="utf-8")
            cam1 = root / "cam1.pkl"
            cam2 = root / "cam2.pkl"
            metadata = {"metadata": {"camera_sync": {"source_pkls": {
                "camera1": str(cam1.resolve()),
                "camera2": str(cam2.resolve()),
            }}}}
            (pose / "metadata" / "pose_data_1.json").write_text(json.dumps(metadata), encoding="utf-8")
            config = {
                "inputs": {"cam1_pkl": str(cam1), "cam2_pkl": str(cam2)},
                "paths": {
                    "pose_output_dir": str(pose),
                    "fused_output_dir": str(fused),
                    "learnable_output_dir": str(root / "learnable"),
                    "learnable_extra_output_dir": str(root / "learnable_extra"),
                },
                "learnable": {"enabled": False},
                "learnable_extra": {"enabled": False},
            }
            self.assertEqual(len(_evaluation_input_dirs(config)), 2)
            _require_evaluation_inputs(config)

            metadata["metadata"]["camera_sync"]["source_pkls"]["camera1"] = str(root / "other.pkl")
            (pose / "metadata" / "pose_data_1.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different inputs"):
                _require_evaluation_inputs(config)

    def test_person_payload_resolution_by_target_name(self):
        from compat import resolve_person_payload
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "harmony_person_mapping.txt").write_text("0 = aria02\n1 = aria01", encoding="utf-8")
            pkl_path = root / "cam.pkl"
            wham_data = {
                0: {"pose": np.zeros((10, 72)), "name": "track_zero"},
                1: {"pose": np.zeros((20, 72)), "name": "track_one"},
            }
            p1 = resolve_person_payload(wham_data, pkl_path=pkl_path, target_person="aria01")
            self.assertEqual(p1["name"], "track_one")
            p2 = resolve_person_payload(wham_data, pkl_path=pkl_path, target_person="aria02")
            self.assertEqual(p2["name"], "track_zero")

class MasterProtectionAndAngleLimitTests(unittest.TestCase):
    def test_occluded_hinge_keeps_both_visible_neighbor_bone_lengths(self):
        from fusion_pipeline.executor import _apply_final_master_protection
        baseline = {
            "left_shoulder": [0.0, 0.0, 0.0],
            "left_elbow": [1.0, 0.0, 0.0],
            "left_wrist": [1.0, 1.0, 0.0],
        }
        points = {
            **baseline,
            "left_elbow": [0.5, 0.5, 1.0],
        }
        _apply_final_master_protection(
            points,
            baseline,
            vis1={"left_shoulder": True, "left_elbow": False, "left_wrist": False},
            vis2={"left_elbow": True, "left_wrist": False},
            h1={"left_elbow": 0.1},
            h2={"left_elbow": 0.9},
            max_displacement=2.0,
            weight=1.0,
        )
        shoulder = np.asarray(points["left_shoulder"])
        elbow = np.asarray(points["left_elbow"])
        wrist = np.asarray(points["left_wrist"])
        self.assertAlmostEqual(float(np.linalg.norm(elbow - shoulder)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(wrist - elbow)), 1.0, places=6)
        self.assertGreater(float(elbow[2]), 0.0)

    def test_final_master_residual_only_updates_occluded_joint_with_fixed_bone_length(self):
        from fusion_pipeline.executor import _apply_final_master_protection
        baseline = {
            "left_elbow": [0.0, 0.0, 0.0],
            "left_wrist": [1.0, 0.0, 0.0],
            "left_hand": [1.0, 0.0, 0.0],
            "right_wrist": [1.0, 0.0, 0.0],
        }
        points = {
            "left_elbow": [0.0, 0.0, 0.0],
            "left_wrist": [1.0, 1.0, 0.0],
            "left_hand": [1.0, 0.0, 0.0],
            "right_wrist": [1.0, 1.0, 0.0],
        }
        _apply_final_master_protection(
            points,
            baseline,
            vis1={"left_wrist": False, "right_wrist": True},
            vis2={"left_wrist": True, "right_wrist": True},
            h1={"left_wrist": 0.1, "right_wrist": 0.8},
            h2={"left_wrist": 0.9, "right_wrist": 0.9},
            max_displacement=2.0,
            weight=1.0,
        )
        self.assertGreater(points["left_wrist"][1], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(points["left_wrist"])), 1.0)
        self.assertEqual(points["left_hand"], points["left_wrist"])
        self.assertEqual(points["right_wrist"], baseline["right_wrist"])

    def test_final_master_protection_requires_a_visible_confidence_margin(self):
        from fusion_pipeline.executor import _apply_final_master_protection
        baseline = {
            "left_wrist": [0.0, 0.0, 0.0],
            "right_wrist": [0.0, 0.0, 0.0],
        }
        points = {
            "left_wrist": [0.02, 0.0, 0.0],
            "right_wrist": [0.1, 0.0, 0.0],
        }
        _apply_final_master_protection(
            points,
            baseline,
            vis1={"left_wrist": True, "right_wrist": True},
            vis2={"left_wrist": True, "right_wrist": True},
            h1={"left_wrist": 0.8, "right_wrist": 0.2},
            h2={"left_wrist": 0.9, "right_wrist": 0.8},
            master_margin=0.15,
        )
        self.assertEqual(points["left_wrist"], baseline["left_wrist"])
        self.assertEqual(points["right_wrist"], [0.05, 0.0, 0.0])

    def test_master_protection_requires_margin_for_visible_transfer(self):
        from fusion_pipeline.correction import apply_confidence_corrections
        cam1 = {"left_wrist": np.array([1.0, 0.0, 0.0])}
        cam2 = {"left_wrist": np.array([1.02, 0.0, 0.0])}
        identity = (1.0, np.eye(3), np.zeros(3))
        # When cam1 is visible and slave is only slightly higher (0.6 vs 0.5 <= margin 0.15), master is preserved
        c1, _ = apply_confidence_corrections(
            cam1, cam2, k1_set=set(), k2_set={"left_wrist"},
            t12=identity, t21=identity,
            h1={"left_wrist": 0.5}, h2={"left_wrist": 0.6},
            vis1={"left_wrist": True}, vis2={"left_wrist": True},
            master_margin=0.15,
        )
        np.testing.assert_array_equal(c1["left_wrist"], cam1["left_wrist"])

        # A clearly better slave may move a visible joint, but only through capped blending.
        c1_upd, _ = apply_confidence_corrections(
            cam1, cam2, k1_set=set(), k2_set={"left_wrist"},
            t12=identity, t21=identity,
            h1={"left_wrist": 0.5}, h2={"left_wrist": 0.8},
            vis1={"left_wrist": True}, vis2={"left_wrist": True},
            master_margin=0.15,
        )
        self.assertGreater(float(c1_upd["left_wrist"][0]), float(cam1["left_wrist"][0]))
        self.assertLess(float(c1_upd["left_wrist"][0]), float(cam2["left_wrist"][0]))

        # A low-confidence master accepts a clearly better slave.
        c1_bad, _ = apply_confidence_corrections(
            cam1, cam2, k1_set=set(), k2_set={"left_wrist"},
            t12=identity, t21=identity,
            h1={"left_wrist": 0.2}, h2={"left_wrist": 0.8},
            vis1={"left_wrist": True}, vis2={"left_wrist": True},
            master_margin=0.15,
        )
        self.assertNotEqual(float(c1_bad["left_wrist"][0]), float(cam1["left_wrist"][0]))

    def test_joint_angle_limit_blocks_impossible_flexion(self):
        from fusion_pipeline.correction import apply_confidence_corrections
        # Shoulder at [0, 1, 0], Elbow at [0, 0, 0], Wrist at [0, -1, 0] (straight arm)
        cam1 = {
            "left_shoulder": np.array([0.0, 1.0, 0.0]),
            "left_elbow": np.array([0.0, 0.0, 0.0]),
            "left_wrist": np.array([0.0, -1.0, 0.0]),
        }
        # Cam2 has wrist folded directly onto shoulder (angle ~ 0 deg, physically impossible)
        cam2 = {
            "left_shoulder": np.array([0.0, 1.0, 0.0]),
            "left_elbow": np.array([0.0, 0.0, 0.0]),
            "left_wrist": np.array([0.0, 0.95, 0.0]),
        }
        identity = (1.0, np.eye(3), np.zeros(3))
        c1, _ = apply_confidence_corrections(
            cam1, cam2, k1_set=set(), k2_set={"left_wrist"},
            t12=identity, t21=identity,
            h1={"left_wrist": 0.2}, h2={"left_wrist": 0.9},
            vis1={"left_wrist": False}, vis2={"left_wrist": True},
            master_margin=0.15,
            uncertainty_gating=False,
        )
        # Rejection of impossible fold preserves cam1
        np.testing.assert_array_equal(c1["left_wrist"], cam1["left_wrist"])


    def test_occluded_master_joint_accepts_high_trust_transfer(self):
        from fusion_pipeline.correction import apply_confidence_corrections
        cam1 = {"left_wrist": np.array([0.0, 0.0, 0.0]), "left_elbow": np.array([0.0, 0.0, 0.0]), "left_shoulder": np.array([0.0, 1.0, 0.0])}
        cam2 = {"left_wrist": np.array([0.1, 0.0, 0.0]), "left_elbow": np.array([0.0, 0.0, 0.0]), "left_shoulder": np.array([0.0, 1.0, 0.0])}
        identity = (1.0, np.eye(3), np.zeros(3))
        # Occluded master uses at most half of the slave residual.
        c1_occ, _ = apply_confidence_corrections(
            cam1, cam2, k1_set=set(), k2_set={"left_wrist"},
            t12=identity, t21=identity,
            h1={"left_wrist": 0.1}, h2={"left_wrist": 0.9},
            vis1={"left_wrist": False}, vis2={"left_wrist": True},
            uncertainty_gating=False,
        )
        self.assertAlmostEqual(float(c1_occ["left_wrist"][0]), 0.05)

    def test_smoothing_uses_master_only_baseline_for_protected_joints(self):
        from scipy.signal import savgol_filter
        from fusion_pipeline.executor import _apply_temporal_smoothing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fused_dir = root / "fused"
            pose_dir = root / "pose"
            metadata_dir = root / "metadata"
            fused_dir.mkdir()
            pose_dir.mkdir()
            metadata_dir.mkdir()
            master_x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

            for frame, x in enumerate(master_x, 1):
                pose = {"camera1": {"protected": [x, 0.0, 0.0], "rescued": [x, 0.0, 0.0]}}
                fused = {"camera1": {"protected": [x + 1.0, 0.0, 0.0], "rescued": [x + 0.1, 0.0, 0.0]}}
                metadata = {
                    "vis1": {"protected": True, "rescued": True},
                    "vis2": {"protected": True, "rescued": True},
                    "joint_confidence": {
                        "camera1": {"protected": 0.8, "rescued": 0.1},
                        "camera2": {"protected": 1.0, "rescued": 0.9},
                    },
                }
                (pose_dir / f"pose_data_{frame}.json").write_text(json.dumps(pose), encoding="utf-8")
                (fused_dir / f"fused_data_{frame}.json").write_text(json.dumps(fused), encoding="utf-8")
                (metadata_dir / f"fused_data_{frame}.json").write_text(json.dumps(metadata), encoding="utf-8")

            _apply_temporal_smoothing(
                fused_dir,
                pose_dir=pose_dir,
                metadata_dir=metadata_dir,
                master_protection={"enabled": True, "confidence_margin": 0.15},
                window_length=5,
                polyorder=2,
            )

            expected = savgol_filter(master_x, 5, 2)
            for frame, x in enumerate(expected, 1):
                result = json.loads((fused_dir / f"fused_data_{frame}.json").read_text(encoding="utf-8"))
                self.assertAlmostEqual(result["camera1"]["protected"][0], x, places=6)
                self.assertAlmostEqual(result["camera1"]["rescued"][0], x + 0.05, places=6)

    def test_clamp_to_baseline_spares_occluded_joints(self):
        from fusion_pipeline.executor import _clamp_to_baseline
        points = {"joint_vis": [0.3, 0.0, 0.0], "joint_occ": [0.3, 0.0, 0.0]}
        baseline = {"joint_vis": [0.0, 0.0, 0.0], "joint_occ": [0.0, 0.0, 0.0]}
        vis = {"joint_vis": True, "joint_occ": False}
        _clamp_to_baseline(points, baseline, vis=vis, max_radius=0.08)
        # Visible joint is clamped to max_radius 0.08m
        self.assertAlmostEqual(points["joint_vis"][0], 0.08, places=5)
        # Occluded joint is spared and remains at 0.30m
        self.assertAlmostEqual(points["joint_occ"][0], 0.30, places=5)


if __name__ == "__main__":
    unittest.main()

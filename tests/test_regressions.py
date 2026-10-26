import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation_pipeline.executor import _compute_pa_mpjpe, _compute_pck
from brute_force_runner import _ranking_key
from fusion_pipeline.correction import apply_multiview_geometric_fusion
from fusion_pipeline.executor import _source_frame_indices
from fusion_pipeline.detector import require_ray_casting_backend
from fusion_pipeline.optimization import optimize_sequence_gpu
from learnable_pipeline.executor import _slice_person
from learnable_pipeline.post_opt import post_optimize_smpl_sequence
from pipeline import _evaluation_input_dirs, _require_evaluation_inputs


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

    def test_safe_consensus_blend_moves_both_views_only_partway(self):
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
        )
        self.assertAlmostEqual(float(fused1["left_wrist"][0]), 1.004)
        self.assertAlmostEqual(float(fused2["left_wrist"][0]), 1.028)

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
        )
        self.assertEqual(unchanged1["left_wrist"], cam1["left_wrist"])
        self.assertEqual(unchanged2["left_wrist"], far_cam2["left_wrist"])

    def test_ranking_prioritizes_both_pose_metrics(self):
        balanced = {
            "% delta_mpjpe": 0.2,
            "% delta_pa_mpjpe": 0.2,
            "% delta_accel": 1.0,
            "% delta_mble": 1.0,
        }
        smooth_but_worse_pose = {
            "% delta_mpjpe": -0.1,
            "% delta_pa_mpjpe": 1.0,
            "% delta_accel": 90.0,
            "% delta_mble": 20.0,
        }
        self.assertGreater(_ranking_key(balanced), _ranking_key(smooth_but_worse_pose))

        same_pose_better_motion = dict(balanced, **{
            "% delta_accel": 5.0,
            "% delta_mble": 2.0,
        })
        self.assertGreater(_ranking_key(same_pose_better_motion), _ranking_key(balanced))


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


if __name__ == "__main__":
    unittest.main()

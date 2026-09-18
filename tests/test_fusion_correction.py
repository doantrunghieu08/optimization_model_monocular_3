import unittest

import numpy as np

from src.pipelines.fusion.correction import (
    apply_confidence_corrections,
    apply_root_relative,
    estimate_sequence_root_similarity,
    estimate_umeyama,
    ransac_umeyama,
)


class FusionCorrectionTest(unittest.TestCase):
    def test_similarity_requires_three_anchors(self):
        camera = {"a": [0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]}

        with self.assertRaisesRegex(ValueError, "At least 3 anchors"):
            ransac_umeyama(camera, camera, ["a", "b"], threshold=0.05, max_combos=10)

    def test_ransac_rejects_a_consensus_smaller_than_three(self):
        rng = np.random.default_rng(0)
        names = [str(i) for i in range(8)]
        camera1 = dict(zip(names, rng.normal(size=(8, 3))))
        camera2 = dict(zip(names, rng.normal(size=(8, 3))))

        with self.assertRaisesRegex(ValueError, "at least 3 inliers"):
            ransac_umeyama(camera1, camera2, names, threshold=0.05, max_combos=500)

    def test_similarity_rejects_collinear_points(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "non-collinear"):
            estimate_umeyama(points, points)

    def test_confidence_correction_rejects_large_jump(self):
        cam1 = {"near": [0.0, 0.0, 0.0], "far": [0.0, 0.0, 0.0]}
        cam2 = {"near": [0.04, 0.0, 0.0], "far": [1.0, 0.0, 0.0]}
        identity = (1.0, np.eye(3), np.zeros(3))

        _, corrected, applied, _ = apply_confidence_corrections(
            cam1, cam2, {"near", "far"}, set(), identity, identity, max_displacement=0.05,
            return_applied=True,
        )

        np.testing.assert_array_equal(corrected["near"], cam1["near"])
        np.testing.assert_array_equal(corrected["far"], cam2["far"])
        self.assertEqual(applied, {"near"})

    def test_confidence_correction_blends_by_source_reliability(self):
        cam1 = {"joint": [0.0, 0.0, 0.0]}
        cam2 = {"joint": [0.04, 0.0, 0.0]}
        identity = (1.0, np.eye(3), np.zeros(3))

        _, corrected = apply_confidence_corrections(
            cam1,
            cam2,
            {"joint"},
            set(),
            identity,
            identity,
            max_displacement=0.05,
            h1={"joint": 0.8},
            h2={"joint": 0.2},
            blend_mode="confidence",
        )

        np.testing.assert_allclose(corrected["joint"], [0.008, 0.0, 0.0])

    def test_sequence_alignment_recovers_root_relative_pose(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        names = ["neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        relative = {
            "neck": np.array([0.0, 0.5, 0.0]),
            "left_shoulder": np.array([-0.2, 0.4, 0.0]),
            "right_shoulder": np.array([0.2, 0.4, 0.0]),
            "left_hip": np.array([-0.1, 0.0, 0.0]),
            "right_hip": np.array([0.1, 0.0, 0.0]),
        }
        frames = []
        for offset in (0.0, 0.1, 0.2):
            root1 = np.array([offset, 0.0, 3.0])
            root2 = np.array([2.0, offset, 5.0])
            cam1 = {name: root1 + point for name, point in relative.items()}
            cam2 = {name: root2 + 1.2 * (rotation @ point) for name, point in relative.items()}
            frames.append({"camera1": cam1, "camera2": cam2})

        t12, _, diagnostics = estimate_sequence_root_similarity(frames, names, threshold=0.01)
        candidate = apply_root_relative(frames[0]["camera1"]["neck"], frames[0]["camera1"], frames[0]["camera2"], t12)

        np.testing.assert_allclose(candidate, frames[0]["camera2"]["neck"], atol=1e-8)
        self.assertEqual(diagnostics["inliers"], diagnostics["observations"])


if __name__ == "__main__":
    unittest.main()

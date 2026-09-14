import unittest

import numpy as np

from fusion_pipeline.correction import apply_confidence_corrections, estimate_umeyama, ransac_umeyama


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

        _, corrected = apply_confidence_corrections(
            cam1, cam2, {"near", "far"}, set(), identity, identity, max_displacement=0.05,
        )

        np.testing.assert_array_equal(corrected["near"], cam1["near"])
        np.testing.assert_array_equal(corrected["far"], cam2["far"])


if __name__ == "__main__":
    unittest.main()

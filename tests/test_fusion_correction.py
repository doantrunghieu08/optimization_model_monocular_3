import unittest

import numpy as np

from fusion_pipeline.correction import estimate_umeyama, ransac_umeyama


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


if __name__ == "__main__":
    unittest.main()

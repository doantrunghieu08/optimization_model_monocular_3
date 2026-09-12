import unittest

from fusion_pipeline.correction import ransac_umeyama


class FusionCorrectionTest(unittest.TestCase):
    def test_similarity_requires_three_anchors(self):
        camera = {"a": [0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]}

        with self.assertRaisesRegex(ValueError, "At least 3 anchors"):
            ransac_umeyama(camera, camera, ["a", "b"], threshold=0.05, max_combos=10)


if __name__ == "__main__":
    unittest.main()

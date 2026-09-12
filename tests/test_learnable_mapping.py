import unittest

from keypoints_map import load_keypoints3d_map
from learnable_pipeline.executor import _get_custom_to_body25


class LearnableMappingTest(unittest.TestCase):
    def test_canonical_joints_map_to_body25(self):
        mapping = _get_custom_to_body25(load_keypoints3d_map("configs/keypoints3D_map.yml"))

        self.assertEqual(len(mapping), 19)
        self.assertEqual(mapping["head"], "nose")
        self.assertEqual(mapping["pelvis"], "mid_hip")
        self.assertEqual(mapping["left_foot"], "left_small_toe")
        self.assertNotIn("left_hand", mapping)


if __name__ == "__main__":
    unittest.main()

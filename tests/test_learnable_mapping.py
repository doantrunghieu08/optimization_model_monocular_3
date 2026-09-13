import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from keypoints_map import load_keypoints3d_map
from learnable_pipeline.executor import _get_custom_to_body25, _load_wham_starts, _slice_person, load_stage_results


class LearnableMappingTest(unittest.TestCase):
    def test_canonical_joints_map_to_body25(self):
        mapping = _get_custom_to_body25(load_keypoints3d_map("configs/keypoints3D_map.yml"))

        self.assertEqual(len(mapping), 19)
        self.assertEqual(mapping["head"], "nose")
        self.assertEqual(mapping["pelvis"], "mid_hip")
        self.assertEqual(mapping["left_foot"], "left_small_toe")
        self.assertNotIn("left_hand", mapping)

    def test_wham_slice_uses_sync_start(self):
        person = {
            "pose": np.arange(10).reshape(5, 2),
            "trans": np.arange(15).reshape(5, 3),
            "betas": np.ones((1, 10)),
        }

        sliced = _slice_person(person, start=2, frame_count=2)

        np.testing.assert_array_equal(sliced["pose"], person["pose"][2:4])
        np.testing.assert_array_equal(sliced["trans"], person["trans"][2:4])
        self.assertIs(sliced["betas"], person["betas"])

    def test_wham_starts_follow_pose_sync_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "pose_data_3.json").write_text(json.dumps({
                "metadata": {"camera_sync": {"left_start": 4, "right_start": 9}}
            }), encoding="utf-8")

            self.assertEqual(
                _load_wham_starts(Path(tmp), first_frame_id=3),
                {"camera1": 6, "camera2": 11},
            )

            (metadata_dir / "pose_data_4.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing camera_sync starts"):
                _load_wham_starts(Path(tmp), first_frame_id=4)

    def test_stage_loader_propagates_input_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keypoints3d").mkdir()
            (root / "metadata").mkdir()
            (root / "keypoints3d" / "fused_data_1.json").write_text(
                json.dumps({"camera1": {}, "camera2": {}}), encoding="utf-8"
            )
            (root / "metadata" / "fused_data_1.json").write_text(
                json.dumps({"metadata": {"fusion_config": {"enabled": True}}}), encoding="utf-8"
            )

            result = load_stage_results(root, "fused_data_", "Learnable")

            self.assertTrue(result[0]["metadata"]["fusion_config"]["enabled"])


if __name__ == "__main__":
    unittest.main()

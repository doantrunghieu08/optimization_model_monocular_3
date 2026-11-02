"""MBLE safety: same 20 skeleton edges as evaluation, no GT used in fitting."""
import unittest
import numpy as np

from fusion_pipeline.executor import _enforce_bone_lengths, KINEMATIC_CHAINS_FULL


class MBLEPreservationTests(unittest.TestCase):
    def make_pose(self):
        # Pose is non-degenerate with 21 joints and varied limb lengths.
        root = {
            "pelvis": [0., 0., 0.], "neck": [0., .48, .02],
            "head": [0., .65, .06],
            "left_shoulder": [-.22, .49, 0.], "right_shoulder": [.21, .49, 0.],
            "left_hip": [-.11, -.05, 0.], "right_hip": [.11, -.05, 0.],
            "left_elbow": [-.42, .34, 0.], "right_elbow": [.39, .35, 0.],
            "left_wrist": [-.52, .17, .04], "right_wrist": [.49, .18, -.02],
            "left_hand": [-.58, .1, .05], "right_hand": [.55, .09, -.03],
            "left_knee": [-.13, -.47, .05], "right_knee": [.12, -.45, -.03],
            "left_ankle": [-.14, -.84, .06], "right_ankle": [.13, -.85, -.02],
            "left_toe": [-.17, -.9, .22], "right_toe": [.15, -.91, .21],
            "left_foot": [-.15, -.86, .21], "right_foot": [.14, -.86, .19],
        }
        return root

    def test_20_edges_match_evaluation_skeleton(self):
        from pathlib import Path
        import yaml
        map_path = Path(__file__).resolve().parents[1] / "configs/keypoints3D_map.yml"
        skel = yaml.safe_load(map_path.read_text(encoding="utf-8"))["skeleton"]
        self.assertEqual(len(skel), 20)
        self.assertEqual(
            {frozenset(x) for x in KINEMATIC_CHAINS_FULL},
            {frozenset(x) for x in skel},
        )

    def test_all_bone_lengths_equal_input_for_both_camera_views(self):
        raw = self.make_pose()
        for camera_seed in (0, 1):
            moved = {}
            rng = np.random.default_rng(camera_seed)
            for name, value in raw.items():
                moved[name] = (np.asarray(value) + rng.normal(0, .06, 3)).tolist()
            _enforce_bone_lengths(moved, raw, full_skeleton=True)
            for child, parent in KINEMATIC_CHAINS_FULL:
                got = np.linalg.norm(np.asarray(moved[child]) - np.asarray(moved[parent]))
                expected = np.linalg.norm(np.asarray(raw[child]) - np.asarray(raw[parent]))
                self.assertLess(abs(got - expected), 2.0e-6, (child, parent))

    def test_robust_median_projection_is_opt_in(self):
        raw = self.make_pose()
        moved = {name: list(v) for name, v in raw.items()}
        _enforce_bone_lengths(moved, raw, {"left_wrist": .4}, full_skeleton=True)
        self.assertAlmostEqual(
            np.linalg.norm(np.asarray(moved["left_wrist"]) - np.asarray(moved["left_elbow"])),
            .4, places=5
        )


if __name__ == "__main__":
    unittest.main()

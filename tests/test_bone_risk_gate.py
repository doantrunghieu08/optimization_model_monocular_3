"""M2 bone length + M4 negative-transfer safety + M5 robust projection tests."""
import unittest
import numpy as np
from fusion_pipeline.bone_quality import estimate_bone_quality
from fusion_pipeline.adaptive_gate import apply_adaptive_joint_fusion


def view(wrist, elbow=(0., 0., 0.)):
    return {
        "left_elbow": list(elbow),
        "left_wrist": list(wrist),
    }


def fuse(cam1, cam2, *, vis1=True, vis2=True, bone1=1.0, bone2=1.0,
         refs=None, anchors=None, config=None):
    identity = (1.0, np.eye(3), np.zeros(3))
    return apply_adaptive_joint_fusion(
        cam1, cam2, list(cam1), identity, identity,
        {"left_wrist": .8}, {"left_wrist": .95},
        {"left_wrist": vis1}, {"left_wrist": vis2},
        joint_quality={"camera1": {"left_wrist": bone1},
                       "camera2": {"left_wrist": bone2}},
        bone_quality={"camera1": {"left_wrist": bone1},
                      "camera2": {"left_wrist": bone2}},
        bone_references=refs, anchor_names=anchors, config=config,
    )


class BoneRiskGateTests(unittest.TestCase):
    def test_bone_quality_flags_single_frame_outlier(self):
        frames = [
            {"camera1": view((2.0 if i == 2 else 1.0, 0, 0)),
             "camera2": view((1., 0, 0))}
            for i in range(5)
        ]
        quality, reference = estimate_bone_quality(frames)
        self.assertAlmostEqual(reference["camera1"]["left_wrist"], 1.0, places=6)
        self.assertLess(quality[2]["camera1"]["left_wrist"], .15)
        self.assertGreater(quality[1]["camera1"]["left_wrist"], .95)
        self.assertGreater(quality[2]["camera2"]["left_wrist"], .95)

    def test_bilateral_detects_persistent_asymmetric_length(self):
        frames = [
            {"camera1": {
                "left_shoulder": [0., 0., 0.], "left_elbow": [1., 0., 0.],
                "right_shoulder": [0., 0., 0.], "right_elbow": [2., 0., 0.]},
             "camera2": {}}
            for _ in range(5)
        ]
        quality, reference = estimate_bone_quality(frames)
        self.assertAlmostEqual(reference["camera1"]["right_elbow"], 2.0)
        self.assertLess(quality[2]["camera1"]["right_elbow"], 1.0)
        self.assertLess(quality[2]["camera1"]["left_elbow"], 1.0)

    def test_missing_reference_neutral_and_bad_segment_zero(self):
        frames = [{"camera1": view((1., 0, 0)), "camera2": view((1., 0, 0))}
                  for _ in range(2)]
        quality, refs = estimate_bone_quality(frames)
        self.assertNotIn("left_wrist", refs["camera1"])
        self.assertEqual(quality[0]["camera1"]["left_wrist"], 1.0)
        broken = [{"camera1": view((0., 0, 0)), "camera2": view((1., 0, 0))} for _ in range(5)]
        qbad, _ = estimate_bone_quality(broken)
        self.assertEqual(qbad[2]["camera1"]["left_wrist"], 0.0)

    def test_occluded_rescue_requires_bone_evidence(self):
        master, slave = view((1., 0, 0)), view((.9, .2, 0))
        _, _, good = fuse(master, slave, vis1=False, bone2=1.)
        _, _, bad = fuse(master, slave, vis1=False, bone2=.1)
        self.assertGreater(good["joint_transfer"]["camera1"]["left_wrist"], 0)
        self.assertEqual(bad["joint_transfer"]["camera1"]["left_wrist"], 0)
        self.assertEqual(bad["adaptive_stats"]["rejected_occluded_bone"], 1)

    def test_alignment_without_three_anchors_disables_transfer(self):
        master, slave = view((1., 0, 0)), view((.9, .2, 0))
        _, _, metadata = fuse(master, slave, anchors=["pelvis", "neck"])
        self.assertEqual(metadata["joint_transfer"]["camera1"]["left_wrist"], 0)
        self.assertGreater(metadata["adaptive_stats"]["rejected_alignment"], 0)

    def test_m5_projects_using_sequence_median_not_bad_current_length(self):
        master, slave = view((1.5, 0, 0)), view((.6, .8, 0))
        a, _, metadata = fuse(
            master, slave, bone1=.1, bone2=1.0,
            refs={"camera1": {"left_wrist": 1.0}, "camera2": {"left_wrist": 1.0}},
        )
        self.assertGreater(metadata["joint_transfer"]["camera1"]["left_wrist"], 0)
        self.assertAlmostEqual(np.linalg.norm(a["left_wrist"] - a["left_elbow"]), 1.0, places=5)

    def test_nonfinite_source_is_rejected(self):
        master, slave = view((1., 0, 0)), view((float("nan"), 0, 0))
        _, _, meta = fuse(master, slave)
        self.assertEqual(meta["joint_transfer"]["camera1"]["left_wrist"], 0)


if __name__ == "__main__":
    unittest.main()

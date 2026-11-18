"""Track -> person labelling used when a dataset has no *_person_mapping.txt."""
import unittest

from preprocessing_data.label_persons import _segments, summarize_track

PERSONS = ["aria01", "aria02"]


def frames(votes):
    return [(i, v, {}) for i, v in enumerate(votes)]


class LabelPersonsTests(unittest.TestCase):
    def test_majority_label_and_switch_frames(self):
        s = summarize_track(frames(["aria01"] * 7 + ["aria02"] * 3), PERSONS, 0.5)
        self.assertEqual(s["label"], "aria01")
        self.assertEqual(s["frames_following_other_person"], [7, 8, 9])

    def test_bystander_track_stays_unlabelled(self):
        s = summarize_track(frames(["none"] * 8 + ["aria01"] * 2), PERSONS, 0.5)
        self.assertIsNone(s["label"])
        self.assertEqual(s["frames_following_other_person"], [])

    def test_ambiguous_frames_do_not_vote(self):
        s = summarize_track(frames(["ambiguous"] * 6 + ["aria02"] * 4), PERSONS, 0.5)
        self.assertIsNone(s["label"])  # 4/10 < 0.5

    def test_empty_track(self):
        s = summarize_track([], PERSONS, 0.5)
        self.assertIsNone(s["label"])
        self.assertEqual(s["frames_with_gt"], 0)

    def test_segments_compress_runs(self):
        self.assertEqual(_segments([(0, "a"), (1, "a"), (2, "b"), (5, "b"), (6, "a")]),
                         [("a", 0, 1, 2), ("b", 2, 5, 2), ("a", 6, 6, 1)])


if __name__ == "__main__":
    unittest.main()

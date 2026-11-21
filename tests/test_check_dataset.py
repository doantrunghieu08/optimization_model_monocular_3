"""preprocessing_data.check_dataset on a tiny synthetic Harmony4D-style sequence."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from preprocessing_data.check_dataset import check_sequence, find_sequences, frame_gaps


def _make_sequence(root, cams=("cam01", "cam02"), mapping=True, gt_ids=range(1, 21), dup_gt=True,
                   shifted_cam=None, extrinsics=False, poses2d=True):
    seq = Path(root) / "099_test"
    for c in cams:
        d = seq / "exo" / c / f"harmony4D_099_test_{c}_wham"
        d.mkdir(parents=True)
        start = 50 if c == shifted_cam else 0
        tracks = {0: {"frame_ids": np.arange(start, start + 20), "verts": np.zeros((20, 4, 3))},
                  3: {"frame_ids": np.arange(start, start + 20), "verts": np.zeros((20, 4, 3))}}
        joblib.dump(tracks, d / f"wham_harmony4D_099_test_{c}_ffmpeg.pkl")
        if mapping:
            (d / f"wham_harmony4D_099_test_{c}_person_mapping.txt").write_text("0 = aria01\n3 = aria02\n")
        (seq / "exo" / c / "images").mkdir()
        (seq / "exo" / c / "images" / f"{c}.mp4").write_bytes(b"")
    gt = seq / "processed_data" / "poses3d"
    gt.mkdir(parents=True)
    for i in gt_ids:
        np.save(gt / f"{i:05d}.npy", {"aria01": np.ones((17, 4)), "aria02": np.ones((17, 4))})
    if dup_gt:
        np.save(gt / "00005(1).npy", {"aria01": np.ones((17, 4))})
    if poses2d:
        (seq / "processed_data" / "poses2d" / cams[0]).mkdir(parents=True)
    w = seq / "colmap" / "workplace"
    w.mkdir(parents=True)
    (w / "cameras.txt").write_text("# header\n1 PINHOLE 10 10 1 1 5 5\n2 PINHOLE 10 10 1 1 5 5\n")
    if extrinsics:
        (w / "images.txt").write_text("# header\n")
    return seq


def _levels(rep, area):
    return {i["level"] for i in rep.items if i["area"] == area}


class CheckDatasetTests(unittest.TestCase):
    def test_clean_sequence_has_no_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = check_sequence(_make_sequence(tmp, dup_gt=False), ["aria01", "aria02"])
            self.assertEqual(rep.n_errors, 0, rep.items)
            self.assertEqual(rep.facts["gt"]["frames_per_person"], {"aria01": 20, "aria02": 20})
            self.assertEqual(rep.facts["cameras"]["cam01"]["gt_overlap"], {"aria01": 20, "aria02": 20})
            self.assertIn("WARN", _levels(rep, "2d"))       # no tracking_results*.pth
            self.assertIn("WARN", _levels(rep, "gt-leak"))  # processed_data/poses2d exists
            self.assertIn("WARN", _levels(rep, "calib"))    # intrinsics only

    def test_missing_mapping_is_error_and_suggests_label_persons(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = check_sequence(_make_sequence(tmp, mapping=False), ["aria01", "aria02"])
            self.assertIn("WARN", _levels(rep, "mapping"))
            self.assertIn("ERROR", _levels(rep, "layout"))  # no camera usable
            self.assertTrue(any("label_persons" in s for s in rep.facts["next_steps"]))

    def test_duplicates_gaps_and_extrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = [i for i in range(1, 21) if i not in (7, 8)]
            rep = check_sequence(_make_sequence(tmp, gt_ids=ids, extrinsics=True), ["aria01", "aria02"])
            msgs = " ".join(i["msg"] for i in rep.items)
            self.assertIn("00005(1).npy", msgs)
            self.assertEqual(rep.facts["gt"]["gaps"], [(5, 8)])
            self.assertIn("OK", _levels(rep, "calib"))

    def test_camera_without_gt_overlap_and_without_shared_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = check_sequence(_make_sequence(tmp, shifted_cam="cam02"), ["aria01", "aria02"])
            self.assertIn("ERROR", _levels(rep, "sync"))
            self.assertTrue(any("share no WHAM frame id" in i["msg"] for i in rep.items))

    def test_find_sequences_and_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = _make_sequence(tmp)
            self.assertEqual(find_sequences(tmp), [seq])
            self.assertEqual(find_sequences(seq), [seq])
        self.assertEqual(frame_gaps([0, 1, 2, 5, 6, 9]), [(2, 5), (6, 9)])


if __name__ == "__main__":
    unittest.main()

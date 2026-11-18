"""Identity-mask evaluation filter (frames where the WHAM track follows the wrong person)."""
import json
import tempfile
import unittest
from pathlib import Path

from evaluation_pipeline.executor import _identity_excluded_ids


class IdentityMaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mask.json"
        self.path.write_text(json.dumps({"margins": {
            "wham_cam02": {"aria01": {"0": -20.0, "5": 9.9, "6": 10.1, "7": 35.0}},
        }}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_disabled_mask_excludes_nothing(self):
        self.assertEqual(_identity_excluded_ids(None, "wham_cam02", "aria01"), set())
        self.assertEqual(_identity_excluded_ids({"path": None}, "wham_cam02", "aria01"), set())

    def test_threshold_is_strict_and_configurable(self):
        cfg = {"path": str(self.path), "margin_mm": 10.0}
        self.assertEqual(_identity_excluded_ids(cfg, "wham_cam02", "aria01"), {6, 7})
        cfg20 = {"path": str(self.path), "margin_mm": 20.0}
        self.assertEqual(_identity_excluded_ids(cfg20, "wham_cam02", "aria01"), {7})

    def test_unknown_camera_or_person_excludes_nothing(self):
        cfg = {"path": str(self.path), "margin_mm": 10.0}
        self.assertEqual(_identity_excluded_ids(cfg, "wham_cam99", "aria01"), set())
        self.assertEqual(_identity_excluded_ids(cfg, "wham_cam02", "aria02"), set())

    def test_multiple_mask_files_are_merged(self):
        second = Path(self.tmp.name) / "mask2.json"
        second.write_text(json.dumps({"margins": {"wham_cam77": {"aria02": {"3": 50.0}}}}))
        cfg = {"path": [str(self.path), str(second)], "margin_mm": 10.0}
        self.assertEqual(_identity_excluded_ids(cfg, "wham_cam02", "aria01"), {6, 7})
        self.assertEqual(_identity_excluded_ids(cfg, "wham_cam77", "aria02"), {3})

    def test_gt_loader_ignores_download_duplicates(self):
        import numpy as np
        from evaluation_pipeline.executor import _load_ground_truth
        gt_dir = Path(self.tmp.name) / "poses3d"
        gt_dir.mkdir()
        for stem, value in (("00001", 1.0), ("00002", 2.0), ("00001(1)", 9.0), ("00003", 3.0)):
            arr = np.full((17, 4), value)
            np.save(gt_dir / f"{stem}.npy", {"aria01": arr}, allow_pickle=True)
        gt = _load_ground_truth(gt_dir / "_.json", truth_dir=gt_dir, target_person="aria01")
        self.assertEqual(sorted(gt), [0, 1, 2])
        self.assertEqual(float(gt[1]["joints"]["left_hip"][0]), 2.0)

    def test_person_name_is_case_insensitive(self):
        cfg = {"path": str(self.path), "margin_mm": 10.0}
        self.assertEqual(_identity_excluded_ids(cfg, "wham_cam02", "ARIA01"), {6, 7})


if __name__ == "__main__":
    unittest.main()

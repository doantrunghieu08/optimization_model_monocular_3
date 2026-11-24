import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch

from train_v4 import (AlignmentError, associate, centered, export_model, fit, prepare,
                      torch_fuse, validate_rows, synchronize)
from keypoints_map import load_keypoints3d_map
from fusion_pipeline.partwise import extract_pair_features, fuse_partwise, slave_directions
from test_partwise import CFG, skeleton


def pair(split="train"):
    master = [skeleton(elbow_lift=0.01 * t) for t in range(10)]
    slave = [skeleton(depth=6, elbow_lift=0.02 * t) for t in range(10)]
    names = list(master[0])
    features, children = extract_pair_features(master, slave)
    return dict(action=split, split=split, M=master, S=slave, names=names,
                features=features, children=children,
                directions=slave_directions(master, slave, 4),
                target=centered(np.array([[f[n] for n in names] for f in slave]), names).astype(np.float32))


class TrainingTests(unittest.TestCase):
    def test_prepare_reports_rejected_camera_and_keeps_valid_pairs(self):
        mapping = load_keypoints3d_map("configs/keypoints3D_map.yml")
        names = [k["name"] for k in mapping["keypoints"]]
        joints = np.array([[skeleton(elbow_lift=0.01*t)[n] for n in names] for t in range(24)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split, count in (("train", 3), ("val", 2)):
                gt_id = f"{split}_ground_truth"
                with (root / gt_id).open("wb") as stream:
                    np.savez(stream, verts=joints)
                for i in range(count):
                    pkl_id = f"{split}_camera_{i}"
                    length = 8 if split == "train" and i == 0 else len(joints)
                    joblib.dump({"0": dict(frame_ids=np.arange(length), verts=joints[:length])}, root / pkl_id)
                    rows.append(dict(action=split, split=split, cam=f"cam{i}",
                                     GT=gt_id, **{"file PKL": pkl_id}))
            report_path = root / "alignment.json"
            with patch("train_v4.download", side_effect=lambda drive, value, cache: root/value), \
                 patch("train_v4.joints_from_vertices", side_effect=lambda vertices, reg: vertices):
                pairs, reports = prepare(rows, None, root, CFG, report_path)
                rejected = [r for r in reports if r["status"] == "rejected"]
                self.assertEqual([(r["action"], r["cam"]) for r in rejected], [("train", "cam0")])
                self.assertEqual(json.loads(report_path.read_text()), reports)
                self.assertEqual({p["split"] for p in pairs}, {"train", "val"})
                self.assertTrue(all("cam0" not in p["cameras"] for p in pairs if p["split"] == "train"))
                with self.assertRaisesRegex(ValueError, "train: need at least two aligned cameras"):
                    prepare([r for r in rows if not (r["split"] == "train" and r["cam"] == "cam2")],
                            None, root, CFG, report_path)
                self.assertEqual(json.loads(report_path.read_text())[0]["status"], "rejected")

    def test_alignment_rejects_insufficient_overlap(self):
        names = pair()["names"]
        gt = np.zeros((100, len(names), 3))
        with self.assertRaisesRegex(AlignmentError, "No alignment covers"):
            associate({"0": (np.arange(16), gt[:16])}, gt, names)

    def test_leakage_and_duplicates_rejected(self):
        rows = [dict(action=a, cam=c, split=s, GT=g, **{"file PKL": "some_id"})
                for a,s,g in [("a","train","g1"),("b","val","g2")] for c in ["cam0","cam1"]]
        validate_rows(rows)
        bad = copy.deepcopy(rows)
        bad[1]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_rows(bad)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_rows(rows + rows[:1])
        bad = copy.deepcopy(rows)
        bad[2]["GT"] = "g1"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_rows(bad)

    def test_differentiable_fusion_matches_inference(self):
        p = pair()
        w = torch.nn.Parameter(torch.arange(8, dtype=torch.float32) / 10)
        std = torch.ones(8)
        actual = torch_fuse(p, w, std)
        cfg = dict(CFG, weight_model=export_model(w.detach().numpy(), std.numpy()))
        result, _ = fuse_partwise(p["M"], p["S"], cfg)
        expected = centered(np.array([[f[n] for n in p["names"]] for f in result]), p["names"])
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-6)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_association_recovers_track_and_offset(self):
        rng = np.random.default_rng(9)
        p = pair()
        names = p["names"]
        gt = rng.normal(size=(45, len(names), 3))
        correct = gt[7:37] @ np.array([[0,-1,0],[1,0,0],[0,0,1]]) + 4
        wrong = rng.normal(size=correct.shape)
        ids, joints, report = associate({"wrong": (np.arange(30), wrong),
                                        "right": (np.arange(30), correct)}, gt, names)
        self.assertEqual(report["track"], "right")
        self.assertEqual(report["ratio"], 1)
        self.assertEqual(report["offset"], 7)
        np.testing.assert_array_equal(ids, np.arange(7,37))

    def test_fragmented_tracks_and_double_rate_gt(self):
        rng = np.random.default_rng(2)
        names = pair()["names"]
        gt = rng.normal(size=(60, len(names), 3))
        joints = gt[::2] + 4
        ids, actual, report = associate({"first": (np.arange(15), joints[:15]),
                                        "second": (np.arange(15,30), joints[15:])}, gt, names)
        self.assertEqual(report["ratio"], 2)
        self.assertEqual(report["offset"], 0)
        self.assertEqual(report["switches"], 1)
        np.testing.assert_array_equal(ids, np.arange(0,60,2))
        np.testing.assert_allclose(actual, joints)

    def test_camera_phase_interpolation(self):
        im, is_ = np.arange(0,20,2), np.arange(1,20,2)
        jm, js = im[:,None,None] * np.ones((1,3,3)), is_[:,None,None] * np.ones((1,3,3))
        ticks, master, slave = synchronize(im,jm,is_,js)
        np.testing.assert_array_equal(ticks, np.arange(1,19))
        np.testing.assert_allclose(master, slave)
        np.testing.assert_allclose(master[:,0,0], ticks)

    def test_checkpoint_best_and_resume_match_uninterrupted(self):
        pairs = [pair(), pair("val")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = fit(pairs, CFG, root/"full", epochs=3)
            fit(pairs, CFG, root/"part", epochs=1)
            resumed = fit(pairs, CFG, root/"resume", epochs=3, resume=root/"part/epoch_0001.pt")
            self.assertEqual(full, resumed)
            best = json.loads((root/"resume/best_weights.json").read_text())
            self.assertEqual(best["val_mpjpe_mm"], min(h["val_mpjpe_mm"] for h in resumed))
            self.assertTrue((root/"resume/epoch_0003.pt").exists())
            for key in ["weights","std"]:
                a = json.loads((root/"full/epoch_0003_weights.json").read_text())[key]
                b = json.loads((root/"resume/epoch_0003_weights.json").read_text())[key]
                np.testing.assert_allclose(a,b)
            with self.assertRaisesRegex(ValueError, "differs"):
                fit(pairs, CFG, root/"bad", epochs=3, lr=0.2, resume=root/"part/epoch_0001.pt")


if __name__ == "__main__":
    unittest.main()

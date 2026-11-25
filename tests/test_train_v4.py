import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch

from train_v4 import (AlignmentError, associate, centered, export_model, fit,
                      index_mounted_files, prepare, resolve_mounted, torch_fuse,
                      validate_rows, synchronize)
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
            with patch("train_v4.resolve_mounted", side_effect=lambda drive, value, mount, state=None: root / value), \
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
                bad_rows = [dict(action="bad", split="train", cam=f"cam{i}",
                                 GT="train_ground_truth", **{"file PKL": "train_camera_0"})
                            for i in range(2)]
                pairs, reports = prepare(rows + bad_rows, None, root, CFG, report_path,
                                         skip_unaligned_actions=True)
                self.assertEqual({p["action"] for p in pairs}, {"train", "val"})
                skipped = [r for r in reports if r["status"] == "skipped_action"]
                self.assertEqual([r["action"] for r in skipped], ["bad"])
                self.assertEqual(json.loads(report_path.read_text()), reports)
                self.assertTrue(all(r["camera_frames"] == 8 and r["gt_frames"] == 24
                                    for r in reports if r["status"] == "rejected"))
                with self.assertRaisesRegex(ValueError, "No usable pairs for one of the splits"):
                    prepare(bad_rows + [r for r in rows if r["split"] == "val"], None, root, CFG,
                            report_path, skip_unaligned_actions=True)

    def test_resolve_mounted_mydrive_shortcut_and_duplicates(self):
        class FakeReq:
            def __init__(self, payload):
                self.payload = payload
            def execute(self, num_retries=0):
                if isinstance(self.payload, Exception):
                    raise self.payload
                return self.payload

        class FakeFiles:
            def __init__(self, db):
                self.db = db
            def get(self, fileId, fields="", supportsAllDrives=False):
                return FakeReq(self.db.get(fileId, RuntimeError("404")))

        class FakeDrive:
            def __init__(self, db):
                self._files = FakeFiles(db)
            def files(self):
                return self._files

        with tempfile.TemporaryDirectory() as tmp:
            mount = Path(tmp) / "drive"
            (mount / "MyDrive/ds/seq").mkdir(parents=True)
            f1 = mount / "MyDrive/ds/seq/wham.pkl"
            f1_dup = mount / "MyDrive/ds/seq/wham (1).pkl"
            f1.write_bytes(b"a")
            f1_dup.write_bytes(b"b")
            sc_dir = mount / ".shortcut-targets-by-id/shared_folder_12345"
            sc_dir.mkdir(parents=True)
            f2 = sc_dir / "sub/gt.npz"
            f2.parent.mkdir(parents=True)
            f2.write_bytes(b"gt")

            db = {
                "root": {"id": "root_0000000000"},
                "ds_000000000000": {"id": "ds_000000000000", "name": "ds", "mimeType": "application/vnd.google-apps.folder", "parents": ["root_0000000000"]},
                "seq_00000000000": {"id": "seq_00000000000", "name": "seq", "mimeType": "application/vnd.google-apps.folder", "parents": ["ds_000000000000"]},
                "file_dup_000000": {"id": "file_dup_000000", "name": "wham.pkl", "parents": ["seq_00000000000"]},
                "sub_00000000000": {"id": "sub_00000000000", "name": "sub", "mimeType": "application/vnd.google-apps.folder", "parents": ["shared_folder_12345"]},
                "gt_shared_00000": {"id": "gt_shared_00000", "name": "gt.npz", "parents": ["sub_00000000000"]},
                "missing_id_0000": {"id": "missing_id_0000", "name": "x.pkl", "parents": ["unshared_folder_99"]},
                "orphan_id_00000": {"id": "orphan_id_00000", "name": "solo.npz", "owners": [{"emailAddress": "owner@lab.org"}]},
            }
            drive = FakeDrive(db)
            xattrs = {str(f1): "other_id_000000", str(f1_dup): "file_dup_000000"}
            state = {}
            with patch("train_v4._xattr_drive_id", side_effect=lambda p: xattrs.get(str(p))):
                # 0-API xattr scan works on an explicit dataset folder
                scanned = index_mounted_files({"file_dup_000000"}, mount / "MyDrive/ds")
                self.assertEqual(scanned["file_dup_000000"], f1_dup)
                self.assertEqual(resolve_mounted(drive, "file_dup_000000", mount, state), f1_dup)
                self.assertEqual(resolve_mounted(drive, "gt_shared_00000", mount, state), f2)
                with self.assertRaisesRegex(FileNotFoundError, "folders/unshared_folder_99"):
                    resolve_mounted(drive, "missing_id_0000", mount, state)
                with self.assertRaisesRegex(FileNotFoundError, "owner@lab.org"):
                    resolve_mounted(drive, "orphan_id_00000", mount, state)

    def test_untrimmed_camera_longer_than_gt_aligns_at_true_rate(self):
        rng = np.random.default_rng(7)
        names = pair()["names"]
        gt = rng.normal(size=(40, len(names), 3))
        extra = rng.normal(size=(60, len(names), 3))
        cam = np.concatenate([gt @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + 2, extra], axis=0)
        ids, _, report = associate({"0": (np.arange(100), cam)}, gt, names)
        self.assertEqual(report["ratio"], 1.0)
        self.assertEqual(report["offset"], 0)
        np.testing.assert_array_equal(ids, np.arange(40))

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

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch

from train_v4 import (BODY, AlignmentError, associate, centered, export_model, fit,
                      index_mounted_files, prepare, resolve_mounted, torch_fuse,
                      validate_rows, synchronize, validation, rotation, training_options,
                      training_config, naive_relative_loss, selection_key, naive_prediction, torch_pa_loss, pa_errors,
                      tracks_from_wham, synchronize_visibility)
from keypoints_map import load_keypoints3d_map
from fusion_pipeline.partwise import eval_weight_model, extract_pair_features, fuse_partwise, slave_directions
from test_partwise import CFG, skeleton, rotate


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
    def test_prepare_learns_mesh_occlusion_without_confidence2d_and_reuses_occ_cache(self):
        names = [k["name"] for k in load_keypoints3d_map("configs/keypoints3D_map.yml")["keypoints"]]
        gt = np.array([[skeleton(elbow_lift=0.01*t)[n] for n in names] for t in range(24)])
        cfg = dict(CFG, occlusion={"enabled": True, "multi_person": True})
        def visibility(pose, mesh, **kwargs):
            return {n: not (n == "left_elbow" and pose["pelvis"][2] < 5) for n in names}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "gt.npz", verts=gt)
            # Reversed fragments and nonzero video frame IDs exercise track/time association.
            joblib.dump({"b": dict(frame_ids=np.arange(112, 124), verts=gt[12:]),
                         "a": dict(frame_ids=np.arange(100, 112), verts=gt[:12])}, root / "cam0.pkl")
            joblib.dump({"c": dict(frame_ids=np.arange(200, 224), verts_cam=gt + [0, 0, 2])}, root / "cam1.pkl")
            rows = [dict(action=s, split=s, cam=c, GT=f"ground_truth_{s}", GT_path=str(root / "gt.npz"),
                         PKL_path=str(root / f"{c}.pkl"), **{"file PKL": f"camera_{s}_{c}_012345"})
                    for s in ("train", "val") for c in ("cam0", "cam1")]
            # Old joint-only caches must not mask the newly available occlusion data.
            for row in rows:
                joblib.dump({}, root / f"{row['file PKL']}.joints.joblib")
            with patch("train_v4.joints_from_vertices", side_effect=lambda v, r: v), \
                 patch("train_v4.require_ray_casting_backend"), \
                 patch("train_v4.load_torso_mask", return_value=np.ones(6890, dtype=bool)), \
                 patch("train_v4.compute_visibility_from_mesh_vertices", side_effect=visibility) as compute:
                pairs, reports = prepare(rows, None, root, cfg)
                compute.assert_called()
                compute.reset_mock()
                cached, _ = prepare(rows, None, root, cfg)
                compute.assert_not_called()
                changed = dict(cfg, occlusion=dict(cfg["occlusion"], tau=0.1))
                prepare(rows, None, root, changed)
                compute.assert_called()
            self.assertEqual(len(pairs), 4)
            forward, reverse = pairs[:2]
            elbow = forward["children"].index("left_elbow")
            np.testing.assert_array_equal(forward["features"][:, elbow, 1], -1)
            np.testing.assert_allclose(forward["features"][:, :, 2], -1 / len(names))
            np.testing.assert_allclose(forward["features"], -reverse["features"], atol=1e-6)
            np.testing.assert_array_equal(forward["features"][:, :, 3], 0)
            np.testing.assert_array_equal(cached[0]["features"], forward["features"])
            self.assertTrue(any(r["occluded_joint_fraction"] > 0 for r in reports))
            with patch("train_v4.fuse_partwise", wraps=fuse_partwise) as fusion:
                validation([forward], export_model(np.ones(8), np.ones(8)), cfg)
            self.assertIs(fusion.call_args.kwargs["vis_master"], forward["vis_master"])
            self.assertIs(fusion.call_args.kwargs["vis_slave"], forward["vis_slave"])

    def test_visibility_interpolation_preserves_exact_ticks_and_requires_both_endpoints(self):
        ids = np.array([0, 2, 4])
        vis = np.array([[True, False], [False, True], [True, True]])
        actual = synchronize_visibility(ids, vis, np.arange(5))
        np.testing.assert_array_equal(actual, [[True, False], [False, False], [False, True],
                                              [False, True], [True, True]])

    def test_mesh_occlusion_uses_camera_vertices_and_other_tracks_at_their_own_frame_ids(self):
        reg = np.zeros((2, 6890))
        reg[0, 8], reg[1, 9] = 1, 1
        cube = np.array(list(itertools.product([-0.3, 0.3], [-0.3, 0.3], [2, 3])))
        mesh = np.zeros((6890, 3))
        mesh[:8] = cube
        mesh[8:10] = [[0, 0, 4], [1, 0, 4]]
        other = np.full((1, 6890, 3), [0.5, 0, 2])
        raw = {"target": dict(frame_ids=[10, 11], verts_cam=np.stack([mesh, mesh]),
                               verts=np.full((2, 6890, 3), 99.0)),
               "other": dict(frame_ids=[11], verts=other)}
        mask = np.zeros(6890, dtype=bool)
        mask[:8] = True
        with patch("fusion_pipeline.detector._TORSO_MASK", mask):
            tracks, vis = tracks_from_wham(raw, reg, ["left_wrist", "right_wrist"],
                                          {"enabled": True, "multi_person": True})
        np.testing.assert_allclose(tracks["target"][1][0], mesh[8:10])
        # Left wrist is behind its own torso; the other track exists only at frame 11.
        self.assertFalse(vis["target"][:, 0].any())
        self.assertTrue(vis["target"][0, 1])
        self.assertFalse(vis["target"][1, 1])
        with patch("train_v4.joints_from_vertices", side_effect=lambda v, r: v):
            with self.assertRaisesRegex(ValueError, "Invalid/duplicate frame ids"):
                tracks_from_wham({"bad": dict(frame_ids=[1, 1], verts=np.stack([mesh, mesh]))},
                                 reg, ["left_wrist", "right_wrist"], {"enabled": False})

    def test_occ_features_receive_gradients_and_are_exported_without_confidence2d(self):
        p = pair()
        vm = [{"left_elbow": t % 2 == 0, "left_wrist": t % 2 == 0} for t in range(len(p["M"]))]
        vs = [{"left_elbow": t % 2 != 0, "left_wrist": t % 2 != 0} for t in range(len(p["M"]))]
        p["vis_master"], p["vis_slave"] = vm, vs
        p["features"], _ = extract_pair_features(p["M"], p["S"], vm, vs)
        target = [p["M"][t] if t % 2 == 0 else p["S"][t] for t in range(len(p["M"]))]
        p["target"] = centered(np.array([[f[n] for n in p["names"]] for f in target]), p["names"]).astype(np.float32)
        w = torch.zeros(8, requires_grad=True)
        loss = (torch_fuse(p, w, torch.ones(8)) - torch.as_tensor(p["target"])).square().sum()
        loss.backward()
        self.assertTrue(torch.all(w.grad[1:3].abs() > 1e-8))
        self.assertEqual(float(w.grad[3]), 0)
        cfg = dict(CFG, occlusion={"enabled": True})
        with tempfile.TemporaryDirectory() as tmp:
            fit([p, dict(p, split="val", action="val")], cfg, tmp, epochs=1)
            model = json.loads((Path(tmp) / "best_weights.json").read_text())
            self.assertEqual(model["active_features"][1:4], [True, True, False])
            self.assertTrue(all(w > 0 for w in model["weights"][1:3]))
            import yaml
            exported = yaml.safe_load((Path(tmp) / "best_partwise.yml").read_text())
            self.assertEqual(exported["fusion"]["occlusion"], cfg["occlusion"])
            self.assertNotIn("occlusion", exported["fusion"]["partwise"])

    def test_prepare_shares_baseline_rotation_across_disconnected_runs(self):
        names = [k["name"] for k in load_keypoints3d_map("configs/keypoints3D_map.yml")["keypoints"]]
        ids = np.r_[np.arange(12), np.arange(20, 32)]
        gt = np.array([[skeleton()[n] for n in names] for _ in range(32)])
        joints = np.array([[rotate(skeleton(), 0 if t < 12 else 60)[n] for n in names] for t in ids])
        body = [names.index(n) for n in BODY]
        expected = centered(gt[ids], names)
        expected = expected @ rotation(expected[:, body], centered(joints, names)[:, body])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "gt.npz", verts=gt)
            joblib.dump({"0": dict(frame_ids=ids, verts=joints)}, root / "cam.pkl")
            rows = [dict(action=s, split=s, cam=c, GT=f"ground_truth_{s}", GT_path=str(root / "gt.npz"),
                         PKL_path=str(root / "cam.pkl"), **{"file PKL": f"camera_{s}_{c}_012345"})
                    for s in ("train", "val") for c in ("cam0", "cam1")]
            with patch("train_v4.joints_from_vertices", side_effect=lambda v, r: v), \
                 patch("train_v4.associate", return_value=(ids, joints, {})):
                pairs, _ = prepare(rows, None, root, CFG)
            self.assertEqual(len(pairs), 8)
            np.testing.assert_allclose(pairs[0]["target"], expected[:12], atol=1e-6)
            np.testing.assert_allclose(pairs[2]["target"], expected[12:], atol=1e-6)

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

    def test_prepare_falls_back_to_download_and_cleans_temp_files(self):
        mapping = load_keypoints3d_map("configs/keypoints3D_map.yml")
        names = [k["name"] for k in mapping["keypoints"]]
        joints = np.array([[skeleton(elbow_lift=0.01 * t)[n] for n in names] for t in range(24)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_file = root / "raw_gt.npz"
            with gt_file.open("wb") as stream:
                np.savez(stream, verts=joints)
            pkl_file = root / "raw_pkl.pkl"
            joblib.dump({"0": dict(frame_ids=np.arange(len(joints)), verts=joints)}, pkl_file)
            rows = [
                dict(action=s, split=s, cam=c, GT=f"gt_{s}_0123456789", **{"file PKL": f"pkl_{s}_{c}_0123456"})
                for s in ("train", "val") for c in ("cam0", "cam1")
            ]
            cache = root / "cache"
            def fake_dl(drive, fid, dest):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((gt_file if fid.startswith("gt_") else pkl_file).read_bytes())
                return dest
            with patch("train_v4.resolve_mounted", side_effect=FileNotFoundError("not mounted")), \
                 patch("train_v4._download_drive_file", side_effect=fake_dl) as dl_mock, \
                 patch("train_v4.joints_from_vertices", side_effect=lambda vertices, reg: vertices):
                pairs, _ = prepare(rows, object(), cache, CFG, root / "alignment.json")
            self.assertEqual(len(pairs), 4)
            self.assertGreaterEqual(dl_mock.call_count, 6)
            self.assertTrue((cache / "gt_train_0123456789.gt_joints.joblib").exists())
            self.assertTrue((cache / "pkl_train_cam0_0123456.joints.joblib").exists())
            self.assertFalse(list((cache / "_tmp").glob("*")))

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
            (mount / "MyDrive/_v4_shortcuts").mkdir(parents=True)
            (mount / "MyDrive/_v4_shortcuts/sc_file_000000").write_bytes(b"sc")
            xattrs = {str(f1): "other_id_000000", str(f1_dup): "file_dup_000000"}
            state = {"unmounted_targets": {}}
            with patch("train_v4._xattr_drive_id", side_effect=lambda p: xattrs.get(str(p))):
                # 0-API xattr scan works on an explicit dataset folder
                scanned = index_mounted_files({"file_dup_000000"}, mount / "MyDrive/ds")
                self.assertEqual(scanned["file_dup_000000"], f1_dup)
                self.assertEqual(resolve_mounted(drive, "sc_file_000000", mount, state),
                                 mount / "MyDrive/_v4_shortcuts/sc_file_000000")
                self.assertEqual(resolve_mounted(drive, "file_dup_000000", mount, state), f1_dup)
                self.assertEqual(resolve_mounted(drive, "gt_shared_00000", mount, state), f2)
                with self.assertRaisesRegex(FileNotFoundError, "folders/unshared_folder_99"):
                    resolve_mounted(drive, "missing_id_0000", mount, state)
                with self.assertRaisesRegex(FileNotFoundError, "owner@lab.org"):
                    resolve_mounted(drive, "orphan_id_00000", mount, state)
                self.assertIn("orphan_id_00000", state["unmounted_targets"])
                self.assertIn("unshared_folder_99", state["unmounted_targets"])

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

    def test_sequence_mpjpe_does_not_refit_fused_orientation(self):
        p = pair("val")
        yawed = [rotate(f, 25) for f in p["S"]]
        with patch("train_v4.fuse_partwise", return_value=(yawed, {})):
            metrics = validation([p], export_model(np.zeros(8), np.ones(8)), CFG)
        self.assertLess(metrics["pa_mm"], 0.001)
        self.assertGreater(metrics["mpjpe_mm"], 20)

    def test_disabled_features_are_ignored_even_when_present_at_inference(self):
        model = export_model(np.ones(8), np.ones(8), [True, False, False, False, True, True, True, True])
        feats = np.zeros((2, 3, 8), dtype=np.float32)
        expected = eval_weight_model(feats, model)
        feats[..., 1:4] = 1000
        np.testing.assert_array_equal(eval_weight_model(feats, model), expected)

    def test_learned_weights_keep_visibility_and_depth_outlier_guards(self):
        master = [skeleton(depth=4, elbow_lift=0.2) for _ in range(10)]
        slave = [skeleton(depth=6) for _ in range(10)]
        wm = dict(enabled=True, **export_model(np.zeros(8), np.ones(8)))
        cfg = dict(CFG, weight_model=wm, multi_person={"enabled": True, "anti_swap_gate": False})
        raw, _ = fuse_partwise(master, slave, dict(cfg, multi_person={"enabled": False}))
        vis_m = [{"left_elbow": False, "left_wrist": False} for _ in master]
        out, diag = fuse_partwise(master, slave, cfg, vis_master=vis_m)
        self.assertGreater(diag["occluded_master_bones"], 0)
        self.assertLess(np.linalg.norm(np.asarray(out[0]["left_wrist"]) - slave[0]["left_wrist"] + [0, 0, 2]),
                        np.linalg.norm(np.asarray(raw[0]["left_wrist"]) - slave[0]["left_wrist"] + [0, 0, 2]))
        _, diag = fuse_partwise(master, [skeleton(depth=18) for _ in master], cfg)
        self.assertEqual(diag["depth_outlier_frames"], 10)
        self.assertGreater(diag["master_weight_min"], 0.95)

    def test_training_options_reject_invalid_values(self):
        for options in ({"patience": 0}, {"patience": 1.5}, {"prior_l2": -1}, {"mpjpe_weight": float("nan")},
                        {"naive_regret_weight": -1}, {"naive_margin_mm": float("inf")}):
            with self.assertRaises(ValueError):
                training_options(dict(CFG, training=options))

    def test_naive_regret_changes_gradients_only_for_losing_frames(self):
        pa = torch.tensor([0.02, 0.08], requires_grad=True)
        mp = torch.tensor([0.07, 0.04], requires_grad=True)
        loss, regret = naive_relative_loss(pa, mp, torch.tensor([0.05, 0.05]),
                                           torch.tensor([0.06, 0.06]), training_options(CFG))
        self.assertAlmostEqual(float(regret), 0.01625, places=7)
        self.assertAlmostEqual(float(loss), 0.08, places=7)
        loss.backward()
        np.testing.assert_allclose(pa.grad.numpy(), [0.5, 1.0])
        np.testing.assert_allclose(mp.grad.numpy(), [0.25, 0.125])

    def test_naive_regret_margin_is_in_mm_and_can_be_disabled(self):
        pa, mp = torch.tensor([0.049]), torch.tensor([0.04])
        opts = training_options(dict(CFG, training={"naive_margin_mm": 2.0}))
        loss, regret = naive_relative_loss(pa, mp, torch.tensor([0.05]), torch.tensor([0.06]), opts)
        self.assertAlmostEqual(float(regret), 0.001, places=7)
        opts["naive_regret_weight"] = 0.0
        plain, _ = naive_relative_loss(pa, mp, torch.tensor([0.05]), torch.tensor([0.06]), opts)
        self.assertAlmostEqual(float(loss - plain), 0.001, places=7)

    def test_selection_requires_both_metrics_to_beat_naive(self):
        naive = dict(pa_mm=50.0, mpjpe_mm=100.0)
        opts = training_options(CFG)
        compromise = selection_key(dict(pa_mm=40.0, mpjpe_mm=110.0), naive, opts)
        both_better = selection_key(dict(pa_mm=49.0, mpjpe_mm=99.0), naive, opts)
        self.assertLess(both_better, compromise)
        self.assertGreater(both_better[1], compromise[1])
        self.assertTrue(selection_key(naive, naive, opts)[0])
        opts["naive_margin_mm"] = 1.0
        self.assertTrue(selection_key(dict(pa_mm=49.0, mpjpe_mm=99.0), naive, opts)[0])

    def test_naive_reference_copies_the_benchmark_smoothing_configuration(self):
        pipeline = {"fusion": {"partwise": copy.deepcopy(CFG),
                               "occlusion": {"enabled": True, "multi_person": True},
                               "naive": {"master_weight": 0.5, "use_aotr_stability": True},
                               "aotr": {"sequence_stability": {"enabled": True}}}}
        before = copy.deepcopy(pipeline)
        cfg = training_config(pipeline)
        self.assertEqual(cfg["occlusion"], pipeline["fusion"]["occlusion"])
        p = pair()
        output = [{k: v.tolist() for k, v in f.items()} for f in p["M"]]
        with patch("fusion_pipeline.aotr_stability.postprocess_aotr_3d", return_value=(output, {})) as smooth:
            actual = naive_prediction(p, cfg)
        self.assertEqual(smooth.call_args.kwargs["config"], {"enabled": True})
        self.assertEqual(pipeline, before)
        np.testing.assert_allclose(actual, np.array([[f[n] for n in p["names"]] for f in p["M"]]))

    def test_frame_pa_errors_match_the_original_mean_loss(self):
        p = pair()
        pred = torch_fuse(p, torch.ones(8), torch.ones(8))
        target = torch.as_tensor(p["target"])
        errors = torch_pa_loss(pred, target, per_frame=True)
        self.assertEqual(tuple(errors.shape), (len(p["M"]),))
        self.assertAlmostEqual(float(errors.mean()), float(torch_pa_loss(pred, target)), places=7)
        np.testing.assert_allclose(errors.detach().numpy(), pa_errors(pred.detach().numpy(), p["target"]), atol=1e-6)

    def test_checkpoint_that_beats_both_metrics_is_kept_over_lower_objective(self):
        def metrics(pa, mp):
            return dict(pa_mm=pa, mpjpe_mm=mp, mono_pa_mm=60.0, mono_mpjpe_mm=120.0)

        scores = [metrics(55, 110), metrics(55, 110), metrics(50, 100),
                  metrics(40, 110), metrics(49, 99), metrics(35, 101)]
        with tempfile.TemporaryDirectory() as tmp, patch("train_v4.validation", side_effect=scores):
            history = fit([pair(), pair("val")], CFG, tmp, epochs=3)
            best = json.loads((Path(tmp) / "best_weights.json").read_text())
            self.assertEqual(best["epoch"], 2)
            self.assertTrue(best["val_beats_naive"])
            self.assertFalse(history[2]["val_beats_naive"])
            self.assertTrue(history[2]["best_val_beats_naive"])

    def test_unbeaten_naive_is_reported_and_training_still_stops_on_plateau(self):
        naive = dict(pa_mm=0.0, mpjpe_mm=0.0, mono_pa_mm=20.0, mono_mpjpe_mm=30.0)
        candidate = dict(pa_mm=10.0, mpjpe_mm=10.0, mono_pa_mm=20.0, mono_mpjpe_mm=30.0)
        cfg = dict(CFG, training={"patience": 2})
        with tempfile.TemporaryDirectory() as tmp, patch("train_v4.validation", side_effect=[candidate, candidate, naive] + [candidate] * 3):
            history = fit([pair(), pair("val")], cfg, tmp, epochs=20)
            self.assertEqual(len(history), 3)
            best = json.loads((Path(tmp) / "best_weights.json").read_text())
            self.assertFalse(best["val_beats_naive"])

    def test_prior_inactive_features_and_early_stop_survive_resume(self):
        cfg = dict(CFG, training={"patience": 2, "min_delta_mm": 1e6})
        # A high learning rate exercises the non-negative depth projection.
        with tempfile.TemporaryDirectory() as tmp:
            history = fit([pair(), pair("val")], cfg, Path(tmp) / "first", epochs=10, lr=2)
            self.assertEqual(len(history), 3)
            saved = json.loads((Path(tmp) / "first/epoch_0003_weights.json").read_text())
            self.assertGreaterEqual(saved["weights"][0], 0)
            for i, active in enumerate(saved["active_features"]):
                if not active:
                    self.assertEqual(saved["std"][i], 1)
                    self.assertEqual(saved["weights"][i], 0)
            resumed = fit([pair(), pair("val")], cfg, Path(tmp) / "resume", epochs=10, lr=2,
                          resume=Path(tmp) / "first/epoch_0003.pt")
            self.assertEqual(history, resumed)
            self.assertTrue((Path(tmp) / "resume/best_partwise.yml").exists())

    def test_depth_outlier_guard_runs_in_features_and_without_multi_person(self):
        master = [skeleton(depth=4) for _ in range(8)]
        exploded = [skeleton(depth=18) for _ in range(8)]
        f_ms, _ = extract_pair_features(master, exploded)
        f_sm, _ = extract_pair_features(exploded, master)
        self.assertTrue(np.all(f_ms[:, :, 0] == 3.0))
        np.testing.assert_allclose(f_ms, -f_sm, atol=1e-6)
        wm = dict(enabled=True, **export_model(np.zeros(8), np.ones(8)))
        cfg = dict(CFG, weight_model=wm, multi_person={"enabled": False})
        _, diag = fuse_partwise(master, exploded, cfg)
        self.assertEqual(diag["depth_outlier_frames"], 8)
        self.assertGreater(diag["master_weight_min"], 0.95)

    def test_zero_prior_and_nonnegative_weights_constraint(self):
        p = pair()
        for t, frame in enumerate(p["M"]):
            for value in frame.values():
                value[2] += 0.1 * t
        p["features"], _ = extract_pair_features(p["M"], p["S"])
        p["directions"] = slave_directions(p["M"], p["S"], 4)
        with tempfile.TemporaryDirectory() as tmp:
            fit([p, pair("val")], CFG, Path(tmp) / "init", epochs=1, lr=1e-9)
            init_model = json.loads((Path(tmp) / "init/epoch_0001_weights.json").read_text())
            fit([p, pair("val")], CFG, Path(tmp) / "high_lr", epochs=3, lr=2.0)
            high_model = json.loads((Path(tmp) / "high_lr/epoch_0003_weights.json").read_text())
        self.assertTrue(init_model["active_features"][0])
        np.testing.assert_allclose(init_model["weights"], np.zeros(8), atol=1e-6)
        self.assertTrue(all(w >= 0.0 for w in high_model["weights"]))

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
        ids, actual, report, sources = associate({"first": (np.arange(15), joints[:15]),
                                        "second": (np.arange(15,30), joints[15:])}, gt, names,
                                        return_sources=True)
        self.assertEqual(report["ratio"], 2)
        self.assertEqual(report["offset"], 0)
        self.assertEqual(report["switches"], 1)
        np.testing.assert_array_equal(ids, np.arange(0,60,2))
        np.testing.assert_allclose(actual, joints)
        self.assertEqual(sources, [("first", i) for i in range(15)] + [("second", i) for i in range(15)])

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
            selected = min(resumed, key=lambda h: (not h["val_beats_naive"], h["val_objective_mm"]))
            self.assertEqual(best["epoch"], selected["epoch"])
            self.assertTrue((root/"resume/epoch_0003.pt").exists())
            for key in ["weights","std"]:
                a = json.loads((root/"full/epoch_0003_weights.json").read_text())[key]
                b = json.loads((root/"resume/epoch_0003_weights.json").read_text())[key]
                np.testing.assert_allclose(a,b)
            with self.assertRaisesRegex(ValueError, "differs"):
                fit(pairs, CFG, root/"bad", epochs=3, lr=0.2, resume=root/"part/epoch_0001.pt")


if __name__ == "__main__":
    unittest.main()

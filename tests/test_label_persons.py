"""Track -> person labelling used when a dataset has no *_person_mapping.txt."""
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import yaml

import compat as lp_compat
import preprocessing_data.label_persons as lp
from preprocessing_data.label_persons import _segments, summarize_track

PERSONS = ["aria01", "aria02"]
GT_ID = "1j2V2c5wFR0Vqb-TLS55HmrEAFJLXKAGC"
PKL_IDS = {"cam01": "1-YLUt9UZNVuwUw3si1aLFENMIw-qAXY1", "cam02": "1HzDcIK6Ha3Q0VzVvHwS0fyQBc2vL9UMS"}


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


def write_gt_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name in members:
            buf = io.BytesIO()
            np.save(buf, np.array({"aria01": np.zeros((17, 4)), "aria02": np.ones((17, 4))}, dtype=object))
            zf.writestr(name, buf.getvalue())
    return path


class SourceTests(unittest.TestCase):
    def test_local_path_relative_to_repo_root(self):
        self.assertEqual(lp.resolve_source("configs/keypoints3D_map.yml", "x"),
                         ("local", (lp.ROOT / "configs/keypoints3D_map.yml").resolve()))

    def test_drive_id_and_link(self):
        self.assertEqual(lp.resolve_source(GT_ID, "x"), ("drive", GT_ID))
        self.assertEqual(lp.resolve_source(f"https://drive.google.com/file/d/{GT_ID}/view?usp=sharing", "x"),
                         ("drive", GT_ID))
        self.assertEqual(lp.resolve_source(f"https://drive.google.com/open?id={GT_ID}", "x"), ("drive", GT_ID))

    def test_missing_local_path_is_an_error_not_a_drive_id(self):
        with self.assertRaises(FileNotFoundError):
            lp.resolve_source("input/043_Grappling2/exo/cam01/missing.pkl", "x")
        with self.assertRaises(ValueError):
            lp.resolve_source("", "x")


class GtZipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_prefers_poses3d_folder(self):
        z = write_gt_zip(self.tmp / "gt.zip", ["seq/processed_data/smpl/00001.npy",
                                               "seq/processed_data/poses3d/00001.npy"])
        self.assertEqual(lp.extract_gt_zip(z, self.tmp / "out").name, "poses3d")

    def test_npy_at_root_and_macos_junk(self):
        z = write_gt_zip(self.tmp / "gt.zip", ["00001.npy", "00002.npy", "__MACOSX/._00001.npy"])
        out = lp.extract_gt_zip(z, self.tmp / "out")
        self.assertEqual(out, (self.tmp / "out").resolve())
        self.assertFalse((out / "__MACOSX").exists())

    def test_bad_zip_unsafe_member_and_empty(self):
        (self.tmp / "bad.zip").write_bytes(b"not a zip")
        with self.assertRaises(ValueError):
            lp.extract_gt_zip(self.tmp / "bad.zip", self.tmp / "o1")
        with self.assertRaises(ValueError):
            lp.extract_gt_zip(write_gt_zip(self.tmp / "evil.zip", ["../evil.npy"]), self.tmp / "o2")
        zipfile.ZipFile(self.tmp / "empty.zip", "w").close()
        with self.assertRaises(ValueError):
            lp.extract_gt_zip(self.tmp / "empty.zip", self.tmp / "o3")


class FakeLabeller:
    """A fake pkl is {track id: label}; each track is labelled as is."""
    params = {"margin_mm": 10.0, "max_pa_mm": 120.0, "min_vote_fraction": 0.5}

    def __init__(self):
        self.gt_npy = self.persons = None

    def load_gt(self, truth_dir, persons):
        self.gt_npy, self.persons = len(list(Path(truth_dir).glob("*.npy"))), list(persons)
        return {p: {} for p in persons}

    def label(self, raw, gt, persons):  # a track is its label, or {"label": ..., "frame_ids": ...}
        return {str(tid): summarize_track(frames([t["label"] if isinstance(t, dict) else t] * 3), persons, 0.5)
                for tid, t in raw.items()}


VIDEO_ID = "1VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
TRACK_ID = "1TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
OTHER_TRACK_ID = "1OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"


def wham_tracks():
    return {0: {"label": "aria01", "frame_ids": np.arange(5)}, 1: {"label": "aria02", "frame_ids": np.arange(5)}}


def tracking_tracks(ids=(0, 1)):
    """Like WHAM's tracking_results.pth: torch tensors, keypoints = (x, y, confidence)."""
    import torch
    return {tid: {"frame_id": torch.arange(5), "keypoints": torch.full((5, 17, 3), 0.8 - 0.3 * k),
                  "bbox": torch.zeros(5, 3)} for k, tid in enumerate(ids)}


def without_local_lines(text):
    return "".join(line for line in text.splitlines(keepends=True) if "_local:" not in line)


class YamlRunTests(unittest.TestCase):
    """YAML with Drive IDs and local paths: downloads kept in the dataset-folder layout,
    mapping next to every pkl, <key>_local written into the same YAML for
    build_identity_mask.py (no other YAML file)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()  # macOS: /var -> /private/var
        self.data = self.tmp / "data"
        self.seq = self.data / "005_sword3"
        self.local_pkl = self.tmp / "local" / "wham_harmony4D_005_sword3_cam03_ffmpeg.pkl"
        self.local_pkl.parent.mkdir()
        joblib.dump({0: "aria01", 1: "aria02"}, self.local_pkl)
        self.downloads = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_config(self, cams, gt=GT_ID, name="brute.yml", targets=PERSONS):
        """Like configs/brute_force_005_sword3.yml: comments and a shared camera list."""
        lines = []
        for cam in cams:
            lines.append(f"  - id: {cam['id']}")
            lines += [f"    {k}: {v}" for k, v in cam.items() if k != "id"]
        text = "segments:\n"
        for i, person in enumerate(targets):
            text += f"- name: 005_sword3_{person}\n  target_person: {person}\n  ground_truth_dir: {gt}\n"
            text += ("  cameras: &id001\n" + "\n".join(lines) + "\n"
                     "  # - id: cam09\n  #   pkl: input/043_Grappling2/exo/cam09/wham_x.pkl\n"
                     if i == 0 else "  cameras: *id001\n")
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def fake_download(self, fail=()):
        def download(file_id, dest_dir, what):
            self.downloads.append(file_id)
            if file_id in fail:
                raise RuntimeError("Drive quota exceeded")
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            if file_id == GT_ID:
                return write_gt_zip(dest_dir / "poses3d.zip", ["poses3d/00001.npy", "poses3d/00002.npy"])
            if file_id == VIDEO_ID:
                path = dest_dir / "harmony4D_005_sword3_cam01_ffmpeg.mp4"
                path.write_bytes(b"mp4")
                return path
            if file_id in (TRACK_ID, OTHER_TRACK_ID):  # Drive name as downloaded twice
                path = dest_dir / "tracking_results (1).pth"
                joblib.dump(tracking_tracks() if file_id == TRACK_ID else tracking_tracks((50, 51)), path)
                return path
            # As on the real Drive: every camera's pkl has the same name.
            path = dest_dir / "wham_output.pkl"
            joblib.dump(wham_tracks() if file_id in PKL_IDS.values() else {7: "aria02"}, path)
            return path
        return download

    def run_yaml(self, cfg, fail=(), persons=None):
        fake, out = FakeLabeller(), io.StringIO()
        with mock.patch.object(lp, "download_drive_file", self.fake_download(fail)), redirect_stdout(out):
            reports, failed = lp.run_config(cfg, persons=persons, data_dir=self.data, labeller=fake)
        self.printed = out.getvalue()
        return reports, failed, fake

    def kept_pkl(self, cam):
        return self.seq / "exo" / cam / f"005_sword3_{cam}_wham" / f"wham_005_sword3_{cam}.pkl"

    def test_drive_inputs_kept_and_local_keys_written_in_same_yaml(self):
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"], "video": VIDEO_ID},
                                 {"id": "cam02", "pkl": PKL_IDS["cam02"], "video": "inputs/missing/cam02.mp4"},
                                 {"id": "cam03", "pkl": str(self.local_pkl)}])
        before = cfg.read_text(encoding="utf-8")
        reports, failed, fake = self.run_yaml(cfg)
        self.assertEqual(failed, [])
        self.assertEqual(self.downloads, [GT_ID, PKL_IDS["cam01"], VIDEO_ID, PKL_IDS["cam02"]])
        self.assertEqual(reports[0]["persons"], PERSONS)
        # Kept in the dataset-folder layout; GT zip extracted then removed; no temp folder left.
        poses = self.seq / "processed_data" / "poses3d"
        video = self.seq / "exo" / "cam01" / "images" / "harmony4D_005_sword3_cam01_ffmpeg.mp4"
        self.assertEqual((fake.gt_npy, sorted(p.name for p in poses.iterdir())), (2, ["00001.npy", "00002.npy"]))
        self.assertEqual(list(self.seq.rglob("*.zip")), [])
        self.assertEqual([p for p in self.seq.rglob(".*") if p.is_dir()], [])
        self.assertTrue(video.is_file())
        for pkl in (self.kept_pkl("cam01"), self.kept_pkl("cam02"), self.local_pkl):
            self.assertTrue(pkl.is_file())  # not deleted: build_identity_mask.py reads it
            self.assertEqual(lp_compat.parse_person_mapping(pkl), {"0": "aria01", "1": "aria02"})
        # Same YAML: only *_local lines added (comments, anchors kept), no other YAML written.
        after = cfg.read_text(encoding="utf-8")
        self.assertEqual(without_local_lines(after), before)
        self.assertIn("    pkl: " + PKL_IDS["cam01"] + "\n    pkl_local: ", after)
        self.assertEqual(sorted(p.name for p in self.tmp.glob("*.yml")), ["brute.yml"])
        segs = yaml.safe_load(after)["segments"]
        for seg in segs:
            self.assertEqual(seg["ground_truth_dir_local"], str(poses))
            cam1, cam2, cam3 = seg["cameras"]
            self.assertEqual((cam1["pkl_local"], cam1["video_local"]), (str(self.kept_pkl("cam01")), str(video)))
            self.assertEqual(cam2["pkl_local"], str(self.kept_pkl("cam02")))
            self.assertNotIn("video_local", cam2)  # missing local video: nothing to download
            self.assertNotIn("pkl_local", cam3)    # local pkl: used as is
        # build_identity_mask.py resolves the same YAML to the kept files.
        import build_identity_mask as bim
        self.assertEqual(bim.input_path(segs[0], "ground_truth_dir", cfg), poses)
        self.assertEqual([bim.input_path(c, "pkl", cfg) for c in segs[1]["cameras"]],
                         [self.kept_pkl("cam01"), self.kept_pkl("cam02"), self.local_pkl])
        # Rerun: nothing downloaded, YAML untouched.
        self.run_yaml(cfg)
        self.assertEqual(len(self.downloads), 4)
        self.assertEqual(cfg.read_text(encoding="utf-8"), after)

    def test_changed_id_replaces_old_pkl_and_updates_local_key(self):
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"]}])
        self.run_yaml(cfg)
        new_id = "1ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"
        cfg.write_text(cfg.read_text(encoding="utf-8").replace(PKL_IDS["cam01"], new_id), encoding="utf-8")
        _, failed, _ = self.run_yaml(cfg)
        self.assertEqual((failed, self.downloads[-1]), ([], new_id))
        # Old pkl and its mapping were removed: the folder now holds only the new download.
        self.assertEqual(joblib.load(self.kept_pkl("cam01")), {7: "aria02"})
        self.assertEqual(lp_compat.parse_person_mapping(self.kept_pkl("cam01")), {"7": "aria02"})
        text = cfg.read_text(encoding="utf-8")
        self.assertEqual(text.count("pkl_local:"), 1)  # still one key, same per-camera path
        self.assertEqual(yaml.safe_load(text)["segments"][0]["cameras"][0]["pkl_local"], str(self.kept_pkl("cam01")))

    def test_same_drive_name_gives_one_pkl_stem_per_camera(self):
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"]}, {"id": "cam02", "pkl": PKL_IDS["cam02"]}])
        _, failed, _ = self.run_yaml(cfg)
        self.assertEqual(failed, [])
        cams = yaml.safe_load(cfg.read_text(encoding="utf-8"))["segments"][0]["cameras"]
        # build_identity_mask.py and the evaluator key the identity mask by this stem.
        self.assertEqual([Path(c["pkl_local"]).stem for c in cams], ["wham_005_sword3_cam01", "wham_005_sword3_cam02"])

    def test_earlier_download_under_drive_name_is_renamed_not_downloaded(self):
        cam_dir = self.seq / "exo" / "cam01"
        old = cam_dir / "output_wham"
        old.mkdir(parents=True)
        joblib.dump({0: "aria01", 1: "aria02"}, old / "wham_output.pkl")
        (old / "wham_output_person_mapping.txt").write_text("0 = aria01\n", encoding="utf-8")
        (old / "wham_output_person_labels.json").write_text("{}", encoding="utf-8")
        (cam_dir / lp.DRIVE_MARKER).write_text(json.dumps(
            {"pkl": {"id": PKL_IDS["cam01"], "path": "output_wham/wham_output.pkl"}}), encoding="utf-8")
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"],
                                  "pkl_local": f'"{old / "wham_output.pkl"}"'}])
        _, failed, _ = self.run_yaml(cfg)
        self.assertEqual((failed, self.downloads), ([], [GT_ID]))  # the pkl itself was not downloaded
        new = self.kept_pkl("cam01")
        self.assertFalse(old.exists())
        self.assertEqual(sorted(p.name for p in new.parent.iterdir()),
                         ["wham_005_sword3_cam01.pkl", "wham_005_sword3_cam01_person_labels.json",
                          "wham_005_sword3_cam01_person_mapping.txt"])
        self.assertEqual(json.loads((cam_dir / lp.DRIVE_MARKER).read_text())["pkl"]["path"],
                         "005_sword3_cam01_wham/wham_005_sword3_cam01.pkl")
        # The YAML now points at the new name; the hand-kept mapping survives, with a hint.
        self.assertEqual(yaml.safe_load(cfg.read_text())["segments"][0]["cameras"][0]["pkl_local"], str(new))
        self.assertEqual(lp_compat.parse_person_mapping(new), {"0": "aria01"})
        self.assertIn("has no track for ['aria02']", self.printed)
        self.assertIn("--overwrite", self.printed)

    def test_tracks_are_labelled_against_every_gt_person(self):
        # The YAML only targets aria01, but the GT also holds aria02 -> vote against both.
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"]}], targets=["aria01"])
        reports, failed, fake = self.run_yaml(cfg)
        self.assertEqual((failed, fake.persons, reports[0]["persons"]), ([], PERSONS, PERSONS))
        self.assertNotIn("only one person", self.printed)

    def test_gt_persons(self):
        poses = self.tmp / "gt" / "poses3d"
        poses.mkdir(parents=True)
        np.save(poses / "00001.npy", np.array({"aria02": np.zeros((17, 4)), "aria01": np.ones((17, 4))}, dtype=object))
        (poses / "00001(1).npy").write_bytes(b"duplicate download")  # ignored like in the GT loader
        self.assertEqual(lp.gt_persons(poses), ["aria01", "aria02"])
        self.assertEqual(lp.gt_persons(poses.parent), ["aria01", "aria02"])
        self.assertEqual(lp.gt_persons(self.tmp / "empty"), [])

    def test_stale_local_key_removed_when_value_becomes_local(self):
        cfg = self.write_config([{"id": "cam03", "pkl": str(self.local_pkl), "pkl_local": '"/old/download.pkl"'}])
        _, failed, _ = self.run_yaml(cfg)
        self.assertEqual(failed, [])
        self.assertNotIn("pkl_local", cfg.read_text(encoding="utf-8"))

    def test_failed_download_then_rerun_downloads_only_what_is_missing(self):
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"]}, {"id": "cam02", "pkl": PKL_IDS["cam02"]}])
        reports, failed, _ = self.run_yaml(cfg, fail={PKL_IDS["cam01"]})
        self.assertEqual(failed, ["005_sword3/cam01"])
        self.assertIn("Drive quota exceeded", reports[0]["cameras"]["cam01"]["error"])
        self.assertEqual(reports[0]["cameras"]["cam02"]["mapping"], {"0": "aria01", "1": "aria02"})
        self.assertEqual(list((self.seq / "exo" / "cam01").iterdir()), [])  # no half download kept
        cams = yaml.safe_load(cfg.read_text(encoding="utf-8"))["segments"][0]["cameras"]
        self.assertEqual(["pkl_local" in c for c in cams], [False, True])
        _, failed, _ = self.run_yaml(cfg)
        self.assertEqual(failed, [])
        self.assertEqual(self.downloads, [GT_ID, PKL_IDS["cam01"], PKL_IDS["cam02"], PKL_IDS["cam01"]])
        cams = yaml.safe_load(cfg.read_text(encoding="utf-8"))["segments"][0]["cameras"]
        self.assertEqual(cams[0]["pkl_local"], str(self.kept_pkl("cam01")))

    def test_foreign_gt_folder_is_never_replaced(self):
        poses = self.seq / "processed_data" / "poses3d"
        poses.mkdir(parents=True)
        (poses / "00001.npy").write_bytes(b"user data")
        _, failed, _ = self.run_yaml(self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"]}]))
        self.assertEqual(failed, ["005_sword3 (GT)"])
        self.assertEqual((poses / "00001.npy").read_bytes(), b"user data")

    def test_all_local_yaml_is_left_untouched(self):
        gt_dir = self.tmp / "poses3d"
        gt_dir.mkdir()
        cfg = self.write_config([{"id": "cam03", "pkl": str(self.local_pkl)}], gt=str(gt_dir))
        before = cfg.read_text(encoding="utf-8")
        _, failed, _ = self.run_yaml(cfg)
        self.assertEqual((failed, self.downloads, cfg.read_text(encoding="utf-8")), ([], [], before))

    def test_crlf_yaml_keeps_its_line_endings(self):
        path = self.tmp / "crlf.yml"
        path.write_bytes(f"segments:\r\n- name: s\r\n  cameras:\r\n  - id: cam01\r\n    pkl: {PKL_IDS['cam01']}\r\n".encode())
        changes = lp.write_local_keys(path, {("drive", PKL_IDS["cam01"]): self.local_pkl})
        self.assertEqual(len(changes), 1)
        raw = path.read_bytes()
        self.assertEqual(raw.count(b"\r\n"), 6)
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))

    def test_config_errors(self):
        with self.assertRaises(FileNotFoundError):  # typo in a local path fails before any download
            lp.load_config(self.write_config([{"id": "cam01", "pkl": "inputs/missing/wham_x.pkl"}], name="bad.yml"))
        segs = [{"name": "s_aria01", "target_person": "aria01", "ground_truth_dir": GT_ID,
                 "cameras": [{"id": "cam01", "pkl": PKL_IDS["cam01"]}]},
                {"name": "s_aria02", "target_person": "aria02", "ground_truth_dir": GT_ID,
                 "cameras": [{"id": "cam01", "pkl": PKL_IDS["cam02"]}]}]
        (self.tmp / "conflict.yml").write_text(yaml.safe_dump({"segments": segs}), encoding="utf-8")
        with self.assertRaises(ValueError):  # same camera id, two different pkls
            lp.load_config(self.tmp / "conflict.yml")

    # ---------------------------------------------------------------- tracking_result

    def test_tracking_result_is_kept_next_to_the_pkl_and_read_by_preprocess(self):
        other = self.tmp / "elsewhere" / "trk.pth"
        other.parent.mkdir()
        joblib.dump(tracking_tracks(), other)
        joblib.dump(wham_tracks(), self.local_pkl)
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"], "tracking_result": TRACK_ID},
                                 {"id": "cam03", "pkl": str(self.local_pkl), "tracking_result": str(other)}])
        reports, failed, _ = self.run_yaml(cfg)
        self.assertEqual(failed, [])
        kept = self.kept_pkl("cam01").parent / "tracking_results.pth"
        self.assertTrue(kept.is_file() and not kept.is_symlink())       # Drive file, next to the Drive pkl
        link = self.local_pkl.parent / "tracking_results.pth"
        self.assertEqual(link.resolve(), other.resolve())                 # local file linked next to the local pkl
        cams = yaml.safe_load(cfg.read_text())["segments"][0]["cameras"]
        self.assertEqual(cams[0]["tracking_result_local"], str(kept))
        self.assertNotIn("tracking_result_local", cams[1])
        cover = reports[0]["cameras"]["cam01"]["tracking_coverage"]
        self.assertEqual(cover["aria01"], {"wham_frames": 5, "frames_with_2d": 5, "mean_confidence": 0.8})
        self.assertEqual(cover["aria02"]["mean_confidence"], 0.5)
        # What the pipeline does next: preprocess_pipeline/extract_2d.py finds it by the pkl.
        from preprocess_pipeline.extract_2d import _load_tracking_payload
        for pkl in (self.kept_pkl("cam01"), self.local_pkl):
            t = _load_tracking_payload(pkl, target_person="aria02")
            np.testing.assert_allclose(np.asarray(t["keypoints"])[:, 0, 2], [0.5] * 5)
        self.run_yaml(cfg)  # rerun: nothing downloaded again
        self.assertEqual(self.downloads.count(TRACK_ID), 1)

    def test_tracking_result_of_another_video_is_reported(self):
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"], "tracking_result": OTHER_TRACK_ID}])
        reports, failed, _ = self.run_yaml(cfg)
        self.assertEqual(failed, [])
        self.assertEqual(reports[0]["cameras"]["cam01"]["tracking_coverage"]["aria01"]["frames_with_2d"], 0)
        self.assertIn("no track of aria01", self.printed)

    # ---------------------------------------------------------------- --persons N

    def test_persons_2_adds_a_segment_per_person_and_persons_1_removes_it(self):
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"]}, {"id": "cam02", "pkl": PKL_IDS["cam02"]}],
                                targets=["aria01"])
        _, failed, fake = self.run_yaml(cfg, persons=2)
        self.assertEqual((failed, fake.persons), ([], PERSONS))
        text = cfg.read_text(encoding="utf-8")
        segs = yaml.safe_load(text)["segments"]
        self.assertEqual([(s["name"], s["target_person"]) for s in segs],
                         [("005_sword3_aria01", "aria01"), ("005_sword3_aria02", "aria02")])
        self.assertEqual(segs[1]["cameras"], segs[0]["cameras"])         # shared through the anchor
        self.assertIn("  cameras: *id001\n", text)
        self.assertEqual(segs[1]["added_by"], lp.ADDED_BY)
        self.assertEqual(segs[1]["ground_truth_dir_local"], segs[0]["ground_truth_dir_local"])
        self.assertIn("# - id: cam09", text)                             # comments kept
        self.assertEqual(lp._brute_force_runs(cfg)[1], 2 * 2 * 1)       # 2 persons x 2 ordered pairs
        self.assertIn("-> 4 camera-pair runs", self.printed)
        self.run_yaml(cfg, persons=2)
        self.assertEqual(cfg.read_text(encoding="utf-8"), text)          # idempotent
        self.run_yaml(cfg, persons=1)
        after = cfg.read_text(encoding="utf-8")
        self.assertEqual([s["target_person"] for s in yaml.safe_load(after)["segments"]], ["aria01"])
        self.assertNotIn("added_by", after)
        self.assertNotIn("005_sword3_aria02", after)

    def test_segments_written_by_hand_are_never_removed(self):
        cfg = self.write_config([{"id": "cam01", "pkl": PKL_IDS["cam01"]}])     # aria01 and aria02 by hand
        before = cfg.read_text(encoding="utf-8")
        _, failed, _ = self.run_yaml(cfg, persons=1)
        self.assertEqual(failed, [])
        self.assertEqual(without_local_lines(cfg.read_text(encoding="utf-8")), before)

    def test_persons_2_adds_an_anchor_when_cameras_have_none(self):
        cfg = self.tmp / "plain.yml"
        cfg.write_text("# my config\nsegments:\n  - name: \"005_sword3_aria01\"\n    target_person: aria01\n"
                       f"    ground_truth_dir: \"{GT_ID}\"\n    ground_truth_dir_2d: \"1qq7GFqBSCXOvHgQ7aWwyee24PWByC_6t\"\n"
                       f"    cameras:\n      - id: \"cam01\"\n        pkl: \"{PKL_IDS['cam01']}\"\n", encoding="utf-8")
        _, failed, _ = self.run_yaml(cfg, persons=2)
        self.assertEqual(failed, [])
        text = cfg.read_text(encoding="utf-8")
        self.assertIn("    cameras: &cams_005_sword3\n", text)
        self.assertIn("    cameras: *cams_005_sword3\n", text)
        segs = yaml.safe_load(text)["segments"]
        self.assertEqual(segs[1]["ground_truth_dir_2d"], "1qq7GFqBSCXOvHgQ7aWwyee24PWByC_6t")
        self.assertEqual(segs[1]["cameras"], segs[0]["cameras"])
        self.assertTrue(text.startswith("# my config\n"))

    def test_choose_and_parse_persons(self):
        gt = ["aria01", "aria02"]
        self.assertEqual(lp.choose_persons(2, gt, ["aria01"]), (gt, gt))
        self.assertEqual(lp.choose_persons(1, gt, ["aria01"]), (gt, ["aria01"]))
        self.assertEqual(lp.choose_persons(None, gt, ["aria02"]), (gt, ["aria02"]))  # the YAML's targets
        self.assertEqual(lp.choose_persons(["aria02"], gt, ["aria01"]), (gt, ["aria02"]))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(lp.choose_persons(3, gt, [])[1], gt)                    # only 2 in the GT
        self.assertEqual((lp.parse_persons(["2"]), lp.parse_persons(["aria01", "aria02"]), lp.parse_persons(None)),
                         (2, ["aria01", "aria02"], None))


class BuildIdentityMaskInputTests(unittest.TestCase):
    def test_local_value_wins_and_missing_local_key_is_explained(self):
        import build_identity_mask as bim
        cfg = "configs/brute_force_005_sword3.yml"
        here = "configs/keypoints3D_map.yml"
        self.assertEqual(bim.input_path({"pkl": here, "pkl_local": "/stale.pkl"}, "pkl", cfg), bim.ROOT / here)
        with self.assertRaises(FileNotFoundError) as err:
            bim.input_path({"pkl": PKL_IDS["cam01"]}, "pkl", cfg)
        self.assertIn("preprocessing_data.label_persons", str(err.exception))


if __name__ == "__main__":
    unittest.main()

"""Track -> person labelling used when a dataset has no *_person_mapping.txt."""
import io
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
        self.gt_npy = None

    def load_gt(self, truth_dir, persons):
        self.gt_npy = len(list(Path(truth_dir).glob("*.npy")))
        return {p: {} for p in persons}

    def label(self, raw, gt, persons):
        return {str(tid): summarize_track(frames([lab] * 3), persons, 0.5) for tid, lab in raw.items()}


VIDEO_ID = "1VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"


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

    def write_config(self, cams, gt=GT_ID, name="brute.yml"):
        """Like configs/brute_force_005_sword3.yml: comments and a shared camera list."""
        lines = []
        for cam in cams:
            lines.append(f"  - id: {cam['id']}")
            lines += [f"    {k}: {v}" for k, v in cam.items() if k != "id"]
        text = (f"segments:\n- name: 005_sword3_aria01\n  target_person: aria01\n"
                f"  ground_truth_dir: {gt}\n  cameras: &id001\n" + "\n".join(lines) + "\n"
                "  # - id: cam09\n  #   pkl: input/043_Grappling2/exo/cam09/wham_x.pkl\n"
                f"- name: 005_sword3_aria02\n  target_person: aria02\n  ground_truth_dir: {gt}\n"
                "  cameras: *id001\n")
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
            cam = next((c for c, i in PKL_IDS.items() if i == file_id), "camX")
            path = dest_dir / f"wham_harmony4D_005_sword3_{cam}_ffmpeg.pkl"
            joblib.dump({0: "aria01", 1: "aria02"}, path)
            return path
        return download

    def run_yaml(self, cfg, fail=()):
        fake = FakeLabeller()
        with mock.patch.object(lp, "download_drive_file", self.fake_download(fail)), \
                redirect_stdout(io.StringIO()):
            reports, failed = lp.run_config(cfg, data_dir=self.data, labeller=fake)
        return reports, failed, fake

    def kept_pkl(self, cam):
        return self.seq / "exo" / cam / f"harmony4D_005_sword3_{cam}_wham" / f"wham_harmony4D_005_sword3_{cam}_ffmpeg.pkl"

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
        self.assertFalse(self.kept_pkl("cam01").parent.exists())  # old pkl + its mapping removed
        new_pkl = self.seq / "exo" / "cam01" / "harmony4D_005_sword3_camX_wham" / "wham_harmony4D_005_sword3_camX_ffmpeg.pkl"
        text = cfg.read_text(encoding="utf-8")
        self.assertEqual(text.count("pkl_local:"), 1)  # updated in place, not added twice
        self.assertEqual(yaml.safe_load(text)["segments"][0]["cameras"][0]["pkl_local"], str(new_pkl))

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

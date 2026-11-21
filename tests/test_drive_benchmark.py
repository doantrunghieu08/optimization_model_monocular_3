import csv
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from run_drive_benchmark import extract_poses, is_drive_segment, main, prepare_segment_inputs, report_complete, run_segment, segment_protocol


class DriveBenchmarkTest(unittest.TestCase):
    def test_keeps_successful_downloads_when_staging_fails(self):
        name = "001_mma5_aria01"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/brute_force_sample.yml").write_text(
                f"segments:\n- name: {name}\n  ground_truth_dir: 1example_drive_id_0123456789\n  cameras: []\n",
                encoding="utf-8",
            )
            dataset_dir = root / "input" / "drive_benchmark" / "001_mma5"

            def fail_after_download(segment, directory, *, local_only):
                self.assertFalse(local_only)
                self.assertEqual(directory, dataset_dir)
                directory.mkdir(parents=True)
                (directory / "poses3d.zip").write_bytes(b"downloaded")
                raise RuntimeError("camera download failed")

            with patch("run_drive_benchmark.ROOT", root), patch("sys.argv", ["run_drive_benchmark.py"]), \
                    patch("run_drive_benchmark.load_config", return_value={"fusion": {}}), \
                    patch("run_drive_benchmark.stage_segment", side_effect=fail_after_download):
                self.assertEqual(main(), 1)
            self.assertEqual((dataset_dir / "poses3d.zip").read_bytes(), b"downloaded")

    def test_local_only_uses_cache_and_reports_missing_inputs(self):
        segment = {
            "name": "001_mma5_aria01", "target_person": "aria01", "ground_truth_dir": "gt-id",
            "cameras": [{"id": "cam01", "pkl": "pkl-id", "video": "video-id"}],
        }
        with TemporaryDirectory() as tmp, patch("run_drive_benchmark.download") as download:
            root = Path(tmp)
            stem = "harmony4D_001_mma5_cam01"
            pkl = root / "exo" / "cam01" / f"{stem}_wham" / f"wham_{stem}_ffmpeg.pkl"
            video = root / "exo" / "cam01" / "images" / f"{stem}_ffmpeg.mp4"
            for path in (root / "poses3d.zip", pkl):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"cached")
            with self.assertRaisesRegex(FileNotFoundError, video.name):
                prepare_segment_inputs(segment, root, local_only=True)
            video.parent.mkdir()
            video.write_bytes(b"cached")
            cameras = prepare_segment_inputs(segment, root, local_only=True)
            self.assertEqual(cameras[0]["pkl"], str(pkl))
            download.assert_not_called()

    def test_extracts_numeric_frames_and_finds_people(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = BytesIO()
            np.save(frame, {"aria01": np.zeros((17, 3)), "aria02": np.zeros((17, 3))})
            with ZipFile(root / "poses3d.zip", "w") as archive:
                archive.writestr("poses3d/00001.npy", frame.getvalue())
                archive.writestr("poses3d/00001(1).npy", b"ignored duplicate")
            poses = root / "ground_truth"
            self.assertEqual(extract_poses(root / "poses3d.zip", poses), ["aria01", "aria02"])
            self.assertEqual([p.name for p in poses.iterdir()], ["00001.npy"])

    def test_completion_requires_both_directions_under_current_protocol(self):
        file_id = "1example_drive_id_0123456789"
        segment = {"name": "test", "ground_truth_dir": file_id, "cameras": [
            {"id": cam_id, "pkl": file_id, "video": file_id} for cam_id in ("cam01", "cam02")
        ]}
        self.assertTrue(is_drive_segment(segment))
        self.assertFalse(is_drive_segment({**segment, "ground_truth_dir": "input/gt"}))
        with TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.csv"
            with report.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["Segment", "Protocol", "Cam Master", "Cam Slave"])
                writer.writerow(["test", "current", "cam01", "cam02"])
                writer.writerow(["test", "stale", "cam02", "cam01"])
            self.assertFalse(report_complete(report, segment, "current"))
            with report.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(["test", "current", "cam02", "cam01"])
            self.assertTrue(report_complete(report, segment, "current"))

    def test_run_segment_defaults_to_6_workers(self):
        segment = {"name": "001_mma5_aria01", "cameras": []}
        base_config = {"paths": {"fused_output_dir": "output/fused"}}
        with patch("run_drive_benchmark.run_brute_force", return_value={"001_mma5_aria01": []}) as run_bf:
            run_segment(base_config, segment, Path("output/drive_benchmark/001_mma5_aria01"))
        self.assertEqual(run_bf.call_args.kwargs["workers"], 6)

    def test_expands_both_persons_into_one_parallel_run(self):
        file_id = "1example_drive_id_0123456789"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/002_sword3.yml").write_text(
                "segments:\n"
                "- name: 002_sword3_aria01\n"
                "  target_person: aria01\n"
                f"  ground_truth_dir: {file_id}\n"
                "  cameras:\n"
                f"  - id: cam01\n    pkl: {file_id}\n    video: {file_id}\n"
                f"  - id: cam02\n    pkl: {file_id}\n    video: {file_id}\n",
                encoding="utf-8",
            )
            dataset_dir = root / "input" / "drive_benchmark" / "002_sword3"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "person_labels_report.json").write_text('{"persons": ["aria01", "aria02"]}', encoding="utf-8")
            base_cfg = {"fusion": {"aotr": {"enabled": True}, "method": "aotr"}, "paths": {"fused_output_dir": "out"}}

            def fake_stage(seg, directory, *, local_only):
                self.assertEqual(directory, dataset_dir)
                return {**seg, "ground_truth_dir": str(directory / "poses3d"), "cameras": seg["cameras"],
                        "_persons": ["aria01", "aria02"]}

            def fake_run_bf(cfg, brute_cfg, *, local_report_dir, workers):
                self.assertEqual(workers, 6)
                names = [s["name"] for s in brute_cfg["segments"]]
                self.assertEqual(names, ["002_sword3_aria01", "002_sword3_aria02"])
                from fusion_pipeline.aotr_benchmark import report_csv_name
                csv_path = Path(local_report_dir) / report_csv_name(cfg)
                with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(["Segment", "Protocol", "Cam Master", "Cam Slave"])
                    for s in brute_cfg["segments"]:
                        proto = segment_protocol(cfg, s)
                        w.writerow([s["name"], proto, "cam01", "cam02"])
                        w.writerow([s["name"], proto, "cam02", "cam01"])
                return {n: [{}, {}] for n in names}

            with patch("run_drive_benchmark.ROOT", root), \
                    patch("sys.argv", ["run_drive_benchmark.py", "--local-only"]), \
                    patch("run_drive_benchmark.load_config", return_value=base_cfg), \
                    patch("run_drive_benchmark.stage_segment", side_effect=fake_stage), \
                    patch("run_drive_benchmark.run_brute_force", side_effect=fake_run_bf):
                self.assertEqual(main(), 0)

    def test_reverse_runs_configs_and_cameras_bottom_to_top(self):
        file_id = "1example_drive_id_0123456789"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            for name in ("001_mma5", "002_sword3"):
                (root / f"configs/{name}.yml").write_text(
                    "segments:\n"
                    f"- name: {name}_aria01\n  target_person: aria01\n  ground_truth_dir: {file_id}\n"
                    f"  cameras:\n  - id: cam01\n    pkl: {file_id}\n    video: {file_id}\n"
                    f"  - id: cam02\n    pkl: {file_id}\n    video: {file_id}\n",
                    encoding="utf-8",
                )
            seen = []

            def fake_stage(seg, directory, *, local_only):
                seen.append((seg["name"], [c["id"] for c in seg["cameras"]]))
                return {**seg, "ground_truth_dir": str(directory / "poses3d"), "cameras": seg["cameras"], "_persons": ["aria01"]}

            with patch("run_drive_benchmark.ROOT", root), \
                    patch("sys.argv", ["run_drive_benchmark.py", "--reverse"]), \
                    patch("run_drive_benchmark.load_config", return_value={"fusion": {}, "paths": {"fused_output_dir": "out"}}), \
                    patch("run_drive_benchmark.stage_segment", side_effect=fake_stage), \
                    patch("run_drive_benchmark.run_brute_force", side_effect=lambda cfg, bc, **kw: {s["name"]: [] for s in bc["segments"]}):
                main()
            self.assertEqual(seen, [("002_sword3_aria01", ["cam02", "cam01"]), ("001_mma5_aria01", ["cam02", "cam01"])])

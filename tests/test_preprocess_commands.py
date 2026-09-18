import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.pipelines.preprocess.extract_2d import _build_2d_camera_payload, _load_tracking_payload
from src.pipelines.preprocess.executor import _get_video_fps, _run_cmd


class PreprocessCommandTest(unittest.TestCase):
    @patch("src.pipelines.preprocess.executor.subprocess.check_output", return_value="30000/1001\n")
    def test_ffprobe_path_is_one_argument(self, check_output):
        video_path = 'video"; dangerous command.avi'

        self.assertAlmostEqual(_get_video_fps(video_path), 30000 / 1001)
        args = check_output.call_args.args[0]
        self.assertIsInstance(args, list)
        self.assertEqual(args[-1], video_path)
        self.assertNotIn("shell", check_output.call_args.kwargs)

    @patch("src.pipelines.preprocess.executor.subprocess.run")
    def test_command_runner_does_not_enable_shell(self, run):
        command = ["ffmpeg", "-version"]
        _run_cmd(command)
        run.assert_called_once_with(command, check=True)

    def test_tracking_falls_back_to_sibling_file(self):
        with TemporaryDirectory() as directory:
            pkl_path = Path(directory) / "result.pkl"
            pkl_path.touch()
            pkl_path.with_name("tracking_results.pth").touch()
            payloads = [{0: {}}, {0: {"frame_id": [7], "keypoints": [[[1, 2, 0.9]]]}}]
            with patch("src.pipelines.preprocess.extract_2d.load_joblib_compat", side_effect=payloads):
                tracking = _load_tracking_payload(pkl_path)

        self.assertEqual(tracking["frame_id"], [7])

    def test_coco17_tracking_skips_unavailable_wholebody_joints(self):
        tracking = {"frame_id": [0], "keypoints": [[[0, 0, 0.9]] * 17]}
        payload, _ = _build_2d_camera_payload(tracking, {"head": 0, "left_toe": 17})

        self.assertEqual(payload["keypoints"]["0"], {"head": [0.0, 0.0, 0.9]})


if __name__ == "__main__":
    unittest.main()

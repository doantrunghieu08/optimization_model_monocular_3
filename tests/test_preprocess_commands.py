import unittest
from unittest.mock import patch

from preprocess_pipeline.executor import _get_video_fps, _run_cmd


class PreprocessCommandTest(unittest.TestCase):
    @patch("preprocess_pipeline.executor.subprocess.check_output", return_value="30000/1001\n")
    def test_ffprobe_path_is_one_argument(self, check_output):
        video_path = 'video"; dangerous command.avi'

        self.assertAlmostEqual(_get_video_fps(video_path), 30000 / 1001)
        args = check_output.call_args.args[0]
        self.assertIsInstance(args, list)
        self.assertEqual(args[-1], video_path)
        self.assertNotIn("shell", check_output.call_args.kwargs)

    @patch("preprocess_pipeline.executor.subprocess.run")
    def test_command_runner_does_not_enable_shell(self, run):
        command = ["ffmpeg", "-version"]
        _run_cmd(command)
        run.assert_called_once_with(command, check=True)


if __name__ == "__main__":
    unittest.main()

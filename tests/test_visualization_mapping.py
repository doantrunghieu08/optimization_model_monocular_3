import json
import tempfile
import unittest
from pathlib import Path

from src.pipelines.visualization.executor import _load_source_frame_map


class VisualizationMappingTest(unittest.TestCase):
    def test_source_frame_map_uses_output_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "pose" / "metadata"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "pose_data_3.json").write_text(json.dumps({
                "metadata": {"source_frame_indices": {"camera2": 8}}
            }), encoding="utf-8")

            self.assertEqual(
                _load_source_frame_map({"pose_output_dir": str(Path(tmp) / "pose")}, "camera2"),
                {3: 8},
            )


if __name__ == "__main__":
    unittest.main()

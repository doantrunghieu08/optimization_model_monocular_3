import csv
import tempfile
import unittest
from pathlib import Path

import benchmark_improvements as benchmark


class BenchmarkImprovementsTest(unittest.TestCase):
    def test_all_pair_directions_are_reproducible(self):
        forward = benchmark.build_pairs("forward")
        reverse = benchmark.build_pairs("reverse")

        self.assertEqual(len(forward), 28)
        self.assertEqual(reverse, [(right, left) for left, right in forward])
        self.assertEqual(len(benchmark.build_pairs("both")), 56)

    def test_variant_is_config_driven_without_mutating_baseline(self):
        original = {"fusion": {"correction": {"orientation_enabled": False}}}
        changed = benchmark.with_variant(original, benchmark.VARIANTS["confidence"])

        self.assertNotIn("blend_mode", original["fusion"]["correction"])
        self.assertTrue(changed["fusion"]["correction"]["enabled"])
        self.assertEqual(changed["fusion"]["correction"]["blend_mode"], "confidence")
        self.assertEqual(changed["fusion"]["correction"]["confidence_delta_cap"], 0.05)

    def test_metric_parser_reads_requested_camera_average(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MPJPE_cam2.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Frame", "fused_priority1_mm"])
                writer.writerow([1, "10.00"])
                writer.writerow(["AVERAGE", "12.34"])

            value = benchmark.parse_metric(Path(directory), "MPJPE", "cam2", "fused")

        self.assertEqual(value, 12.34)


if __name__ == "__main__":
    unittest.main()

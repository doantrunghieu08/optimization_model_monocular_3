"""Independent tests for AOTR benchmark provenance and stale cache safety."""
import json
import tempfile
import unittest
from pathlib import Path

from fusion_pipeline.aotr_benchmark import (
    LEGACY_PROTOCOL, cache_compatible, evaluation_protocol,
    report_csv_name, verify_aotr_execution,
)


class AOTRBenchmarkTests(unittest.TestCase):
    def test_legacy_cache_must_not_be_reused_for_aotr(self):
        cfg = {"fusion": {"aotr": {"enabled": True, "observed_prior_weight": 0.0001}}}
        protocol = evaluation_protocol(cfg)
        self.assertIn("__aotr_fk_trustregion_v2_output_isolated__", protocol)
        self.assertFalse(cache_compatible({
            "protocol": LEGACY_PROTOCOL + "__aotr_fk_trustregion_v1__e83f43ea9b57"
        }, protocol))
        self.assertFalse(cache_compatible({"protocol": LEGACY_PROTOCOL}, protocol))
        self.assertTrue(cache_compatible({"protocol": protocol}, protocol))
        self.assertTrue(report_csv_name(cfg).startswith("brute_force_report_aotr_"))
        self.assertTrue(report_csv_name(cfg).endswith(".csv"))

    def test_different_parameters_cannot_share_cache(self):
        first = {"fusion": {"aotr": {"enabled": True, "trust_initial_radius": 0.25}}}
        second = {"fusion": {"aotr": {"enabled": True, "trust_initial_radius": 0.50}}}
        a = evaluation_protocol(first)
        b = evaluation_protocol(second)
        self.assertNotEqual(a, b)
        self.assertNotEqual(report_csv_name(first), report_csv_name(second))
        self.assertFalse(cache_compatible({"protocol": a}, b))

    def test_legacy_still_uses_original_report(self):
        cfg = {"fusion": {"aotr": {"enabled": False}}}
        self.assertEqual(evaluation_protocol(cfg), LEGACY_PROTOCOL)
        self.assertEqual(report_csv_name(cfg), "brute_force_report.csv")

    def test_aotr_requires_valid_new_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = {"fusion": {"aotr": {"enabled": True}},
                   "paths": {"fused_output_dir": temp}}
            with self.assertRaisesRegex(RuntimeError, "absent"):
                verify_aotr_execution(cfg)
            diag = Path(temp) / "aotr_diagnostics.json"
            diag.write_text(json.dumps({
                "mode": "aotr", "frames": 20, "corrected_frames": 0,
                "accepted_bone_frames": 0, "fallback_frames": 20,
                "reason_counts": {"missing_intrinsics": 20},
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "All AOTR frames"):
                verify_aotr_execution(cfg)
            diag.write_text(json.dumps({
                "mode": "aotr", "frames": 20, "corrected_frames": 5,
                "accepted_bone_frames": 9, "fallback_frames": 2,
                "reason_counts": {"objective_improved": 5, "no_supported_improvement": 13},
            }), encoding="utf-8")
            summary = verify_aotr_execution(cfg)
            self.assertEqual(summary["accepted_bone_frames"], 9)


if __name__ == "__main__":
    unittest.main()

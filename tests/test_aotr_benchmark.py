"""Independent tests for AOTR benchmark provenance and stale cache safety."""
import json
import tempfile
import unittest
from pathlib import Path

from fusion_pipeline.aotr_benchmark import (
    LEGACY_PROTOCOL, cache_compatible, evaluation_protocol, identity_mask_exclusions,
    report_csv_name, verify_aotr_execution,
)


class AOTRBenchmarkTests(unittest.TestCase):
    def test_legacy_cache_must_not_be_reused_for_aotr(self):
        cfg = {"fusion": {"aotr": {"enabled": True, "observed_prior_weight": 0.0001}}}
        protocol = evaluation_protocol(cfg)
        self.assertIn("__aotr_fk_trustregion_v3_all_evidence_stable__", protocol)
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

    def test_person_and_height_cannot_share_cache(self):
        first = {
            "inputs": {"target_person": "aria01", "person_heights": {"aria01": 1.737}},
            "fusion": {"aotr": {"enabled": True}},
        }
        second = {
            "inputs": {"target_person": "aria02", "person_heights": {"aria02": 1.700}},
            "fusion": {"aotr": {"enabled": True}},
        }
        self.assertNotEqual(evaluation_protocol(first), evaluation_protocol(second))
        self.assertNotEqual(report_csv_name(first), report_csv_name(second))

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
                "mode": "aotr", "frames": 20, "corrected_frames": 0,
                "accepted_bone_frames": 0, "fallback_frames": 20,
                "reason_counts": {"pair_mode_posed_fallback": 20},
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not an AOTR benchmark"):
                verify_aotr_execution(cfg)
            diag.write_text(json.dumps({
                "mode": "aotr", "frames": 20, "corrected_frames": 5,
                "accepted_bone_frames": 9, "fallback_frames": 2,
                "reason_counts": {"objective_improved": 5, "no_supported_improvement": 13},
            }), encoding="utf-8")
            summary = verify_aotr_execution(cfg)
            self.assertEqual(summary["accepted_bone_frames"], 9)

    def test_enabled_sequence_stability_cannot_be_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = {
                "fusion": {"aotr": {
                    "enabled": True,
                    "sequence_stability": {"enabled": True},
                }},
                "paths": {"fused_output_dir": temp},
            }
            (Path(temp) / "aotr_diagnostics.json").write_text(json.dumps({
                "mode": "aotr", "frames": 20, "corrected_frames": 5,
                "accepted_bone_frames": 9, "fallback_frames": 2,
                "sequence_stability": {"status": "skipped_nonuniform_3d_evidence"},
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "enabled but did not run"):
                verify_aotr_execution(cfg)

    def test_adaptive_naive_stability_follows_use_aotr_stability(self):
        """The ablation adaptive_naive@use_aotr_stability=false must pass (stability off is intended),
        while the default run still has to apply AOTR's stability pass, and an ablation that asked
        for it to be off must not silently run it."""
        def check(temp, adaptive_cfg, status):
            cfg = {"fusion": {"method": "adaptive_naive", "adaptive_naive": adaptive_cfg,
                              "aotr": {"enabled": True, "sequence_stability": {"enabled": True}}},
                   "paths": {"fused_output_dir": temp}}
            (Path(temp) / "aotr_diagnostics.json").write_text(json.dumps({
                "mode": "adaptive_naive", "frames": 20, "fallback_frames": 0,
            }), encoding="utf-8")
            (Path(temp) / "adaptive_naive_diagnostics.json").write_text(json.dumps({
                "mode": "adaptive_naive", "frames": 20, "fallback_frames": 0,
                "sequence_stability": {"status": status},
            }), encoding="utf-8")
            return verify_aotr_execution(cfg)

        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(check(temp, {}, "applied")["frames"], 20)
            with self.assertRaisesRegex(RuntimeError, "sequence stability did not run"):
                check(temp, {}, "disabled")
            self.assertEqual(check(temp, {"use_aotr_stability": False}, "disabled")["frames"], 20)
            with self.assertRaisesRegex(RuntimeError, "use_aotr_stability=false but the stability pass ran"):
                check(temp, {"use_aotr_stability": False}, "applied")


class IdentityMaskProtocolTests(unittest.TestCase):
    """The identity mask's content is part of the protocol: results cached with another
    set of excluded frames must not be reused (same mask file name, rebuilt content)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mask = Path(self.tmp.name) / "identity_mask_005_sword3.json"

    def tearDown(self):
        self.tmp.cleanup()

    def protocol(self, margins, source="configs/brute_force_005_sword3.yml", margin_mm=10.0, paths=None):
        if margins is not None:
            self.mask.write_text(json.dumps({"source": source, "margins": margins}), encoding="utf-8")
        return evaluation_protocol({"fusion": {"aotr": {"enabled": True}}, "evaluation": {
            "identity_mask": {"path": paths or [str(self.mask)], "margin_mm": margin_mm}}})

    def test_rebuilt_mask_with_other_excluded_frames_invalidates_cache(self):
        # Before the fix: every camera shared one key, nothing excluded.
        old = self.protocol({"wham_output": {"aria01": {"0": 2.0, "1": 3.0}}})
        # After: one key per camera, cam04 follows aria02 in frame 1.
        new = self.protocol({"wham_005_sword3_cam01": {"aria01": {"0": 2.0, "1": 3.0}},
                             "wham_005_sword3_cam04": {"aria01": {"0": 4.0, "1": 41.5}}})
        self.assertNotEqual(old, new)
        self.assertFalse(cache_compatible({"protocol": old}, new))

    def test_same_excluded_frames_keep_cache(self):
        a = self.protocol({"cam04": {"aria01": {"0": 4.0, "1": 41.5}}})
        b = self.protocol({"cam04": {"aria01": {"0": 6.0, "1": 30.0}}, "cam01": {"aria01": {"0": 1.0}}},
                          source="/abs/other/path.yml")  # margins move but stay on the same side of 10 mm
        self.assertEqual(a, b)

    def test_threshold_and_merge_order_follow_the_evaluator(self):
        self.protocol({"cam04": {"aria01": {"3": 10.0, "1": 10.5, "2": -50}}})  # writes the mask
        excluded, problems = identity_mask_exclusions({"path": [str(self.mask)], "margin_mm": 10.0})
        self.assertEqual((excluded, problems), ({"cam04": {"aria01": [1]}}, []))  # > margin, not >=
        later = Path(self.tmp.name) / "later.json"
        later.write_text(json.dumps({"margins": {"cam04": {"aria01": {"7": 99}}}}), encoding="utf-8")
        merged, _ = identity_mask_exclusions({"path": [str(self.mask), str(later)]})
        self.assertEqual(merged, {"cam04": {"aria01": [7]}})  # a later file replaces the stem, as in evaluation

    def test_missing_mask_file_does_not_crash_and_differs(self):
        missing = self.protocol(None, paths=[str(Path(self.tmp.name) / "not_built_yet.json")])
        present = self.protocol({"cam04": {"aria01": {"1": 41.5}}})
        self.assertIn("__aotr_", missing)
        self.assertNotEqual(missing, present)


if __name__ == "__main__":
    unittest.main()

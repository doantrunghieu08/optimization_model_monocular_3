"""Fast checks for restoring pre-V3 behavior and for truthful diagnostics."""
import re
import unittest
from pathlib import Path

from fusion_pipeline.diagnostics import summarize_selective_frames


ROOT = Path(__file__).resolve().parents[1]


class SelectiveRegressionGuardTests(unittest.TestCase):
    def test_experimental_v3_must_not_be_default(self):
        cfg = (ROOT / "configs" / "pipeline.yml").read_text(encoding="utf-8")
        self.assertRegex(cfg, r"selective_joint_fusion:\s*\n\s+enabled:\s+false\b")

    def test_diagnostics_distinguish_reject_and_fallback(self):
        frames = [
            {
                "joint_selection": {"head": "KEEP", "left_hand": "BLEND"},
                "joint_selection_details": {"head": "quality_ambiguous"},
                "selection_summary": {"alignment_valid": True, "alignment_median_m": .03},
            },
            {
                "joint_selection": {"head": "KEEP", "left_hand": "REPLACE"},
                "joint_selection_details": {"head": "unsafe_local_geometry"},
                "selection_summary": {"alignment_valid": False, "alignment_median_m": None},
            },
            {"error": "missing frame; raw fallback"},
        ]
        summary = summarize_selective_frames(frames)
        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["frames_with_invalid_alignment"], 1)
        self.assertEqual(summary["frames_with_missing_decisions_or_fallback"], 1)
        self.assertEqual(summary["joint_frames_accepted"], 2)
        self.assertEqual(summary["joint_frames_evaluated"], 4)
        self.assertAlmostEqual(summary["joint_acceptance_rate"], 0.5)
        self.assertEqual(summary["keep_reasons"]["quality_ambiguous"], 1)
        self.assertEqual(summary["keep_reasons"]["unsafe_local_geometry"], 1)
        self.assertEqual(summary["decisions"], {"BLEND": 1, "KEEP": 2, "REPLACE": 1})

    def test_empty_diagnostics_are_not_misreported_as_accuracy(self):
        summary = summarize_selective_frames([])
        self.assertEqual(summary["joint_acceptance_rate"], 0)
        self.assertEqual(summary["joint_frames_accepted"], 0)
        self.assertIn("do not", summary["warning"].lower())
        self.assertIn("PA-MPJPE", summary["warning"])

if __name__ == "__main__":
    unittest.main()

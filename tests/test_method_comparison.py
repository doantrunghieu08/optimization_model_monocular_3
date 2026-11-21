"""run_method_comparison: unit construction and summary on synthetic results (no pipeline run)."""
import unittest

import numpy as np
import pandas as pd

from run_method_comparison import build_units, parse_split, summarize

BRUTE = {"segments": [{
    "name": "seq", "ground_truth_dir": "gt", "target_person": "aria01",
    "cameras": [{"id": c, "pkl": f"{c}.pkl"} for c in ("cam01", "cam02", "cam03", "cam04")],
}]}


class UnitTests(unittest.TestCase):
    def test_parse_split(self):
        self.assertEqual(parse_split("heldout=ARIA02:cam03, cam05"), ("heldout", "aria02", ["cam03", "cam05"]))

    def test_split_units_are_ordered_pairs_within_split(self):
        units = build_units(BRUTE, [("d", "aria01", ["cam01", "cam02"]), ("h", "aria02", ["cam03", "cam04"])])
        pairs = {(u["split"], u["person"], u["A"]["id"], u["B"]["id"]) for u in units}
        self.assertEqual(pairs, {("d", "aria01", "cam01", "cam02"), ("d", "aria01", "cam02", "cam01"),
                                 ("h", "aria02", "cam03", "cam04"), ("h", "aria02", "cam04", "cam03")})

    def test_default_units_use_all_pairs_and_segment_person(self):
        units = build_units(BRUTE)
        self.assertEqual(len(units), 12)
        self.assertTrue(all(u["person"] == "aria01" and u["split"] == "all" for u in units))
        self.assertEqual(len(build_units(BRUTE, persons=["aria01", "aria02"])), 24)

    def test_unknown_camera_in_split_is_an_error(self):
        with self.assertRaises(ValueError):
            build_units(BRUTE, [("d", "aria01", ["cam01", "cam99"])])


def _results(n=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mono = {"pa": 50 + rng.normal(), "mpjpe": 110 + rng.normal(), "mble": 22.0, "accel": 20.0}
        for method, gain in (("aotr", 3.0), ("partwise", 6.0), ("naive", 1.0)):
            r = {"segment": "s", "split": "heldout", "person": "aria02", "master": f"c{i}", "slave": "x",
                 "method": method, "mask": "masked", "frames": 100, "status": "ok"}
            for k, v in mono.items():
                r[f"{k}_mono"] = v
                r[k] = v - (gain if k == "pa" else 0.0)
            rows.append(r)
    rows.append({"segment": "s", "split": "heldout", "person": "aria02", "master": "bad", "slave": "x",
                 "status": "skip: no shared frames"})
    return pd.DataFrame(rows)


class SummaryTests(unittest.TestCase):
    def test_table_means_wins_and_counts(self):
        text = summarize(_results())
        self.assertIn("split=heldout, masked  (runs=8)", text)
        self.assertIn("| partwise |", text)
        self.assertIn("8/8", text)
        self.assertIn("| partwise | naive | -5.00", text)
        self.assertIn("Units: 8 ok, 1 skipped, 0 errors.", text)
        self.assertNotIn("WARNING", text)

    def test_macro_table_over_sequences(self):
        df = _results(n=8)
        df["sequence"] = np.where(df.master.isin(["c0", "c1", "c2", "c3"]), "seqA", "seqB")
        text = summarize(df)
        self.assertIn("Macro (mean over 2 sequences", text)
        self.assertIn("| partwise | naive | -5.00", text)
        self.assertIn("| 2/2 |", text)

    def test_sequence_name_from_gt_dir(self):
        from run_method_comparison import sequence_name
        self.assertEqual(sequence_name("input/043_Grappling2/processed_data/poses3d"), "043_Grappling2")
        self.assertEqual(sequence_name("gt"), "gt")

    def test_warns_when_mono_differs_between_methods(self):
        df = _results()
        df.loc[df.method == "naive", "pa_mono"] += 1.0
        self.assertIn("WARNING: mono differs", summarize(df))


if __name__ == "__main__":
    unittest.main()

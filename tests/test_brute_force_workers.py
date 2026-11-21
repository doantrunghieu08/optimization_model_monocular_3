"""--workers N: the parallel runner runs the same pairs and writes the same report as the
sequential loop, each worker with its own output directories and one BLAS/torch thread."""
import csv
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import brute_force_runner as bfr

BASE_CFG = {
    "fusion": {"aotr": {"enabled": True}, "belief": {"alpha": 0.001, "beta": 0.8}},
    "paths": {
        "smpl_model": "models/SMPL_NEUTRAL.pkl",
        "preprocess_output_dir": "output/preprocess_results",
        "fused_output_dir": "output/fused_results",
        "learnable_output_dir": "output/learnable_results",
        "learnable_extra_output_dir": "output/learnable_extra_results",
        "visualization_output_dir": "output/visualize_results",
        "evaluation_output_dir": "output/evaluation_results",
    },
}


def fake_evaluate(cam_a, cam_b, cfg, gt_dir, workspace, seg_name, target_person=None):
    """Stands in for _evaluate_camera_pair in a worker process (must be importable)."""
    pair = (cam_a["id"], cam_b["id"])
    print(f"pipeline chatter {pair}")  # goes to the pair log, not the console
    if pair == ("c3", "c1"):
        return None  # e.g. cameras without shared frames
    if pair == ("c3", "c2"):
        raise RuntimeError("worker blew up")
    Path(cfg["paths"]["fused_output_dir"]).mkdir(parents=True, exist_ok=True)
    return {
        "master": cam_a["id"], "supplement": cam_b["id"], "set": "Seq1", "target_person": target_person,
        "protocol": bfr._segment_protocol(BASE_CFG, {"target_person": target_person}),
        "mpjpe": 50.0, "pa_mpjpe": 40.0, "% delta_pa_mpjpe": float(int(cam_a["id"][1]) * 10 + int(cam_b["id"][1])),
        "omp": os.environ.get("OMP_NUM_THREADS"), "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
        "fused_dir": cfg["paths"]["fused_output_dir"], "smpl": cfg["paths"]["smpl_model"],
        "gt": gt_dir, "pid": os.getpid(),
    }


class WorkerConfigTests(unittest.TestCase):
    def test_every_output_dir_moves_into_the_worker_dir(self):
        cfg = bfr._worker_config(BASE_CFG, Path("/tmp/w03"))
        outputs = {k: v for k, v in cfg["paths"].items() if k.endswith("_output_dir")}
        self.assertEqual(len(outputs), 6)
        self.assertTrue(all(Path(v).parent == Path("/tmp/w03") for v in outputs.values()))
        self.assertEqual(len(set(outputs.values())), 6)  # no two stages share a directory
        self.assertEqual(cfg["paths"]["smpl_model"], "models/SMPL_NEUTRAL.pkl")  # inputs untouched
        self.assertEqual(BASE_CFG["paths"]["fused_output_dir"], "output/fused_results")  # not mutated

    def test_progress_helpers(self):
        self.assertEqual(bfr._format_duration(75), "1m15s")
        self.assertEqual(bfr._format_duration(3 * 3600 + 5 * 60 + 9), "3h05m")
        self.assertEqual(bfr._pair_log_name("001 mma5/aria01", "cam01", "cam02"), "001_mma5_aria01__cam01-cam02.log")
        line = bfr._result_summary({"% delta_pa_mpjpe": 2.5, "pa_mpjpe": 41.234, "accel": float("inf"), "mble": None})
        self.assertIn("PA +2.50%", line)
        self.assertIn("PA-MPJPE=41.23", line)
        self.assertIn("Accel=N/A", line)
        self.assertIn("MBLE=N/A", line)


class ParallelRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.saved = (bfr.ACTIVE_LOCAL_REPORT_DIR, bfr.ACTIVE_REPORT_CSV_NAME, os.environ.get("OMP_NUM_THREADS"))
        bfr.ACTIVE_LOCAL_REPORT_DIR, bfr.ACTIVE_REPORT_CSV_NAME = self.tmp, "report.csv"
        os.environ["OMP_NUM_THREADS"] = "7"
        (self.tmp / "gt").mkdir()

    def tearDown(self):
        bfr.ACTIVE_LOCAL_REPORT_DIR, bfr.ACTIVE_REPORT_CSV_NAME, omp = self.saved
        if omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = omp
        shutil.rmtree(self.tmp, ignore_errors=True)

    def segment(self, name, person, cam_ids):
        return {"name": name, "target_person": person, "ground_truth_dir": "1" * 33,
                "ground_truth_dir_local": str(self.tmp / "gt"),
                "cameras": [{"id": c, "pkl": f"/data/{c}.pkl"} for c in cam_ids]}

    def test_parallel_run_skips_cache_survives_failures_and_isolates_workers(self):
        aria01 = self.segment("s_aria01", "aria01", ["c1", "c2", "c3"])
        aria02 = self.segment("s_aria02", "aria02", ["c1", "c2"])
        p1, p2 = bfr._segment_protocol(BASE_CFG, aria01), bfr._segment_protocol(BASE_CFG, aria02)
        existing = {
            ("s_aria01", "c1", "c2"): {"protocol": p1, "master": "c1", "supplement": "c2", "mpjpe": 9.0},
            ("s_aria01", "c2", "c1"): {"protocol": "stale", "master": "c2", "supplement": "c1"},
            # aria02 rows have their own protocol and must be reused as well
            ("s_aria02", "c1", "c2"): {"protocol": p2, "master": "c1", "supplement": "c2", "mpjpe": 9.0},
            ("s_aria02", "c2", "c1"): {"protocol": p2, "master": "c2", "supplement": "c1", "mpjpe": 9.0},
        }
        ranked = bfr._run_pairs_parallel([aria01, aria02], existing, BASE_CFG, self.tmp,
                                         "sheet", "Run_test", workers=2, evaluate=fake_evaluate)

        self.assertEqual(set(ranked), {"s_aria01", "s_aria02"})
        self.assertEqual(len(ranked["s_aria02"]), 2)  # all cached, nothing run
        rows = ranked["s_aria01"]
        # c1->c2 cached; c2->c1 (stale) + c1->c3 + c2->c3 run; c3->c1 gave None; c3->c2 raised
        self.assertEqual({(r["master"], r["supplement"]) for r in rows},
                         {("c1", "c2"), ("c2", "c1"), ("c1", "c3"), ("c2", "c3")})
        new = [r for r in rows if "pid" in r]
        self.assertEqual(len(new), 3)
        self.assertEqual([r["% delta_pa_mpjpe"] for r in new], [23.0, 21.0, 13.0])  # ranked best first
        root = self.tmp / bfr.WORKERS_DIR_NAME
        for r in new:
            self.assertEqual((r["omp"], r["openblas"]), ("1", "1"))
            self.assertIn(Path(r["fused_dir"]).parent.name, {"w00", "w01"})
            self.assertEqual(Path(r["fused_dir"]).parent.parent, root)
            self.assertEqual(r["smpl"], "models/SMPL_NEUTRAL.pkl")
            self.assertEqual(r["gt"], str((self.tmp / "gt").resolve()))
            self.assertNotEqual(r["pid"], os.getpid())
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "7")  # parent environment restored

        log = (root / "logs" / "s_aria01__c1-c3.log").read_text(encoding="utf-8")
        self.assertIn("pipeline chatter ('c1', 'c3')", log)
        with (self.tmp / "report.csv").open(encoding="utf-8-sig") as f:
            report = list(csv.DictReader(f))
        self.assertEqual(len(report), 6)  # 4 aria01 + 2 aria02, cached rows included
        self.assertEqual(sorted(r["Segment"] for r in report), ["s_aria01"] * 4 + ["s_aria02"] * 2)

    def test_nothing_to_run_starts_no_worker(self):
        seg = self.segment("s", "aria01", ["c1", "c2"])
        p = bfr._segment_protocol(BASE_CFG, seg)
        existing = {("s", a, b): {"protocol": p, "master": a, "supplement": b}
                    for a, b in (("c1", "c2"), ("c2", "c1"))}
        ranked = bfr._run_pairs_parallel([seg], existing, BASE_CFG, self.tmp, "sheet", "Run",
                                         workers=4, evaluate=fake_evaluate)
        self.assertEqual(len(ranked["s"]), 2)
        self.assertFalse((self.tmp / bfr.WORKERS_DIR_NAME).exists())


if __name__ == "__main__":
    unittest.main()

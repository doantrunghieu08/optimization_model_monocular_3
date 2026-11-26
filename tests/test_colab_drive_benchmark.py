"""Check the Colab config path without installing packages or contacting Drive."""
import ast
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from fusion_pipeline.aotr_benchmark import report_csv_name


ROOT = Path(__file__).resolve().parents[1]


class ColabConfigTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((ROOT / "colab_drive_benchmark.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "colab_drive_benchmark.ipynb", "exec")
        tree = ast.parse("".join(notebook["cells"][1]["source"]))
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                  t.id == "PIPELINE_BEST_CONFIG" for t in n.targets)) or
                 (isinstance(n, ast.FunctionDef) and n.name == "apply_method_config")]
        self.ns = dict(copy=copy, yaml=yaml, report_csv_name=report_csv_name, Path=Path,
                       MULTI_PERSON=True, STATIC_CAMERA=False, USE_WEIGHT_MODEL=True)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-config", "exec"), self.ns)

    def test_benchmark_keeps_trained_weights_after_naive_run(self):
        source = copy.deepcopy(self.ns["PIPELINE_BEST_CONFIG"])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            # A stale default pipeline must never supply the benchmark weights.
            path = root / "configs/pipeline.yml"
            path.write_text("fusion: {}\n", encoding="utf-8")
            self.ns["REPO_DIR"] = root
            self.ns["apply_method_config"]("naive")
            name, _ = self.ns["apply_method_config"]("partwise")
            actual = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(actual["fusion"]["partwise"], source["fusion"]["partwise"])
            self.assertEqual(actual["evaluation"], source["evaluation"])
            self.assertEqual(actual["fusion"]["method"], "partwise")
            self.assertEqual(name, report_csv_name(actual))
            self.assertEqual(self.ns["PIPELINE_BEST_CONFIG"], source)

    def test_explicit_depth_ablation_keeps_trained_snapshot(self):
        source = copy.deepcopy(self.ns["PIPELINE_BEST_CONFIG"])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            self.ns.update(REPO_DIR=root, USE_WEIGHT_MODEL=False)
            self.ns["apply_method_config"]("partwise")
            actual = yaml.safe_load((root / "configs/pipeline.yml").read_text(encoding="utf-8"))
            self.assertFalse(actual["fusion"]["partwise"]["weight_model"]["enabled"])
            self.assertEqual(self.ns["PIPELINE_BEST_CONFIG"], source)

    def test_missing_trained_weights_fail_instead_of_falling_back(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            self.ns["REPO_DIR"] = root
            self.ns["PIPELINE_BEST_CONFIG"]["fusion"]["partwise"].pop("weight_model")
            with self.assertRaisesRegex(ValueError, "Missing trained"):
                self.ns["apply_method_config"]("partwise")


if __name__ == "__main__":
    unittest.main()

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
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ("load_trained_config", "apply_method_config")]
        config = yaml.safe_load((ROOT / "configs/pipeline.yml").read_text(encoding="utf-8"))
        config["fusion"]["partwise"]["weight_model"].update(
            enabled=True, weights=[1.0, 0.7, 0.4, 0.0, 0.2, 0.1, 0.1, 0.2],
            std=[1.0] * 8, active_features=[True, True, True, False, True, True, True, True])
        self.ns = dict(copy=copy, yaml=yaml, report_csv_name=report_csv_name, Path=Path,
                       MULTI_PERSON=True, STATIC_CAMERA=False, USE_WEIGHT_MODEL=True,
                       PIPELINE_BEST_CONFIG=config)
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

    def test_loads_new_occ_weights_from_training_output_instead_of_old_repo_config(self):
        source = copy.deepcopy(self.ns["PIPELINE_BEST_CONFIG"])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            path = root / "pipeline_best.yml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            self.ns["PIPELINE_BEST_CONFIG"] = self.ns["load_trained_config"](path)
            self.ns["REPO_DIR"] = root
            self.ns["apply_method_config"]("naive")
            self.ns["apply_method_config"]("partwise")
            actual = yaml.safe_load((root / "configs/pipeline.yml").read_text(encoding="utf-8"))
            model = actual["fusion"]["partwise"]["weight_model"]
            self.assertEqual(model["weights"][1:4], [0.7, 0.4, 0.0])
            self.assertEqual(model["active_features"][1:4], [True, True, False])
            self.assertEqual(actual["fusion"]["occlusion"], source["fusion"]["occlusion"])

    def test_missing_or_invalid_training_output_fails(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline_best.yml"
            with self.assertRaisesRegex(FileNotFoundError, "TRAINED_CONFIG_PATH"):
                self.ns["load_trained_config"](path)
            for invalid in ("null", "[]", "fusion: {}"):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "fusion.partwise"):
                    self.ns["load_trained_config"](path)

    def test_train_exports_latest_child_run_config_to_shared_benchmark_path(self):
        notebook = json.loads((ROOT / "colab_train_v4.ipynb").read_text(encoding="utf-8"))
        export_cell = next("".join(c["source"]) for c in notebook["cells"]
                           if c["cell_type"] == "code" and "out_cfg = RUN_OUT" in "".join(c["source"]))
        config = copy.deepcopy(self.ns["PIPELINE_BEST_CONFIG"])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/pipeline.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
            output = root / "run_005_occ"
            child = output / "lr_0.02"
            child.mkdir(parents=True)
            (child / "best_partwise.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
            namespace = dict(REPO_DIR=root, OUTPUT=output, ACTIVE_OUT=child)
            exec(compile(export_cell, "train-export", "exec"), namespace)
            shared = self.ns["load_trained_config"](output / "pipeline_best.yml")
            self.assertEqual(shared["fusion"]["partwise"], config["fusion"]["partwise"])
            self.assertEqual(shared["fusion"]["occlusion"], config["fusion"]["occlusion"])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from evaluation_pipeline.executor import _compute_acceleration_error, _compute_mble


class EvaluationMetricTests(unittest.TestCase):
    def test_mble_and_acceleration_error(self):
        truth = {"a": np.array([0.0, 0.0, 0.0]), "b": np.array([1.0, 0.0, 0.0])}
        pred = {"a": truth["a"], "b": np.array([1.1, 0.0, 0.0])}
        mean_mble, details = _compute_mble(pred, truth, [["a", "b"]])
        self.assertAlmostEqual(mean_mble, 100.0)
        self.assertAlmostEqual(details["a-b"]["error_mm"], 100.0)

        linear = tuple({"a": np.array([float(x), 0.0, 0.0])} for x in (0, 1, 2))
        mean_accel, _ = _compute_acceleration_error(linear, linear, ["a"])
        self.assertAlmostEqual(mean_accel, 0.0)

        accelerated = (linear[0], linear[1], {"a": np.array([3.0, 0.0, 0.0])})
        mean_accel, details = _compute_acceleration_error(accelerated, linear, ["a"])
        self.assertAlmostEqual(mean_accel, 1000.0)
        self.assertAlmostEqual(details["a"]["error_mm_s2"], 1000.0)


if __name__ == "__main__":
    unittest.main()

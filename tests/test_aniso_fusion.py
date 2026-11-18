import unittest
from pathlib import Path

import numpy as np

from fusion_pipeline import aniso


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


NAMES = ["left_hip", "right_hip", "neck", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "head"]


class AnisoFusionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_kabsch_recovers_rotation(self):
        B = self.rng.normal(size=(20, 3))
        R = _rot([1, 2, 3], 40)
        self.assertTrue(np.allclose(aniso.kabsch(B @ R.T, B, np.ones(20)), R, atol=1e-8))

    def test_robust_rotation_ignores_outlier_joint(self):
        B = self.rng.normal(size=(20, 3)) * 0.3
        R = _rot([0, 1, 0], 30)
        A = B @ R.T
        A[0] += np.array([1.0, -1.0, 0.5])  # grossly mispredicted joint
        R_plain = aniso.kabsch(A, B, np.ones(20))
        R_rob = aniso.robust_rotation(A, B, np.ones(20), iters=10)
        err = lambda X: np.linalg.norm(X - R)  # noqa: E731
        self.assertLess(err(R_rob), err(R_plain))
        self.assertLess(err(R_rob), 0.05)

    def test_low_belief_view_defers_to_other_view(self):
        X1 = np.array([[[0.1, 0.2, 0.3]]])
        X2 = np.array([[[0.5, -0.2, 0.1]]])
        D = np.array([[[0.0, 0.0, 1.0]]])
        F = aniso.fuse(X1, X2, D, D, np.array([[1.0]]), np.array([[1e-4]]), 0.03, 0.06, 0.03, 0.06)
        self.assertTrue(np.allclose(F, X1, atol=1e-3))

    def test_depth_axis_is_taken_from_the_other_camera(self):
        # cam1 looks along z, cam2 along x. Each is wrong only along its own ray.
        truth = np.array([0.2, 0.1, 0.3])
        X1 = (truth + [0, 0, 0.2])[None, None]
        X2 = (truth + [0.2, 0, 0])[None, None]
        D1 = np.array([[[0.0, 0.0, 1.0]]])
        D2 = np.array([[[1.0, 0.0, 0.0]]])
        w = np.ones((1, 1))
        F = aniso.fuse(X1, X2, D1, D2, w, w, 0.01, 0.2, 0.01, 0.2)
        self.assertLess(np.linalg.norm(F[0, 0] - truth), 0.02)
        self.assertLess(np.linalg.norm(F[0, 0] - truth), np.linalg.norm(0.5 * (X1 + X2)[0, 0] - truth))

    def test_self_calibration_recovers_depth_noise_ratio(self):
        # Ray directions must vary across samples, otherwise only a1^2 + a2^2 is identifiable.
        n = 6000
        D1 = self.rng.normal(size=(n, 3))
        D1 /= np.linalg.norm(D1, axis=1, keepdims=True)
        D2 = self.rng.normal(size=(n, 3))
        D2 /= np.linalg.norm(D2, axis=1, keepdims=True)
        a1, b1, a2, b2 = 0.02, 0.06, 0.03, 0.09

        def noise(D, a, b):
            e = self.rng.normal(size=(n, 3)) * a
            depth = (e * D).sum(-1, keepdims=True)
            return e - depth * D + D * self.rng.normal(size=(n, 1)) * b

        X = self.rng.normal(size=(n, 3)) * 0.3
        X1 = X + noise(D1, a1, b1)
        X2 = X + noise(D2, a2, b2)
        est = aniso.self_calibrate_noise(X1, X2, D1, D2, np.ones(n), np.ones(n))
        # Depth noise is well identified; the lateral split between the two cameras lies on a
        # nearly flat likelihood ridge, so only the summed lateral variance is checked.
        self.assertAlmostEqual(est[1], b1, delta=0.1 * b1)
        self.assertAlmostEqual(est[3], b2, delta=0.1 * b2)
        self.assertAlmostEqual(est[0] ** 2 + est[2] ** 2, a1 ** 2 + a2 ** 2, delta=0.1 * (a1 ** 2 + a2 ** 2))

    def test_banded_smoothing_matches_dense_solution(self):
        for T in (3, 4, 9, 50):
            X = self.rng.normal(size=(T, 4, 3))
            lam = 0.7
            Dm = np.zeros((T - 2, T))
            for i in range(T - 2):
                Dm[i, i:i + 3] = (1.0, -2.0, 1.0)
            dense = np.linalg.solve(np.eye(T) + lam * Dm.T @ Dm, X.reshape(T, -1)).reshape(X.shape)
            self.assertTrue(np.allclose(aniso.smooth_trajectory(X, lam), dense, atol=1e-9), T)
        X = self.rng.normal(size=(10, 2, 3))
        self.assertIs(aniso.smooth_trajectory(X, 0.0), X)

    def test_fuse_sequence_reduces_error_of_two_noisy_views(self):
        T, J = 60, len(NAMES)
        skel = self.rng.normal(size=(J, 3)) * 0.3
        truth = np.stack([skel @ _rot([0, 1, 0], 2 * t).T for t in range(T)])
        R12 = _rot([0, 1, 0], 70)  # camera2 -> camera1
        c1 = np.array([0.0, 0.0, 4.0])
        P1 = truth + c1
        P2 = truth @ R12 + np.array([0.0, 0.0, 4.0])  # R12^T x per row
        D1 = P1 / np.linalg.norm(P1, axis=-1, keepdims=True)
        D2 = P2 / np.linalg.norm(P2, axis=-1, keepdims=True)
        P1n = P1 + D1 * self.rng.normal(size=(T, J, 1)) * 0.08 + self.rng.normal(size=(T, J, 3)) * 0.01
        P2n = P2 + D2 * self.rng.normal(size=(T, J, 1)) * 0.08 + self.rng.normal(size=(T, J, 3)) * 0.01
        W = np.ones((T, J))
        Y1, _, info = aniso.fuse_sequence(P1n, P2n, W, W, NAMES, smoothing_lambda=0.0)

        def rel_err(P):
            root = 0.5 * (P[:, 0] + P[:, 1])
            root_t = 0.5 * (P1[:, 0] + P1[:, 1])
            return np.linalg.norm((P - root[:, None]) - (P1 - root_t[:, None]), axis=-1).mean()

        self.assertLess(rel_err(Y1), 0.8 * rel_err(P1n))
        self.assertGreater(info["sigma_mm"]["camera1_depth"], info["sigma_mm"]["camera1_lateral"])

    def test_aniso_module_is_calibration_and_annotation_free(self):
        source = (Path(__file__).resolve().parents[1] / "fusion_pipeline" / "aniso.py").read_text(encoding="utf-8").lower()
        for forbidden in ("ground_truth", "poses3d", "intrinsics_by_cam", "_load_occlusion_intrinsics"):
            self.assertNotIn(forbidden, source)

    def test_pipeline_config_defaults_to_aniso(self):
        from config_loader import load_config
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "pipeline.yml")
        self.assertEqual(cfg["fusion"]["method"], "aniso")


if __name__ == "__main__":
    unittest.main()


class AnisoOptionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_per_joint_noise_identifies_noisier_camera_under_white_noise(self):
        T, J = 2000, 4
        t = np.linspace(0, 20, T)[:, None] + np.arange(J)[None, :]
        X = np.stack([np.sin(t), np.cos(t), 0 * t], axis=-1) * 0.2
        s1 = np.array([0.01, 0.03, 0.01, 0.02])
        s2 = np.array([0.03, 0.01, 0.01, 0.02])
        X1 = X + self.rng.normal(size=(T, J, 3)) * s1[None, :, None]
        X2 = X + self.rng.normal(size=(T, J, 3)) * s2[None, :, None]
        v1, v2 = aniso.per_joint_noise(X1, X2)
        self.assertGreater(v2[0], v1[0])
        self.assertGreater(v1[1], v2[1])
        self.assertTrue(np.allclose(np.sqrt(v1 / 3), s1, rtol=0.25))

    def test_gcv_smooths_more_for_noisier_signal(self):
        T = 400
        y = np.sin(np.linspace(0, 60, T))[:, None, None] * np.ones((1, 2, 3))
        lam_lo = aniso.gcv_lambda(y + self.rng.normal(size=y.shape) * 0.01)
        lam_hi = aniso.gcv_lambda(y + self.rng.normal(size=y.shape) * 0.2)
        self.assertGreater(lam_hi, lam_lo)

    def test_shared_bone_projection_sets_constant_lengths(self):
        names = ["a", "b"]
        F = np.zeros((5, 2, 3))
        F[:, 1, 0] = [0.9, 1.0, 1.1, 1.0, 1.2]
        lengths = aniso.shared_bone_lengths(F, names, [("b", "a")])
        self.assertAlmostEqual(lengths[("b", "a")], 1.0)
        pts = {"a": [0.0, 0.0, 0.0], "b": [0.0, 2.0, 0.0]}
        aniso.project_bone_lengths(pts, lengths)
        self.assertAlmostEqual(np.linalg.norm(pts["b"]), 1.0, places=5)

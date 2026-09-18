import unittest
import numpy as np
from scipy.optimize import minimize
from src.pipelines.fusion.optimization import optimize_f_points, calculate_stats

class OptimizationDeepDiveTest(unittest.TestCase):
    def setUp(self):
        self.anchors = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        self.f_list = ["left_elbow", "left_wrist", "right_elbow", "right_wrist"]
        
        # Base realistic 3D joint positions (in meters)
        self.base_pose = {
            "left_shoulder": [-0.2, 1.4, 2.0],
            "right_shoulder": [0.2, 1.4, 2.0],
            "left_hip": [-0.15, 0.9, 2.0],
            "right_hip": [0.15, 0.9, 2.0],
            "left_elbow": [-0.4, 1.1, 2.0],
            "left_wrist": [-0.5, 0.8, 2.0],
            "right_elbow": [0.4, 1.1, 2.0],
            "right_wrist": [0.5, 0.8, 2.0],
        }

    def test_huber_vs_mse_loss_outlier_resistance(self):
        """Test that Huber loss resists large outlier errors far better than MSE loss."""
        cam1 = {k: np.array(v) for k, v in self.base_pose.items()}
        cam2 = {k: np.array(v) for k, v in self.base_pose.items()}
        
        # Introduce a large outlier error (0.5m displacement) in left_wrist for cam2
        cam2["left_wrist"] = np.array([-0.5, 0.8 + 0.5, 2.0])
        
        huber_stats = calculate_stats(cam1, cam2, self.f_list, self.anchors, loss_type="huber")
        mse_stats = calculate_stats(cam1, cam2, self.f_list, self.anchors, loss_type="mse")
        
        huber_loss = huber_stats[4]
        mse_loss = mse_stats[4]
        
        # Mean diff is 0.125m (0.5m / 4 joints). MSE loss = 0.125^2 = 0.015625. Huber loss < MSE loss
        self.assertLess(huber_loss, mse_loss)
        print(f"\n[Test] Huber Loss ({huber_loss:.6f}) < MSE Loss ({mse_loss:.6f}) under outlier displacement")

    def test_slsqp_optimization_with_kinematic_constraints(self):
        """Test that SLSQP optimization runs successfully with kinematic constraints enabled."""
        cam1 = {k: np.array(v) for k, v in self.base_pose.items()}
        cam2 = {k: np.array(v) for k, v in self.base_pose.items()}
        
        # Add slight noise to cam2
        rng = np.random.default_rng(42)
        for k in self.f_list:
            cam2[k] = cam2[k] + rng.normal(scale=0.03, size=3)
            
        result_kinematic, res_k = optimize_f_points(
            data={"camera1": cam1, "camera2": cam2},
            anchors=self.anchors,
            f_list=self.f_list,
            use_kinematic_constraints=True,
            loss_type="huber",
            max_iter=100
        )
        
        result_unconstrained, res_u = optimize_f_points(
            data={"camera1": cam1, "camera2": cam2},
            anchors=self.anchors,
            f_list=self.f_list,
            use_kinematic_constraints=False,
            loss_type="huber",
            max_iter=100
        )
        
        self.assertIn("camera1", result_kinematic)
        self.assertIn("camera2", result_kinematic)
        print(f"\n[Test] Kinematic SLSQP Success: {res_k.success}, Unconstrained SLSQP Success: {res_u.success}")

    def test_temporal_smoothing(self):
        """Test that temporal lambda reduces frame-to-frame joint jitter."""
        cam1_t0 = {k: np.array(v) for k, v in self.base_pose.items()}
        cam2_t0 = {k: np.array(v) for k, v in self.base_pose.items()}
        
        # Frame t1 has noisy jump
        cam1_t1 = {k: np.array(v) for k, v in self.base_pose.items()}
        cam2_t1 = {k: np.array(v) for k, v in self.base_pose.items()}
        cam1_t1["left_wrist"] = cam1_t1["left_wrist"] + np.array([0.1, 0.1, 0.0])
        cam2_t1["left_wrist"] = cam2_t1["left_wrist"] + np.array([0.1, 0.1, 0.0])
        
        # Optimize t1 with temporal smoothing from t0
        res_smoothed, _ = optimize_f_points(
            data={"camera1": cam1_t1, "camera2": cam2_t1},
            anchors=self.anchors,
            f_list=self.f_list,
            prev_data={"camera1": cam1_t0, "camera2": cam2_t0},
            temporal_lambda=10.0,
            use_kinematic_constraints=False,
            loss_type="huber"
        )
        
        # Wrist position should be smoothed closer to t0
        wrist_smoothed = np.array(res_smoothed["camera1"]["left_wrist"])
        wrist_raw = cam1_t1["left_wrist"]
        wrist_t0 = cam1_t0["left_wrist"]
        
        dist_smoothed = np.linalg.norm(wrist_smoothed - wrist_t0)
        dist_raw = np.linalg.norm(wrist_raw - wrist_t0)
        
        self.assertLess(dist_smoothed, dist_raw)
        print(f"\n[Test] Temporal smoothing reduced jitter: {dist_raw:.4f}m -> {dist_smoothed:.4f}m from prev frame")

if __name__ == "__main__":
    unittest.main()

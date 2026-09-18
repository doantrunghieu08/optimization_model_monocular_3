import unittest
import numpy as np
from src.pipelines.fusion.correction import _pose_root as correction_pose_root
from src.pipelines.fusion.optimization import _pose_root as optimization_pose_root
from src.pipelines.fusion.optimization import optimize_f_points, calculate_stats

class SolutionValidationTest(unittest.TestCase):
    def setUp(self):
        self.pose_full = {
            "neck": [0.0, 1.4, 2.0],
            "left_hip": [-0.15, 0.9, 2.0],
            "right_hip": [0.15, 0.9, 2.0],
            "left_elbow": [-0.4, 1.1, 2.0],
            "right_elbow": [0.4, 1.1, 2.0],
        }

    def test_pose_root_uses_neck_and_hips_triplet(self):
        """Verify that _pose_root computes centroid over neck and hips for enhanced root stability."""
        root_corr = correction_pose_root(self.pose_full)
        root_opt = optimization_pose_root(self.pose_full)
        
        expected_root = np.array([0.0, (1.4 + 0.9 + 0.9) / 3.0, 2.0])
        
        np.testing.assert_allclose(root_corr, expected_root)
        np.testing.assert_allclose(root_opt, expected_root)
        print(f"\n[Validation] Root Centroid (Neck+Hips) = {root_corr.round(4).tolist()}")

    def test_root_fallback_when_neck_missing(self):
        """Verify fallback to hips when neck keypoint is missing."""
        pose_no_neck = {
            "left_hip": [-0.15, 0.9, 2.0],
            "right_hip": [0.15, 0.9, 2.0],
        }
        root_corr = correction_pose_root(pose_no_neck)
        expected_root = np.array([0.0, 0.9, 2.0])
        np.testing.assert_allclose(root_corr, expected_root)
        print(f"[Validation] Root Fallback (Hips-only) = {root_corr.round(4).tolist()}")

    def test_full_fusion_optimization_pipeline(self):
        """Verify SLSQP optimization with updated root and Huber loss."""
        cam1 = {k: np.array(v) for k, v in self.pose_full.items()}
        cam2 = {k: np.array(v) + np.array([0.01, -0.01, 0.02]) for k, v in self.pose_full.items()}
        
        anchors = ["neck", "left_hip", "right_hip"]
        f_list = ["left_elbow", "right_elbow"]
        
        opt_data, res = optimize_f_points(
            data={"camera1": cam1, "camera2": cam2},
            anchors=anchors,
            f_list=f_list,
            loss_type="huber",
            use_kinematic_constraints=True
        )
        
        self.assertTrue(res.success)
        self.assertIn("camera1", opt_data)
        self.assertIn("camera2", opt_data)
        print("[Validation] Full Fusion Optimization with Kinematic Constraints & Huber Loss PASSED cleanly!")

if __name__ == "__main__":
    unittest.main()

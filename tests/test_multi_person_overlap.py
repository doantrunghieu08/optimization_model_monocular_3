"""Regression & unit tests for multi-person overlap, occlusion, anti-swap gating, and person mapping."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from compat import parse_person_mapping, resolve_person_payload, resolve_target_track_ids
from fusion_pipeline.aotr import optimize_aotr
from fusion_pipeline.aotr_3d import optimize_aotr_3d
from fusion_pipeline.aotr_benchmark import evaluation_protocol
from fusion_pipeline.detector import compute_inter_person_occlusion, compute_visibility_from_mesh_vertices
from fusion_pipeline.executor import _extract_other_persons_by_frame
from fusion_pipeline.partwise import fuse_partwise


def _skeleton(depth=4.0, left_arm_err=(0.0, 0.0, 0.0), right_arm_err=(0.0, 0.0, 0.0)):
    base = {
        "pelvis": [0.0, -0.10, 0.0], "neck": [0.0, 0.50, 0.0], "head": [0.0, 0.70, 0.0],
        "left_hip": [-0.12, -0.12, 0.0], "right_hip": [0.12, -0.12, 0.0],
        "left_knee": [-0.13, -0.55, 0.02], "right_knee": [0.13, -0.55, 0.02],
        "left_ankle": [-0.13, -0.95, 0.0], "right_ankle": [0.13, -0.95, 0.0],
        "left_foot": [-0.13, -1.00, 0.08], "right_foot": [0.13, -1.00, 0.08],
        "left_toe": [-0.13, -1.00, 0.14], "right_toe": [0.13, -1.00, 0.14],
        "left_shoulder": [-0.20, 0.45, 0.0], "right_shoulder": [0.20, 0.45, 0.0],
        "left_elbow": [-0.45, 0.30, 0.05], "right_elbow": [0.45, 0.30, 0.05],
        "left_wrist": [-0.60, 0.10, 0.15], "right_wrist": [0.60, 0.10, 0.15],
        "left_hand": [-0.65, 0.05, 0.18], "right_hand": [0.65, 0.05, 0.18],
    }
    out = {k: np.asarray(v, dtype=float) + np.array([0.0, 0.0, depth]) for k, v in base.items()}
    la = np.asarray(left_arm_err, dtype=float)
    ra = np.asarray(right_arm_err, dtype=float)
    for k in ("left_elbow", "left_wrist", "left_hand"):
        out[k] = out[k] + la
    for k in ("right_elbow", "right_wrist", "right_hand"):
        out[k] = out[k] + ra
    return out


def _rotate(pose, deg, shift=(0.0, 0.0, 0.0)):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c = pose["pelvis"]
    return {k: R @ (v - c) + c + np.asarray(shift, dtype=float) for k, v in pose.items()}


def _blob_at(center, n=40, radius=0.02):
    rng = np.random.default_rng(7)
    return np.asarray(center, dtype=float)[None, :] + rng.uniform(-radius, radius, size=(n, 3))


class MultiPersonOverlapTests(unittest.TestCase):
    def test_inter_person_occlusion_checks_all_joints_and_n_people(self):
        target = _skeleton(depth=4.0)
        # Person B stands halfway along the ray to left_shoulder; Person C blocks right_wrist.
        person_b = _blob_at(target["left_shoulder"] * 0.5)
        person_c = _blob_at(target["right_wrist"] * 0.6)
        occ = compute_inter_person_occlusion(target, [person_b, person_c], occlusion_tau=0.05, ray_radius=0.045, vertex_stride=1, min_hits=3)
        self.assertTrue(occ["left_shoulder"])
        self.assertTrue(occ["right_wrist"])
        self.assertFalse(occ["right_shoulder"])
        self.assertFalse(occ["pelvis"])

        # Behind target should NOT occlude
        person_behind = _blob_at(target["left_shoulder"] * 1.3)
        occ_behind = compute_inter_person_occlusion(target, [person_behind], occlusion_tau=0.05, ray_radius=0.045, vertex_stride=1, min_hits=3)
        self.assertFalse(any(occ_behind.values()))

        vis = compute_visibility_from_mesh_vertices(target, verts=None, extra_verts=[person_b, person_c])
        self.assertFalse(vis["left_shoulder"])
        self.assertFalse(vis["right_wrist"])
        self.assertTrue(vis["pelvis"])

    def test_extract_other_persons_indexes_by_true_frame_id_for_n_people(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkl = root / "wham_cam01.pkl"
            (root / "wham_cam01_person_mapping.txt").write_text("0 = aria01\n1 = aria02\n2 = aria03\n", encoding="utf-8")
            raw = {
                0: {"frame_ids": np.array([10, 11, 12]), "verts": np.zeros((3, 5, 3))},
                1: {"frame_ids": np.array([11, 12, 13]), "verts": np.ones((3, 5, 3))},
                2: {"frame_ids": np.array([12, 13]), "verts": np.full((2, 5, 3), 2.0)},
            }
            by_frame = _extract_other_persons_by_frame(raw, target_person="aria01", pkl_path=pkl)
            self.assertNotIn(10, by_frame)
            self.assertEqual(len(by_frame[11]), 1)
            self.assertEqual(len(by_frame[12]), 2)
            self.assertTrue(np.allclose(by_frame[11][0], 1.0))
            self.assertTrue(np.allclose(by_frame[12][1], 2.0))

    def test_parse_person_mapping_does_not_steal_sibling_camera_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam1 = root / "exo" / "cam01"
            cam2 = root / "exo" / "cam02"
            cam1.mkdir(parents=True)
            cam2.mkdir(parents=True)
            (cam1 / "wham_cam01_person_mapping.txt").write_text("0 = aria01\n", encoding="utf-8")
            pkl2 = cam2 / "wham_cam02.pkl"
            pkl2.write_bytes(b"")
            self.assertEqual(parse_person_mapping(pkl2), {})

    def test_partwise_rescues_occluded_master_and_protects_from_occluded_slave(self):
        gt = [_skeleton(depth=4.0) for _ in range(8)]
        # Master has bad left_elbow/wrist and is occluded on left arm; Slave matches GT.
        master = [_skeleton(depth=5.0, left_arm_err=(0.0, 0.25, -0.20)) for _ in range(8)]
        slave = [_rotate(_skeleton(depth=3.0), 35) for _ in range(8)]
        vis_m = [{k: (k not in ("left_elbow", "left_wrist", "left_hand")) for k in gt[0]} for _ in range(8)]
        vis_s = [{k: True for k in gt[0]} for _ in range(8)]

        base_cfg = {"depth_logit": {"a": 0.0, "b": 1.7, "mu": 0.0, "sd": 1.4},
                    "rotation_half_window": 2, "sequence_stability": {"enabled": False}}
        mp_cfg = {**base_cfg, "multi_person": {
            "enabled": True, "visibility_weighted_rotation": True,
            "master_occluded_logit_bias": 2.5, "slave_occluded_logit_bias": 2.5,
            "anti_swap_gate": True, "max_median_bone_angle_deg": 50.0,
        }}
        out_old, _ = fuse_partwise(master, slave, base_cfg)
        out_new, diag = fuse_partwise(master, slave, mp_cfg, vis_master=vis_m, vis_slave=vis_s)
        err_old = np.mean([np.linalg.norm(np.asarray(o["left_wrist"]) - (g["left_wrist"] + np.array([0, 0, 1.0]))) for o, g in zip(out_old, gt)])
        err_new = np.mean([np.linalg.norm(np.asarray(o["left_wrist"]) - (g["left_wrist"] + np.array([0, 0, 1.0]))) for o, g in zip(out_new, gt)])
        self.assertLess(err_new, err_old)
        self.assertGreater(diag["occluded_master_bones"], 0)

        # Now Slave is corrupted & occluded on left arm while Master is clean & visible
        master_clean = [_skeleton(depth=3.0) for _ in range(8)]
        slave_bad = [_rotate(_skeleton(depth=5.0, left_arm_err=(0.0, 0.30, -0.25)), 35) for _ in range(8)]
        out_old2, _ = fuse_partwise(master_clean, slave_bad, base_cfg)
        out_new2, diag2 = fuse_partwise(master_clean, slave_bad, mp_cfg, vis_master=vis_s, vis_slave=vis_m)
        dev_old = np.mean([np.linalg.norm(np.asarray(o["left_wrist"]) - m["left_wrist"]) for o, m in zip(out_old2, master_clean)])
        dev_new = np.mean([np.linalg.norm(np.asarray(o["left_wrist"]) - m["left_wrist"]) for o, m in zip(out_new2, master_clean)])
        self.assertLess(dev_new, dev_old * 0.5)
        self.assertGreater(diag2["occluded_slave_bones"], 0)

    def test_partwise_and_aotr_gate_identity_swap_frames(self):
        # Master = upright person A; Slave = person B with completely opposite limbs (same torso!)
        master = [_skeleton(depth=4.0) for _ in range(6)]
        swapped_slave = []
        for _ in range(6):
            s = _skeleton(depth=4.0)
            for side in ("left", "right"):
                # Flip arms up and bend knees backward
                s[f"{side}_elbow"] = s[f"{side}_shoulder"] + np.array([0.0, 0.30, 0.0])
                s[f"{side}_wrist"] = s[f"{side}_elbow"] + np.array([0.0, 0.25, 0.0])
                s[f"{side}_hand"] = s[f"{side}_wrist"] + np.array([0.0, 0.05, 0.0])
                s[f"{side}_knee"] = s[f"{side}_hip"] + np.array([0.0, 0.20, -0.35])
                s[f"{side}_ankle"] = s[f"{side}_knee"] + np.array([0.0, -0.35, -0.20])
            swapped_slave.append(_rotate(s, 25))

        mp_cfg = {
            "depth_logit": {"a": 0.0, "b": 1.7, "mu": 0.0, "sd": 1.4},
            "rotation_half_window": 2,
            "sequence_stability": {"enabled": False},
            "multi_person": {"enabled": True, "anti_swap_gate": True, "max_median_bone_angle_deg": 45.0},
        }
        out_pw, diag_pw = fuse_partwise(master, swapped_slave, mp_cfg)
        self.assertEqual(diag_pw["swap_gated_frames"], 6)
        for o, m in zip(out_pw, master):
            self.assertLess(np.linalg.norm(np.asarray(o["left_wrist"]) - m["left_wrist"]), 1e-9)

        out_aotr, diag_aotr = optimize_aotr_3d(
            master[0], swapped_slave[0],
            config={"max_cross_view_bone_angle_deg": 45.0},
        )
        self.assertEqual(diag_aotr["status"], "fallback")
        self.assertEqual(diag_aotr["reason"], "cross_view_pose_mismatch")
        self.assertLess(np.linalg.norm(np.asarray(out_aotr["left_wrist"]) - master[0]["left_wrist"]), 1e-9)

    def test_aotr_temporal_alignment_fallback_rescues_when_torso_occluded(self):
        clean_m = _skeleton(depth=4.0)
        clean_s = _rotate(_skeleton(depth=4.0), 40, shift=(0.6, 0.0, 0.2))
        _, diag_clean = optimize_aotr_3d(clean_m, clean_s)
        prev_tf = diag_clean.get("_transform")
        self.assertIsNotNone(prev_tf)

        # Now Person B hides 5 of 6 torso joints + left arm in Master, while Slave is clean
        occ_m = _skeleton(depth=4.0, left_arm_err=(0.0, 0.25, -0.15))
        vis_m = {k: True for k in occ_m}
        for j in ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "left_elbow", "left_wrist"):
            vis_m[j] = False

        # Without fallback: fails closed on insufficient_torso_anchors
        _, d_no = optimize_aotr_3d(occ_m, clean_s, vis_master=vis_m, config={"temporal_alignment_fallback": False})
        self.assertEqual(d_no["status"], "fallback")
        self.assertEqual(d_no["reason"], "insufficient_torso_anchors")

        # With temporal fallback: uses prev_tf and rescues left arm
        fused, d_yes = optimize_aotr_3d(
            occ_m, clean_s, vis_master=vis_m,
            config={"temporal_alignment_fallback": True},
            prev_transform=prev_tf,
        )
        self.assertEqual(d_yes["alignment_source"], "temporal_fallback")
        self.assertEqual(d_yes["status"], "corrected")
        err_before = np.linalg.norm(occ_m["left_wrist"] - clean_m["left_wrist"])
        err_after = np.linalg.norm(np.asarray(fused["left_wrist"]) - clean_m["left_wrist"])
        self.assertLess(err_after, err_before)

    def test_partwise_guards_against_occlusion_bbox_collapse_depth_explosion(self):
        # Clean Master at z=6.0m; Slave collapses under occlusion to z=16.0m with huge limb error.
        master = [_skeleton(depth=6.0) for _ in range(8)]
        slave = [_rotate(_skeleton(depth=16.0, left_arm_err=(0.0, 0.35, -0.25)), 30) for _ in range(8)]
        base_cfg = {"depth_logit": {"a": 0.0, "b": 1.7, "mu": 0.0, "sd": 1.4},
                    "rotation_half_window": 2, "sequence_stability": {"enabled": False}}
        mp_cfg = {**base_cfg, "multi_person": {"enabled": True, "anti_swap_gate": False}}
        out_old, _ = fuse_partwise(master, slave, base_cfg)
        out_new, diag = fuse_partwise(master, slave, mp_cfg)
        err_old = np.mean([np.linalg.norm(np.asarray(o["left_wrist"]) - m["left_wrist"]) for o, m in zip(out_old, master)])
        err_new = np.mean([np.linalg.norm(np.asarray(o["left_wrist"]) - m["left_wrist"]) for o, m in zip(out_new, master)])
        self.assertEqual(diag["depth_outlier_frames"], 8)
        self.assertLess(err_new, err_old * 0.2)

    def test_protocol_hash_changes_when_multi_person_occlusion_toggled(self):
        base = {"fusion": {"method": "aotr", "aotr": {"enabled": True}, "occlusion": {"enabled": True, "multi_person": False}},
                "inputs": {"target_person": "aria01"}}
        mp = {"fusion": {"method": "aotr", "aotr": {"enabled": True}, "occlusion": {"enabled": True, "multi_person": True}},
              "inputs": {"target_person": "aria01"}}
        self.assertNotEqual(evaluation_protocol(base), evaluation_protocol(mp))


if __name__ == "__main__":
    unittest.main()

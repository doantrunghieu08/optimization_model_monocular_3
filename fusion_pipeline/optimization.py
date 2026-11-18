import json
import time
from pathlib import Path
import numpy as np
import torch
from fusion_pipeline.config import BONE_LENGTH_MIN_SCALE
from fusion_pipeline.config import BONE_LENGTH_MAX_SCALE
from fusion_pipeline.detector import as_xyz
from fusion_pipeline.config import DEFAULT_OCCLUDED_FACTOR
from fusion_pipeline.config import RIGID_BONES_RATIO
from fusion_pipeline.config import HEIGHT
from fusion_pipeline.config import HUBER_DELTA
from fusion_pipeline.config import HINGE_TRIPLETS
from fusion_pipeline.config import DEFAULT_MIN_COS
from fusion_pipeline.config import DEFAULT_MAX_COS


def get_diff_f(f_name, anchors, cam1, cam2, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR):
    p1_f = as_xyz(cam1[f_name])
    p2_f = as_xyz(cam2[f_name])
    sum_wdiff, sum_w = 0.0, 0.0
    for a in anchors:
        ca1 = float(conf1.get(a, 1.0)) if conf1 else 1.0
        ca2 = float(conf2.get(a, 1.0)) if conf2 else 1.0
        va1 = 1.0 if (vis1 and vis1.get(a, True)) else float(occluded_factor)
        va2 = 1.0 if (vis2 and vis2.get(a, True)) else float(occluded_factor)
        w = ((ca1 + ca2) / 2.0) * (va1 * va2)
        d1 = np.linalg.norm(p1_f - as_xyz(cam1[a]))
        d2 = np.linalg.norm(p2_f - as_xyz(cam2[a]))
        sum_wdiff += w * abs(d1 - d2)
        sum_w += w
    return sum_wdiff / max(sum_w, 1e-12)


def calculate_stats(cam1, cam2, f_list, anchors, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR, f_weights=None, huber_delta=HUBER_DELTA):
    if not f_list or not anchors:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = [get_diff_f(f, anchors, cam1, cam2, conf1=conf1, conf2=conf2, vis1=vis1, vis2=vis2, occluded_factor=occluded_factor) for f in f_list]
    if f_weights:
        diffs = [d * f_weights[n] for d, n in zip(diffs, f_list)]
    arr_diffs = np.array(diffs, dtype=float)
    q1 = float(np.percentile(arr_diffs, 25))
    q3 = float(np.percentile(arr_diffs, 75))
    mean_val = float(np.mean(arr_diffs))
    median_val = float(np.median(arr_diffs))
    huber = np.where(arr_diffs <= huber_delta, 0.5 * arr_diffs ** 2, huber_delta * (arr_diffs - 0.5 * huber_delta))
    huber_loss = float(np.mean(huber))
    return q1, q3, mean_val, median_val, huber_loss


def compute_dynamic_scale(cam_dict, f_list, ratios):
    sum_len, sum_ratio = 0.0, 0.0
    for (c, p), r in ratios.items():
        if c not in f_list and p not in f_list and c in cam_dict and p in cam_dict:
            sum_len += float(np.linalg.norm(cam_dict[c] - cam_dict[p]))
            sum_ratio += r
    if sum_ratio > 0:
        return sum_len / sum_ratio
    if "left_shoulder" in cam_dict and "right_shoulder" in cam_dict:
        shoulder_dist = float(np.linalg.norm(as_xyz(cam_dict["left_shoulder"]) - as_xyz(cam_dict["right_shoulder"])))
        if shoulder_dist > 0.1:
            return shoulder_dist / 0.22
    return HEIGHT


def optimize_f_points(data, anchors, f_list, conf1=None, conf2=None, vis1=None, vis2=None, occluded_factor=DEFAULT_OCCLUDED_FACTOR, regularization=False, regularization_lambda=1.0, prev_data=None, prev_prev_data=None, temporal_lambda=1.0, max_iter=35, uncertainty_gating=True, device=None):
    """
    Tối ưu hóa nhóm điểm F bằng PyTorch Autograd + Adam trên GPU (nhanh gấp ~100x so với SciPy SLSQP).
    """
    cam1 = {k: as_xyz(v) for k, v in data["camera1"].items()}
    cam2 = {k: as_xyz(v) for k, v in data["camera2"].items()}
    if not f_list or not anchors:
        return {"camera1": cam1, "camera2": cam2}, None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_floor = 0.2 if uncertainty_gating else 0.0

    w_a = []
    for a in anchors:
        ca1 = float(conf1.get(a, 1.0)) if conf1 else 1.0
        ca2 = float(conf2.get(a, 1.0)) if conf2 else 1.0
        va1 = 1.0 if (vis1 and vis1.get(a, True)) else float(occluded_factor)
        va2 = 1.0 if (vis2 and vis2.get(a, True)) else float(occluded_factor)
        w_a.append(((ca1 + ca2) / 2.0) * (va1 * va2))
    w_a_t = torch.tensor(w_a, dtype=torch.float32, device=device)
    w_a_t = w_a_t / torch.clamp(w_a_t.sum(), min=1e-12)

    w_f, c1_f, c2_f = [], [], []
    for f in f_list:
        c1 = float(conf1.get(f, 1.0)) if conf1 else 1.0
        c2 = float(conf2.get(f, 1.0)) if conf2 else 1.0
        v1 = 1.0 if (vis1 and vis1.get(f, True)) else float(occluded_factor)
        v2 = 1.0 if (vis2 and vis2.get(f, True)) else float(occluded_factor)
        w_f.append(((c1 + c2) / 2.0) * (v1 * v2))
        c1_f.append(max(c1, min_floor))
        c2_f.append(max(c2, min_floor))
    w_f_t = torch.tensor(w_f, dtype=torch.float32, device=device)
    c1_f_t = torch.tensor(c1_f, dtype=torch.float32, device=device)
    c2_f_t = torch.tensor(c2_f, dtype=torch.float32, device=device)

    p1_f_init = torch.tensor([cam1[f] for f in f_list], dtype=torch.float32, device=device)
    p2_f_init = torch.tensor([cam2[f] for f in f_list], dtype=torch.float32, device=device)
    p1_a_t = torch.tensor([cam1[a] for a in anchors], dtype=torch.float32, device=device)
    p2_a_t = torch.tensor([cam2[a] for a in anchors], dtype=torch.float32, device=device)

    dyn_scale_cam1 = compute_dynamic_scale(cam1, f_list, RIGID_BONES_RATIO)
    dyn_scale_cam2 = compute_dynamic_scale(cam2, f_list, RIGID_BONES_RATIO)

    f_indices = {name: i for i, name in enumerate(f_list)}
    anchor_indices = {name: i for i, name in enumerate(anchors)}

    bone_info1, bone_info2 = [], []
    for (child, parent), ratio in RIGID_BONES_RATIO.items():
        if child not in cam1 or parent not in cam1 or (child not in f_list and parent not in f_list):
            continue
        ref1 = float(np.linalg.norm(cam1[child] - cam1[parent]))
        target1 = ref1 if ref1 > 0.05 else (ratio * dyn_scale_cam1)
        bone_info1.append((child, parent, (BONE_LENGTH_MIN_SCALE * target1) ** 2, (BONE_LENGTH_MAX_SCALE * target1) ** 2))

        ref2 = float(np.linalg.norm(cam2[child] - cam2[parent]))
        target2 = ref2 if ref2 > 0.05 else (ratio * dyn_scale_cam2)
        bone_info2.append((child, parent, (BONE_LENGTH_MIN_SCALE * target2) ** 2, (BONE_LENGTH_MAX_SCALE * target2) ** 2))

    prev_p1_t, prev_p2_t = None, None
    has_prev_prev = prev_prev_data is not None
    if prev_data:
        p1_prev_list, p2_prev_list = [], []
        p1_pp_list, p2_pp_list = [], []
        prev_cam1 = prev_data.get("camera1", {})
        prev_cam2 = prev_data.get("camera2", {})
        pp_cam1 = prev_prev_data.get("camera1", {}) if has_prev_prev else {}
        pp_cam2 = prev_prev_data.get("camera2", {}) if has_prev_prev else {}
        for f in f_list:
            p1_prev_list.append(prev_cam1.get(f, cam1[f]))
            p2_prev_list.append(prev_cam2.get(f, cam2[f]))
            if has_prev_prev:
                p1_pp_list.append(pp_cam1.get(f, p1_prev_list[-1]))
                p2_pp_list.append(pp_cam2.get(f, p2_prev_list[-1]))
        prev_p1_t = torch.tensor(p1_prev_list, dtype=torch.float32, device=device)
        prev_p2_t = torch.tensor(p2_prev_list, dtype=torch.float32, device=device)
        if has_prev_prev:
            pp_p1_t = torch.tensor(p1_pp_list, dtype=torch.float32, device=device)
            pp_p2_t = torch.tensor(p2_pp_list, dtype=torch.float32, device=device)

    delta1 = torch.zeros_like(p1_f_init, requires_grad=True)
    delta2 = torch.zeros_like(p2_f_init, requires_grad=True)

    opt = torch.optim.Adam([delta1, delta2], lr=0.01)
    n_iters = min(int(max_iter), 35)

    for _ in range(n_iters):
        opt.zero_grad()
        p1_cur = p1_f_init + delta1
        p2_cur = p2_f_init + delta2

        d1 = torch.cdist(p1_cur, p1_a_t)
        d2 = torch.cdist(p2_cur, p2_a_t)
        diff = torch.abs(d1 - d2)
        diff_f = (diff * w_a_t.unsqueeze(0)).sum(dim=-1) * w_f_t
        huber = torch.where(diff_f <= HUBER_DELTA, 0.5 * (diff_f ** 2), HUBER_DELTA * (diff_f - 0.5 * HUBER_DELTA)).mean()

        loss = huber
        if regularization:
            prox = (c1_f_t * (delta1 ** 2).sum(dim=-1) + c2_f_t * (delta2 ** 2).sum(dim=-1)).sum()
            loss = loss + float(regularization_lambda) * prox

        if prev_p1_t is not None:
            if has_prev_prev:
                t_loss = ((p1_cur - 2.0 * prev_p1_t + pp_p1_t) ** 2).sum() + ((p2_cur - 2.0 * prev_p2_t + pp_p2_t) ** 2).sum()
            else:
                t_loss = ((p1_cur - prev_p1_t) ** 2).sum() + ((p2_cur - prev_p2_t) ** 2).sum()
            loss = loss + float(temporal_lambda) * t_loss

        b_penalty = 0.0
        for c, p, low, up in bone_info1:
            p_c = p1_cur[f_indices[c]] if c in f_indices else p1_a_t[anchor_indices[c]]
            p_p = p1_cur[f_indices[p]] if p in f_indices else p1_a_t[anchor_indices[p]]
            d_sq = ((p_c - p_p) ** 2).sum()
            b_penalty = b_penalty + torch.relu(low - d_sq) ** 2 + torch.relu(d_sq - up) ** 2

        for c, p, low, up in bone_info2:
            p_c = p2_cur[f_indices[c]] if c in f_indices else p2_a_t[anchor_indices[c]]
            p_p = p2_cur[f_indices[p]] if p in f_indices else p2_a_t[anchor_indices[p]]
            d_sq = ((p_c - p_p) ** 2).sum()
            b_penalty = b_penalty + torch.relu(low - d_sq) ** 2 + torch.relu(d_sq - up) ** 2

        loss = loss + 10.0 * b_penalty
        loss.backward()
        opt.step()

    p1_res = (p1_f_init + delta1).detach().cpu().numpy()
    p2_res = (p2_f_init + delta2).detach().cpu().numpy()

    p1_opt = dict(cam1)
    p2_opt = dict(cam2)
    for i, name in enumerate(f_list):
        p1_opt[name] = p1_res[i].tolist()
        p2_opt[name] = p2_res[i].tolist()

    return {"camera1": p1_opt, "camera2": p2_opt}, None


def optimize_sequence_gpu(
    keypoints_dir: Path,
    metadata_dir: Path,
    pose_dir: Path | None = None,
    config: dict = None,
    device: str = None,
    master_only: bool = False,
    anchor_cam: str = "camera1",
    selection_only: bool = False,
    protect_master: bool | None = None,
) -> None:
    """
    Tối ưu hóa toàn bộ chuỗi frame (Batch Sequence Optimization) bằng PyTorch Autograd trên GPU.
    Giải bài toán đa mục tiêu (Multi-view Consistency + Bone Constraints + Temporal Acceleration + Proximity)
    cho toàn bộ video trong một pass duy nhất (~0.3-0.7s trên GPU thay vì 3-4 phút CPU SLSQP).
    """
    keypoints_dir = Path(keypoints_dir)
    metadata_dir = Path(metadata_dir)
    files = sorted(keypoints_dir.glob("fused_data_*.json"), key=lambda f: int(f.stem.split("_")[-1]))
    T = len(files)
    if T < 3:
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    all_fused = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    all_meta = [json.loads((metadata_dir / f.name).read_text(encoding="utf-8")) if (metadata_dir / f.name).exists() else {} for f in files]

    joint_names = list(all_fused[0]["camera1"].keys())
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    N = len(joint_names)
    # These are the only coordinates selective mode may change. KEEP joints
    # are frozen in the optimizer and are not length-projected afterwards.
    selection_np = np.array([
        [1.0 if meta.get("joint_selection", {}).get(name, "KEEP") in ("BLEND", "REPLACE") else 0.0
         for name in joint_names]
        for meta in all_meta
    ], dtype=np.float32)
    if selection_only:
        master_only = True
        if not np.any(selection_np):
            print("[Fusion] No accepted slave joints; selective optimizer skipped.")
            return

    bone_pairs = []
    for (c, p) in RIGID_BONES_RATIO.keys():
        if c in name_to_idx and p in name_to_idx:
            bone_pairs.append((name_to_idx[c], name_to_idx[p]))

    p1_fused_np = np.array([[frame["camera1"][j] for j in joint_names] for frame in all_fused], dtype=np.float32)
    p2_fused_np = np.array([[frame["camera2"][j] for j in joint_names] for frame in all_fused], dtype=np.float32)

    # Reference bone lengths from initial pose
    all_pose = None
    if pose_dir and Path(pose_dir).exists():
        pose_files = sorted(Path(pose_dir).glob("pose_data_*.json"), key=lambda f: int(f.stem.split("_")[-1]))
        if len(pose_files) == T:
            all_pose = [json.loads(f.read_text(encoding="utf-8")) for f in pose_files]
            p1_ref_np = np.array([[frame["camera1"][j] for j in joint_names] for frame in all_pose], dtype=np.float32)
            p2_ref_np = np.array([[frame["camera2"][j] for j in joint_names] for frame in all_pose], dtype=np.float32)
        else:
            p1_ref_np, p2_ref_np = p1_fused_np, p2_fused_np
    else:
        p1_ref_np, p2_ref_np = p1_fused_np, p2_fused_np

    ref_bones1, ref_bones2 = [], []
    for (c_idx, p_idx) in bone_pairs:
        d1 = float(np.linalg.norm(p1_ref_np[:, c_idx] - p1_ref_np[:, p_idx], axis=-1).mean())
        d2 = float(np.linalg.norm(p2_ref_np[:, c_idx] - p2_ref_np[:, p_idx], axis=-1).mean())
        ref_bones1.append(d1)
        ref_bones2.append(d2)

    ref_bones1_t = torch.tensor(ref_bones1, device=device, dtype=torch.float32)
    ref_bones2_t = torch.tensor(ref_bones2, device=device, dtype=torch.float32)

    p1_init = torch.tensor(p1_fused_np, device=device, dtype=torch.float32)
    p2_init = torch.tensor(p2_fused_np, device=device, dtype=torch.float32)

    conf1_np = np.ones((T, N), dtype=np.float32)
    conf2_np = np.ones((T, N), dtype=np.float32)
    for t, meta in enumerate(all_meta):
        jc = meta.get("joint_confidence", {})
        c1 = jc.get("camera1", {}) if isinstance(jc, dict) else {}
        c2 = jc.get("camera2", {}) if isinstance(jc, dict) else {}
        v1 = meta.get("vis1", {})
        v2 = meta.get("vis2", {})
        for j_name, j_idx in name_to_idx.items():
            val1 = float(c1.get(j_name, 1.0)) if isinstance(c1, dict) else 1.0
            val2 = float(c2.get(j_name, 1.0)) if isinstance(c2, dict) else 1.0
            vis_w1 = 1.0 if (not v1 or v1.get(j_name, True)) else DEFAULT_OCCLUDED_FACTOR
            vis_w2 = 1.0 if (not v2 or v2.get(j_name, True)) else DEFAULT_OCCLUDED_FACTOR
            conf1_np[t, j_idx] = val1 * vis_w1
            conf2_np[t, j_idx] = val2 * vis_w2

    c1_t = torch.clamp(torch.tensor(conf1_np, device=device, dtype=torch.float32), min=0.05, max=1.0)
    c2_t = torch.clamp(torch.tensor(conf2_np, device=device, dtype=torch.float32), min=0.05, max=1.0)

    torso_anchors = [name_to_idx[n] for n in ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip") if n in name_to_idx]
    anchor_idx_t = torch.tensor(torso_anchors, device=device, dtype=torch.long)

    opt_cfg = (config or {}).get("fusion", {}).get("optimization", {})
    fusion_cfg = (config or {}).get("fusion", {})
    regularization_enabled = bool(opt_cfg.get("regularization", True))
    reg_lambda = float(opt_cfg.get("regularization_lambda", 1.0))
    temp_lambda = float(opt_cfg.get("temporal_lambda", 0.03))
    n_iters = min(int(opt_cfg.get("max_iter", 35)), 50)

    if selection_only:
        mask1 = torch.tensor(selection_np, device=device).unsqueeze(-1)
        mask2 = torch.zeros_like(p2_init)
    else:
        mask1 = torch.ones_like(p1_init)
        mask2 = torch.zeros_like(p2_init) if master_only else torch.ones_like(p2_init)

    delta1 = torch.zeros_like(p1_init, requires_grad=True)
    delta2 = torch.zeros_like(p2_init, requires_grad=(not master_only and not selection_only))
    params = [delta1] if (master_only or selection_only) else [delta1, delta2]
    optimizer = torch.optim.Adam(params, lr=0.01)

    for _ in range(n_iters):
        optimizer.zero_grad()
        p1 = p1_init + delta1 * mask1
        p2 = p2_init + delta2 * mask2

        # 1. Belief-weighted data likelihood anchor
        loss_prox = (c1_t.unsqueeze(-1) * ((delta1 * mask1) ** 2)).mean()
        if not master_only and not selection_only:
            loss_prox = loss_prox + (c2_t.unsqueeze(-1) * ((delta2 * mask2) ** 2)).mean()

        # 2. 2nd-order acceleration smoothness
        accel1 = p1[2:] - 2.0 * p1[1:-1] + p1[:-2]
        loss_accel = (accel1 ** 2).mean()
        if not master_only and not selection_only:
            accel2 = p2[2:] - 2.0 * p2[1:-1] + p2[:-2]
            loss_accel = loss_accel + (accel2 ** 2).mean()

        # 3. Kinematic bone length constraint
        loss_bone = torch.zeros((), device=device)
        if bone_pairs:
            for b_idx, (c_i, p_i) in enumerate(bone_pairs):
                d1 = torch.norm(p1[:, c_i] - p1[:, p_i], dim=-1)
                loss_bone = loss_bone + ((d1 - ref_bones1_t[b_idx]) ** 2).mean()
                if not master_only and not selection_only:
                    d2 = torch.norm(p2[:, c_i] - p2[:, p_i], dim=-1)
                    loss_bone = loss_bone + ((d2 - ref_bones2_t[b_idx]) ** 2).mean()

        # 4. Multi-view torso-anchor distance consistency
        loss_mv = torch.zeros((), device=device)
        if not master_only and not selection_only and anchor_idx_t.numel() > 0:
            cd1 = torch.cdist(p1, p1[:, anchor_idx_t])
            cd2 = torch.cdist(p2, p2[:, anchor_idx_t])
            diff_mv = torch.abs(cd1 - cd2)
            w_mv = torch.min(c1_t.unsqueeze(-1), c2_t.unsqueeze(-1))
            loss_mv = (w_mv * diff_mv).mean()

        total_loss = (reg_lambda * loss_prox if regularization_enabled else 0.0) + temp_lambda * loss_accel + 5.0 * loss_bone + 0.5 * loss_mv
        total_loss.backward()
        optimizer.step()

    p1_opt = (p1_init + delta1 * mask1).detach().cpu().numpy()
    p2_opt = (p2_init + delta2 * mask2).detach().cpu().numpy()
    if selection_only:
        # Cap optimizer drift beyond the already vetted slave proposal.
        original = p1_init.detach().cpu().numpy()
        drift = np.linalg.norm(p1_opt - original, axis=-1, keepdims=True)
        ratio = np.minimum(1.0, 0.02 / np.maximum(drift, 1e-8))
        p1_opt = original + (p1_opt - original) * ratio
        p1_opt = np.where(selection_np[..., None] > 0, p1_opt, original)

    for t, f in enumerate(files):
        d = all_fused[t]
        for j_idx, j_name in enumerate(joint_names):
            d["camera1"][j_name] = [round(float(x), 6) for x in p1_opt[t, j_idx]]
            if not master_only and not selection_only:
                d["camera2"][j_name] = [round(float(x), 6) for x in p2_opt[t, j_idx]]
        if all_pose is not None and not selection_only and bool(fusion_cfg.get("enforce_bone_lengths", True)):
            from fusion_pipeline.executor import _enforce_bone_lengths
            for cam in (("camera1",) if master_only else ("camera1", "camera2")):
                if cam in d and cam in all_pose[t]:
                    _enforce_bone_lengths(d[cam], all_pose[t][cam])
        f.write_text(json.dumps(d), encoding="utf-8")

    dt = time.time() - t0
    print(f"[Fusion] GPU Sequence Optimization completed in {dt:.3f}s on {device}")



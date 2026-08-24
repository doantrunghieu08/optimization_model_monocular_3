from pathlib import Path
import re
import copy
import joblib
import numpy as np
import sys
from json_io import read_json
from config_loader import resolve_inputs
from fusion_pipeline.detector import compute_visibility_from_mesh_vertices
from keypoints_map import load_keypoints3d_map
from json_io import write_json
from fusion_pipeline.detector import detect_cross_view_errors
from preprocess_pipeline.calib import resolve_selected_intrinsics
from compat import load_joblib_compat
from fusion_pipeline.optimization import calculate_stats
from config_loader import resolve_preprocess_output_dir
from fusion_pipeline.correction import estimate_bidirectional_similarity
from fusion_pipeline.detector import get_orientation_flag
from fusion_pipeline.config import OUTPUT_SUBDIRS
from fusion_pipeline.detector import load_torso_mask
from fusion_pipeline.detector import as_xyz
from fusion_pipeline.detector import make_raw_judgement_fallback
from fusion_pipeline.correction import apply_rotation_mismatch_corrections
from fusion_pipeline.correction import apply_confidence_corrections
from fusion_pipeline.optimization import optimize_f_points
from fusion_pipeline import context
import pdb

LEARNABLE_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "_learnable_backend"

#From ThanhNT: 23-08-2026
#def calculate_sequence_average_belief
def calculate_sequence_average_belief(sequence_results: list) -> tuple[float, float]:
    """
    Tính trung bình local belief của tất cả các khớp trong 1 frame, 
    sau đó trung bình cộng cho tất cả các frames.
    """
    c1_frame_beliefs, c2_frame_beliefs = [], []
    
    for frame_data in sequence_results:
        conf = frame_data.get("joint_confidence", {})
        c1, c2 = conf.get("camera1", {}), conf.get("camera2", {})
        
        # Tính trung bình các khớp (poses) trong nội bộ 1 frame
        if c1: c1_frame_beliefs.append(sum(c1.values()) / len(c1))
        if c2: c2_frame_beliefs.append(sum(c2.values()) / len(c2))
        
    # Lấy ra danh sách các giá trị đã tính trung bình (theo đúng thứ tự gốc)
    avg_h1 = [round(context.H1[k] / context.count_of_frames, 2) for k in context.H1]
    avg_h2 = [round(context.H2[k] / context.count_of_frames, 2) for k in context.H2]
    
    return avg_c1, avg_c2
#end of calculate_sequence_average_belief From ThanhNT: 23-08-2026


def _frame_index(path: Path) -> int:
    match = re.search(r"\d+", path.name)
    if not match:
        raise ValueError(f"Cannot extract frame index from {path.name}")
    return int(match.group())


def _clean_output(output_dir: Path, pattern: str = "*.json", create_split_dirs: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob(pattern):
        old_json.unlink(missing_ok=True)
    if create_split_dirs:
        for subdir in OUTPUT_SUBDIRS:
            target_dir = output_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            for old_json in target_dir.glob(pattern):
                old_json.unlink(missing_ok=True)


def _extract_person_payload(wham_data):
    if isinstance(wham_data, dict):
        if 0 in wham_data:
            return wham_data[0]
        if "0" in wham_data:
            return wham_data["0"]
        for value in wham_data.values():
            if isinstance(value, dict):
                return value
    if isinstance(wham_data, list):
        for value in wham_data:
            if isinstance(value, dict):
                return value
    return None


def _load_verts_if_available(paths: dict, occlusion_enabled: bool):
    if not occlusion_enabled:
        return False, None, None, 0, 0

    wham_path_1 = Path(paths["cam1_pkl"])
    wham_path_2 = Path(paths["cam2_pkl"])
    if not wham_path_1.exists():
        raise FileNotFoundError(f"WHAM PKL file not found: {wham_path_1}")
    if not wham_path_2.exists():
        raise FileNotFoundError(f"WHAM PKL file not found: {wham_path_2}")

    person_1 = _extract_person_payload(load_joblib_compat(wham_path_1))
    person_2 = _extract_person_payload(load_joblib_compat(wham_path_2))

    if person_1 is None or person_2 is None or "verts_cam" not in person_1 or "verts_cam" not in person_2:
        raise ValueError("verts_cam not found in WHAM PKL inputs; fusion.occlusion.enabled requires verts_cam")

    verts_cam1 = person_1["verts_cam"]
    verts_cam2 = person_2["verts_cam"]
    print(f"[Fusion] WHAM cam1 verts: {verts_cam1.shape[0]} frames")
    print(f"[Fusion] WHAM cam2 verts: {verts_cam2.shape[0]} frames")
    return True, verts_cam1, verts_cam2, verts_cam1.shape[0], verts_cam2.shape[0]


def _load_pose_frame(path: Path, metadata_dir: Path):
    data = read_json(path)
    if "camera1" in data and "camera2" in data:
        metadata_path = metadata_dir / path.name
        if metadata_path.exists():
            metadata_data = read_json(metadata_path)
            data.update(metadata_data)
        return data
    return data


def _load_2d_profile(config: dict, cam_id: str):
    preprocess_dir = Path(resolve_preprocess_output_dir(config))
    profile_path = preprocess_dir / f"data_{cam_id}.json"
    if not profile_path.exists():
        print(f"[Fusion] 2D confidence not found: {profile_path}")
        return None
    profile = read_json(profile_path)
    payload = profile.get(f"2D_camera_{cam_id}")
    if not isinstance(payload, dict):
        print(f"[Fusion] 2D confidence missing in {profile_path.name}")
        return None
    keypoints = payload.get("keypoints")
    if not isinstance(keypoints, dict):
        print(f"[Fusion] 2D confidence keypoints missing in {profile_path.name}")
        return None
    return keypoints


def _load_2d_profiles(config: dict) -> dict:
    return {
        "camera1": _load_2d_profile(config, "cam1"),
        "camera2": _load_2d_profile(config, "cam2"),
    }


def _load_occlusion_intrinsics(config: dict) -> dict:
    return {
        "camera1": np.asarray(resolve_selected_intrinsics(config, "cam1"), dtype=float),
        "camera2": np.asarray(resolve_selected_intrinsics(config, "cam2"), dtype=float),
    }


def _frame_confidence_from_profile(profile, source_idx, frame_idx: int):
    if not profile:
        return None
    candidates = []
    if source_idx is not None:
        candidates.append(int(source_idx))
    candidates.extend([frame_idx - 1, frame_idx])
    for candidate in candidates:
        data = profile.get(str(candidate))
        if isinstance(data, dict):
            return data
    return None


def _confidence2d_for_frame(data: dict, frame_idx: int, profiles: dict) -> dict:
    source_indices = data.get("metadata", {}).get("source_frame_indices", {})
    return {
        "camera1": _frame_confidence_from_profile(profiles.get("camera1"), source_indices.get("camera1"), frame_idx),
        "camera2": _frame_confidence_from_profile(profiles.get("camera2"), source_indices.get("camera2"), frame_idx),
    }


def run_phase3_pipeline(
    data_in,
    map_path,
    occlusion_tau,
    regularization,
    regularization_lambda,
    temporal_lambda,
    max_iter,
    ransac_threshold,
    ransac_max_combos,
    belief_alpha,
    belief_beta,
    verts_by_cam=None,
    intrinsics_by_cam=None,
    frame_idx=None,
    prev_optimized_data=None,
    confidence2d_by_cam=None,
):
    cam1 = {k: as_xyz(v) for k, v in data_in["camera1"].items()}
    cam2 = {k: as_xyz(v) for k, v in data_in["camera2"].items()}

    map_data = load_keypoints3d_map(map_path)
    expected_names = [kp["name"] for kp in map_data["keypoints"]]

    if set(cam1.keys()) != set(expected_names) or set(cam2.keys()) != set(expected_names):
        raise ValueError("Input does not have exactly the 21 expected keys for both cameras")

    names = expected_names
    cam1 = {k: cam1[k] for k in names}
    cam2 = {k: cam2[k] for k in names}

    if verts_by_cam is not None:
        if intrinsics_by_cam is None:
            raise ValueError("intrinsics_by_cam is required when verts_by_cam is provided")
        vis1 = compute_visibility_from_mesh_vertices(cam1, verts_by_cam["camera1"], intrinsics_by_cam["camera1"], occlusion_tau)
        vis2 = compute_visibility_from_mesh_vertices(cam2, verts_by_cam["camera2"], intrinsics_by_cam["camera2"], occlusion_tau)
    else:
        vis1 = {n: True for n in names}
        vis2 = {n: True for n in names}

    confidence2d_by_cam = confidence2d_by_cam or {}
    if belief_alpha is None or belief_beta is None:
        raise ValueError("fusion.belief.alpha and fusion.belief.beta must be provided in config")
    detected = detect_cross_view_errors(
        cam1,
        cam2,
        names,
        vis1,
        vis2,
        confidence2d1=confidence2d_by_cam.get("camera1"),
        confidence2d2=confidence2d_by_cam.get("camera2"),
        alpha=belief_alpha,
        beta=belief_beta,
    )
    m_set = detected["M"]
    k1_set = detected["K1"]
    k2_set = detected["K2"]
    l_list = detected["L"]
    all_weights = detected["weights"]
    H1_all = detected["H1"]
    H2_all = detected["H2"]

    t12, t21, a_list = estimate_bidirectional_similarity(
        cam1,
        cam2,
        l_list,
        threshold=ransac_threshold,
        max_combos=ransac_max_combos,
    )

    cam1_corr, cam2_corr = apply_confidence_corrections(cam1, cam2, k1_set, k2_set, t12, t21)
    cam1_corr, cam2_corr = apply_rotation_mismatch_corrections(
        cam1_corr,
        cam2_corr,
        cam1,
        cam2,
        m_set,
        k1_set,
        k2_set,
        H1_all,
        H2_all,
        t12,
        t21,
    )

    a_new = sorted(set(a_list) | k1_set | k2_set)
    f_list = [n for n in names if n not in set(a_new)]
    before_stats = calculate_stats(cam1_corr, cam2_corr, names, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)
    optimized_data, _ = optimize_f_points(
        {"camera1": cam1_corr, "camera2": cam2_corr},
        a_new,
        f_list,
        conf1=H1_all,
        conf2=H2_all,
        vis1=vis1,
        vis2=vis2,
        regularization=regularization,
        regularization_lambda=regularization_lambda,
        prev_data=prev_optimized_data,
        temporal_lambda=temporal_lambda,
        max_iter=max_iter,
    )
    after_stats = calculate_stats(optimized_data["camera1"], optimized_data["camera2"], names, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)

    flags1_after = get_orientation_flag(optimized_data["camera1"])
    flags2_after = get_orientation_flag(optimized_data["camera2"])
    m_after = {n for n in names if (flags1_after.get(n, 0) == 1 and flags2_after.get(n, 0) == -1) or (flags1_after.get(n, 0) == -1 and flags2_after.get(n, 0) == 1)}

    return {
        "M": sorted(m_set),
        "M_after": sorted(m_after),
        "M_resolved": len(m_after) == 0 and len(m_set) > 0,
        "K1": sorted(k1_set),
        "K2": sorted(k2_set),
        "A_new": a_new,
        "F": f_list,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "optimized": {"camera1": {k: list(v) for k, v in optimized_data["camera1"].items()}, "camera2": {k: list(v) for k, v in optimized_data["camera2"].items()}},
        "joint_confidence": {"camera1": H1_all, "camera2": H2_all},
        "vis1": {k: bool(v) for k, v in vis1.items()},
        "vis2": {k: bool(v) for k, v in vis2.items()},
    }
    #end of def run_phase3_pipeline


def run_fusion(config: dict) -> None:
    paths = config["paths"]
    inputs = resolve_inputs(config)
    runtime_cfg = config.get("runtime", {})
    fusion_cfg = config.get("fusion", {})

    if not fusion_cfg["enabled"]:
        print("[Fusion] Disabled by config: fusion.enabled=false")
        return

    input_dir = Path(paths["pose_output_dir"])
    output_dir = Path(paths["fused_output_dir"])

    if not input_dir.exists():
        raise FileNotFoundError(f"Pose JSON directory not found: {input_dir}")

    if runtime_cfg.get("clean_output", True):
        _clean_output(output_dir, "fused_data_*.json", create_split_dirs=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    occlusion_cfg = fusion_cfg["occlusion"]
    belief_cfg = fusion_cfg["belief"]
    occlusion_enabled = occlusion_cfg["enabled"]
    if occlusion_enabled:
        load_torso_mask(paths["segmentation"])

    wham_loaded, verts_cam1, verts_cam2, n_frames_1, n_frames_2 = _load_verts_if_available(inputs, occlusion_enabled)
    confidence2d_profiles = _load_2d_profiles(config)
    occlusion_intrinsics = _load_occlusion_intrinsics(config) if occlusion_enabled and wham_loaded else None

    keypoints_dir = input_dir / "keypoints3d"
    metadata_dir = input_dir / "metadata"
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Pose keypoints directory not found: {keypoints_dir}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Pose metadata directory not found: {metadata_dir}")
    file_paths = sorted(keypoints_dir.glob("pose_data_*.json"), key=_frame_index)
    print(f"[Fusion] Found {len(file_paths)} pose JSON files")

    ransac_cfg = fusion_cfg["ransac"]
    opt_cfg = fusion_cfg["optimization"]

    prev_result = None
    for path in file_paths:
        frame_idx = _frame_index(path)
        out_name = f"fused_data_{frame_idx}.json"
        data = _load_pose_frame(path, metadata_dir=metadata_dir)

        if wham_loaded:
            wham_frame = frame_idx - 1
            if 0 <= wham_frame < n_frames_1 and wham_frame < n_frames_2:
                verts_input = {"camera1": verts_cam1[wham_frame], "camera2": verts_cam2[wham_frame]}
            else:
                print(f"[Fusion] Frame {frame_idx}: WHAM frame out of range. Occlusion skipped.")
                verts_input = None
        else:
            verts_input = None

        try:
            prev_opt = prev_result["optimized"] if prev_result and "optimized" in prev_result else None
            confidence2d_by_cam = _confidence2d_for_frame(data, frame_idx, confidence2d_profiles)
            result = run_phase3_pipeline(
                data,
                map_path=paths["keypoints3d_map"],
                verts_by_cam=verts_input,
                intrinsics_by_cam=occlusion_intrinsics,
                occlusion_tau=occlusion_cfg["tau"],
                regularization=opt_cfg["regularization"],
                regularization_lambda=opt_cfg["regularization_lambda"],
                temporal_lambda=opt_cfg["temporal_lambda"],
                max_iter=opt_cfg["max_iter"],
                ransac_threshold=ransac_cfg["threshold"],
                ransac_max_combos=ransac_cfg["max_combos"],
                frame_idx=frame_idx,
                prev_optimized_data=prev_opt,
                confidence2d_by_cam=confidence2d_by_cam,
                belief_alpha=belief_cfg["alpha"],
                belief_beta=belief_cfg["beta"],
            )
            # 1. Lấy dữ liệu an toàn (chú ý đổi [] thành {})
            joint_conf = result.get("joint_confidence", {})
            cam1 = np.array(joint_conf.get("camera1", []))
            cam2 = np.array(joint_conf.get("camera2", []))
            
            # 2. Cộng dồn từng phần tử
            if prev_result is not None:
                # Duyệt qua từng key ('head', 'neck'...) và cộng giá trị tương ứng
                context.H1 = {k: context.H1.get(k, 0) + cam1.get(k, 0) for k in cam1}
                context.H2 = {k: context.H2.get(k, 0) + cam2.get(k, 0) for k in cam2}
                context.count_of_frames += 1
            else:
                # Dùng .copy() để tránh lỗi tham chiếu bộ nhớ trong Python
                context.H1 = cam1.copy()
                context.H2 = cam2.copy()
                context.count_of_frames = 1
            
            occluded_cam1 = sorted(name for name, visible in result.get("vis1", {}).items() if not visible)
            occluded_cam2 = sorted(name for name, visible in result.get("vis2", {}).items() if not visible)
            occlusion_parts = []
            if occluded_cam1:
                occlusion_parts.append(f"cam1: {', '.join(occluded_cam1)}")
            if occluded_cam2:
                occlusion_parts.append(f"cam2: {', '.join(occluded_cam2)}")
            if occlusion_parts:
                print(f"[Fusion] Frame {frame_idx}: Occlusion: {' | '.join(occlusion_parts)}")
            prev_result = result
        except Exception as e:
            print(f"[Fusion] Frame {frame_idx}: FAILED ({e}) -> fallback")
            result = copy.deepcopy(prev_result) if prev_result is not None else make_raw_judgement_fallback(data, frame_idx, e)

        fused_keypoints = {
            "camera1": result.get("optimized", {}).get("camera1", {}),
            "camera2": result.get("optimized", {}).get("camera2", {}),
        }
        fused_metadata = {k: v for k, v in result.items() if k not in ("camera1", "camera2", "optimized")}
        write_json(output_dir / "keypoints3d" / out_name, fused_keypoints)
        write_json(output_dir / "metadata" / out_name, fused_metadata)

    print(f"[Fusion] Done. Output: {output_dir}")

"""## 9. Chuan bi Learnable backend

Pha learnable can backend noi bo cua Learnable-SMPLify. Cell nay chi tao phan backend can dung.

"""

LEARNABLE_VENDOR_FILES = {
    'common/geometry.py': "import torch\n\n\ndef rotation_matrix_to_angle_axis(rotation_matrix):\n    \"\"\"\n    This function is borrowed from https://github.com/kornia/kornia\n    Convert 3x4 rotation matrix to Rodrigues vector\n    Args:\n        rotation_matrix (Tensor): rotation matrix.\n    Returns:\n        Tensor: Rodrigues vector transformation.\n    Shape:\n        - Input: :math:`(N, 3, 4)`\n        - Output: :math:`(N, 3)`\n    Example:\n        >>> input = torch.rand(2, 3, 4)  # Nx3x4\n        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3\n    \"\"\"\n    if rotation_matrix.shape[1:] == (3, 3):\n        rot_mat = rotation_matrix.reshape(-1, 3, 3)\n        hom = torch.tensor([0, 0, 1],\n                           dtype=torch.float32,\n                           device=rotation_matrix.device)\n        hom = hom.reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)\n        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)\n\n    quaternion = rotation_matrix_to_quaternion(rotation_matrix)\n    aa = quaternion_to_angle_axis(quaternion)\n    aa[torch.isnan(aa)] = 0.0\n    return aa\n\n\ndef quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:\n    \"\"\"\n    This function is borrowed from https://github.com/kornia/kornia\n    Convert quaternion vector to angle axis of rotation.\n    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h\n    Args:\n        quaternion (torch.Tensor): tensor with quaternions.\n    Return:\n        torch.Tensor: tensor with angle axis of rotation.\n    Shape:\n        - Input: :math:`(*, 4)` where `*` means, any number of dimensions\n        - Output: :math:`(*, 3)`\n    Example:\n        >>> quaternion = torch.rand(2, 4)  # Nx4\n        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3\n    \"\"\"\n    if not torch.is_tensor(quaternion):\n        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(\n            type(quaternion)))\n\n    if not quaternion.shape[-1] == 4:\n        raise ValueError(\n            'Input must be a tensor of shape Nx4 or 4. Got {}'.format(\n                quaternion.shape))\n    # unpack input and compute conversion\n    q1: torch.Tensor = quaternion[..., 1]\n    q2: torch.Tensor = quaternion[..., 2]\n    q3: torch.Tensor = quaternion[..., 3]\n    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3\n\n    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)\n    cos_theta: torch.Tensor = quaternion[..., 0]\n    two_theta: torch.Tensor = 2.0 * torch.where(\n        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta),\n        torch.atan2(sin_theta, cos_theta))\n\n    k_pos: torch.Tensor = two_theta / sin_theta\n    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)\n    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)\n\n    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]\n    angle_axis[..., 0] += q1 * k\n    angle_axis[..., 1] += q2 * k\n    angle_axis[..., 2] += q3 * k\n    return angle_axis\n\n\ndef rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):\n    \"\"\"\n    This function is borrowed from https://github.com/kornia/kornia\n    Convert 3x4 rotation matrix to 4d quaternion vector\n    This algorithm is based on algorithm described in\n    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201\n    Args:\n        rotation_matrix (Tensor): the rotation matrix to convert.\n    Return:\n        Tensor: the rotation in quaternion\n    Shape:\n        - Input: :math:`(N, 3, 4)`\n        - Output: :math:`(N, 4)`\n    Example:\n        >>> input = torch.rand(4, 3, 4)  # Nx3x4\n        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4\n    \"\"\"\n    if not torch.is_tensor(rotation_matrix):\n        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(\n            type(rotation_matrix)))\n\n    if len(rotation_matrix.shape) > 3:\n        raise ValueError(\n            'Input size must be a three dimensional tensor. Got {}'.format(\n                rotation_matrix.shape))\n    if not rotation_matrix.shape[-2:] == (3, 4):\n        raise ValueError(\n            'Input size must be a N x 3 x 4  tensor. Got {}'.format(\n                rotation_matrix.shape))\n\n    rmat_t = torch.transpose(rotation_matrix, 1, 2)\n\n    mask_d2 = rmat_t[:, 2, 2] < eps\n\n    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]\n    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]\n\n    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]\n    q0 = torch.stack([\n        rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0,\n        rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]\n    ], -1)\n    t0_rep = t0.repeat(4, 1).t()\n\n    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]\n    q1 = torch.stack([\n        rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0],\n        t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]\n    ], -1)\n    t1_rep = t1.repeat(4, 1).t()\n\n    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]\n    q2 = torch.stack([\n        rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2],\n        rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2\n    ], -1)\n    t2_rep = t2.repeat(4, 1).t()\n\n    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]\n    q3 = torch.stack([\n        t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],\n        rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]\n    ], -1)\n    t3_rep = t3.repeat(4, 1).t()\n\n    mask_c0 = mask_d2 * mask_d0_d1\n    mask_c1 = mask_d2 * ~mask_d0_d1\n    mask_c2 = ~mask_d2 * mask_d0_nd1\n    mask_c3 = ~mask_d2 * ~mask_d0_nd1\n    mask_c0 = mask_c0.view(-1, 1).type_as(q0)\n    mask_c1 = mask_c1.view(-1, 1).type_as(q1)\n    mask_c2 = mask_c2.view(-1, 1).type_as(q2)\n    mask_c3 = mask_c3.view(-1, 1).type_as(q3)\n\n    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3\n    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa\n                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa\n    q *= 0.5\n    return q",
    'common/human_models.py': "import pickle\nimport smplx\nimport torch\n\nimport numpy as np\nimport os.path as osp\nfrom contextlib import redirect_stdout, redirect_stderr\nimport io\n\n\nclass SMPLX(object):\n    def __init__(self, human_model_path):\n        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}\n        sink = io.StringIO()\n        with redirect_stdout(sink), redirect_stderr(sink):\n            self.layer = {'neutral': smplx.create(human_model_path, 'smplx', gender='NEUTRAL', use_face_contour=True, **self.layer_arg),\n                            'male': smplx.create(human_model_path, 'smplx', gender='MALE', use_face_contour=True, **self.layer_arg),\n                            'female': smplx.create(human_model_path, 'smplx', gender='FEMALE', use_face_contour=True, **self.layer_arg)\n                            }\n        self.vertex_num = 10475\n        self.face = self.layer['neutral'].faces\n        self.shape_param_dim = 10\n        self.expr_code_dim = 10\n        with open(osp.join(human_model_path, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:\n            self.j14_regressor = pickle.load(f, encoding='latin1')\n        with open(osp.join(human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:\n            self.hand_vertex_idx = pickle.load(f, encoding='latin1')\n        self.face_vertex_idx = np.load(osp.join(human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))\n        self.J_regressor = self.layer['neutral'].J_regressor.numpy()\n        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}\n        self.orig_hand_regressor = self.make_hand_regressor()\n        #self.orig_hand_regressor = {'left': self.layer.J_regressor.numpy()[[20,37,38,39,25,26,27,28,29,30,34,35,36,31,32,33],:], 'right': self.layer.J_regressor.numpy()[[21,52,53,54,40,41,42,43,44,45,49,50,51,46,47,48],:]}\n\n        # original SMPLX joint set\n        self.orig_joint_num = 53 # 22 (body joints) + 30 (hand joints) + 1 (face jaw joint)\n        self.orig_joints_name = \\\n        ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints\n        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints\n        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', # right hand joints\n        'Jaw' # face jaw joint\n        )\n        self.orig_flip_pairs = \\\n        ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints\n        (22,37), (23,38), (24,39), (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51) # hand joints\n        )\n        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')\n        self.orig_joint_part = \\\n        {'body': range(self.orig_joints_name.index('Pelvis'), self.orig_joints_name.index('R_Wrist')+1),\n        'lhand': range(self.orig_joints_name.index('L_Index_1'), self.orig_joints_name.index('L_Thumb_3')+1),\n        'rhand': range(self.orig_joints_name.index('R_Index_1'), self.orig_joints_name.index('R_Thumb_3')+1),\n        'face': range(self.orig_joints_name.index('Jaw'), self.orig_joints_name.index('Jaw')+1)}\n\n        # changed SMPLX joint set for the supervision\n        self.joint_num = 137 # 25 (body joints) + 40 (hand joints) + 72 (face keypoints)\n        self.joints_name = \\\n        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body joints\n         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints\n         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints\n         *['Face_' + str(i) for i in range(1,73)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)\n         )\n        self.root_joint_idx = self.joints_name.index('Pelvis')\n        self.lwrist_idx = self.joints_name.index('L_Wrist')\n        self.rwrist_idx = self.joints_name.index('R_Wrist')\n        self.neck_idx = self.joints_name.index('Neck')\n        self.flip_pairs = \\\n        ( (1,2), (3,4), (5,6), (8,9), (10,11), (12,13), (14,17), (15,18), (16,19), (20,21), (22,23), # body joints\n        (25,45), (26,46), (27,47), (28,48), (29,49), (30,50), (31,51), (32,52), (33,53), (34,54), (35,55), (36,56), (37,57), (38,58), (39,59), (40,60), (41,61), (42,62), (43,63), (44,64), # hand joints\n        (67,68), # face eyeballs\n        (69,78), (70,77), (71,76), (72,75), (73,74), # face eyebrow\n        (83,87), (84,86), # face below nose\n        (88,97), (89,96), (90,95), (91,94), (92,99), (93,98), # face eyes\n        (100,106), (101,105), (102,104), (107,111), (108,110), # face mouth\n        (112,116), (113,115), (117,119), # face lip\n        (120,136), (121,135), (122,134), (123,133), (124,132), (125,131), (126,130), (127,129) # face contours\n        )\n        self.joint_idx = \\\n        (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints\n        37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand joints\n        52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints\n        22,15, # jaw, head\n        57,56, # eyeballs\n        76,77,78,79,80,81,82,83,84,85, # eyebrow\n        86,87,88,89, # nose\n        90,91,92,93,94, # below nose\n        95,96,97,98,99,100,101,102,103,104,105,106, # eyes\n        107, # right mouth\n        108,109,110,111,112, # upper mouth\n        113, # left mouth\n        114,115,116,117,118, # lower mouth\n        119, # right lip\n        120,121,122, # upper lip\n        123, # left lip\n        124,125,126, # lower lip\n        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour\n        )\n        self.joint_part = \\\n        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('Nose')+1),\n        'lhand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('L_Pinky_4')+1),\n        'rhand': range(self.joints_name.index('R_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),\n        'hand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),\n        'face': range(self.joints_name.index('Face_1'), self.joints_name.index('Face_72')+1)}\n        \n        # changed SMPLX joint set for PositionNet prediction\n        self.pos_joint_num = 65 # 25 (body joints) + 40 (hand joints)\n        self.pos_joints_name = \\\n        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body joints\n         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints\n         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints\n         )\n        self.pos_joint_part = \\\n        {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),\n        'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),\n        'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),\n        'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}\n        self.pos_joint_part['L_MCP'] = [self.pos_joints_name.index('L_Index_1') - len(self.pos_joint_part['body']),\n                                        self.pos_joints_name.index('L_Middle_1') - len(self.pos_joint_part['body']),\n                                        self.pos_joints_name.index('L_Ring_1') - len(self.pos_joint_part['body']),\n                                        self.pos_joints_name.index('L_Pinky_1') - len(self.pos_joint_part['body'])]\n        self.pos_joint_part['R_MCP'] = [self.pos_joints_name.index('R_Index_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),\n                                        self.pos_joints_name.index('R_Middle_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),\n                                        self.pos_joints_name.index('R_Ring_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),\n                                        self.pos_joints_name.index('R_Pinky_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand'])]\n    \n    def make_hand_regressor(self):\n        regressor = self.layer['neutral'].J_regressor.numpy()\n        lhand_regressor = np.concatenate((regressor[[20,37,38,39],:],\n                                            np.eye(self.vertex_num)[5361,None],\n                                                regressor[[25,26,27],:],\n                                                np.eye(self.vertex_num)[4933,None],\n                                                regressor[[28,29,30],:],\n                                                np.eye(self.vertex_num)[5058,None],\n                                                regressor[[34,35,36],:],\n                                                np.eye(self.vertex_num)[5169,None],\n                                                regressor[[31,32,33],:],\n                                                np.eye(self.vertex_num)[5286,None]))\n        rhand_regressor = np.concatenate((regressor[[21,52,53,54],:],\n                                            np.eye(self.vertex_num)[8079,None],\n                                                regressor[[40,41,42],:],\n                                                np.eye(self.vertex_num)[7669,None],\n                                                regressor[[43,44,45],:],\n                                                np.eye(self.vertex_num)[7794,None],\n                                                regressor[[49,50,51],:],\n                                                np.eye(self.vertex_num)[7905,None],\n                                                regressor[[46,47,48],:],\n                                                np.eye(self.vertex_num)[8022,None]))\n        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}\n        return hand_regressor\n\n        \n    def reduce_joint_set(self, joint):\n        new_joint = []\n        for name in self.pos_joints_name:\n            idx = self.joints_name.index(name)\n            new_joint.append(joint[:,idx,:])\n        new_joint = torch.stack(new_joint,1)\n        return new_joint\n\nclass SMPL(object):\n    def __init__(self, human_model_path):\n        self.layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False}\n        sink = io.StringIO()\n        with redirect_stdout(sink), redirect_stderr(sink):\n            self.layer = {'neutral': smplx.create(human_model_path, 'smpl', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(human_model_path, 'smpl', gender='MALE', **self.layer_arg), 'female': smplx.create(human_model_path, 'smpl', gender='FEMALE', **self.layer_arg)}\n        self.vertex_num = 6890\n        self.face = self.layer['neutral'].faces\n        self.shape_param_dim = 10\n        self.vposer_code_dim = 32\n\n        # original SMPL joint set\n        self.orig_joint_num = 24\n        self.orig_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')\n        self.orig_flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) )\n        self.orig_parent_ids = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]\n        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')\n        self.orig_joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)\n\n        self.J_regressor_idx = {'pelvis': 0}\n        self.joint_num = self.orig_joint_num\n        self.joints_name = self.orig_joints_name\n        self.flip_pairs = self.orig_flip_pairs\n        self.parent_ids = self.orig_parent_ids\n        self.root_joint_idx = self.orig_root_joint_idx\n        self.joint_regressor = self.orig_joint_regressor\n\n        # for twist decomposition\n        self.kintree_table = np.array([[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],\n                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])\n\nif __name__ == '__main__':\n    smpl_x = SMPLX('data/SMPL-family')\n    smpl = SMPL('data/SMPL-family')\n",
    'common/keypoint_geo.py': "import torch\n\n\ndef normalize(v, eps=1e-8):\n    return v / (v.norm(dim=-1, keepdim=True) + eps)\n\n\ndef build_local_frame(left_hip, right_hip, thorax, pelvis):\n    y_axis = normalize(left_hip - right_hip)  # (B, 1, 3)\n\n    torso_vec = normalize(thorax - pelvis)\n\n    # standardization\n    proj = (torso_vec * y_axis).sum(dim=-1, keepdim=True) * y_axis\n    z_axis = normalize(torso_vec - proj)\n\n    x_axis = normalize(torch.cross(z_axis, y_axis, dim=-1))  # (B, 1, 3)\n\n    R = torch.cat([x_axis, y_axis, z_axis], dim=1)  # (B, 3, 3)\n    R = R.transpose(1, 2)\n    return R\n\n\ndef normalize_kp(kp, invalid_mask, kp_index, R=None, T=None):\n    \"\"\"\n        Normalize openpose 25 keypoints using human-centric coordinates.\n    \"\"\"\n\n    kp = kp.clone()\n    if R is not None and T is not None:\n        kp = torch.matmul(kp - T, R)\n    else:\n        T = kp[:, [kp_index['pelvis']], :]\n        kp = kp - T\n        if R is None:\n            R = build_local_frame(\n                kp[:, [kp_index['left_hip']], :],\n                kp[:, [kp_index['right_hip']], :],\n                kp[:, [kp_index['thorax']], :],\n                kp[:, [kp_index['pelvis']], :]\n            )\n        kp = torch.matmul(kp, R)\n    if invalid_mask is not None:\n        kp[:, invalid_mask, :] = 0.0\n    return kp, R, T",
    'common/transforms.py': "import torch\nfrom torch.nn import functional as F\nfrom einops.einops import rearrange\n\nfrom common.geometry import rotation_matrix_to_angle_axis\n\ndef rot6d_to_axis_angle(x):\n    batch_size = x.shape[0]\n\n    x = x.view(-1, 3, 2)\n    a1 = x[:, :, 0]\n    a2 = x[:, :, 1]\n    b1 = F.normalize(a1)\n    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)\n    b3 = torch.cross(b1, b2, dim=-1)\n    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix\n\n    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix\n    axis_angle = rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle\n    axis_angle[torch.isnan(axis_angle)] = 0.0\n    return axis_angle\n\ndef rot6d_to_rotmat(x):\n    \"\"\"Convert 6D rotation representation to 3x3 rotation matrix.\n    Based on Zhou et al., \"On the Continuity of Rotation Representations in Neural Networks\", CVPR 2019\n    Input:\n        (B,6) Batch of 6-D rotation representations\n    Output:\n        (B,3,3) Batch of corresponding rotation matrices\n    \"\"\"\n    if x.shape[-1] == 6:\n        batch_size = x.shape[0]\n        if len(x.shape) == 3:\n            num = x.shape[1]\n            x = rearrange(x, 'b n d -> (b n) d', d=6)\n        else:\n            num = 1\n        x = rearrange(x, 'b (k l) -> b k l', k=3, l=2)\n        # x = x.view(-1,3,2)\n        a1 = x[:, :, 0]\n        a2 = x[:, :, 1]\n        b1 = F.normalize(a1)\n        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)\n        b3 = torch.cross(b1, b2, dim=-1)\n\n        mat = torch.stack((b1, b2, b3), dim=-1)\n        if num > 1:\n            mat = rearrange(mat, '(b n) h w-> b n h w', b=batch_size, n=num, h=3, w=3)\n    else:\n        x = x.view(-1,3,2)\n        a1 = x[:, :, 0]\n        a2 = x[:, :, 1]\n        b1 = F.normalize(a1)\n        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)\n        b3 = torch.cross(b1, b2, dim=-1)\n        mat = torch.stack((b1, b2, b3), dim=-1)\n    return mat",
    'config/net.yaml': "dataset:\n  train_dataset_list: ['AMASS']\n  test_dataset: 'AMASS' # NOTE(yyc): you can change this to other structured datasets\n  data_dir: 'data'\n  stride: [1, 3, 5, 7, 9] # NOTE(yyc): for default testing during training, s will be fixed to max(stride)\n  max_stride: 10\n  downsample_rate: 10\n\n\nmodel_params:\n  model_name: 'NetBody25'\n  backbone:\n    name: 'stgcn'\n    graph_args:\n        labeling_mode: 'spatial'\n    params:\n      kp_dim: 3\n      window_size: 2\n      num_point: 25\n  head:\n    pred_pose_num: 22\n    feat_dim: 335\n    input_dim: 256\n    hidden_dim: 256\n    output_dim: 6\n  human_model:\n    smpl_dir: 'data/SMPL-family'\n\n  loss_config:\n    kp3d: 5.0\n    pose: 1.0\n    verts: 1.0\n\ntrain_params:\n  batch_size: 128\n  num_epochs: 100\n  optimizer:\n    name: 'AdamW'\n    lr: 0.0001\n    weight_decay: 0.1\n  scheduler:\n    name: 'CosineAnnealLR'\n  ckpt_save_freq: 20",
    'module/backbone/basic_modules.py': "import torch\nimport torch.nn as nn\n\nimport math\n\ndef conv_init(module):\n    # he_normal\n    n = module.out_channels\n    for k in module.kernel_size:\n        n *= k\n    module.weight.data.normal_(0, math.sqrt(2. / n))\n\n\ndef import_class(name):\n    components = name.split('.')\n    mod = __import__(components[0])\n    for comp in components[1:]:\n        mod = getattr(mod, comp)\n    return mod\n\n\nclass Unit2D(nn.Module):\n    def __init__(self,\n                 D_in,\n                 D_out,\n                 kernel_size,\n                 stride=1,\n                 dim=2,\n                 dropout=0,\n                 bias=True):\n        super(Unit2D, self).__init__()\n        pad = int((kernel_size - 1) / 2)\n        if dim == 2:\n            self.conv = nn.Conv2d(\n                D_in,\n                D_out,\n                kernel_size=(kernel_size, 1),\n                padding=(pad, 0),\n                stride=(stride, 1),\n                bias=bias)\n        elif dim == 3:\n            self.conv = nn.Conv2d(\n                D_in,\n                D_out,\n                kernel_size=(1, kernel_size),\n                padding=(0, pad),\n                stride=(1, stride),\n                bias=bias)\n        else:\n            raise ValueError()\n\n        self.bn = nn.SyncBatchNorm(D_out)\n        self.relu = nn.ReLU()\n        self.dropout = nn.Dropout(dropout)\n\n        # initialize\n        conv_init(self.conv)\n\n    def forward(self, x):\n        x = self.dropout(x)\n        x = self.relu(self.bn(self.conv(x)))\n        return x\n\n\nclass Unit_GCN(nn.Module):\n    def __init__(self,\n                 in_channels,\n                 out_channels,\n                 A,\n                 use_local_bn=False,\n                 kernel_size=1,\n                 stride=1,\n                 mask_learning=False):\n        super(Unit_GCN, self).__init__()\n\n        # ==========================================\n        # number of nodes\n        self.V = A.size()[-1]\n\n        # the adjacency matrixes of the graph\n        self.A = nn.Parameter(\n            A.clone(), requires_grad=False).view(-1, self.V, self.V)\n\n        # number of input channels\n        self.in_channels = in_channels\n\n        # number of output channels\n        self.out_channels = out_channels\n\n        # if true, use mask matrix to reweight the adjacency matrix\n        self.mask_learning = mask_learning\n\n        # number of adjacency matrix (number of partitions)\n        self.num_A = self.A.size()[0]\n\n        # if true, each node have specific parameters of batch normalizaion layer.\n        # if false, all nodes share parameters.\n        self.use_local_bn = use_local_bn\n        # ==========================================\n\n        self.conv_list = nn.ModuleList([\n            nn.Conv2d(\n                self.in_channels,\n                self.out_channels,\n                kernel_size=(kernel_size, 1),\n                padding=(int((kernel_size - 1) / 2), 0),\n                stride=(stride, 1)) for i in range(self.num_A)\n        ])\n\n        if mask_learning:\n            self.mask = nn.Parameter(torch.ones(self.A.size()))\n        if use_local_bn:\n            self.bn = nn.SyncBatchNorm(self.out_channels * self.V)\n        else:\n            self.bn = nn.SyncBatchNorm(self.out_channels)\n\n        self.relu = nn.ReLU()\n\n        # initialize\n        for conv in self.conv_list:\n            conv_init(conv)\n\n    def forward(self, x):\n        N, C, T, V = x.size()\n        self.A = self.A.cuda(x.get_device())\n        A = self.A\n\n        # reweight adjacency matrix\n        if self.mask_learning:\n            A = A * self.mask\n\n        # graph convolution\n        for i, a in enumerate(A):\n            xa = x.view(-1, V).mm(a).view(N, C, T, V)\n\n            if i == 0:\n                y = self.conv_list[i](xa)\n            else:\n                y = y + self.conv_list[i](xa)\n\n        # batch normalization\n        if self.use_local_bn:\n            y = y.permute(0, 1, 3, 2).contiguous().view(\n                N, self.out_channels * V, T)\n            y = self.bn(y)\n            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)\n        else:\n            y = self.bn(y)\n\n        # nonliner\n        y = self.relu(y)\n\n        return y\n\n\nclass TCN_GCN_unit(nn.Module):\n    def __init__(self,\n                 in_channel,\n                 out_channel,\n                 A,\n                 kernel_size=9,\n                 stride=1,\n                 dropout=0.5,\n                 use_local_bn=False,\n                 mask_learning=False):\n        super(TCN_GCN_unit, self).__init__()\n        half_out_channel = out_channel / 2\n        self.A = A\n        self.V = A.size()[-1]\n        self.C = in_channel\n\n        self.gcn1 = Unit_GCN(\n            in_channel,\n            out_channel,\n            A,\n            use_local_bn=use_local_bn,\n            mask_learning=mask_learning)\n        self.tcn1 = Unit2D(\n            out_channel,\n            out_channel,\n            kernel_size=kernel_size,\n            dropout=dropout,\n            stride=stride)\n        if (in_channel != out_channel) or (stride != 1):\n            self.down1 = Unit2D(\n                in_channel, out_channel, kernel_size=1, stride=stride)\n        else:\n            self.down1 = None\n\n    def forward(self, x):\n        # N, C, T, V = x.size()\n        x = self.tcn1(self.gcn1(x)) + (x if\n                                       (self.down1 is None) else self.down1(x))\n        return x\n\n\nclass TCN_GCN_unit_multiscale(nn.Module):\n    def __init__(self,\n                 in_channels,\n                 out_channels,\n                 A,\n                 kernel_size=9,\n                 stride=1,\n                 **kwargs):\n        super(TCN_GCN_unit_multiscale, self).__init__()\n        self.unit_1 = TCN_GCN_unit(\n            in_channels,\n            out_channels / 2,\n            A,\n            kernel_size=kernel_size,\n            stride=stride,\n            **kwargs)\n        self.unit_2 = TCN_GCN_unit(\n            in_channels,\n            out_channels - out_channels / 2,\n            A,\n            kernel_size=kernel_size * 2 - 1,\n            stride=stride,\n            **kwargs)\n\n    def forward(self, x):\n        return torch.cat((self.unit_1(x), self.unit_2(x)), dim=1)",
    'module/backbone/gcn.py': "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nfrom .basic_modules import Unit2D, Unit_GCN, TCN_GCN_unit, TCN_GCN_unit_multiscale\n\n\ndefault_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),\n                    (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]\n\n\nclass STGCN(nn.Module):\n    \"\"\" Spatial temporal graph convolutional networks\n                        for skeleton-based action recognition.\n\n    Input shape:\n        Input shape should be (N, C, T, V)\n        where N is the number of samples,\n              C is the number of input channels,\n              T is the length of the sequence,\n              V is the number of joints or graph nodes\n    \n    Arguments:\n        About shape:\n            kp_dim (int): Number of channels in the input keypoints\n            window_size (int): Length of input sequence\n            num_point (int): Number of joints or graph nodes\n        About net:\n            backbone_config: The structure of backbone networks\n        About graph convolution:\n            graph: The graph of skeleton, represented by a adjacency matrix\n            mask_learning: If true, use mask matrices to reweight the adjacency matrices\n            use_local_bn: If true, each node in the graph have specific parameters of batch normalzation layer\n        About temporal convolution:\n            multiscale: If true, use multi-scale temporal convolution\n            temporal_kernel_size: The kernel size of temporal convolution\n            dropout: The drop out rate of the dropout layer in front of each temporal convolution layer\n\n    \"\"\"\n\n    def __init__(self,\n                 kp_dim,\n                 window_size,\n                 num_point,\n                 graph,\n                 backbone_config=None,\n                 mask_learning=True,\n                 use_local_bn=False,\n                 multiscale=False,\n                 temporal_kernel_size=9,\n                 dropout=0.5):\n        super(STGCN, self).__init__()\n\n        self.kp_dim = kp_dim\n        self.num_point = num_point\n        self.graph = graph\n        self.A = torch.from_numpy(self.graph.A).float()\n\n        self.multiscale = multiscale\n\n        self.data_bn = nn.SyncBatchNorm(kp_dim * num_point)\n\n        kwargs = dict(\n            A=self.A,\n            mask_learning=mask_learning,\n            use_local_bn=use_local_bn,\n            dropout=dropout,\n            kernel_size=temporal_kernel_size)\n\n        if self.multiscale:\n            unit = TCN_GCN_unit_multiscale\n        else:\n            unit = TCN_GCN_unit\n\n        # backbone\n        if backbone_config is None:\n            backbone_config = default_backbone\n\n        backbone_in_c = backbone_config[0][0]\n        backbone_out_c = backbone_config[-1][1]\n        backbone_out_t = window_size\n        backbone = []\n        for in_c, out_c, stride in backbone_config:\n            backbone.append(unit(in_c, out_c, stride=stride, **kwargs))\n            if backbone_out_t % stride == 0:\n                backbone_out_t = backbone_out_t // stride\n            else:\n                backbone_out_t = backbone_out_t // stride + 1\n        self.backbone = nn.ModuleList(backbone)\n\n        # head\n        self.gcn0 = Unit_GCN(\n            kp_dim,\n            backbone_in_c,\n            self.A,\n            mask_learning=mask_learning,\n            use_local_bn=use_local_bn)\n\n        self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9)\n\n    def forward(self, x):\n        N, C, T, V = x.size()\n\n        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)\n\n        x = self.data_bn(x)\n\n        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()\n\n        # model\n        x = self.gcn0(x)\n        x = self.tcn0(x)\n        for m in self.backbone:\n            x = m(x)\n\n        # V pooling, x in [N, C, T // downsample, V]\n        x = F.avg_pool2d(x, kernel_size=(1, V)).squeeze(-1)\n\n        return x",
    'module/backbone/graph/openpose_graph.py': "import numpy as np\nfrom . import tools\n\n# reference : https://arxiv.org/abs/1604.02808\n# edge format: (origin, neighbor)\n\nnum_node = 25\nself_link = [(i, i) for i in range(num_node)]\ninward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),\n                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),\n                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),\n                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]\ninward = [(i - 1, j - 1) for (i, j) in inward_ori_index]\noutward = [(j, i) for (i, j) in inward]\nneighbor = inward + outward\n\n\nclass Graph():\n    \"\"\" The Graph to model the skeletons in OpenPose25\n\n    Arguments:\n        labeling_mode: must be one of the follow candidates\n            uniform: Uniform Labeling\n            dastance*: Distance Partitioning*\n            dastance: Distance Partitioning\n            spatial: Spatial Configuration\n            DAD: normalized graph adjacency matrix\n            DLD: normalized graph laplacian matrix\n\n    For more information, please refer to the section 'Partition Strategies' in our paper.\n\n    \"\"\"\n\n    def __init__(self, labeling_mode='uniform'):\n        self.A = self.get_adjacency_matrix(labeling_mode)\n        self.num_node = num_node\n        self.self_link = self_link\n        self.inward = inward\n        self.outward = outward\n        self.neighbor = neighbor\n\n    def get_adjacency_matrix(self, labeling_mode=None):\n        if labeling_mode is None:\n            return self.A\n        if labeling_mode == 'uniform':\n            A = tools.get_uniform_graph(num_node, self_link, neighbor)\n        elif labeling_mode == 'distance*':\n            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)\n        elif labeling_mode == 'distance':\n            A = tools.get_distance_graph(num_node, self_link, neighbor)\n        elif labeling_mode == 'spatial':\n            A = tools.get_spatial_graph(num_node, self_link, inward, outward)\n        elif labeling_mode == 'DAD':\n            A = tools.get_DAD_graph(num_node, self_link, neighbor)\n        elif labeling_mode == 'DLD':\n            A = tools.get_DLD_graph(num_node, self_link, neighbor)\n        # elif labeling_mode == 'customer_mode':\n        #     pass\n        else:\n            raise ValueError()\n        return A\n\n\ndef main():\n    mode = ['uniform', 'distance*', 'distance', 'spatial', 'DAD', 'DLD']\n    np.set_printoptions(threshold=np.nan)\n    for m in mode:\n        print('=' * 10 + m + '=' * 10)\n        print(Graph(m).get_adjacency_matrix())\n\n\nif __name__ == '__main__':\n    main()",
    'module/backbone/graph/tools.py': "import numpy as np\n\n\ndef edge2mat(link, num_node):\n    A = np.zeros((num_node, num_node))\n    for i, j in link:\n        A[j, i] = 1\n    return A\n\n\ndef normalize_digraph(A):\n    Dl = np.sum(A, 0)\n    num_node = A.shape[0]\n    Dn = np.zeros((num_node, num_node))\n    for i in range(num_node):\n        if Dl[i] > 0:\n            Dn[i, i] = Dl[i]**(-1)\n    AD = np.dot(A, Dn)\n    return AD\n\n\ndef normalize_undigraph(A):\n    Dl = np.sum(A, 0)\n    num_node = A.shape[0]\n    Dn = np.zeros((num_node, num_node))\n    for i in range(num_node):\n        if Dl[i] > 0:\n            Dn[i, i] = Dl[i]**(-0.5)\n    DAD = np.dot(np.dot(Dn, A), Dn)\n    return DAD\n\n\ndef get_uniform_graph(num_node, self_link, neighbor):\n    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))\n    return A\n\n\ndef get_uniform_distance_graph(num_node, self_link, neighbor):\n    I = edge2mat(self_link, num_node)\n    N = normalize_digraph(edge2mat(neighbor, num_node))\n    A = I - N\n    return A\n\n\ndef get_distance_graph(num_node, self_link, neighbor):\n    I = edge2mat(self_link, num_node)\n    N = normalize_digraph(edge2mat(neighbor, num_node))\n    A = np.stack((I, N))\n    return A\n\n\ndef get_spatial_graph(num_node, self_link, inward, outward):\n    I = edge2mat(self_link, num_node)\n    In = normalize_digraph(edge2mat(inward, num_node))\n    Out = normalize_digraph(edge2mat(outward, num_node))\n    A = np.stack((I, In, Out))\n    return A\n\n\ndef get_DAD_graph(num_node, self_link, neighbor):\n    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))\n    return A\n\n\ndef get_DLD_graph(num_node, self_link, neighbor):\n    I = edge2mat(self_link, num_node)\n    A = I - normalize_undigraph(edge2mat(neighbor, num_node))\n    return A",
    'module/head/regressor.py': "import math\nimport torch\nimport torch.nn as nn\nfrom torch import Tensor\nfrom torch.nn import init\nfrom torch.nn.parameter import Parameter\n\nclass MultiLinear(nn.Module):\n    r\"\"\"Applies a linear transformation to the incoming data: :math:`y = xA^T + b`\n    \"\"\"\n    __constants__ = ['n_head', 'in_features', 'out_features']\n    n_head: int\n    in_features: int\n    out_features: int\n    weight: Tensor\n\n    def __init__(self, n_head: int, in_features: int, out_features: int, bias: bool = True,\n                 device=None, dtype=None) -> None:\n        factory_kwargs = {'device': device, 'dtype': dtype}\n        super(MultiLinear, self).__init__()\n        self.n_head = n_head\n        self.in_features = in_features\n        self.out_features = out_features\n\n        self.weight = Parameter(torch.empty((n_head, out_features, in_features), **factory_kwargs))\n        if bias:\n            self.bias = Parameter(torch.empty(n_head, out_features, **factory_kwargs))\n        else:\n            self.register_parameter('bias', None)\n            \n        self.reset_parameters()\n\n    def reset_parameters(self) -> None:\n        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with\n        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see\n        # https://github.com/pytorch/pytorch/issues/57109\n        init.kaiming_uniform_(self.weight, a=math.sqrt(5))\n        if self.bias is not None:\n            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)\n            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0\n            init.uniform_(self.bias, -bound, bound)\n\n    def forward(self, input: Tensor) -> Tensor:\n        out = torch.einsum('kij, bkj -> bki', self.weight, input)\n        if self.bias is not None:\n            out += self.bias\n        return out.contiguous()\n\n    def extra_repr(self) -> str:\n        return 'n_head={}, in_features={}, out_features={}, bias={}'.format(\n            self.n_head, self.in_features, self.out_features, self.bias is not None\n        )\n\nclass Regressor(nn.Module):\n    def __init__(self, pred_pose_num, input_dim, hidden_dim, output_dim, **kwargs):\n        super(Regressor, self).__init__()\n        self.layers = nn.Sequential(\n            self._make_multilinear(1, pred_pose_num, input_dim , hidden_dim),\n            MultiLinear(pred_pose_num, hidden_dim, output_dim)\n        )\n\n    def _make_multilinear(self, num, n_head, input_dim, hidden_dim):\n        plane = input_dim\n        layers = []\n        for i in range(num):\n            layer = [MultiLinear(n_head, plane, hidden_dim),\n                     nn.ReLU(inplace=True)]\n            layers.extend(layer)\n\n            plane = hidden_dim\n\n        return nn.Sequential(*layers)\n\n    def forward(self, x):\n        return self.layers(x)",
    'module/loss.py': "import torch\nimport torch.nn as nn\n\n\ndef rotation_matrix_geodesic_loss(R_pred, R_gt):\n    \"\"\"\n    R_pred: (B, J, 3, 3)\n    R_gt:   (B, J, 3, 3)\n    \"\"\"\n    # R_diff = torch.bmm(R_gt.transpose(1, 2), R_pred)  # R_gt^T * R_pred\n    R_diff = torch.matmul(R_gt.transpose(-1, -2), R_pred)  # R_gt^T * R_pred\n    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]\n    # Clamp trace to valid arccos domain to avoid NaNs\n    trace = torch.clamp((trace - 1) / 2, min=-1 + 1e-6, max=1 - 1e-6)\n    theta = torch.acos(trace) * 2 / torch.pi\n    return theta\n\n\nclass ParamLoss(nn.Module):\n    def __init__(self):\n        super(ParamLoss, self).__init__()\n\n    def forward(self, param_out, param_gt, valid=None):\n        if valid is None:\n            loss = rotation_matrix_geodesic_loss(param_out, param_gt)\n        else:\n            loss = rotation_matrix_geodesic_loss(param_out, param_gt) * valid[:, :, 0, 0]\n        return loss\n\n\nclass ParamL2Loss(nn.Module):\n    def __init__(self):\n        super(ParamL2Loss, self).__init__()\n\n    def forward(self, param_out, param_gt, valid=None, pelvis_idx=None):\n        if pelvis_idx is not None:\n            param_out = param_out - param_out[:, [pelvis_idx], :]\n            param_gt = param_gt - param_gt[:, [pelvis_idx], :]\n\n        if valid is None:\n            loss = torch.norm(param_out - param_gt, p=2, dim=-1)\n        else:\n            loss = torch.norm((param_out - param_gt) * valid, p=2, dim=-1)\n        return loss",
    'module/net_body25.py': "import os\nimport numpy as np\nimport torch\nimport torch.nn as nn\n\nfrom smplx.lbs import batch_rodrigues\n\nfrom module.backbone.gcn import STGCN\nfrom module.backbone.graph.openpose_graph import Graph\nfrom module.head.regressor import Regressor\nfrom module.loss import ParamLoss, ParamL2Loss\n\nfrom common.human_models import SMPL\nfrom common.keypoint_geo import normalize_kp\nfrom common.transforms import rot6d_to_rotmat, rot6d_to_axis_angle\n\nSMPL_BODY_POSE_NUM = 23\n\n\nclass NetBody25(nn.Module):\n    def __init__(self, config):\n        super(NetBody25, self).__init__()\n        self.config = config\n        self.graph = Graph(**config.backbone.graph_args)\n        self.backbone = STGCN(graph=self.graph, **config.backbone.params)\n\n        self.linear = nn.Sequential(\n            nn.Linear(config.head.feat_dim, config.head.input_dim),\n            nn.ReLU(),\n            nn.Linear(config.head.input_dim, config.head.input_dim * config.head.pred_pose_num)\n        )\n        self.regressor = Regressor(**config.head)\n        self.pred_pose_num = config.head.pred_pose_num\n\n        # SMPL model\n        self.human_model = SMPL(config.human_model.smpl_dir)\n\n        # replace human model regressor with Openpose regressor\n        openpose_regressor = np.load(os.path.join(config.human_model.smpl_dir, 'smpl', 'J_regressor_body25.npy')).astype(np.float32)\n        self.human_model.J_regressor_idx = {'mid_hip': 8}\n        self.human_model.joint_num = 25\n        self.human_model.joints_name = ('nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',\n                                        'left_shoulder', 'left_elbow', 'left_wrist', 'mid_hip', 'right_hip',\n                                        'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',\n                                        'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe',\n                                        'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel')\n        self.human_model.flip_pairs = ((2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14), (15, 16), (17, 18),\n                                       (24, 21), (22, 19), (23, 20))\n        self.human_model.parent_ids = [0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11]\n        self.human_model.root_joint_idx = [8]\n        self.human_model.joint_regressor = openpose_regressor\n\n        self.openpose_regressor = nn.Parameter(torch.tensor(openpose_regressor, dtype=torch.float32), requires_grad=False)\n\n        # loss\n        self.param_loss = ParamLoss()\n        self.param_l2_loss = ParamL2Loss()\n\n        self.kp_index = {\n            'pelvis': 8,\n            'thorax': 1,\n            'left_hip': 12,\n            'right_hip': 9\n        }\n\n\n    @staticmethod\n    def split_pose_from_smplh(pose):\n        pose = pose.clone()\n        B = pose.shape[0]\n        device = pose.device\n        body_pose = torch.zeros([B, SMPL_BODY_POSE_NUM, 3]).to(device)\n\n        root_orient = pose[:, :3].view(B, -1, 3)\n        body_pose[:, :21, :] = pose[:, 3:66].view(B, -1, 3)\n        hand_pose = pose[:, 66:].view(B, -1, 3)\n\n        return root_orient, body_pose, hand_pose\n\n\n    def forward(self, x, is_training=True):\n        info_dict = {}\n        start_root_orient, start_body_pose, start_hand_pose = self.split_pose_from_smplh(x['start_pose'])\n        end_root_orient, end_body_pose, end_hand_pose = self.split_pose_from_smplh(x['end_pose'])\n        betas = x['betas'][:, :10]\n\n        start_smpl = self.human_model.layer['neutral'](\n            betas=betas,\n            global_orient=start_root_orient,\n            body_pose=start_body_pose\n        )\n\n        end_smpl = self.human_model.layer['neutral'](\n            betas=betas,\n            global_orient=end_root_orient,\n            body_pose=end_body_pose\n        )\n\n        # NOTE(yyc): use openpose 25 joints\n        start_joints = torch.einsum('bvc,jv->bjc', start_smpl.vertices, self.openpose_regressor)\n        end_joints = torch.einsum('bvc,jv->bjc', end_smpl.vertices, self.openpose_regressor)\n\n        # For vis\n        info_dict['start_joints'] = start_joints[0].clone() if is_training else start_joints.clone()\n        info_dict['end_joints'] = end_joints[0].clone() if is_training else end_joints.clone()\n        info_dict['start_verts'] = start_smpl.vertices[0].clone() if is_training else start_smpl.vertices.clone()\n        info_dict['end_verts'] = end_smpl.vertices[0].clone() if is_training else end_smpl.vertices.clone()\n\n        # add translation\n        start_joints = start_joints + x['start_trans'].unsqueeze(1)\n        end_joints = end_joints + x['end_trans'].unsqueeze(1)\n\n        # -- normalize joints --\n        invalid_mask = None\n        start_joints, R, T = normalize_kp(start_joints, invalid_mask, self.kp_index, R=None, T=None)\n        end_joints, _, _ = normalize_kp(end_joints, invalid_mask, self.kp_index, R=R, T=T)\n\n        input_joints = torch.stack([start_joints, end_joints], dim=1)\n        input_joints = input_joints.permute(0, 3, 1, 2) # N, T, V, C -> N, C, T, V\n\n        # -- network process --\n        if not is_training:\n            start_time = torch.cuda.Event(enable_timing=True)\n            end_time = torch.cuda.Event(enable_timing=True)\n            start_time.record()\n        pred_smpl, pred_joints, pred_rotmat_wo_hands, pred_body_pose, pred_root_orient = self.predict(input_joints, start_root_orient, start_body_pose, betas)\n\n        if not is_training:\n            end_time.record()\n            torch.cuda.synchronize()\n            info_dict['pred_body_pose'] = pred_body_pose\n            info_dict['pred_root_orient'] = pred_root_orient\n            info_dict['infer_time'] = start_time.elapsed_time(end_time) / 1000.0  # in seconds\n\n        info_dict['pred_verts'] = pred_smpl.vertices[0].clone() if is_training else pred_smpl.vertices.clone()\n        info_dict['pred_joints'] = pred_joints[0].clone() if is_training else pred_joints.clone()\n\n        # -- loss --\n        loss_dict = {}\n\n        pred_joints = pred_joints + x['end_trans'].unsqueeze(1)\n        pred_joints, _, _ = normalize_kp(pred_joints, invalid_mask, self.kp_index, R=R, T=T)\n\n        B = pred_joints.shape[0]\n        end_rotmat_wo_hands = batch_rodrigues(torch.cat([end_root_orient, end_body_pose[:, :-2]], dim=1).view(-1, 3)).view(B, -1, 3, 3)\n\n        if 'pose' in self.config.loss_config:\n            loss_dict['smpl_pose'] = self.param_loss(pred_rotmat_wo_hands, end_rotmat_wo_hands).mean()\n            loss_dict['smpl_pose'] = loss_dict['smpl_pose'] * self.config.loss_config['pose']\n        if 'verts' in self.config.loss_config:\n            loss_dict['smpl_verts'] = self.param_l2_loss(pred_smpl.vertices, end_smpl.vertices).mean()\n            loss_dict['smpl_verts'] = loss_dict['smpl_verts'] * self.config.loss_config['verts']\n        if 'kp3d' in self.config.loss_config:\n            loss_dict['smpl_kp3d'] = self.param_l2_loss(pred_joints, end_joints).mean()\n            loss_dict['smpl_kp3d'] = loss_dict['smpl_kp3d'] * self.config.loss_config['kp3d']\n\n        return loss_dict, info_dict\n\n\n    def predict(self, input_joints, start_root_orient, start_body_pose, betas):\n        # -- network process --\n        B = input_joints.shape[0]\n        feat = self.backbone(input_joints).squeeze(2)\n\n        input_feats = torch.cat([feat, start_body_pose.view(B, -1), betas], dim=1)\n\n        input_feats = self.linear(input_feats).view(B, self.pred_pose_num, -1)\n        pred_dpose_6d = self.regressor(input_feats)\n        pred_d_rotmat = rot6d_to_rotmat(pred_dpose_6d)\n\n        start_rotmat_wo_hands = batch_rodrigues(torch.cat([start_root_orient, start_body_pose[:, :-2]], dim=1).view(-1, 3)).view(B, -1, 3, 3)\n\n        pred_rotmat_wo_hands = torch.matmul(start_rotmat_wo_hands, pred_d_rotmat)\n\n        pred_rot_angle_wo_hands = rot6d_to_axis_angle(pred_rotmat_wo_hands[..., :2].flatten(0, 1).flatten(1, 2)).view(B, -1, 3)\n\n        pred_root_orient = pred_rot_angle_wo_hands[:, [0]]\n        pred_body_pose = torch.zeros([B, SMPL_BODY_POSE_NUM, 3]).to(pred_rotmat_wo_hands.device)\n        pred_body_pose[:, :SMPL_BODY_POSE_NUM - 2] = pred_rot_angle_wo_hands[:, 1:]\n\n        pred_smpl = self.human_model.layer['neutral'](\n            betas=betas,\n            global_orient=pred_root_orient,\n            body_pose=pred_body_pose\n        )\n        pred_joints = torch.einsum('bvc,jv->bjc', pred_smpl.vertices, self.openpose_regressor)\n\n        return pred_smpl, pred_joints, pred_rotmat_wo_hands, pred_body_pose, pred_root_orient",
}

for rel_path, text in LEARNABLE_VENDOR_FILES.items():
    target = LEARNABLE_VENDOR_ROOT / "src" / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

LEARNABLE_SRC = LEARNABLE_VENDOR_ROOT / "src"
LEARNABLE_CONFIG = LEARNABLE_SRC / "config" / "net.yaml"
for candidate in [LEARNABLE_VENDOR_ROOT, LEARNABLE_SRC]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

print(f"Learnable backend ready: {LEARNABLE_VENDOR_ROOT}")
print(f" - files: {len(LEARNABLE_VENDOR_FILES)}")
print(f" - src: {LEARNABLE_SRC}")
print(f" - config: {LEARNABLE_CONFIG}")

"""## 10. Learnable refinement phase"""

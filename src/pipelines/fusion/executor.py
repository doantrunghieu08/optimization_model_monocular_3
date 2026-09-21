from pathlib import Path
import re
import numpy as np
from tqdm.auto import tqdm
from src.core.json_io import read_json
from src.pipelines.fusion.detector import compute_visibility_from_mesh_vertices
from src.core.keypoints_map import load_keypoints3d_map
from src.core.json_io import write_json
from src.pipelines.fusion.detector import detect_cross_view_errors
from src.pipelines.fusion.optimization import calculate_stats
from src.core.config_loader import resolve_preprocess_output_dir
from src.pipelines.fusion.correction import estimate_bidirectional_similarity
from src.pipelines.fusion.correction import estimate_sequence_root_similarity
from src.pipelines.fusion.correction import fuse_aligned_poses
from src.pipelines.fusion.correction import apply_root_relative
from src.pipelines.fusion.correction import apply_similarity
from src.pipelines.fusion.detector import get_orientation_flag
from src.pipelines.fusion.config import OUTPUT_SUBDIRS
from src.pipelines.fusion.config import NON_REPLACEABLE_ANCHORS
from src.pipelines.fusion.config import MIN_SLAVE_BELIEF_THRESHOLD
from src.pipelines.fusion.config import MIN_LLIST_FOR_RANSAC
from src.pipelines.fusion.detector import load_torso_faces
from src.pipelines.fusion.detector import as_xyz
from src.pipelines.fusion.detector import make_raw_judgement_fallback
from src.pipelines.fusion.correction import apply_rotation_mismatch_corrections
from src.pipelines.fusion.correction import apply_belief_corrections
from src.pipelines.fusion.correction import apply_limb_winner_corrections
from src.pipelines.fusion.correction import select_limb_winner_sets
from src.pipelines.fusion.config import DEFAULT_MAX_BONE_ANGLE_DEG
from src.pipelines.fusion.optimization import optimize_f_points

LEARNABLE_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "_learnable_backend"


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


def _load_pose_meshes(paths: dict, occlusion_enabled: bool):
    if not occlusion_enabled:
        return False, None, None, None, None, 0

    mesh_path = Path(paths["pose_output_dir"]) / "camera_meshes.npz"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Pose mesh cache not found: {mesh_path}. Run pose export first.")
    with np.load(mesh_path, allow_pickle=False) as meshes:
        verts_cam1 = np.asarray(meshes["camera1"])
        verts_cam2 = np.asarray(meshes["camera2"])
        faces = np.asarray(meshes["faces"], dtype=np.int64)
        vertex_parts = np.asarray(meshes["vertex_parts"], dtype=np.int32)
    if verts_cam1.ndim != 3 or verts_cam2.ndim != 3 or verts_cam1.shape[1:] != (6890, 3) or verts_cam2.shape[1:] != (6890, 3):
        raise ValueError(f"Invalid pose mesh cache: {mesh_path}")
    if len(verts_cam1) != len(verts_cam2):
        raise ValueError(f"Camera mesh frame counts do not match in {mesh_path}")
    if vertex_parts.shape != (6890,):
        raise ValueError(f"Invalid SMPL vertex parts in {mesh_path}")
    frame_count = len(verts_cam1)
    print(f"[Fusion] Camera-space mesh cache: {frame_count} synced frames")
    return True, verts_cam1, verts_cam2, faces, vertex_parts, frame_count


def _load_pose_frame(path: Path, metadata_dir: Path):
    data = read_json(path)
    if "camera1" not in data or "camera2" not in data:
        raise ValueError(f"Missing camera pose data in {path}")
    metadata_path = metadata_dir / path.name
    if not metadata_path.exists():
        raise FileNotFoundError(f"Pose metadata not found: {metadata_path}")
    metadata_data = read_json(metadata_path)
    metadata = metadata_data.get("metadata", {})
    for key in ("source_pkl_stems", "source_frame_indices"):
        if not isinstance(metadata.get(key), dict) or not all(camera in metadata[key] for camera in ("camera1", "camera2")):
            raise ValueError(f"Invalid pose metadata {key} in {metadata_path}")
    data.update(metadata_data)
    return data


def _load_2d_profile(config: dict, cam_id: str):
    preprocess_dir = Path(resolve_preprocess_output_dir(config))
    profile_path = preprocess_dir / f"data_{cam_id}.json"
    if not profile_path.exists():
        print(f"[Fusion] 2D belief not found: {profile_path}")
        return None
    profile = read_json(profile_path)
    payload = profile.get(f"2D_camera_{cam_id}")
    if not isinstance(payload, dict):
        print(f"[Fusion] 2D belief missing in {profile_path.name}")
        return None
    keypoints = payload.get("keypoints")
    if not isinstance(keypoints, dict):
        print(f"[Fusion] 2D belief keypoints missing in {profile_path.name}")
        return None
    return keypoints


def _load_2d_profiles(config: dict) -> dict:
    return {
        "camera1": _load_2d_profile(config, "cam1"),
        "camera2": _load_2d_profile(config, "cam2"),
    }


def _frame_belief_from_profile(profile, source_idx, frame_idx: int):
    if not profile:
        return None
    if source_idx is not None:
        data = profile.get(str(int(source_idx)))
        return data if isinstance(data, dict) else None
    candidates = [frame_idx - 1, frame_idx]
    for candidate in candidates:
        data = profile.get(str(candidate))
        if isinstance(data, dict):
            return data
    return None


def _belief2d_for_frame(data: dict, frame_idx: int, profiles: dict) -> dict:
    source_indices = data.get("metadata", {}).get("source_frame_indices", {})
    return {
        "camera1": _frame_belief_from_profile(profiles.get("camera1"), source_indices.get("camera1"), frame_idx),
        "camera2": _frame_belief_from_profile(profiles.get("camera2"), source_indices.get("camera2"), frame_idx),
    }


def _orientation_mismatches(cam1: dict, cam2: dict, names: list[str]) -> set[str]:
    flags1 = get_orientation_flag(cam1)
    flags2 = get_orientation_flag(cam2)
    return {
        name
        for name in names
        if (flags1.get(name, 0) == 1 and flags2.get(name, 0) == -1)
        or (flags1.get(name, 0) == -1 and flags2.get(name, 0) == 1)
    }


def _change_diagnostics(before: dict, after: dict, names: list[str]) -> dict:
    changed = []
    displacements = []
    for name in names:
        displacement_mm = float(np.linalg.norm(as_xyz(after[name]) - as_xyz(before[name])) * 1000.0)
        if displacement_mm > 1e-6:
            changed.append(name)
            displacements.append(displacement_mm)
    return {
        "changed_joints": changed,
        "changed_joint_count": len(changed),
        "mean_displacement_mm": float(np.mean(displacements)) if displacements else 0.0,
        "max_displacement_mm": max(displacements, default=0.0),
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
    belief_alpha=None,
    belief_beta=None,
    verts_by_cam=None,
    torso_faces=None,
    vertex_parts=None,
    frame_idx=None,
    prev_optimized_data=None,
    prev_prev_optimized_data=None,
    belief2d_by_cam=None,
    accel_lambda=3.0,
    orientation_correction_enabled=False,
    optimization_enabled=False,
    reject_new_mismatches=True,
    global_belief=True,
    local_method="naive_distance_belief",
    use_kinematic_constraints=True,
    loss_type="huber",
    belief_delta_cap=0.05,
    correction_blend_mode="belief",
    belief_correction_enabled=True,
    precomputed_transforms=None,
    root_relative_correction=False,
    correction_selector="belief",
    shared_body_pose_enabled=False,
    max_bone_angle_deg=DEFAULT_MAX_BONE_ANGLE_DEG,
    fusion_method="proposed",
    pre_fuse_f_points=False,
    cross_view_lambda=1.0,
):
    if fusion_method not in ("aligned_averaging", "higher_belief_selection", "proposed"):
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    raw_cam1 = {k: as_xyz(v) for k, v in data_in["camera1"].items()}
    raw_cam2 = {k: as_xyz(v) for k, v in data_in["camera2"].items()}
    source = data_in.get("shared_body_pose") if shared_body_pose_enabled else data_in
    if not isinstance(source, dict) or "camera1" not in source or "camera2" not in source:
        raise ValueError("Shared SMPL body pose was not exported; set pose_export.shared_body_pose=true")
    cam1 = {k: as_xyz(v) for k, v in source["camera1"].items()}
    cam2 = {k: as_xyz(v) for k, v in source["camera2"].items()}
    map_data = load_keypoints3d_map(map_path)
    expected_names = [kp["name"] for kp in map_data["keypoints"]]

    if set(cam1.keys()) != set(expected_names) or set(cam2.keys()) != set(expected_names):
        raise ValueError("Input does not have exactly the 21 expected keys for both cameras")

    names = expected_names
    cam1 = {k: cam1[k] for k in names}
    cam2 = {k: cam2[k] for k in names}

    if fusion_method != "aligned_averaging" and verts_by_cam is not None:
        if torso_faces is None:
            raise ValueError("torso_faces is required when verts_by_cam is provided")
        vis1 = compute_visibility_from_mesh_vertices(cam1, verts_by_cam["camera1"], torso_faces, occlusion_tau, vertex_parts=vertex_parts)
        vis2 = compute_visibility_from_mesh_vertices(cam2, verts_by_cam["camera2"], torso_faces, occlusion_tau, vertex_parts=vertex_parts)
    else:
        vis1 = {n: True for n in names}
        vis2 = {n: True for n in names}

    belief2d_by_cam = belief2d_by_cam or {}
    if fusion_method == "aligned_averaging":
        m_set = k1_set = k2_set = set()
        l_list = names
        all_weights = H1_all = H2_all = dict.fromkeys(names, 1.0)
    else:
        if belief_alpha is None or belief_beta is None:
            raise ValueError("fusion.belief.alpha and fusion.belief.beta are required for Higher-Belief and Proposed")
        detected = detect_cross_view_errors(
            cam1,
            cam2,
            names,
            vis1,
            vis2,
            belief2d1=belief2d_by_cam.get("camera1"),
            belief2d2=belief2d_by_cam.get("camera2"),
            alpha=belief_alpha,
            beta=belief_beta,
            global_belief=global_belief,
            local_method=local_method,
            belief_delta_cap=belief_delta_cap,
        )
        m_set, k1_set, k2_set = detected["M"], detected["K1"], detected["K2"]
        l_list, H1_all, H2_all = detected["L"], detected["H1"], detected["H2"]
        all_weights = detected["weights"]

    limb_decisions = []
    if fusion_method == "proposed":
        if correction_selector == "occlusion":
            k1_set = {name for name in names if vis1[name] and not vis2[name]} - NON_REPLACEABLE_ANCHORS
            k2_set = {name for name in names if vis2[name] and not vis1[name]} - NON_REPLACEABLE_ANCHORS
            l_list = [name for name in names if name not in (m_set | k1_set | k2_set)]
        elif correction_selector == "limb_winner":
            k1_set, k2_set, limb_decisions = select_limb_winner_sets(
                H1_all, H2_all, vis1, vis2, belief_delta_cap
            )
            l_list = [name for name in names if name not in (m_set | k1_set | k2_set)]
        elif correction_selector != "belief":
            raise ValueError("fusion.correction.selector must be belief, occlusion, or limb_winner")

        slave_mean_belief = float(np.mean(list(H2_all.values()))) if H2_all else 0.0
        slave_belief_sufficient = (
            correction_selector == "limb_winner"
            or slave_mean_belief >= MIN_SLAVE_BELIEF_THRESHOLD
        )
        if not slave_belief_sufficient:
            import warnings
            warnings.warn(
                f"[Fusion] Slave mean belief={slave_mean_belief:.4f} < threshold={MIN_SLAVE_BELIEF_THRESHOLD}. "
                "Bỏ qua belief correction và optimization để tránh áp transform kém."
            )
            effective_correction = False
            effective_optimization = False
        else:
            effective_correction = belief_correction_enabled
            effective_optimization = optimization_enabled
    else:
        effective_correction = effective_optimization = False

    identity = (1.0, np.eye(3), np.zeros(3))
    needs_transform = (
        fusion_method in ("aligned_averaging", "higher_belief_selection")
        or effective_correction
        or orientation_correction_enabled
        or effective_optimization
    )
    transform_names = names if fusion_method == "aligned_averaging" else l_list
    if precomputed_transforms is not None:
        t12, t21 = precomputed_transforms
        a_list = transform_names
    elif needs_transform:
        if len(transform_names) < MIN_LLIST_FOR_RANSAC:
            import warnings
            warnings.warn(
                f"[Fusion] Chỉ có {len(transform_names)} joint (< {MIN_LLIST_FOR_RANSAC} tối thiểu). "
                "Dùng identity transform thay vì RANSAC để tránh transform kém."
            )
            t12, t21, a_list = identity, identity, transform_names
        else:
            t12, t21, a_list = estimate_bidirectional_similarity(
                cam1,
                cam2,
                transform_names,
                threshold=ransac_threshold,
                max_combos=ransac_max_combos,
            )
    else:
        t12, t21, a_list = identity, identity, transform_names

    if fusion_method in ("aligned_averaging", "higher_belief_selection"):
        cam1_corr, cam2_corr = fuse_aligned_poses(
            cam1, cam2, H1_all, H2_all, t12, t21, fusion_method,
            root_relative=root_relative_correction,
        )
        applied_k1, applied_k2 = set(), set()
    elif effective_correction:
        if correction_selector == "limb_winner":
            cam1_corr, cam2_corr, applied_k1, applied_k2 = apply_limb_winner_corrections(
                cam1, cam2, k1_set, k2_set, t12, t21,
                max_bone_angle_deg=max_bone_angle_deg,
                root_relative=root_relative_correction,
                return_applied=True,
                decisions=limb_decisions,
            )
        else:
            cam1_corr, cam2_corr, applied_k1, applied_k2 = apply_belief_corrections(
                cam1, cam2, k1_set, k2_set, t12, t21, max_displacement=ransac_threshold,
                h1=H1_all, h2=H2_all, blend_mode=correction_blend_mode,
                root_relative=root_relative_correction,
                return_applied=True,
            )
    else:
        cam1_corr, cam2_corr = dict(cam1), dict(cam2)
        applied_k1, applied_k2 = set(), set()

    if fusion_method == "proposed" and orientation_correction_enabled:
        cam1_corr, cam2_corr = apply_rotation_mismatch_corrections(
            cam1_corr, cam2_corr, cam1, cam2, m_set, applied_k1, applied_k2,
            H1_all, H2_all, t12, t21, root_relative=root_relative_correction,
        )
        orientation_applied = m_set - applied_k1 - applied_k2
    else:
        orientation_applied = set()

    a_new = sorted(set(a_list) | applied_k1 | applied_k2 | orientation_applied | NON_REPLACEABLE_ANCHORS)
    skipped_corrections = (k1_set | k2_set) - applied_k1 - applied_k2 if fusion_method == "proposed" else set()
    f_list = [n for n in names if n not in set(a_new) | skipped_corrections]

    if fusion_method == "proposed" and pre_fuse_f_points:
        pre_fuse_names = set(l_list) | set(f_list)
        for name in pre_fuse_names:
            if vis1.get(name, True) and vis2.get(name, True) and name not in (applied_k1 | applied_k2):
                h1_v = float(H1_all.get(name, 0.5))
                h2_v = float(H2_all.get(name, 0.5))
                w1 = h1_v / max(h1_v + h2_v, 1e-12)
                w2 = 1.0 - w1
                cam2_in_1 = apply_root_relative(cam2[name], cam2, cam1, t21) if root_relative_correction else apply_similarity(cam2[name], t21)
                cam1_in_2 = apply_root_relative(cam1[name], cam1, cam2, t12) if root_relative_correction else apply_similarity(cam1[name], t12)
                cam1_corr[name] = w1 * as_xyz(cam1[name]) + w2 * cam2_in_1
                cam2_corr[name] = w2 * as_xyz(cam2[name]) + w1 * cam1_in_2

    before_stats = calculate_stats(cam1_corr, cam2_corr, f_list, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights, loss_type=loss_type)
    mismatches_before_optimization = _orientation_mismatches(cam1_corr, cam2_corr, names)
    if effective_optimization:
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
            prev_prev_data=prev_prev_optimized_data,
            temporal_lambda=temporal_lambda,
            accel_lambda=accel_lambda,
            max_iter=max_iter,
            use_kinematic_constraints=use_kinematic_constraints,
            loss_type=loss_type,
            t12=t12,
            t21=t21,
            cross_view_lambda=cross_view_lambda,
        )
    else:
        optimized_data = {"camera1": dict(cam1_corr), "camera2": dict(cam2_corr)}

    m_after = _orientation_mismatches(optimized_data["camera1"], optimized_data["camera2"], names)
    rejected_mismatches = sorted(m_after - mismatches_before_optimization) if fusion_method == "proposed" and reject_new_mismatches else []
    for name in rejected_mismatches:
        optimized_data["camera1"][name] = cam1_corr[name]
        optimized_data["camera2"][name] = cam2_corr[name]
    if rejected_mismatches:
        m_after = _orientation_mismatches(optimized_data["camera1"], optimized_data["camera2"], names)
    after_stats = calculate_stats(optimized_data["camera1"], optimized_data["camera2"], f_list, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights, loss_type=loss_type)
    changes = {
        "camera1": _change_diagnostics(raw_cam1, optimized_data["camera1"], names),
        "camera2": _change_diagnostics(raw_cam2, optimized_data["camera2"], names),
    }

    return {
        "M": sorted(m_set),
        "M_after": sorted(m_after),
        "M_resolved": len(m_after) == 0 and len(m_set) > 0,
        "K1": sorted(k1_set),
        "K2": sorted(k2_set),
        "K1_applied": sorted(applied_k1),
        "K2_applied": sorted(applied_k2),
        "corrections_skipped": sorted(skipped_corrections),
        "A_new": a_new,
        "F": f_list,
        "F_optimized": f_list if effective_optimization else [],
        "orientation_correction_enabled": bool(orientation_correction_enabled),
        "orientation_applied": sorted(orientation_applied),
        "belief_correction_enabled": bool(effective_correction),
        "alignment_mode": "sequence_root" if root_relative_correction else "frame",
        "correction_selector": correction_selector,
        "fusion_method": fusion_method,
        "limb_decisions": limb_decisions,
        "shared_body_pose_enabled": bool(shared_body_pose_enabled),
        "rejected_new_mismatches": rejected_mismatches,
        "changes": changes,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "optimized": {"camera1": {k: list(v) for k, v in optimized_data["camera1"].items()}, "camera2": {k: list(v) for k, v in optimized_data["camera2"].items()}},
        "joint_belief": {"camera1": H1_all, "camera2": H2_all},
        "vis1": {k: bool(v) for k, v in vis1.items()},
        "vis2": {k: bool(v) for k, v in vis2.items()},
    }
    #end of def run_phase3_pipeline


def run_fusion(config: dict) -> None:
    paths = config["paths"]
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

    fusion_method = fusion_cfg.get("method", "proposed")
    occlusion_cfg = fusion_cfg["occlusion"]
    belief_cfg = fusion_cfg.get("belief", {})
    occlusion_enabled = occlusion_cfg["enabled"] and fusion_method != "aligned_averaging"
    mesh_loaded, verts_cam1, verts_cam2, faces, vertex_parts, mesh_frame_count = _load_pose_meshes(paths, occlusion_enabled)
    torso_faces = load_torso_faces(vertex_parts, faces) if mesh_loaded else None
    belief2d_profiles = (
        _load_2d_profiles(config)
        if fusion_method != "aligned_averaging"
        else {"camera1": None, "camera2": None}
    )

    keypoints_dir = input_dir / "keypoints3d"
    metadata_dir = input_dir / "metadata"
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Pose keypoints directory not found: {keypoints_dir}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Pose metadata directory not found: {metadata_dir}")
    file_paths = sorted(keypoints_dir.glob("pose_data_*.json"), key=_frame_index)
    print(f"[Fusion] Found {len(file_paths)} pose JSON files")
    if not file_paths:
        raise ValueError(f"No pose JSON files found in {keypoints_dir}")
    if mesh_loaded and mesh_frame_count != len(file_paths):
        raise ValueError(f"Pose mesh cache has {mesh_frame_count} frames but pose output has {len(file_paths)}")

    ransac_cfg = fusion_cfg["ransac"]
    opt_cfg = fusion_cfg["optimization"]
    correction_cfg = fusion_cfg.get("correction", {})
    alignment_mode = correction_cfg.get("alignment_mode", "frame")
    loaded_frames = [(path, _load_pose_frame(path, metadata_dir=metadata_dir)) for path in file_paths]
    sequence_transforms = None
    sequence_alignment = None
    if alignment_mode == "sequence_root" and (
        fusion_method in ("aligned_averaging", "higher_belief_selection")
        or correction_cfg.get("enabled", False)
        or correction_cfg.get("orientation_enabled", False)
        or opt_cfg.get("enabled", False)
    ):
        alignment_joints = ("neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
        t12, t21, sequence_alignment = estimate_sequence_root_similarity(
            [data for _, data in loaded_frames], alignment_joints, ransac_cfg["threshold"]
        )
        sequence_transforms = (t12, t21)

    max_fallback_ratio = fusion_cfg.get("max_fallback_ratio", 0.0)
    fallback_count = 0
    prev_result = None
    prev_prev_result = None
    pending_outputs = []
    # --- THÊM TQDM Ở ĐÂY ---
    for path, data in tqdm(loaded_frames, desc="[Fusion] Processing", unit="frame", dynamic_ncols=True):
        frame_idx = _frame_index(path)
        out_name = f"fused_data_{frame_idx}.json"
        if mesh_loaded:
            mesh_frame = frame_idx - 1
            if 0 <= mesh_frame < mesh_frame_count:
                verts_input = {"camera1": verts_cam1[mesh_frame], "camera2": verts_cam2[mesh_frame]}
            else:
                # Dùng tqdm.write thay cho print trong vòng lặp
                tqdm.write(f"[Fusion] Frame {frame_idx}: mesh frame out of range. Occlusion skipped.")
                verts_input = None
        else:
            verts_input = None

        try:
            prev_opt = prev_result["optimized"] if prev_result and "optimized" in prev_result else None
            prev_prev_opt = prev_prev_result["optimized"] if prev_prev_result and "optimized" in prev_prev_result else None
            belief2d_by_cam = _belief2d_for_frame(data, frame_idx, belief2d_profiles)
            result = run_phase3_pipeline(
                data,
                map_path=paths["keypoints3d_map"],
                verts_by_cam=verts_input,
                torso_faces=torso_faces,
                vertex_parts=vertex_parts,
                occlusion_tau=occlusion_cfg["tau"],
                regularization=opt_cfg["regularization"],
                regularization_lambda=opt_cfg["regularization_lambda"],
                temporal_lambda=opt_cfg["temporal_lambda"],
                accel_lambda=opt_cfg.get("accel_lambda", 3.0),
                max_iter=opt_cfg["max_iter"],
                ransac_threshold=ransac_cfg["threshold"],
                ransac_max_combos=ransac_cfg["max_combos"],
                frame_idx=frame_idx,
                prev_optimized_data=prev_opt,
                prev_prev_optimized_data=prev_prev_opt,
                belief2d_by_cam=belief2d_by_cam,
                belief_alpha=belief_cfg.get("alpha") if fusion_method != "aligned_averaging" else None,
                belief_beta=belief_cfg.get("beta") if fusion_method != "aligned_averaging" else None,
                global_belief=belief_cfg.get("global", True),
                local_method=belief_cfg.get("local_method", "naive_distance_belief"),
                orientation_correction_enabled=correction_cfg.get("orientation_enabled", False),
                optimization_enabled=opt_cfg.get("enabled", False),
                use_kinematic_constraints=opt_cfg["use_kinematic_constraints"],
                loss_type=opt_cfg["loss_type"],
                reject_new_mismatches=correction_cfg.get("reject_new_mismatches", True),
                belief_delta_cap=correction_cfg.get("belief_delta_cap", 0.05),
                correction_blend_mode=correction_cfg.get("blend_mode", "belief"),
                belief_correction_enabled=correction_cfg.get("enabled", False),
                precomputed_transforms=sequence_transforms,
                root_relative_correction=alignment_mode == "sequence_root",
                correction_selector=correction_cfg.get("selector", "belief"),
                shared_body_pose_enabled=fusion_cfg.get("shared_body_pose_enabled", False),
                max_bone_angle_deg=correction_cfg.get("max_bone_angle_deg", DEFAULT_MAX_BONE_ANGLE_DEG),
                fusion_method=fusion_method,
                pre_fuse_f_points=opt_cfg.get("pre_fuse_f_points", True),
                cross_view_lambda=opt_cfg.get("cross_view_lambda", 1.0),
            )
            occluded_cam1 = sorted(name for name, visible in result.get("vis1", {}).items() if not visible)
            occluded_cam2 = sorted(name for name, visible in result.get("vis2", {}).items() if not visible)
            occlusion_parts = []
            if occluded_cam1:
                occlusion_parts.append(f"cam1: {', '.join(occluded_cam1)}")
            if occluded_cam2:
                occlusion_parts.append(f"cam2: {', '.join(occluded_cam2)}")
            if occlusion_parts:
                # Dùng tqdm.write thay cho print
                tqdm.write(f"[Fusion] Frame {frame_idx}: Occlusion: {' | '.join(occlusion_parts)}")
        except (ValueError, KeyError) as e:
            # Chỉ fallback cho lỗi dữ liệu đã biết; lỗi lập trình sẽ raise
            tqdm.write(f"[Fusion] Frame {frame_idx}: FAILED ({e}) -> fallback")
            result = make_raw_judgement_fallback(data, frame_idx, e)
            fallback_count += 1
        else:
            prev_prev_result = prev_result
            prev_result = result

        fused_keypoints = {
            "camera1": result.get("optimized", {}).get("camera1", {}),
            "camera2": result.get("optimized", {}).get("camera2", {}),
        }
        fused_metadata = {k: v for k, v in result.items() if k not in ("camera1", "camera2", "optimized")}
        fused_metadata["metadata"] = {
            **data.get("metadata", {}),
            "fusion_config": fusion_cfg,
            "sequence_alignment": sequence_alignment,
        }
        pending_outputs.append((out_name, fused_keypoints, fused_metadata))

    if fallback_count > 0:
        fallback_ratio = fallback_count / len(file_paths)
        print(f"\n[Fusion] WARNING: {fallback_count}/{len(file_paths)} frames used raw-pose fallback ({fallback_ratio:.1%})")
        if fallback_ratio > max_fallback_ratio:
            raise RuntimeError(
                f"[Fusion] Fallback ratio {fallback_ratio:.1%} exceeds max_fallback_ratio={max_fallback_ratio:.1%}. "
                f"{fallback_count} out of {len(file_paths)} frames fell back to raw pose data. "
                "Investigate the data errors or increase fusion.max_fallback_ratio in config."
            )

    for out_name, fused_keypoints, fused_metadata in pending_outputs:
        write_json(output_dir / "keypoints3d" / out_name, fused_keypoints)
        write_json(output_dir / "metadata" / out_name, fused_metadata)

    print(f"\n[Fusion] Done. Output: {output_dir}")


"""## 10. Learnable refinement phase"""

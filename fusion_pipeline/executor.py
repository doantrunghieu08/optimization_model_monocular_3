from pathlib import Path
import re
import numpy as np
from tqdm.auto import tqdm
from json_io import read_json
from fusion_pipeline.detector import compute_visibility_from_mesh_vertices
from keypoints_map import load_keypoints3d_map
from json_io import write_json
from fusion_pipeline.detector import detect_cross_view_errors
from fusion_pipeline.optimization import calculate_stats
from config_loader import resolve_preprocess_output_dir
from fusion_pipeline.correction import estimate_bidirectional_similarity
from fusion_pipeline.detector import get_orientation_flag
from fusion_pipeline.config import OUTPUT_SUBDIRS
from fusion_pipeline.detector import load_torso_faces
from fusion_pipeline.detector import as_xyz
from fusion_pipeline.detector import make_raw_judgement_fallback
from fusion_pipeline.correction import apply_rotation_mismatch_corrections
from fusion_pipeline.correction import apply_confidence_corrections
from fusion_pipeline.optimization import optimize_f_points

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


def _frame_confidence_from_profile(profile, source_idx, frame_idx: int):
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


def _confidence2d_for_frame(data: dict, frame_idx: int, profiles: dict) -> dict:
    source_indices = data.get("metadata", {}).get("source_frame_indices", {})
    return {
        "camera1": _frame_confidence_from_profile(profiles.get("camera1"), source_indices.get("camera1"), frame_idx),
        "camera2": _frame_confidence_from_profile(profiles.get("camera2"), source_indices.get("camera2"), frame_idx),
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
    torso_faces=None,
    frame_idx=None,
    prev_optimized_data=None,
    confidence2d_by_cam=None,
    orientation_correction_enabled=False,
    optimization_enabled=False,
    reject_new_mismatches=True,
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
        if torso_faces is None:
            raise ValueError("torso_faces is required when verts_by_cam is provided")
        vis1 = compute_visibility_from_mesh_vertices(cam1, verts_by_cam["camera1"], torso_faces, occlusion_tau)
        vis2 = compute_visibility_from_mesh_vertices(cam2, verts_by_cam["camera2"], torso_faces, occlusion_tau)
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
    if orientation_correction_enabled:
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
    before_stats = calculate_stats(cam1_corr, cam2_corr, f_list, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)
    if optimization_enabled:
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
    else:
        optimized_data = {"camera1": dict(cam1_corr), "camera2": dict(cam2_corr)}

    m_after = _orientation_mismatches(optimized_data["camera1"], optimized_data["camera2"], names)
    rejected_mismatches = sorted(m_after - m_set) if reject_new_mismatches else []
    for name in rejected_mismatches:
        optimized_data["camera1"][name] = cam1[name]
        optimized_data["camera2"][name] = cam2[name]
    if rejected_mismatches:
        m_after = _orientation_mismatches(optimized_data["camera1"], optimized_data["camera2"], names)
    after_stats = calculate_stats(optimized_data["camera1"], optimized_data["camera2"], f_list, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)

    return {
        "M": sorted(m_set),
        "M_after": sorted(m_after),
        "M_resolved": len(m_after) == 0 and len(m_set) > 0,
        "K1": sorted(k1_set),
        "K2": sorted(k2_set),
        "A_new": a_new,
        "F": f_list,
        "F_optimized": f_list if optimization_enabled else [],
        "orientation_correction_enabled": bool(orientation_correction_enabled),
        "rejected_new_mismatches": rejected_mismatches,
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
    mesh_loaded, verts_cam1, verts_cam2, faces, vertex_parts, mesh_frame_count = _load_pose_meshes(paths, occlusion_enabled)
    torso_faces = load_torso_faces(vertex_parts, faces) if mesh_loaded else None
    confidence2d_profiles = _load_2d_profiles(config)

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

    max_fallback_ratio = fusion_cfg.get("max_fallback_ratio", 0.0)
    fallback_count = 0
    prev_result = None
    pending_outputs = []
    # --- THÊM TQDM Ở ĐÂY ---
    for path in tqdm(file_paths, desc="[Fusion] Processing", unit="frame", dynamic_ncols=True):
        frame_idx = _frame_index(path)
        out_name = f"fused_data_{frame_idx}.json"
        data = _load_pose_frame(path, metadata_dir=metadata_dir)

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
            confidence2d_by_cam = _confidence2d_for_frame(data, frame_idx, confidence2d_profiles)
            result = run_phase3_pipeline(
                data,
                map_path=paths["keypoints3d_map"],
                verts_by_cam=verts_input,
                torso_faces=torso_faces,
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
                orientation_correction_enabled=correction_cfg.get("orientation_enabled", False),
                optimization_enabled=opt_cfg.get("enabled", False),
                reject_new_mismatches=correction_cfg.get("reject_new_mismatches", True),
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
            prev_result = result

        fused_keypoints = {
            "camera1": result.get("optimized", {}).get("camera1", {}),
            "camera2": result.get("optimized", {}).get("camera2", {}),
        }
        fused_metadata = {k: v for k, v in result.items() if k not in ("camera1", "camera2", "optimized")}
        fused_metadata["metadata"] = {
            **data.get("metadata", {}),
            "fusion_config": fusion_cfg,
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

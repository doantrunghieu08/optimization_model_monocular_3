from pathlib import Path
import re
import numpy as np
from tqdm.auto import tqdm
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
from fusion_pipeline.detector import require_ray_casting_backend
from fusion_pipeline.detector import as_xyz
from fusion_pipeline.detector import make_raw_judgement_fallback
from fusion_pipeline.correction import apply_rotation_mismatch_corrections
from fusion_pipeline.correction import apply_confidence_corrections
from fusion_pipeline.correction import apply_multiview_geometric_fusion

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


from compat import load_joblib_compat, resolve_person_payload


def _extract_person_payload(wham_data):
    return resolve_person_payload(wham_data)


def _load_verts_if_available(paths: dict, occlusion_enabled: bool, target_person=None):
    if not occlusion_enabled:
        return False, None, None, 0, 0

    wham_path_1 = Path(paths["cam1_pkl"])
    wham_path_2 = Path(paths["cam2_pkl"])
    if not wham_path_1.exists():
        raise FileNotFoundError(f"WHAM PKL file not found: {wham_path_1}")
    if not wham_path_2.exists():
        raise FileNotFoundError(f"WHAM PKL file not found: {wham_path_2}")

    person_1 = resolve_person_payload(load_joblib_compat(wham_path_1), pkl_path=wham_path_1, target_person=target_person)
    person_2 = resolve_person_payload(load_joblib_compat(wham_path_2), pkl_path=wham_path_2, target_person=target_person)

    verts_cam1 = person_1.get("verts_cam", person_1.get("verts")) if person_1 else None
    verts_cam2 = person_2.get("verts_cam", person_2.get("verts")) if person_2 else None
    if person_1 is None or person_2 is None or verts_cam1 is None or verts_cam2 is None:
        raise ValueError("Neither 'verts_cam' nor 'verts' found in WHAM PKL inputs; fusion.occlusion.enabled requires mesh vertices")
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


def _load_2d_profile(config: dict, cam_id: str, pkl_path: Path | None = None):
    preprocess_dir = Path(resolve_preprocess_output_dir(config))
    profile_path = preprocess_dir / f"data_{cam_id}.json"
    if profile_path.exists():
        profile = read_json(profile_path)
        payload = profile.get(f"2D_camera_{cam_id}")
        if isinstance(payload, dict) and isinstance(payload.get("keypoints"), dict):
            return payload["keypoints"]

    # ponytail: fallback directly to companion tracking_results.pth if not in preprocess JSON
    if pkl_path:
        pth_path = Path(pkl_path).parent / "tracking_results.pth"
        if pth_path.exists():
            try:
                from preprocess_pipeline.extract_2d import _build_2d_camera_payload, _load_keypoints2d_map
                kmap = _load_keypoints2d_map(Path(config["paths"]["keypoints2d_map"]))
                t_raw = load_joblib_compat(pth_path)
                t = t_raw[0] if (isinstance(t_raw, dict) and 0 in t_raw) else (next(iter(t_raw.values())) if isinstance(t_raw, dict) else t_raw)
                payload, _ = _build_2d_camera_payload(t, kmap)
                if profile_path.exists():
                    profile = read_json(profile_path)
                    profile[f"2D_camera_{cam_id}"] = payload
                    write_json(profile_path, profile)
                print(f"[Fusion] Loaded 2D confidence directly from {pth_path.name} for {cam_id}")
                return payload.get("keypoints")
            except Exception as exc:
                print(f"[Fusion] 2D confidence load failed from {pth_path.name}: {exc}")

    print(f"[Fusion] 2D confidence missing for {cam_id}")
    return None


def _load_2d_profiles(config: dict, inputs: dict | None = None) -> dict:
    inputs = inputs or resolve_inputs(config)
    return {
        "camera1": _load_2d_profile(config, "cam1", Path(inputs.get("cam1_pkl", "")) if inputs.get("cam1_pkl") else None),
        "camera2": _load_2d_profile(config, "cam2", Path(inputs.get("cam2_pkl", "")) if inputs.get("cam2_pkl") else None),
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


def _source_frame_indices(data: dict, frame_idx: int) -> dict[str, int]:
    source = data.get("metadata", {}).get("source_frame_indices", {})
    fallback = frame_idx - 1
    return {
        "camera1": int(source.get("camera1", fallback)),
        "camera2": int(source.get("camera2", fallback)),
    }


def run_phase3_pipeline(
    data_in,
    map_path,
    occlusion_tau,
    ransac_threshold,
    ransac_max_combos,
    belief_alpha,
    belief_beta,
    verts_by_cam=None,
    intrinsics_by_cam=None,
    confidence2d_by_cam=None,
    blend=True,
    conflict_detection=True,
    uncertainty_gating=True,
    max_displacement=0.35,
    min_confidence=0.1,
    consensus_threshold=0.18,
    consensus_blend_strength=0.25,
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
        uncertainty_gating=uncertainty_gating,
        min_confidence=min_confidence,
    )
    m_set = detected["M"] if conflict_detection else set()
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

    torso_check = [n for n in ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip") if n in cam1 and n in cam2]
    skip_cross_view = False
    if torso_check and t12 is not None:
        from fusion_pipeline.correction import apply_similarity
        residuals = [float(np.linalg.norm(apply_similarity(np.asarray(cam1[n], dtype=float), t12) - np.asarray(cam2[n], dtype=float))) for n in torso_check]
        if np.mean(residuals) > 0.12:
            skip_cross_view = True

    if not skip_cross_view:
        cam1_corr, cam2_corr = apply_confidence_corrections(
            cam1,
            cam2,
            k1_set,
            k2_set,
            t12=t12,
            t21=t21,
            h1=H1_all,
            h2=H2_all,
            blend=blend,
            uncertainty_gating=uncertainty_gating,
            max_displacement=max_displacement,
            vis1=vis1,
            vis2=vis2,
        )
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
            blend=blend,
            uncertainty_gating=uncertainty_gating,
            max_displacement=max_displacement,
            vis1=vis1,
            vis2=vis2,
        )
        cam1_corr, cam2_corr = apply_multiview_geometric_fusion(
            cam1_corr,
            cam2_corr,
            l_list,
            t12,
            t21,
            h1=H1_all,
            h2=H2_all,
            vis1=vis1,
            vis2=vis2,
            max_displacement=max_displacement,
            consensus_threshold=consensus_threshold,
            consensus_blend_strength=consensus_blend_strength,
        )
    else:
        cam1_corr, cam2_corr = dict(cam1), dict(cam2)

    # Preserve conflict-corrected joints in anchors so SLSQP optimizer does not overwrite them
    a_new = sorted(set(a_list) | k1_set | k2_set | m_set)
    f_list = [n for n in names if n not in set(a_new)]
    before_stats = calculate_stats(cam1_corr, cam2_corr, names, a_new, conf1=H1_all, conf2=H2_all, vis1=vis1, vis2=vis2, f_weights=all_weights)
    optimized_data = {"camera1": cam1_corr, "camera2": cam2_corr}
    after_stats = before_stats

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


KINEMATIC_CHAINS = (
    ("head", "neck"),
    ("left_elbow", "left_shoulder"),
    ("right_elbow", "right_shoulder"),
    ("left_wrist", "left_elbow"),
    ("right_wrist", "right_elbow"),
    ("left_hand", "left_wrist"),
    ("right_hand", "right_wrist"),
    ("left_knee", "left_hip"),
    ("right_knee", "right_hip"),
    ("left_ankle", "left_knee"),
    ("right_ankle", "right_knee"),
    ("left_toe", "left_ankle"),
    ("left_foot", "left_ankle"),
    ("right_toe", "right_ankle"),
    ("right_foot", "right_ankle"),
)


def _enforce_bone_lengths(points: dict, ref_points: dict) -> None:
    # ponytail: minimal forward kinematic bone projection, locks lengths to SMPL baseline
    for child, parent in KINEMATIC_CHAINS:
        if child not in points or parent not in points or child not in ref_points or parent not in ref_points:
            continue
        p_c = np.asarray(points[child], dtype=float)
        p_p = np.asarray(points[parent], dtype=float)
        r_c = np.asarray(ref_points[child], dtype=float)
        r_p = np.asarray(ref_points[parent], dtype=float)
        l_ref = float(np.linalg.norm(r_c - r_p))
        if l_ref < 1e-4:
            continue
        v = p_c - p_p
        dist = float(np.linalg.norm(v))
        if dist > 1e-6:
            points[child] = [round(float(x), 6) for x in (p_p + v * (l_ref / dist))]


def _apply_temporal_smoothing(keypoints_dir: Path, pose_dir: Path | None = None, window_length: int = 5, polyorder: int = 2) -> None:
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return
    files = sorted(keypoints_dir.glob("fused_data_*.json"), key=_frame_index)
    if len(files) < 3:
        return
    all_data = [read_json(f) for f in files]
    w_len = min(window_length, len(files))
    if w_len % 2 == 0:
        w_len -= 1
    if w_len <= polyorder:
        return

    for cam in ("camera1", "camera2"):
        if not all_data or cam not in all_data[0]:
            continue
        joint_names = list(all_data[0][cam].keys())
        coords = np.array([[d[cam][j] for j in joint_names] for d in all_data], dtype=float)
        if np.isnan(coords).any():
            continue
        smoothed = savgol_filter(coords, window_length=w_len, polyorder=polyorder, axis=0)
        for t, d in enumerate(all_data):
            for j_idx, j_name in enumerate(joint_names):
                d[cam][j_name] = [round(float(x), 6) for x in smoothed[t, j_idx]]

    if pose_dir and pose_dir.exists():
        pose_files = sorted(pose_dir.glob("*.json"), key=_frame_index)
        pose_map = {_frame_index(pf): read_json(pf) for pf in pose_files}
        for f, d in zip(files, all_data):
            f_idx = _frame_index(f)
            ref_d = pose_map.get(f_idx)
            if not ref_d:
                continue
            for cam in ("camera1", "camera2"):
                if cam in d and cam in ref_d:
                    _enforce_bone_lengths(d[cam], ref_d[cam])

    for f, d in zip(files, all_data):
        write_json(f, d)
    print(f"[Fusion] Applied Savitzky-Golay smoothing (window={w_len}, polyorder={polyorder}) with rigid bone projection")


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
        require_ray_casting_backend()
        if load_torso_mask(paths["segmentation"]) is None:
            raise RuntimeError("fusion.occlusion.enabled requires a valid SMPL torso mask")

    wham_loaded, verts_cam1, verts_cam2, n_frames_1, n_frames_2 = _load_verts_if_available(
        inputs, occlusion_enabled, target_person=inputs.get("target_person")
    )
    confidence2d_profiles = _load_2d_profiles(config, inputs)
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
    # --- THÊM TQDM Ở ĐÂY ---
    for path in tqdm(file_paths, desc="[Fusion] Processing", unit="frame", dynamic_ncols=True):
        frame_idx = _frame_index(path)
        out_name = f"fused_data_{frame_idx}.json"
        data = _load_pose_frame(path, metadata_dir=metadata_dir)

        if wham_loaded:
            source_frames = _source_frame_indices(data, frame_idx)
            cam1_frame = source_frames["camera1"]
            cam2_frame = source_frames["camera2"]
            if 0 <= cam1_frame < n_frames_1 and 0 <= cam2_frame < n_frames_2:
                verts_input = {"camera1": verts_cam1[cam1_frame], "camera2": verts_cam2[cam2_frame]}
            else:
                # Dùng tqdm.write thay cho print trong vòng lặp
                tqdm.write(f"[Fusion] Frame {frame_idx}: WHAM frame out of range. Occlusion skipped.")
                verts_input = None
        else:
            verts_input = None

        try:
            confidence2d_by_cam = _confidence2d_for_frame(data, frame_idx, confidence2d_profiles)
            result = run_phase3_pipeline(
                data,
                map_path=paths["keypoints3d_map"],
                verts_by_cam=verts_input,
                intrinsics_by_cam=occlusion_intrinsics,
                occlusion_tau=occlusion_cfg["tau"],
                ransac_threshold=ransac_cfg["threshold"],
                ransac_max_combos=ransac_cfg["max_combos"],
                confidence2d_by_cam=confidence2d_by_cam,
                belief_alpha=belief_cfg["alpha"],
                belief_beta=belief_cfg["beta"],
                blend=fusion_cfg.get("blend", True),
                conflict_detection=fusion_cfg.get("conflict_detection", True),
                uncertainty_gating=fusion_cfg.get("uncertainty_gating", {}).get("enabled", True) if isinstance(fusion_cfg.get("uncertainty_gating"), dict) else bool(fusion_cfg.get("uncertainty_gating", True)),
                max_displacement=fusion_cfg.get("uncertainty_gating", {}).get("max_displacement", 0.35) if isinstance(fusion_cfg.get("uncertainty_gating"), dict) else 0.35,
                min_confidence=fusion_cfg.get("uncertainty_gating", {}).get("min_confidence", 0.1) if isinstance(fusion_cfg.get("uncertainty_gating"), dict) else 0.1,
                consensus_threshold=fusion_cfg.get("uncertainty_gating", {}).get("consensus_threshold", 0.18) if isinstance(fusion_cfg.get("uncertainty_gating"), dict) else 0.18,
                consensus_blend_strength=fusion_cfg.get("uncertainty_gating", {}).get("consensus_blend_strength", 0.25) if isinstance(fusion_cfg.get("uncertainty_gating"), dict) else 0.25,
            )
            # 1. Lấy dữ liệu an toàn
                
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
        except Exception as e:
            # Dùng tqdm.write thay cho print
            tqdm.write(f"[Fusion] Frame {frame_idx}: FAILED ({e}) -> fallback")
            result = make_raw_judgement_fallback(data, frame_idx, e)

        fused_keypoints = {
            "camera1": result.get("optimized", {}).get("camera1", {}),
            "camera2": result.get("optimized", {}).get("camera2", {}),
        }
        for cam_k in ("camera1", "camera2"):
            if cam_k in fused_keypoints and cam_k in data:
                _enforce_bone_lengths(fused_keypoints[cam_k], data[cam_k])
        fused_metadata = {k: v for k, v in result.items() if k not in ("camera1", "camera2", "optimized")}
        write_json(output_dir / "keypoints3d" / out_name, fused_keypoints)
        write_json(output_dir / "metadata" / out_name, fused_metadata)

    opt_cfg = fusion_cfg.get("optimization", {})
    if opt_cfg.get("enabled", True):
        from fusion_pipeline.optimization import optimize_sequence_gpu
        optimize_sequence_gpu(
            output_dir / "keypoints3d",
            metadata_dir=output_dir / "metadata",
            pose_dir=input_dir / "keypoints3d",
            config=config,
        )

    smoothing_cfg = fusion_cfg.get("smoothing", {})
    if smoothing_cfg.get("enabled", True):
        _apply_temporal_smoothing(
            output_dir / "keypoints3d",
            pose_dir=input_dir / "keypoints3d",
            window_length=int(smoothing_cfg.get("window_length", 5)),
            polyorder=int(smoothing_cfg.get("polyorder", 2)),
        )

    print(f"\n[Fusion] Done. Output: {output_dir}")


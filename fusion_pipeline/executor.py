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
from fusion_pipeline.correction import apply_similarity
from fusion_pipeline.config import DEFAULT_MASTER_BLEND
from fusion_pipeline.config import DEFAULT_MASTER_MAX_DISPLACEMENT
from fusion_pipeline.config import DEFAULT_MASTER_MIN_CONFIDENCE

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


def _extract_other_person_verts(wham_raw, target_person=None, pkl_path=None):
    if not isinstance(wham_raw, dict):
        return None
    from compat import parse_person_mapping
    target_str = str(target_person).strip().lower() if target_person else None
    if target_str in ("aria1", "aria_1", "aria-1"):
        target_str = "aria01"
    elif target_str in ("aria2", "aria_2", "aria-2"):
        target_str = "aria02"
    mapping = parse_person_mapping(pkl_path) if pkl_path else {}
    other_person = "aria02" if target_str == "aria01" else ("aria01" if target_str == "aria02" else None)
    if other_person and mapping:
        for k, v in wham_raw.items():
            if mapping.get(str(k), mapping.get(k, "")) == other_person and isinstance(v, dict):
                verts = v.get("verts_cam", v.get("verts"))
                if verts is not None:
                    return verts
        return None
    for k, v in wham_raw.items():
        person_name = mapping.get(str(k), mapping.get(k, ""))
        if target_str and person_name == target_str:
            continue
        if isinstance(v, dict):
            verts = v.get("verts_cam", v.get("verts"))
            if verts is not None:
                return verts
    return None


def _load_verts_if_available(paths: dict, occlusion_enabled: bool, target_person=None, include_other_person=False):
    if not occlusion_enabled:
        return False, None, None, 0, 0, None, None

    wham_path_1 = Path(paths["cam1_pkl"])
    wham_path_2 = Path(paths["cam2_pkl"])
    if not wham_path_1.exists():
        raise FileNotFoundError(f"WHAM PKL file not found: {wham_path_1}")
    if not wham_path_2.exists():
        raise FileNotFoundError(f"WHAM PKL file not found: {wham_path_2}")

    raw_1 = load_joblib_compat(wham_path_1)
    raw_2 = load_joblib_compat(wham_path_2)
    person_1 = resolve_person_payload(raw_1, pkl_path=wham_path_1, target_person=target_person)
    person_2 = resolve_person_payload(raw_2, pkl_path=wham_path_2, target_person=target_person)

    verts_cam1 = person_1.get("verts_cam", person_1.get("verts")) if person_1 else None
    verts_cam2 = person_2.get("verts_cam", person_2.get("verts")) if person_2 else None
    if person_1 is None or person_2 is None or verts_cam1 is None or verts_cam2 is None:
        raise ValueError("Neither 'verts_cam' nor 'verts' found in WHAM PKL inputs; fusion.occlusion.enabled requires mesh vertices")

    extra_1 = _extract_other_person_verts(raw_1, target_person=target_person, pkl_path=wham_path_1) if include_other_person else None
    extra_2 = _extract_other_person_verts(raw_2, target_person=target_person, pkl_path=wham_path_2) if include_other_person else None
    if extra_1 is not None and extra_2 is not None:
        print(f"[Fusion] Loaded Person 2 3D mesh for multi-person occlusion ray-casting")

    print(f"[Fusion] WHAM cam1 verts: {verts_cam1.shape[0]} frames")
    print(f"[Fusion] WHAM cam2 verts: {verts_cam2.shape[0]} frames")
    return True, verts_cam1, verts_cam2, verts_cam1.shape[0], verts_cam2.shape[0], extra_1, extra_2


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

    # ponytail: fallback directly to a companion tracking-results file if preprocessing was skipped
    if pkl_path:
        tracking_files = sorted(Path(pkl_path).parent.glob("tracking_results*.pth"))
        if tracking_files:
            pth_path = tracking_files[0]
            try:
                from preprocess_pipeline.extract_2d import _build_2d_camera_payload, _load_keypoints2d_map
                kmap = _load_keypoints2d_map(Path(config["paths"]["keypoints2d_map"]))
                t_raw = load_joblib_compat(pth_path)
                t = resolve_person_payload(
                    t_raw,
                    pkl_path=pkl_path,
                    target_person=config.get("inputs", {}).get("target_person"),
                )
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


def detect_adaptive_fusion_mode(cam1, cam2, t12, h1, h2, vis1, vis2, conflicts):
    names = [name for name in cam1 if name in cam2 and vis1.get(name, True) and vis2.get(name, True)]
    if t12 is None or len(names) < 6:
        return "conservative", {"confidence_advantage": 0.0, "median_residual": float("inf"), "q75_residual": float("inf")}

    residuals = np.array([
        np.linalg.norm(apply_similarity(np.asarray(cam1[name], dtype=float), t12) - np.asarray(cam2[name], dtype=float))
        for name in names
    ])
    confidence_advantage = float(np.mean([float(h2.get(name, 0.0)) - float(h1.get(name, 0.0)) for name in names]))
    median_residual = float(np.median(residuals))
    q75_residual = float(np.percentile(residuals, 75))
    conflict_ratio = len(set(conflicts) & set(names)) / len(names)

    # shortcut: thresholds are calibrated for this camera rig; recalibrate on held-out data when the rig changes.
    aggressive = (
        confidence_advantage >= 0.05
        and median_residual <= 0.10
        and q75_residual <= 0.15
        and conflict_ratio <= 0.10
    )
    return ("aggressive" if aggressive else "conservative"), {
        "confidence_advantage": confidence_advantage,
        "median_residual": median_residual,
        "q75_residual": q75_residual,
        "conflict_ratio": conflict_ratio,
    }


def detect_pair_fusion_gate(frame_results, min_slave_advantage: float = 0.05, master_distorted: bool = False):
    if master_distorted:
        return True, {
            "fusion_applied": True,
            "median_slave_advantage": None,
            "mean_slave_advantage": None,
            "min_slave_advantage": float(min_slave_advantage),
            "valid_frames": len(frame_results),
            "reason": "slave_structurally_better",
        }
    frame_advantages = []
    for result in frame_results:
        confidence = result.get("joint_confidence", {})
        master = confidence.get("camera1", {})
        slave = confidence.get("camera2", {})
        names = set(master) & set(slave)
        if names:
            frame_advantages.append(float(np.mean([
                float(slave[name]) - float(master[name]) for name in names
            ])))

    median_advantage = float(np.median(frame_advantages)) if frame_advantages else float("-inf")
    slave_better = median_advantage >= float(min_slave_advantage)
    return slave_better, {
        "fusion_applied": slave_better,
        "median_slave_advantage": median_advantage if frame_advantages else None,
        "mean_slave_advantage": float(np.mean(frame_advantages)) if frame_advantages else None,
        "min_slave_advantage": float(min_slave_advantage),
        "valid_frames": len(frame_advantages),
        "reason": "slave_better" if slave_better else "master_better_or_equal",
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
    master_margin=0.15,
    extra_verts_by_cam=None,
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
        extra1 = extra_verts_by_cam.get("camera1") if extra_verts_by_cam else None
        extra2 = extra_verts_by_cam.get("camera2") if extra_verts_by_cam else None
        vis1 = compute_visibility_from_mesh_vertices(cam1, verts_by_cam["camera1"], intrinsics_by_cam["camera1"], occlusion_tau, extra_verts=extra1)
        vis2 = compute_visibility_from_mesh_vertices(cam2, verts_by_cam["camera2"], intrinsics_by_cam["camera2"], occlusion_tau, extra_verts=extra2)
    else:
        vis1 = {n: True for n in names}
        vis2 = {n: True for n in names}

    confidence2d_by_cam = confidence2d_by_cam or {}
    # ponytail: 2D detector vote - if 2D detector sees joint is missing/occluded (score < 0.25), mark vis = False
    for cam_k, vis_dict in (("camera1", vis1), ("camera2", vis2)):
        c2d = confidence2d_by_cam.get(cam_k) or {}
        for n in names:
            val = c2d.get(n)
            score = val[2] if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 3 else val
            if score is not None and float(score) < 0.25:
                vis_dict[n] = False
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
    fusion_mode, fusion_mode_stats = detect_adaptive_fusion_mode(
        cam1, cam2, t12, H1_all, H2_all, vis1, vis2, m_set
    )

    torso_check = [n for n in ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip") if n in cam1 and n in cam2]
    skip_cross_view = False
    if torso_check and t12 is not None:
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
            master_margin=master_margin,
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
            master_margin=master_margin,
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
            consensus_threshold=max(consensus_threshold, 0.15) if fusion_mode == "aggressive" else consensus_threshold,
            consensus_blend_strength=max(consensus_blend_strength, 1.0) if fusion_mode == "aggressive" else consensus_blend_strength,
            master_margin=master_margin,
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
        "fusion_mode": fusion_mode,
        "fusion_mode_stats": fusion_mode_stats,
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


def _clamp_to_baseline(points: dict, baseline: dict, vis: dict | None = None, max_radius: float = 0.08) -> None:
    # ponytail: master safety net - only clamp joints that were visible in master; occluded joints must be allowed to be corrected by slave
    for k, p in points.items():
        if k not in baseline:
            continue
        if vis is not None and not vis.get(k, True):
            continue
        p_arr = np.asarray(p, dtype=float)
        b_arr = np.asarray(baseline[k], dtype=float)
        diff = p_arr - b_arr
        dist = float(np.linalg.norm(diff))
        if dist > max_radius:
            points[k] = [round(float(x), 6) for x in (b_arr + diff * (max_radius / dist))]


def _apply_final_master_protection(
    points: dict,
    baseline: dict,
    vis1: dict,
    vis2: dict,
    h1: dict | None = None,
    h2: dict | None = None,
    master_margin: float = 0.15,
    master_min_confidence: float = DEFAULT_MASTER_MIN_CONFIDENCE,
    max_displacement: float = DEFAULT_MASTER_MAX_DISPLACEMENT,
    weight: float = DEFAULT_MASTER_BLEND,
) -> None:
    fused = {name: np.asarray(value, dtype=float) for name, value in points.items()}
    points.update({name: list(value) for name, value in baseline.items()})
    eligible = {
        name for name in fused
        if name in baseline and vis2.get(name, True)
        and float((h2 or {}).get(name, 0.0)) > float((h1 or {}).get(name, 1.0))
        and (
            not vis1.get(name, True)
            or float((h1 or {}).get(name, 1.0)) < master_min_confidence
            or float((h2 or {}).get(name, 0.0)) > float((h1 or {}).get(name, 1.0)) + master_margin
        )
    }
    parent_by_child = dict(KINEMATIC_CHAINS)
    children_by_parent = {}
    for child, parent in KINEMATIC_CHAINS:
        children_by_parent.setdefault(parent, []).append(child)
    ordered = [child for child, _ in KINEMATIC_CHAINS]
    ordered.extend(sorted(eligible.difference(ordered)))

    for name in ordered:
        if name not in eligible:
            continue
        base = np.asarray(baseline[name], dtype=float)
        residual = fused[name] - base
        # ponytail: cam1 cannot see this joint at all → trust slave fully, no distance cap
        limit = float("inf") if (not vis1.get(name, True)) else float(max_displacement)
        if float(np.linalg.norm(residual)) > limit:
            continue
        effective_weight = 1.0 if (not vis1.get(name, True)) else float(weight)
        candidate = base + effective_weight * residual
        parent = parent_by_child.get(name)
        if parent in points and parent in baseline:
            parent_out = np.asarray(points[parent], dtype=float)
            parent_base = np.asarray(baseline[parent], dtype=float)
            bone_length = float(np.linalg.norm(base - parent_base))
            direction = candidate - parent_out
            direction_length = float(np.linalg.norm(direction))
            if bone_length > 1e-8 and direction_length > 1e-8:
                candidate = parent_out + bone_length * direction / direction_length

            anchored_children = [
                child for child in children_by_parent.get(name, [])
                if child in points and child in baseline and child not in eligible
                and float(np.linalg.norm(
                    np.asarray(baseline[child], dtype=float) - base
                )) > 1e-8
            ]
            if len(anchored_children) == 1:
                child = anchored_children[0]
                child_out = np.asarray(points[child], dtype=float)
                child_length = float(np.linalg.norm(np.asarray(baseline[child], dtype=float) - base))
                axis = child_out - parent_out
                anchor_distance = float(np.linalg.norm(axis))
                if abs(bone_length - child_length) <= anchor_distance <= bone_length + child_length:
                    axis /= max(anchor_distance, 1e-8)
                    along = (
                        bone_length ** 2 - child_length ** 2 + anchor_distance ** 2
                    ) / (2.0 * max(anchor_distance, 1e-8))
                    center = parent_out + along * axis
                    radius = np.sqrt(max(bone_length ** 2 - along ** 2, 0.0))
                    radial = candidate - center
                    radial -= np.dot(radial, axis) * axis
                    radial_length = float(np.linalg.norm(radial))
                    if radial_length > 1e-8:
                        candidate = center + radius * radial / radial_length
            elif len(anchored_children) > 1:
                candidate = base
        points[name] = [float(x) for x in candidate]

    for child, parent in KINEMATIC_CHAINS:
        if child not in points or parent not in points or child not in baseline or parent not in baseline:
            continue
        base_length = float(np.linalg.norm(
            np.asarray(baseline[child], dtype=float) - np.asarray(baseline[parent], dtype=float)
        ))
        if base_length <= 1e-8:
            points[child] = list(points[parent])


def _apply_temporal_smoothing(
    keypoints_dir: Path,
    pose_dir: Path | None = None,
    metadata_dir: Path | None = None,
    master_protection: dict | None = None,
    window_length: int = 5,
    polyorder: int = 2,
) -> None:
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

    pose_map = {}
    if pose_dir and pose_dir.exists():
        pose_map = {
            _frame_index(path): read_json(path)
            for path in pose_dir.glob("*.json")
        }
    metadata_map = {}
    if metadata_dir and metadata_dir.exists():
        metadata_map = {
            _frame_index(path): read_json(path)
            for path in metadata_dir.glob("*.json")
        }
    master_cfg = master_protection or {}
    protected_baselines = None
    if master_cfg.get("enabled", False) and pose_map:
        joint_names = list(all_data[0].get("camera1", {}))
        pose_frames = [pose_map.get(_frame_index(f)) for f in files]
        if joint_names and all(
            frame and all(name in frame.get("camera1", {}) for name in joint_names)
            for frame in pose_frames
        ):
            pose_coords = np.array([
                [frame["camera1"][name] for name in joint_names]
                for frame in pose_frames
            ], dtype=float)
            protected_baselines = savgol_filter(pose_coords, window_length=w_len, polyorder=polyorder, axis=0)

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

    if pose_map:
        for t, (f, d) in enumerate(zip(files, all_data)):
            f_idx = _frame_index(f)
            ref_d = pose_map.get(f_idx)
            if not ref_d:
                continue
            if protected_baselines is not None and "camera1" in d:
                meta = metadata_map.get(f_idx, {})
                confidence = meta.get("joint_confidence", {})
                baseline = {
                    name: protected_baselines[t, j_idx]
                    for j_idx, name in enumerate(joint_names)
                }
                _apply_final_master_protection(
                    d["camera1"],
                    baseline,
                    meta.get("vis1", {}),
                    meta.get("vis2", {}),
                    confidence.get("camera1", {}),
                    confidence.get("camera2", {}),
                    master_margin=float(master_cfg.get("confidence_margin", 0.15)),
                )
            for cam in ("camera1", "camera2"):
                if cam in d and cam in ref_d:
                    _enforce_bone_lengths(d[cam], ref_d[cam])

    for f, d in zip(files, all_data):
        write_json(f, d)
    print(f"[Fusion] Applied Savitzky-Golay smoothing (window={w_len}, polyorder={polyorder}) with rigid bone projection")


def _restore_master_pose(keypoints_dir: Path, pose_dir: Path) -> None:
    _restore_pose_cameras(keypoints_dir, pose_dir, ("camera1",))


def _restore_pair_pose(keypoints_dir: Path, pose_dir: Path) -> None:
    _restore_pose_cameras(keypoints_dir, pose_dir, ("camera1", "camera2"))


def _restore_pose_cameras(keypoints_dir: Path, pose_dir: Path, cameras: tuple[str, ...]) -> None:
    for fused_path in keypoints_dir.glob("fused_data_*.json"):
        pose_path = pose_dir / fused_path.name.replace("fused_data_", "pose_data_")
        if not pose_path.exists():
            continue
        fused = read_json(fused_path)
        pose = read_json(pose_path)
        changed = False
        for camera in cameras:
            if camera in fused and camera in pose:
                fused[camera] = pose[camera]
                changed = True
        if changed:
            write_json(fused_path, fused)


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

    multi_person_enabled = bool(occlusion_cfg.get("multi_person", False))
    wham_loaded, verts_cam1, verts_cam2, n_frames_1, n_frames_2, extra_1, extra_2 = _load_verts_if_available(
        inputs, occlusion_enabled, target_person=inputs.get("target_person"), include_other_person=multi_person_enabled
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
    frame_results = []

    # ponytail: zero-GT temporal jitter & tracking continuity check to select anchor camera
    master_is_distorted = False
    slave_is_noisy = False
    if wham_loaded and verts_cam1 is not None and verts_cam2 is not None and n_frames_1 > 2 and n_frames_2 > 2:
        n_chk = min(n_frames_1, n_frames_2, 60)
        a1 = float(np.mean(np.linalg.norm(verts_cam1[2:n_chk] - 2 * verts_cam1[1:n_chk - 1] + verts_cam1[:n_chk - 2], axis=-1))) * 1000.0
        a2 = float(np.mean(np.linalg.norm(verts_cam2[2:n_chk] - 2 * verts_cam2[1:n_chk - 1] + verts_cam2[:n_chk - 2], axis=-1))) * 1000.0
        master_is_distorted = (n_frames_1 < 250 and n_frames_2 >= 250) or (a1 > 1.4 * a2 and a1 > 25.0)
        slave_is_noisy = (n_frames_1 >= 250 and n_frames_2 < 250) or (a2 > 1.4 * a1 and a2 > 25.0)
        print(f"[Fusion] Zero-GT Quality: Cam1(jitter={a1:.1f}, frames={n_frames_1}), Cam2(jitter={a2:.1f}, frames={n_frames_2}) -> master_distorted={master_is_distorted}, slave_noisy={slave_is_noisy}")
    else:
        d1_pts, d2_pts = [], []
        for p in file_paths[:min(len(file_paths), 50)]:
            raw_pose = _load_pose_frame(p, metadata_dir=metadata_dir)
            if "camera1" in raw_pose and "camera2" in raw_pose:
                d1_pts.append([raw_pose["camera1"][j] for j in sorted(raw_pose["camera1"])])
                d2_pts.append([raw_pose["camera2"][j] for j in sorted(raw_pose["camera2"])])
        if len(d1_pts) > 2 and len(d2_pts) > 2:
            p1, p2 = np.array(d1_pts), np.array(d2_pts)
            a1 = float(np.mean(np.linalg.norm(p1[2:] - 2 * p1[1:-1] + p1[:-2], axis=-1)))
            a2 = float(np.mean(np.linalg.norm(p2[2:] - 2 * p2[1:-1] + p2[:-2], axis=-1)))
            master_is_distorted = a1 > 1.4 * a2 and a1 > 0.025
            slave_is_noisy = a2 > 1.4 * a1 and a2 > 0.025
            print(f"[Fusion] Zero-GT Fallback: Cam1(jitter={a1:.4f}), Cam2(jitter={a2:.4f}) -> master_distorted={master_is_distorted}, slave_noisy={slave_is_noisy}")

    master_cfg = fusion_cfg.get("master_protection", {})
    effective_master_margin = 0.0 if master_is_distorted else (float(master_cfg.get("confidence_margin", 0.15)) if master_cfg.get("enabled", True) else 0.0)
    print(f"[Fusion] Master Protection: margin={effective_master_margin}")

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
                extra_input = None
                if extra_1 is not None and extra_2 is not None:
                    if 0 <= cam1_frame < len(extra_1) and 0 <= cam2_frame < len(extra_2):
                        extra_input = {"camera1": extra_1[cam1_frame], "camera2": extra_2[cam2_frame]}
            else:
                # Dùng tqdm.write thay cho print trong vòng lặp
                tqdm.write(f"[Fusion] Frame {frame_idx}: WHAM frame out of range. Occlusion skipped.")
                verts_input = None
                extra_input = None
        else:
            verts_input = None
            extra_input = None

        try:
            confidence2d_by_cam = _confidence2d_for_frame(data, frame_idx, confidence2d_profiles)
            result = run_phase3_pipeline(
                data,
                map_path=paths["keypoints3d_map"],
                verts_by_cam=verts_input,
                intrinsics_by_cam=occlusion_intrinsics,
                extra_verts_by_cam=extra_input,
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
                master_margin=effective_master_margin,
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
        frame_results.append(result)

    slave_better, pair_gate = detect_pair_fusion_gate(frame_results, master_distorted=master_is_distorted)
    for metadata_path in (output_dir / "metadata").glob("fused_data_*.json"):
        metadata = read_json(metadata_path)
        metadata["pair_quality_gate"] = pair_gate
        write_json(metadata_path, metadata)

    # ponytail: do NOT blindly destroy cross-view fusion; joint-level gating and min-confidence loss protect master.
    # only fallback when explicitly requested by pair_mode == "posed_fallback".

    smoothing_cfg = fusion_cfg.get("smoothing", {})
    smoothing_enabled = smoothing_cfg.get("enabled", True)
    if smoothing_enabled:
        smoothing_master_cfg = dict(master_cfg)
        smoothing_master_cfg["confidence_margin"] = effective_master_margin
        if master_is_distorted:
            smoothing_master_cfg["enabled"] = False
        _apply_temporal_smoothing(
            output_dir / "keypoints3d",
            pose_dir=input_dir / "keypoints3d",
            metadata_dir=output_dir / "metadata",
            master_protection=smoothing_master_cfg,
            window_length=int(smoothing_cfg.get("window_length", 5)),
            polyorder=int(smoothing_cfg.get("polyorder", 2)),
        )

    opt_cfg = fusion_cfg.get("optimization", {})
    if opt_cfg.get("enabled", True):
        from fusion_pipeline.optimization import optimize_sequence_gpu
        optimize_sequence_gpu(
            output_dir / "keypoints3d",
            metadata_dir=output_dir / "metadata",
            pose_dir=input_dir / "keypoints3d",
            config=config,
            master_only=not slave_better,
            anchor_cam="camera2" if master_is_distorted else "camera1",
        )

    if fusion_cfg.get("pair_mode") == "posed_fallback":
        _restore_master_pose(output_dir / "keypoints3d", input_dir / "keypoints3d")

    pose_keypoints_dir = input_dir / "keypoints3d"
    if not smoothing_enabled and pose_keypoints_dir.exists():
        for f_path in (output_dir / "keypoints3d").glob("fused_data_*.json"):
            ref_path = pose_keypoints_dir / f_path.name.replace("fused_data_", "pose_data_")
            metadata_path = output_dir / "metadata" / f_path.name
            if ref_path.exists() and metadata_path.exists():
                d_f = read_json(f_path)
                d_p = read_json(ref_path)
                d_m = read_json(metadata_path)
                if "camera1" in d_f and "camera1" in d_p:
                    _enforce_bone_lengths(d_f["camera1"], d_p["camera1"])
                    write_json(f_path, d_f)

    print(f"\n[Fusion] Done. Output: {output_dir}")


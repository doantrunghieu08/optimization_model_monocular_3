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
from fusion_pipeline.selective import reference_bone_lengths, select_slave_joints
from fusion_pipeline.aotr import optimize_aotr
from fusion_pipeline.diagnostics import summarize_selective_frames
from fusion_pipeline.detector import compute_bone_distortion_score
from fusion_pipeline.correction import apply_rotation_mismatch_corrections
from fusion_pipeline.correction import apply_confidence_corrections
from fusion_pipeline.correction import apply_multiview_geometric_fusion
from fusion_pipeline.correction import apply_similarity
from fusion_pipeline.config import DEFAULT_MASTER_BLEND
from fusion_pipeline.config import DEFAULT_MASTER_MAX_DISPLACEMENT
from fusion_pipeline.config import DEFAULT_MASTER_MIN_CONFIDENCE
from fusion_pipeline.config import HEIGHT, RIGID_BONES_RATIO

def _frame_index(path: Path) -> int:
    match = re.search(r"\d+", path.name)
    if not match:
        raise ValueError(f"Cannot extract frame index from {path.name}")
    return int(match.group())


def _clean_output(output_dir: Path, pattern: str = "*.json", create_split_dirs: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_json in output_dir.glob(pattern):
        try:
            old_json.unlink(missing_ok=True)
        except Exception:
            pass
    if create_split_dirs:
        for subdir in OUTPUT_SUBDIRS:
            target_dir = output_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            for old_json in target_dir.glob(pattern):
                try:
                    old_json.unlink(missing_ok=True)
                except Exception:
                    pass


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
        return False, None, None, 0, 0, None, None, 0, 0

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

    # ponytail: video timestamp offset for indexed mesh lookups
    frame_rows_1 = {int(frame_id): row for row, frame_id in enumerate(person_1.get("frame_ids", range(verts_cam1.shape[0])))}
    frame_rows_2 = {int(frame_id): row for row, frame_id in enumerate(person_2.get("frame_ids", range(verts_cam2.shape[0])))}

    print(f"[Fusion] WHAM cam1 verts: {verts_cam1.shape[0]} frames")
    print(f"[Fusion] WHAM cam2 verts: {verts_cam2.shape[0]} frames")
    return True, verts_cam1, verts_cam2, verts_cam1.shape[0], verts_cam2.shape[0], extra_1, extra_2, frame_rows_1, frame_rows_2


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


def detect_adaptive_fusion_mode(cam1, cam2, t12, h1, h2, vis1, vis2, conflicts, master_distorted=False):
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
        (master_distorted or confidence_advantage >= 0.05)
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


def detect_camera_pair_quality(cam1_points, cam2_points, names):
    """Compare two zero-GT tracks using skeletal proportions and temporal acceleration."""
    tracks = [np.asarray(cam1_points, dtype=float), np.asarray(cam2_points, dtype=float)]
    metrics = []
    for track in tracks:
        bone_scores = [
            compute_bone_distortion_score(dict(zip(names, frame)))
            for frame in track
        ]
        # ponytail: robust acceleration ignoring initial track acquisition transient
        acc_samples = np.linalg.norm(track[2:] - 2 * track[1:-1] + track[:-2], axis=-1)
        if len(acc_samples) > 8:
            acc_samples = acc_samples[5:]
        acceleration = (
            float(np.mean(acc_samples)) * 1000.0
            if len(acc_samples) > 0 else float("inf")
        )
        metrics.append({
            "bone_distortion": float(np.median(bone_scores)) if bone_scores else float("inf"),
            "acceleration": acceleration,
            "frames": len(track),
        })

    def worse_vote(key, ratio, minimum_gap=0.0, minimum_value=0.0):
        a, b = metrics[0][key], metrics[1][key]
        if a > ratio * b and a - b > minimum_gap and a > minimum_value:
            return "camera1"
        if b > ratio * a and b - a > minimum_gap and b > minimum_value:
            return "camera2"
        return None

    # shortcut: thresholds are calibrated on hugging_002, recalibrate before using a different rig or frame rate.
    votes = [
        worse_vote("bone_distortion", 1.15, minimum_gap=0.02),
        worse_vote("acceleration", 1.35, minimum_value=25.0),
    ]
    # ponytail: zero-GT physics only (bone distortion & acceleration); no frame-count heuristics
    decisive = [vote for vote in votes if vote]
    worse_camera = decisive[0] if decisive and len(set(decisive)) == 1 else (votes[0] or votes[1])
    return {
        "camera1": metrics[0],
        "camera2": metrics[1],
        "bone_vote": votes[0],
        "acceleration_vote": votes[1],
        "continuity_vote": None,
        "worse_camera": worse_camera,
    }


def _quality_joints_from_vertices(vertices, regressor_path, map_path):
    regressor = np.load(regressor_path)
    mapping = load_keypoints3d_map(map_path)["keypoints"]
    indices = [item["regressor_index"] for item in mapping]
    names = [item["name"] for item in mapping]
    joints = np.einsum("jv,fvc->fjc", regressor, np.asarray(vertices, dtype=float))[:, indices]
    return joints, names


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
    master_distorted=False,
    selective_cfg=None,
    reference_bones=None,
    prev_raw=None,
    prev_prev_raw=None,
    aotr_cfg=None,
    pair_mode="aggressive",
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
    # Experimental AOTR is a SEPARATE optimization-only path; legacy belief,
    # selective gating and downstream global postprocessors must never overwrite it.
    if bool((aotr_cfg or {}).get("enabled", False)):
        if pair_mode == "posed_fallback":
            diag = {
                "status": "fallback", "reason": "pair_mode_posed_fallback",
                "joint_selection": {j: "KEEP" for j in cam1},
                "accepted_bones": [], "evidence_mode": "posed_fallback",
                "alignment_median_m": None, "cost_before": None, "cost_after": None,
            }
            return {
                "M": [], "M_after": [], "M_resolved": False,
                "K1": [], "K2": [], "A_new": [], "F": [],
                "before_stats": (0., 0., 0., 0., 0.),
                "after_stats": (0., 0., 0., 0., 0.),
                "optimized": {
                    "camera1": {j: list(np.asarray(v, dtype=float)) for j, v in cam1.items()},
                    "camera2": {j: list(np.asarray(v, dtype=float)) for j, v in cam2.items()},
                },
                "joint_confidence": {"camera1": {}, "camera2": {}},
                "vis1": {j: bool(v) for j, v in vis1.items()},
                "vis2": {j: bool(v) for j, v in vis2.items()},
                "joint_selection": diag["joint_selection"],
                "aotr": diag,
                "fusion_mode": "aotr",
                "fusion_mode_stats": {"evidence_mode": "posed_fallback", "accepted_bones": 0, "alignment_median_m": None},
            }
        fused, diag = optimize_aotr(
            cam1, cam2,
            observations_master=confidence2d_by_cam.get("camera1"),
            observations_slave=confidence2d_by_cam.get("camera2"),
            K_master=(intrinsics_by_cam or {}).get("camera1"),
            K_slave=(intrinsics_by_cam or {}).get("camera2"),
            vis_master=vis1, vis_slave=vis2, config=aotr_cfg,
            prev_master=(prev_raw or {}).get("camera1"),
            prev_slave=(prev_raw or {}).get("camera2"),
            prev_prev_master=(prev_prev_raw or {}).get("camera1"),
            prev_prev_slave=(prev_prev_raw or {}).get("camera2"),
        )
        changed = [j for j, mode in diag["joint_selection"].items() if mode != "KEEP"]
        return {
            "M": [], "M_after": [], "M_resolved": False,
            "K1": [], "K2": changed, "A_new": [], "F": [],
            "before_stats": (0., 0., 0., 0., 0.),
            "after_stats": (0., 0., 0., 0., 0.),
            "optimized": {
                "camera1": fused,
                "camera2": {j: v.tolist() for j, v in cam2.items()},
            },
            "joint_confidence": {"camera1": {}, "camera2": {}},
            "vis1": {j: bool(v) for j, v in vis1.items()},
            "vis2": {j: bool(v) for j, v in vis2.items()},
            "joint_selection": diag["joint_selection"],
            "aotr": {k: v for k, v in diag.items() if k != "joint_selection"},
            "fusion_mode": "aotr",
            "fusion_mode_stats": {
                "evidence_mode": diag.get("evidence_mode", "unknown"),
                "accepted_bones": len(diag["accepted_bones"]),
                "alignment_median_m": diag["alignment_median_m"],
            },
        }

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

    selective_enabled = bool((selective_cfg or {}).get("enabled", False))
    # Select stable visible torso points BEFORE joint replacement; never fit
    # extrinsics on the very limb whose quality we are trying to determine.
    torso_anchors = [
        n for n in ("pelvis", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if n in cam1 and n in cam2 and vis1.get(n, True) and vis2.get(n, True)
    ]
    alignment_names = torso_anchors if selective_enabled else l_list
    t12, t21, a_list = estimate_bidirectional_similarity(
        cam1, cam2, alignment_names,
        threshold=ransac_threshold, max_combos=ransac_max_combos,
    )
    if selective_enabled:
        # Fail closed on ambiguous/degenerate camera alignment.
        enough_anchors = len(torso_anchors) >= 3 and len(a_list) >= 3
        if enough_anchors:
            centered = np.array([cam1[n] for n in a_list]) - np.mean([cam1[n] for n in a_list], axis=0)
            enough_anchors = np.linalg.matrix_rank(centered, tol=1e-5) >= 2
        residuals = [
            float(np.linalg.norm(apply_similarity(cam1[n], t12) - cam2[n]))
            for n in torso_anchors
        ] if enough_anchors else []
        alignment_valid = (
            bool(residuals)
            and float(np.median(residuals)) <= float(selective_cfg.get("alignment_median_max_m", 0.08))
            and float(np.percentile(residuals, 75)) <= float(selective_cfg.get("alignment_q75_max_m", 0.12))
        )
        if alignment_valid:
            c2d_master = confidence2d_by_cam.get("camera1")
            c2d_slave = confidence2d_by_cam.get("camera2")
            fused_master, joint_decisions, selection_details = select_slave_joints(
                cam1, cam2, t21, vis1, vis2, H1_all, H2_all,
                keypoints2d_master=c2d_master,
                keypoints2d_slave=c2d_slave,
                K_master=(intrinsics_by_cam or {}).get("camera1"),
                K_slave=(intrinsics_by_cam or {}).get("camera2"),
                lengths=reference_bones,
                prev_master=(prev_raw or {}).get("camera1"),
                prev_prev_master=(prev_prev_raw or {}).get("camera1"),
                prev_slave=(prev_raw or {}).get("camera2"),
                prev_prev_slave=(prev_prev_raw or {}).get("camera2"),
                config=selective_cfg,
            )
        else:
            fused_master = {n: list(cam1[n]) for n in names}
            joint_decisions = {n: "KEEP" for n in names}
            selection_details = {"_alignment": "unsafe_or_degenerate"}

        accepted = {n for n, mode in joint_decisions.items() if mode != "KEEP"}
        return {
            "M": sorted(m_set), "M_after": sorted(m_set), "M_resolved": False,
            "K1": [], "K2": sorted(accepted),
            "A_new": sorted(a_list), "F": [n for n in names if n not in accepted],
            "before_stats": (0., 0., 0., 0., 0.),
            "after_stats": (0., 0., 0., 0., 0.),
            "optimized": {
                "camera1": fused_master,
                "camera2": {n: list(cam2[n]) for n in names},
            },
            "joint_confidence": {"camera1": H1_all, "camera2": H2_all},
            "vis1": {k: bool(v) for k, v in vis1.items()},
            "vis2": {k: bool(v) for k, v in vis2.items()},
            "joint_selection": joint_decisions,
            "joint_selection_details": selection_details,
            "selection_summary": {
                "accepted": len(accepted), "kept": len(names) - len(accepted),
                "alignment_valid": alignment_valid,
                "alignment_median_m": float(np.median(residuals)) if residuals else None,
            },
            "fusion_mode": "selective",
            "fusion_mode_stats": {"alignment_valid": alignment_valid},
        }

    fusion_mode, fusion_mode_stats = detect_adaptive_fusion_mode(
        cam1, cam2, t12, H1_all, H2_all, vis1, vis2, m_set,
        master_distorted=master_distorted,
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


def _enforce_anatomical_limb_proportions(points: dict) -> None:
    # ponytail: partitions limb lengths anatomically (humerus 54%, femur 53%) to eliminate monocular hinge bias
    for side in ("left", "right"):
        sh, el, wr, hd = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", f"{side}_hand"
        if sh in points and el in points and wr in points:
            p_s, p_e, p_w = np.asarray(points[sh], dtype=float), np.asarray(points[el], dtype=float), np.asarray(points[wr], dtype=float)
            v_up, v_lo = p_e - p_s, p_w - p_e
            d_up, d_lo = float(np.linalg.norm(v_up)), float(np.linalg.norm(v_lo))
            l_tot = d_up + d_lo
            if d_up > 1e-6 and d_lo > 1e-6 and l_tot > 0.1:
                new_el = p_s + v_up * (0.54 * l_tot / d_up)
                new_wr = new_el + v_lo * (0.46 * l_tot / d_lo)
                d_wr = new_wr - p_w
                points[el] = [round(float(x), 6) for x in new_el]
                points[wr] = [round(float(x), 6) for x in new_wr]
                if hd in points:
                    points[hd] = [round(float(x), 6) for x in (np.asarray(points[hd], dtype=float) + d_wr)]

        hip, kn, an = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
        ft, toe = f"{side}_foot", f"{side}_toe"
        if hip in points and kn in points and an in points:
            p_h, p_k, p_a = np.asarray(points[hip], dtype=float), np.asarray(points[kn], dtype=float), np.asarray(points[an], dtype=float)
            v_th, v_sh = p_k - p_h, p_a - p_k
            d_th, d_sh = float(np.linalg.norm(v_th)), float(np.linalg.norm(v_sh))
            l_tot = d_th + d_sh
            if d_th > 1e-6 and d_sh > 1e-6 and l_tot > 0.1:
                new_kn = p_h + v_th * (0.53 * l_tot / d_th)
                new_an = new_kn + v_sh * (0.47 * l_tot / d_sh)
                d_an = new_an - p_a
                points[kn] = [round(float(x), 6) for x in new_kn]
                points[an] = [round(float(x), 6) for x in new_an]
                if ft in points:
                    points[ft] = [round(float(x), 6) for x in (np.asarray(points[ft], dtype=float) + d_an)]
                if toe in points:
                    points[toe] = [round(float(x), 6) for x in (np.asarray(points[toe], dtype=float) + d_an)]


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
        limit = float("inf") if not vis1.get(name, True) else float(max_displacement)
        if float(np.linalg.norm(residual)) > limit:
            continue
        effective_weight = 1.0 if not vis1.get(name, True) else float(weight)
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
            protected_baselines = savgol_filter(
                pose_coords,
                window_length=w_len,
                polyorder=polyorder,
                axis=0,
            )

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

    # AOTR diagnostics must belong to the CURRENT pair/run. Do not allow
    # a stale summary to masquerade as successful execution after failure.
    if bool(fusion_cfg.get("aotr", {}).get("enabled", False)):
        (output_dir / "aotr_diagnostics.json").unlink(missing_ok=True)

    occlusion_cfg = fusion_cfg["occlusion"]
    belief_cfg = fusion_cfg["belief"]
    occlusion_enabled = occlusion_cfg["enabled"]
    if occlusion_enabled:
        require_ray_casting_backend()
        if load_torso_mask(paths["segmentation"]) is None:
            raise RuntimeError("fusion.occlusion.enabled requires a valid SMPL torso mask")

    multi_person_enabled = bool(occlusion_cfg.get("multi_person", False))
    wham_loaded, verts_cam1, verts_cam2, n_frames_1, n_frames_2, extra_1, extra_2, frame_rows_1, frame_rows_2 = _load_verts_if_available(
        inputs, occlusion_enabled, target_person=inputs.get("target_person"), include_other_person=multi_person_enabled
    )
    confidence2d_profiles = _load_2d_profiles(config, inputs)
    occlusion_intrinsics = _load_occlusion_intrinsics(config) if (occlusion_enabled and wham_loaded) or bool(fusion_cfg.get("aotr", {}).get("enabled", False)) else None

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
    selective_cfg = fusion_cfg.get("selective_joint_fusion", {})
    selective_enabled = bool(selective_cfg.get("enabled", False))
    aotr_cfg = fusion_cfg.get("aotr", {})
    aotr_enabled = bool(aotr_cfg.get("enabled", False))
    if aotr_enabled and selective_enabled:
        raise ValueError("fusion.aotr and fusion.selective_joint_fusion cannot both be enabled")
    if aotr_enabled:
        print("[Fusion] EXPERIMENTAL AOTR enabled: legacy gating, smoothing and global optimization bypassed")
    reference_bones = (
        reference_bone_lengths([read_json(p).get("camera1", {}) for p in file_paths])
        if selective_enabled else {}
    )
    previous_raw, previous_previous_raw = None, None

    # ponytail: zero-GT skeletal distortion and acceleration select the safe anchor.
    master_is_distorted = False
    slave_is_noisy = False
    pair_quality = None
    if wham_loaded and verts_cam1 is not None and verts_cam2 is not None and n_frames_1 > 2 and n_frames_2 > 2:
        q1, quality_names = _quality_joints_from_vertices(verts_cam1, paths["j_regressor_3d"], paths["keypoints3d_map"])
        q2, _ = _quality_joints_from_vertices(verts_cam2, paths["j_regressor_3d"], paths["keypoints3d_map"])
        pair_quality = detect_camera_pair_quality(q1, q2, quality_names)
    else:
        d1_pts, d2_pts = [], []
        quality_names = None
        for p in file_paths:
            raw_pose = _load_pose_frame(p, metadata_dir=metadata_dir)
            if "camera1" in raw_pose and "camera2" in raw_pose:
                quality_names = sorted(set(raw_pose["camera1"]) & set(raw_pose["camera2"]))
                d1_pts.append([raw_pose["camera1"][j] for j in quality_names])
                d2_pts.append([raw_pose["camera2"][j] for j in quality_names])
        if len(d1_pts) > 2 and len(d2_pts) > 2:
            pair_quality = detect_camera_pair_quality(d1_pts, d2_pts, quality_names)

    if pair_quality:
        master_is_distorted = pair_quality["worse_camera"] == "camera1"
        slave_is_noisy = pair_quality["worse_camera"] == "camera2"
        print(f"[Fusion] Zero-GT Quality: {pair_quality} -> master_distorted={master_is_distorted}, slave_noisy={slave_is_noisy}")

    master_cfg = fusion_cfg.get("master_protection", {})
    pair_mode = fusion_cfg.get("pair_mode", "aggressive")
    # ponytail: only activate master protection when slave is actually noisy/degraded or explicitly requested
    should_protect_master = (slave_is_noisy and not master_is_distorted) or (pair_mode == "protected")
    effective_master_margin = float(master_cfg.get("confidence_margin", 0.15)) if (should_protect_master and master_cfg.get("enabled", True)) else 0.0
    print(f"[Fusion] Master Protection: should_protect={should_protect_master}, margin={effective_master_margin}")

    # --- THÊM TQDM Ở ĐÂY ---
    for path in tqdm(file_paths, desc="[Fusion] Processing", unit="frame", dynamic_ncols=True):
        frame_idx = _frame_index(path)
        out_name = f"fused_data_{frame_idx}.json"
        data = _load_pose_frame(path, metadata_dir=metadata_dir)

        if wham_loaded:
            source_frames = _source_frame_indices(data, frame_idx)
            cam1_frame = source_frames["camera1"]
            cam2_frame = source_frames["camera2"]
            idx1 = frame_rows_1.get(cam1_frame)
            idx2 = frame_rows_2.get(cam2_frame)
            if idx1 is not None and idx2 is not None:
                verts_input = {"camera1": verts_cam1[idx1], "camera2": verts_cam2[idx2]}
                extra_input = None
                if extra_1 is not None and extra_2 is not None:
                    if 0 <= idx1 < len(extra_1) and 0 <= idx2 < len(extra_2):
                        extra_input = {"camera1": extra_1[idx1], "camera2": extra_2[idx2]}
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
                master_distorted=master_is_distorted,
                selective_cfg=selective_cfg,
                reference_bones=reference_bones,
                prev_raw=previous_raw,
                prev_prev_raw=previous_previous_raw,
                aotr_cfg=aotr_cfg,
                pair_mode=pair_mode,
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
            if selective_enabled:
                result["selective_error"] = f"{type(e).__name__}: {e}"
            if aotr_enabled:
                result["aotr_error"] = f"{type(e).__name__}: {e}"

        fused_keypoints = {
            "camera1": result.get("optimized", {}).get("camera1", {}),
            "camera2": result.get("optimized", {}).get("camera2", {}),
        }
        # In selective mode KEEP must be byte-for-byte equivalent to Master.
        # The old per-frame length projection can move even untouched joints.
        if not selective_enabled and not aotr_enabled:
            for cam_k in ("camera1", "camera2"):
                if cam_k in fused_keypoints and cam_k in data:
                    _enforce_bone_lengths(fused_keypoints[cam_k], data[cam_k])
        fused_metadata = {k: v for k, v in result.items() if k not in ("camera1", "camera2", "optimized")}
        write_json(output_dir / "keypoints3d" / out_name, fused_keypoints)
        write_json(output_dir / "metadata" / out_name, fused_metadata)
        frame_results.append(result)
        previous_previous_raw, previous_raw = previous_raw, data

    aotr_postprocessing = {"status": "disabled"}
    if aotr_enabled and bool(aotr_cfg.get("sequence_stability", {}).get("enabled", False)):
        # Offline second pass. Master/Fused sequences are synchronized and
        # evaluated independently from any ground-truth data.
        evidence = [r.get("aotr", {}).get("evidence_mode") for r in frame_results]
        if evidence:
            from fusion_pipeline.aotr_stability import postprocess_aotr_3d
            master_sequence = [
                _load_pose_frame(p, metadata_dir)["camera1"] for p in file_paths
            ]
            target_paths = [
                output_dir / "keypoints3d" / f"fused_data_{_frame_index(p)}.json"
                for p in file_paths
            ]
            fused_sequence = [read_json(p)["camera1"] for p in target_paths]
            stability_cfg = dict(aotr_cfg.get("sequence_stability", {}))
            anat_cfg = dict(stability_cfg.get("anatomical_calibration", {}))
            inputs_cfg = config.get("inputs", {})
            target_p = str(inputs_cfg.get("target_person", "aria01")).strip().lower()
            heights = inputs_cfg.get("person_heights") or {}
            heights_lower = {str(k).strip().lower(): v for k, v in heights.items()} if isinstance(heights, dict) else {}
            p_entry = heights_lower.get(target_p)
            if isinstance(p_entry, dict):
                if "height_m" in p_entry:
                    anat_cfg["person_height_m"] = float(p_entry["height_m"])
                for frac_key in ("leg_thigh_fraction", "arm_upper_fraction", "leg_height_fraction", "arm_height_fraction"):
                    if frac_key in p_entry:
                        anat_cfg[frac_key] = float(p_entry[frac_key])
            elif p_entry is not None:
                anat_cfg["person_height_m"] = float(p_entry)
            elif inputs_cfg.get("person_height_m") is not None:
                anat_cfg["person_height_m"] = float(inputs_cfg["person_height_m"])
            stability_cfg["anatomical_calibration"] = anat_cfg
            corrected_sequence, aotr_postprocessing = postprocess_aotr_3d(
                master_sequence, fused_sequence,
                config=stability_cfg,
                editable_mask=[
                    r.get("aotr", {}).get("status") != "fallback"
                    for r in frame_results
                ],
            )
            for i, target_path in enumerate(target_paths):
                record = read_json(target_path)
                record["camera1"] = corrected_sequence[i]
                write_json(target_path, record)
                metadata_path = output_dir / "metadata" / target_path.name
                metadata = read_json(metadata_path)
                original = master_sequence[i]
                moved = [
                    j for j in original
                    if float(np.linalg.norm(
                        np.asarray(corrected_sequence[i][j], dtype=float)
                        - np.asarray(fused_sequence[i][j], dtype=float)
                    )) > 1e-7
                ]
                if moved:
                    metadata["aotr_stability_changed_joints"] = moved
                    # A calibrated KEEP is no longer a byte-exact KEEP.
                    decisions = metadata.get("joint_selection", {})
                    for j in moved:
                        if decisions.get(j) == "KEEP":
                            decisions[j] = "POSTPROCESS"
                    metadata["joint_selection"] = decisions
                write_json(metadata_path, metadata)
            print(f"[Fusion] AOTR sequence stability: {aotr_postprocessing}")
        else:
            aotr_postprocessing = {"status": "skipped_no_frames"}
            print(f"[Fusion] AOTR sequence stability skipped: {aotr_postprocessing}")

    if selective_enabled:
        n_selected = sum(
            sum(mode != "KEEP" for mode in result.get("joint_selection", {}).values())
            for result in frame_results
        )
        slave_better = n_selected > 0
        pair_gate = {
            "fusion_applied": slave_better,
            "reason": "joint_level_selection",
            "accepted_joint_frames": n_selected,
            "total_joint_frames": len(frame_results) * len(frame_results[0].get("joint_selection", {})) if frame_results else 0,
        }
    elif aotr_enabled:
        corrected = sum(
            len(r.get("aotr", {}).get("accepted_bones", [])) for r in frame_results
        )
        pair_gate = {
            "fusion_applied": corrected > 0,
            "reason": "aotr_geometric_objective",
            "accepted_bone_frames": corrected,
            "total_frames": len(frame_results),
        }
        slave_better = corrected > 0
    else:
        slave_better, pair_gate = detect_pair_fusion_gate(frame_results, master_distorted=master_is_distorted)
    pair_gate["zero_gt_quality"] = pair_quality
    for metadata_path in (output_dir / "metadata").glob("fused_data_*.json"):
        metadata = read_json(metadata_path)
        metadata["pair_quality_gate"] = pair_gate
        write_json(metadata_path, metadata)

    quality_fallback = (slave_is_noisy and not master_is_distorted) and not selective_enabled and not aotr_enabled
    if quality_fallback and pair_mode == "posed_fallback":
        # Keep single-view smoothing and bone constraints, but remove every slave contribution first.
        _restore_master_pose(output_dir / "keypoints3d", input_dir / "keypoints3d")

    smoothing_cfg = fusion_cfg.get("smoothing", {})
    smoothing_enabled = smoothing_cfg.get("enabled", True)
    if smoothing_enabled and not selective_enabled and not aotr_enabled:
        smoothing_master_cfg = dict(master_cfg)
        smoothing_master_cfg["confidence_margin"] = effective_master_margin
        if not should_protect_master or aotr_enabled:
            # ponytail: AOTR handles master protection via its own trust-region;
            # bypass legacy confidence margin clamp to preserve vetted corrections.
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
    if not aotr_enabled and opt_cfg.get("enabled", True) and (not selective_enabled or selective_cfg.get("optimize_selected", False)):
        from fusion_pipeline.optimization import optimize_sequence_gpu
        optimize_sequence_gpu(
            output_dir / "keypoints3d",
            metadata_dir=output_dir / "metadata",
            pose_dir=input_dir / "keypoints3d",
            config=config,
            master_only=pair_mode == "posed_fallback",
            anchor_cam="camera1" if selective_enabled else ("camera2" if master_is_distorted else "camera1"),
            selection_only=selective_enabled,
        )

    if not aotr_enabled and fusion_cfg.get("pair_mode") == "posed_fallback":
        _restore_master_pose(output_dir / "keypoints3d", input_dir / "keypoints3d")

    pose_keypoints_dir = input_dir / "keypoints3d"
    if not smoothing_enabled and not selective_enabled and not aotr_enabled and pose_keypoints_dir.exists():
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

    if selective_enabled:
        diagnostics = summarize_selective_frames(frame_results)
        write_json(output_dir / "selective_diagnostics.json", diagnostics)
        print(
            "[Fusion] V3 diagnostics: "
            f"{diagnostics['joint_frames_accepted']}/{diagnostics['joint_frames_evaluated']} "
            f"joint-frames accepted ({diagnostics['joint_acceptance_rate']:.1%}); "
            f"{diagnostics['frames_with_invalid_alignment']} invalid-alignment frames; "
            f"{diagnostics['frames_with_missing_decisions_or_fallback']} fallback frames."
        )
        print(f"[Fusion] Reasons for KEEP: {diagnostics['keep_reasons']}")
        if diagnostics["frames_with_missing_decisions_or_fallback"]:
            print("[Fusion] WARNING: selective mode fell back to raw Master in some frames.")

    if aotr_enabled:
        from collections import Counter
        diagnostics = [r.get("aotr", {}) for r in frame_results]
        summary = {
            "mode": "aotr", "frames": len(frame_results),
            "corrected_frames": sum(d.get("status") == "corrected" for d in diagnostics),
            "accepted_bone_frames": sum(len(d.get("accepted_bones", [])) for d in diagnostics),
            "fallback_frames": sum(not d for d in diagnostics) + sum(d.get("status") == "fallback" for d in diagnostics),
            "reason_counts": dict(Counter(d.get("reason", "pipeline_exception") for d in diagnostics)),
            "evidence_counts": dict(Counter(d.get("evidence_mode", "pipeline_exception") for d in diagnostics)),
            "sequence_stability": aotr_postprocessing,
        }
        write_json(output_dir / "aotr_diagnostics.json", summary)
        print(f"[Fusion] AOTR diagnostics: {summary}")
        print(f"\n[Fusion] Done. Output: {output_dir}")
        return

    # ponytail: enforce anatomical limb proportions across all fused frames
    for f_path in (output_dir / "keypoints3d").glob("fused_data_*.json"):
        d_f = read_json(f_path)
        for cam in ("camera1", "camera2"):
            if cam in d_f:
                _enforce_anatomical_limb_proportions(d_f[cam])
        write_json(f_path, d_f)

    print(f"\n[Fusion] Done. Output: {output_dir}")


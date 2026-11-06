from pathlib import Path
import json
from preprocess_pipeline.executor import run_preprocess
from pose_pipeline.executor import run_pose_export, load_pkl_data, _resolve_tracking_frame_ids
from fusion_pipeline.executor import run_fusion
from learnable_pipeline.executor import run_learnable_smplify, run_learnable_smplify_extra
from evaluation_pipeline.executor import run_evaluation
from visualization_pipeline.executor import run_visualization

def _evaluation_input_dirs(config):
    paths = config.get("paths", {})
    module_dirs = [
        Path(paths["pose_output_dir"]) / "keypoints3d",
        Path(paths["fused_output_dir"]) / "keypoints3d",
    ]
    if config.get("learnable", {}).get("enabled", False):
        module_dirs.append(Path(paths["learnable_output_dir"]) / "keypoints3d")
    learnable_extra_cfg = config.get("learnable_extra", {})
    if learnable_extra_cfg.get("enabled", False):
        module_dirs.append(Path(paths["learnable_extra_output_dir"]) / "keypoints3d")
    return module_dirs

def _require_evaluation_inputs(config):
    missing = []
    for module_dir in _evaluation_input_dirs(config):
        if not module_dir.exists() or not any(module_dir.glob("*.json")):
            missing.append(str(module_dir))
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Evaluation stage requires existing outputs for every enabled module. "
            f"Missing or empty directories: {joined}"
        )

    metadata_dir = Path(config["paths"]["pose_output_dir"]) / "metadata"
    metadata_files = sorted(metadata_dir.glob("pose_data_*.json"))
    if not metadata_files:
        raise FileNotFoundError(f"Evaluation stage requires pose metadata in {metadata_dir}")
    with metadata_files[0].open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    recorded = metadata.get("metadata", {}).get("camera_sync", {}).get("source_pkls")
    if not isinstance(recorded, dict):
        raise ValueError("Pose output has no source provenance; rerun the full pipeline before evaluation")
    expected = {
        "camera1": str(Path(config["inputs"]["cam1_pkl"]).resolve()),
        "camera2": str(Path(config["inputs"]["cam2_pkl"]).resolve()),
    }
    actual = {key: str(Path(recorded.get(key, "")).resolve()) for key in expected}
    if actual != expected:
        raise ValueError(f"Pose output belongs to different inputs: expected {expected}, found {actual}")

def is_skip_unsynchronized(config: dict) -> bool:
    for section in ("runtime", "preprocess", "inputs"):
        sub = config.get(section, {})
        if isinstance(sub, dict) and sub.get("skip_unsynchronized_cameras"):
            return True
    return False

def check_cameras_synchronized(config: dict) -> tuple[bool, str]:
    inputs = config.get("inputs", {})
    cam1_pkl, cam2_pkl = inputs.get("cam1_pkl"), inputs.get("cam2_pkl")
    if not cam1_pkl or not cam2_pkl:
        return True, "No pkl files to check"
    p1, p2 = Path(cam1_pkl), Path(cam2_pkl)
    if not p1.exists() or not p2.exists():
        return False, f"Missing pkl file: {cam1_pkl} or {cam2_pkl}"
    target_person = inputs.get("target_person")
    d1 = load_pkl_data(p1, target_person=target_person)
    d2 = load_pkl_data(p2, target_person=target_person)
    fids1 = _resolve_tracking_frame_ids(d1)
    fids2 = _resolve_tracking_frame_ids(d2)
    if fids1 and fids2:
        overlap = len(set(fids1) & set(fids2))
        if not overlap:
            return False, "Hai camera không có frame_id chung"
        return True, f"Synchronized on {overlap} shared frame_ids"
    l1, l2 = len(d1.get("pose", [])), len(d2.get("pose", []))
    if l1 != l2:
        return False, f"Số frame không khớp và không có frame_ids: cam1={l1}, cam2={l2}"
    return True, "Synchronized"

def _run_full_pipeline(config, include_visualization: bool) -> bool:
    if is_skip_unsynchronized(config):
        synced, reason = check_cameras_synchronized(config)
        if not synced:
            print(f"[Pipeline] Bỏ qua vì 2 camera không đồng bộ khung hình: {reason}")
            return False

    visualization_enabled = include_visualization and config.get("visualization", {}).get("enabled", False)
    selected_offset, _ = run_preprocess(config, extract_frames=visualization_enabled)
    print(f"[Pipeline] Running pose with offset={selected_offset}")
    run_pose_export(config)
    print(f"[Pipeline] Running fusion with offset={selected_offset}")
    run_fusion(config)
    print(f"[Pipeline] Running learnable with offset={selected_offset}")
    run_learnable_smplify(config)
    learnable_extra_cfg = config.get("learnable_extra", {})
    if learnable_extra_cfg.get("enabled", False):
        print(f"[Pipeline] Running learnable extra with offset={selected_offset}")
        run_learnable_smplify_extra(config)
    print(f"[Pipeline] Running evaluation with offset={selected_offset}")
    run_evaluation(config)
    if visualization_enabled:
        print(f"[Pipeline] Running visualization with offset={selected_offset}")
        run_visualization(config)
    return True

def run_pipeline(config, stage_override=None):
    if stage_override == "evaluation":
        _require_evaluation_inputs(config)
        print("[Pipeline] Running evaluation only")
        run_evaluation(config)
        return True

    if stage_override is None:
        return _run_full_pipeline(config, include_visualization=False)

    if stage_override == "visualization":
        return _run_full_pipeline(config, include_visualization=True)

    raise ValueError(f"Unsupported runtime.stage: {stage_override}")

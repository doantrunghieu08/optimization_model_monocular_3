from pathlib import Path
import json
from preprocess_pipeline.executor import run_preprocess
from pose_pipeline.executor import run_pose_export
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

def _run_full_pipeline(config, include_visualization: bool) -> None:
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

def run_pipeline(config, stage_override=None):
    if stage_override == "evaluation":
        _require_evaluation_inputs(config)
        print("[Pipeline] Running evaluation only")
        run_evaluation(config)
        return

    if stage_override is None:
        _run_full_pipeline(config, include_visualization=False)
        return

    if stage_override == "visualization":
        _run_full_pipeline(config, include_visualization=True)
        return

    raise ValueError(f"Unsupported runtime.stage: {stage_override}")

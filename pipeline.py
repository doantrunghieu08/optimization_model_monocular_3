from pathlib import Path
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
        Path(paths["learnable_output_dir"]) / "keypoints3d",
    ]
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
            "Evaluation stage requires existing pose/fusion/learnable outputs. "
            f"Missing or empty directories: {joined}"
        )

def _run_full_pipeline(config, include_visualization: bool) -> None:
    selected_offset, _ = run_preprocess(config, extract_frames=include_visualization)
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
    if include_visualization:
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

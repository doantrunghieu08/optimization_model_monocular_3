"""Reproducible MPI stability ablation for the best completed camera pair."""
import copy
import csv
import statistics
from pathlib import Path

from config_loader import absolutize_config_paths, load_config
from pipeline import run_pipeline


ROOT = Path(__file__).parent.resolve()
PAIR = {
    "cam1_pkl": "input/MPI/Segments/wham_out_video_0_seg_1/video_0_seg_1.pkl",
    "cam2_pkl": "input/MPI/Segments/wham_out_video_8_seg_1/video_8_seg_1.pkl",
    "camera1_video": "input/MPI/Segments/wham_out_video_0_seg_1/output.mp4",
    "camera2_video": "input/MPI/Segments/wham_out_video_8_seg_1/output.mp4",
}
VARIANTS = {
    "raw_aotr": {"sequence": False, "smooth": False},
    "temporal_only": {"sequence": True, "smooth": False},
    "temporal_smoothing": {"sequence": True, "smooth": True},
}


def average(csv_path, column):
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return float(next(row[column] for row in rows if row["Frame"] == "AVERAGE"))


def module_average(csv_path, column, module):
    with csv_path.open(encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return statistics.mean(float(row[column]) for row in rows if row["Module"] == module)


def run_variant(base, name, settings):
    cfg = copy.deepcopy(base)
    cfg["inputs"].update(PAIR)
    cfg["runtime"]["stage"] = "default"
    cfg["visualization"]["enabled"] = False
    stability = cfg["fusion"]["aotr"]["sequence_stability"]
    stability["enabled"] = settings["sequence"]
    stability["temporal_stability"]["smooth_trajectories"] = settings["smooth"]
    stability["anatomical_calibration"]["enabled"] = False
    root = ROOT / "output" / "mpi_stability_ablation" / name
    for key in (
        "preprocess_output_dir", "pose_output_dir", "fused_output_dir",
        "learnable_output_dir", "learnable_extra_output_dir",
        "visualization_output_dir", "evaluation_output_dir",
    ):
        cfg["paths"][key] = str(root / key.removesuffix("_dir"))
    cfg = absolutize_config_paths(cfg, ROOT)
    assert Path(cfg["paths"]["fused_output_dir"]).is_relative_to(root)
    run_pipeline(cfg, stage_override=None)
    evaluation = Path(cfg["paths"]["evaluation_output_dir"])
    diagnostics = Path(cfg["paths"]["fused_output_dir"]) / "aotr_diagnostics.json"
    return {
        "variant": name,
        "mpjpe": average(evaluation / "MPJPE_cam1.csv", "fused_priority1_mm"),
        "pa_mpjpe": average(evaluation / "PA-MPJPE_cam1.csv", "fused_priority1_mm"),
        "baseline_mpjpe": average(evaluation / "MPJPE_cam1.csv", "posed_priority1_mm"),
        "baseline_pa_mpjpe": average(evaluation / "PA-MPJPE_cam1.csv", "posed_priority1_mm"),
        "accel": module_average(evaluation / "Accel_cam1.csv", "Frame_Accel_Error_mm_frame2", "fused"),
        "baseline_accel": module_average(evaluation / "Accel_cam1.csv", "Frame_Accel_Error_mm_frame2", "posed"),
        "mble": module_average(evaluation / "MBLE_cam1.csv", "Frame_MBLE_mm", "fused"),
        "baseline_mble": module_average(evaluation / "MBLE_cam1.csv", "Frame_MBLE_mm", "posed"),
        "diagnostics": str(diagnostics),
    }


if __name__ == "__main__":
    base_config = load_config(ROOT / "configs" / "pipeline_mpi.yml")
    for variant, options in VARIANTS.items():
        print("ABLATION_RESULT", run_variant(base_config, variant, options), flush=True)

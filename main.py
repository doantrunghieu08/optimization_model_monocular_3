import argparse
import sys
from pathlib import Path
import yaml
import subprocess
import shutil

# Import pipeline functions
from pipeline import run_pipeline

# Import utility functions for configuration
from config_loader import absolutize_config_paths, print_path_summary

def main():
    parser = argparse.ArgumentParser(description="Monocular Optimization Pipeline")
    parser.add_argument("--config", type=str, default="configs/pipeline.yml", help="Path to config file")
    args = parser.parse_args()

    WORKSPACE_DIR = Path(__file__).parent.resolve()
    config_path = WORKSPACE_DIR / args.config

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        REFERENCE_CONFIG = yaml.safe_load(f)

    # Resolve paths relative to WORKSPACE_DIR
    CONFIG = absolutize_config_paths(REFERENCE_CONFIG, WORKSPACE_DIR)
    print_path_summary(CONFIG)
    
    stage = CONFIG.get('runtime', {}).get('stage', 'visualization')
    print(f"[Config] runtime.stage = {stage}")

    learnable_checkpoint = WORKSPACE_DIR / "models" / "best_ckpt.pth.tar"
    CONFIG.setdefault("learnable", {})["checkpoint"] = str(learnable_checkpoint)
    CONFIG.setdefault("paths", {})["preprocess_output_dir"] = CONFIG["paths"].get("preprocess_output_dir", str(WORKSPACE_DIR / "output" / "preprocess_results"))
    CONFIG.setdefault("paths", {})["learnable_extra_output_dir"] = CONFIG["paths"].get("learnable_extra_output_dir", str(WORKSPACE_DIR / "output" / "learnable_extra_results"))
    CONFIG.setdefault("learnable_extra", {})["enabled"] = bool(CONFIG.get("learnable_extra", {}).get("enabled", False))

    # Run the main pipeline
    run_pipeline(CONFIG, stage_override=stage)

    # Convert results if visualization is enabled
    if stage == "visualization" and CONFIG.get("visualization", {}).get("enabled", False):
        print("\n--- Converting Output Videos ---")
        visualize_output = Path(CONFIG["paths"]["visualization_output_dir"])
        for cam in ["camera1", "camera2"]:
            input_video = visualize_output / f"project_{cam}_pose_fusion_learnable.mp4"
            output_video = visualize_output / f"project_{cam}_viewable.mp4"
            if input_video.exists():
                print(f"Converting {input_video.name} for web viewable format...")
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(input_video),
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    str(output_video)
                ])
                print(f"Created {output_video.name}")

    print("\n--- Packaging Evaluation Results ---")
    evaluation_dir = Path(CONFIG["paths"]["evaluation_output_dir"])
    if evaluation_dir.exists():
        output_zip = str(evaluation_dir.parent / "evaluation_results")
        shutil.make_archive(
            base_name=output_zip,
            format="zip",
            root_dir=str(evaluation_dir.parent),
            base_dir=evaluation_dir.name
        )
        print(f"Created: {output_zip}.zip")
    else:
        print("No evaluation output found to zip.")

if __name__ == "__main__":
    main()

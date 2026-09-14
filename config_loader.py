import os
import re
from ruamel.yaml import YAML
import json
from pathlib import Path
import sys
from compat import patch_numpy_and_inspect

# --- LOGIC PHÂN GIẢI BIẾN MÔI TRƯỜNG Ở ĐÂY ---
env_pattern = re.compile(r'^\${([a-zA-Z0-9_]+)(?::-([^}]+))?}$')

def set_env_from_filename(notebook_filename: str):
    """
    Hàm này lấy alpha, beta từ tên file notebook và nạp vào os.environ
    để tệp pipeline.yml có thể đọc được.
    """
    pattern = r"_alpha(\d+[Ee]-?\d+)_beta(\d+[Ee]-?\d+)"
    match = re.search(pattern, notebook_filename)
    
    if match:
        alpha_val = float(match.group(1))
        beta_val = float(match.group(2))
        
        # Nạp vào os.environ (cùng tên biến với file pipeline.yml)
        os.environ["ALPHA"] = str(alpha_val)
        os.environ["BETA"] = str(beta_val)
        
        print(f"[INFO] Đã nạp biến môi trường từ tên file:")
        print(f"       ALPHA = {alpha_val}")
        print(f"       BETA  = {beta_val}")
    else:
        print("[WARNING] Không tìm thấy alpha/beta trong tên file, sẽ dùng giá trị default trong tệp YML.")

ALLOWED_STAGES = {
    "visualization",
    "evaluation",
}

INPUT_KEYS = (
    "cam1_pkl",
    "cam2_pkl",
    "camera1_video",
    "camera2_video",
    "ground_truth_dir",
)

def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    match = env_pattern.match(value)
    if match:
        env_var = match.group(1)
        default_value = match.group(2)
        result = os.environ.get(env_var, default_value)
        
        if result is None or result == 'null': return None
        if isinstance(result, str):
            result = result.strip('"').strip("'")
            if result.lower() == 'true': return True
            if result.lower() == 'false': return False
            if result.isdigit(): return int(result)
            try:
                return float(result)
            except ValueError:
                pass
        return result
    return value

# Khởi tạo đối tượng yaml mới
custom_yaml = YAML(typ='safe')
custom_yaml.resolver.add_implicit_resolver('!env_var', env_pattern, None)
custom_yaml.constructor.add_constructor('!env_var', env_var_constructor)
# ----------------------------------------------------

def resolve_inputs(config):
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Missing config section: inputs")
    for key in INPUT_KEYS:
        if key not in inputs or not inputs[key]:
            raise ValueError("Missing config input: inputs.{}".format(key))
    return inputs


def resolve_preprocess_output_dir(config):
    paths = config.get("paths")
    if not isinstance(paths, dict) or not paths.get("preprocess_output_dir"):
        raise ValueError("Missing config path: paths.preprocess_output_dir")
    return paths["preprocess_output_dir"]


def load_config(config_path):
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError("Config file not found: {}".format(path))

    with path.open("r", encoding="utf-8") as f:
        # dùng custom_yaml thay cho yaml mặc định
        config = custom_yaml.load(f) or {}

    validate_config(config)
    return config


def validate_config(config):
    if "runtime" not in config:
        raise ValueError("Missing config section: runtime")

    if "paths" not in config:
        raise ValueError("Missing config section: paths")

    runtime_cfg = config["runtime"]
    for key in ("stage", "clean_output"):
        if key not in runtime_cfg or runtime_cfg[key] is None:
            raise ValueError("Missing config runtime parameter: runtime.{}".format(key))

    stage = runtime_cfg["stage"]
    if stage not in ALLOWED_STAGES:
        raise ValueError(
            "Invalid runtime.stage={!r}. Allowed: {}".format(
                stage, sorted(ALLOWED_STAGES)
            )
        )

    paths = config["paths"]
    inputs = resolve_inputs(config)
    evaluation_cfg = config.get("evaluation")
    if not isinstance(evaluation_cfg, dict):
        raise ValueError("Missing config section: evaluation")
    metrics_cfg = evaluation_cfg.get("metrics")
    if not isinstance(metrics_cfg, dict):
        raise ValueError("Missing config section: evaluation.metrics")

    required_paths = [
        "smpl_model",
        "keypoints3d_map",
        "keypoints2d_map",
        "j_regressor_3d",
        "preprocess_output_dir",
        "pose_output_dir",
        "fused_output_dir",
        "learnable_output_dir",
        "learnable_extra_output_dir",
        "visualization_output_dir",
        "evaluation_output_dir",
    ]

    for key in required_paths:
        if key not in paths:
            raise ValueError("Missing config path: paths.{}".format(key))

    for key in INPUT_KEYS:
        if key not in inputs or not inputs[key]:
            raise ValueError("Missing config input: inputs.{}".format(key))

    fusion_cfg = config.get("fusion", {})
    for key in ("enabled", "belief", "occlusion", "ransac", "correction", "optimization"):
        if key not in fusion_cfg:
            raise ValueError("Missing config section: fusion.{}".format(key))
    if not isinstance(fusion_cfg["enabled"], bool):
        raise ValueError("fusion.enabled must be a boolean")
    max_fallback_ratio = fusion_cfg.get("max_fallback_ratio", 0.0)
    if not isinstance(max_fallback_ratio, (int, float)) or isinstance(max_fallback_ratio, bool) or not 0 <= max_fallback_ratio <= 1:
        raise ValueError("fusion.max_fallback_ratio must be a number between 0 and 1")

    if "enabled" not in evaluation_cfg:
        raise ValueError("Missing config section: evaluation.enabled")

    learnable_cfg = config.get("learnable")
    if not isinstance(learnable_cfg, dict):
        raise ValueError("Missing config section: learnable")
    if "enabled" not in learnable_cfg:
        raise ValueError("Missing config section: learnable.enabled")
    if "checkpoint" not in learnable_cfg or not learnable_cfg["checkpoint"]:
        raise ValueError("Missing config learnable parameter: learnable.checkpoint")
    if learnable_cfg["enabled"] and not fusion_cfg["enabled"]:
        raise ValueError("learnable.enabled=true requires fusion.enabled=true")

    learnable_extra_cfg = config.get("learnable_extra")
    if not isinstance(learnable_extra_cfg, dict):
        raise ValueError("Missing config section: learnable_extra")
    if "enabled" not in learnable_extra_cfg:
        raise ValueError("Missing config section: learnable_extra.enabled")

    visualization_cfg = config.get("visualization")
    if not isinstance(visualization_cfg, dict):
        raise ValueError("Missing config section: visualization")
    if "enabled" not in visualization_cfg:
        raise ValueError("Missing config section: visualization.enabled")
    for key in ("target_fps", "dpi", "max_frames", "cameras"):
        if key not in visualization_cfg:
            raise ValueError("Missing config visualization parameter: visualization.{}".format(key))

    belief_cfg = fusion_cfg["belief"]
    for key in ("alpha", "beta", "global", "local_method"):
        if key not in belief_cfg or belief_cfg[key] is None:
            raise ValueError("Missing config fusion belief parameter: fusion.belief.{}".format(key))
    alpha, beta = belief_cfg["alpha"], belief_cfg["beta"]
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0:
        raise ValueError("fusion.belief.alpha must be a non-negative number")
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or not 0 < beta <= 1:
        raise ValueError("fusion.belief.beta must be a number greater than 0 and at most 1")
    if not isinstance(belief_cfg["global"], bool):
        raise ValueError("fusion.belief.global must be a boolean")
    if belief_cfg["local_method"] not in ("naive_distance_belief", "optical_aware_belief"):
        raise ValueError("fusion.belief.local_method must be naive_distance_belief or optical_aware_belief")

    occlusion_cfg = fusion_cfg["occlusion"]
    for key in ("enabled", "tau"):
        if key not in occlusion_cfg or occlusion_cfg[key] is None:
            raise ValueError("Missing config fusion occlusion parameter: fusion.occlusion.{}".format(key))
    if not isinstance(occlusion_cfg["enabled"], bool):
        raise ValueError("fusion.occlusion.enabled must be a boolean")
    if not isinstance(occlusion_cfg["tau"], (int, float)) or occlusion_cfg["tau"] < 0:
        raise ValueError("fusion.occlusion.tau must be a non-negative number")

    correction_cfg = fusion_cfg["correction"]
    for key in ("orientation_enabled", "reject_new_mismatches"):
        if not isinstance(correction_cfg.get(key), bool):
            raise ValueError(f"fusion.correction.{key} must be a boolean")

    ransac_cfg = fusion_cfg["ransac"]
    for key in ("threshold", "max_combos"):
        if key not in ransac_cfg or ransac_cfg[key] is None:
            raise ValueError("Missing config fusion ransac parameter: fusion.ransac.{}".format(key))
    if not isinstance(ransac_cfg["threshold"], (int, float)) or ransac_cfg["threshold"] <= 0:
        raise ValueError("fusion.ransac.threshold must be a positive number")
    if not isinstance(ransac_cfg["max_combos"], int) or isinstance(ransac_cfg["max_combos"], bool) or ransac_cfg["max_combos"] <= 0:
        raise ValueError("fusion.ransac.max_combos must be a positive integer")

    opt_cfg = fusion_cfg["optimization"]
    for key in ("enabled", "use_kinematic_constraints", "loss_type", "regularization", "regularization_lambda", "temporal_lambda", "max_iter"):
        if key not in opt_cfg or opt_cfg[key] is None:
            raise ValueError("Missing config fusion optimization parameter: fusion.optimization.{}".format(key))
    for key in ("enabled", "use_kinematic_constraints", "regularization"):
        if not isinstance(opt_cfg[key], bool):
            raise ValueError(f"fusion.optimization.{key} must be a boolean")
    if opt_cfg["loss_type"] not in ("huber", "mse"):
        raise ValueError("fusion.optimization.loss_type must be huber or mse")
    for key in ("regularization_lambda", "temporal_lambda"):
        if not isinstance(opt_cfg[key], (int, float)) or isinstance(opt_cfg[key], bool) or opt_cfg[key] < 0:
            raise ValueError(f"fusion.optimization.{key} must be a non-negative number")
    if not isinstance(opt_cfg["max_iter"], int) or isinstance(opt_cfg["max_iter"], bool) or opt_cfg["max_iter"] <= 0:
        raise ValueError("fusion.optimization.max_iter must be a positive integer")

    for key in ("pa_mpjpe", "mpjpe", "pck"):
        if key not in metrics_cfg or metrics_cfg[key] is None:
            raise ValueError("Missing config evaluation metric flag: evaluation.metrics.{}".format(key))

"""## 5. chuẩn hóa config để chạy ổn định

"""

def configure_stdout_encoding():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

patch_numpy_and_inspect()
configure_stdout_encoding()



def absolutize_config_paths(config: dict, workspace_dir: Path) -> dict:
    config = json.loads(json.dumps(config))
    for section_name in ("inputs", "paths"):
        section = config.get(section_name, {})
        for key, value in list(section.items()):
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                continue
            if value.startswith("optimization_model_monocular/"):
                section[key] = str(workspace_dir.parent / value)
            else:
                section[key] = str(workspace_dir / value)

    vis_cfg = config.setdefault("visualization", {})
    search_dir = vis_cfg.get("video_search_dir", ".")
    if search_dir == ".":
        vis_cfg["video_search_dir"] = str(workspace_dir)
    elif isinstance(search_dir, str) and not Path(search_dir).is_absolute():
        if search_dir.startswith("optimization_model_monocular/"):
            vis_cfg["video_search_dir"] = str(workspace_dir.parent / search_dir)
        else:
            vis_cfg["video_search_dir"] = str(workspace_dir / search_dir)
    return config


def print_path_summary(config: dict) -> None:
    print("[Config] Inputs:")
    for key, value in config.get("inputs", {}).items():
        print(f" - {key}: {value}")
    print("[Config] Paths:")
    for key, value in config.get("paths", {}).items():
        print(f" - {key}: {value}")

"""## 6. Preprocess phase"""

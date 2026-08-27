import os
import re
from ruamel.yaml import YAML
import json
from pathlib import Path
import sys
import numpy as np

# --- LOGIC PHÂN GIẢI BIẾN MÔI TRƯỜNG Ở ĐÂY ---
env_pattern = re.compile(r'^\${([a-zA-Z0-9_]+)(?::-([^}]+))?}$')

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
        config = yaml.safe_load(f) or {}

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
        "segmentation",
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
    for key in ("enabled", "belief", "occlusion", "ransac", "optimization"):
        if key not in fusion_cfg:
            raise ValueError("Missing config section: fusion.{}".format(key))

    if "enabled" not in evaluation_cfg:
        raise ValueError("Missing config section: evaluation.enabled")

    learnable_cfg = config.get("learnable")
    if not isinstance(learnable_cfg, dict):
        raise ValueError("Missing config section: learnable")
    if "enabled" not in learnable_cfg:
        raise ValueError("Missing config section: learnable.enabled")
    if "checkpoint" not in learnable_cfg or not learnable_cfg["checkpoint"]:
        raise ValueError("Missing config learnable parameter: learnable.checkpoint")

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
    for key in ("alpha", "beta"):
        if key not in belief_cfg or belief_cfg[key] is None:
            raise ValueError("Missing config fusion belief parameter: fusion.belief.{}".format(key))

    occlusion_cfg = fusion_cfg["occlusion"]
    for key in ("enabled", "tau"):
        if key not in occlusion_cfg or occlusion_cfg[key] is None:
            raise ValueError("Missing config fusion occlusion parameter: fusion.occlusion.{}".format(key))

    ransac_cfg = fusion_cfg["ransac"]
    for key in ("threshold", "max_combos"):
        if key not in ransac_cfg or ransac_cfg[key] is None:
            raise ValueError("Missing config fusion ransac parameter: fusion.ransac.{}".format(key))

    opt_cfg = fusion_cfg["optimization"]
    for key in ("regularization", "regularization_lambda", "temporal_lambda", "max_iter"):
        if key not in opt_cfg or opt_cfg[key] is None:
            raise ValueError("Missing config fusion optimization parameter: fusion.optimization.{}".format(key))

    for key in ("pa_mpjpe", "mpjpe", "pck"):
        if key not in metrics_cfg or metrics_cfg[key] is None:
            raise ValueError("Missing config evaluation metric flag: evaluation.metrics.{}".format(key))

"""## 5. chuẩn hóa config để chạy ổn định

"""

def patch_numpy_and_inspect():
    import inspect
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'object'):
        np.object = object
    if not hasattr(np, 'typeDict'):
        np.typeDict = np.sctypeDict
    if not hasattr(np, 'complex'):
        np.complex = complex
    if not hasattr(np, 'unicode'):
        np.unicode = str
    if not hasattr(np, 'str'):
        np.str = str
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec

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
            if not isinstance(value, str) or not value or value.startswith("/"):
                continue
            if value.startswith("optimization_model_monocular/"):
                section[key] = str(workspace_dir.parent / value)
            else:
                section[key] = str(workspace_dir / value)

    vis_cfg = config.setdefault("visualization", {})
    search_dir = vis_cfg.get("video_search_dir", ".")
    if search_dir == ".":
        vis_cfg["video_search_dir"] = str(workspace_dir)
    elif isinstance(search_dir, str) and not search_dir.startswith("/"):
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

import os
import re
from ruamel.yaml import YAML
import json
from pathlib import Path
import sys
from src.core.compat import patch_numpy_and_inspect

# --- LOGIC PHÂN GIẢI BIẾN MÔI TRƯỜNG Ở ĐÂY ---
env_pattern = re.compile(r'^\${([a-zA-Z0-9_]+)(?::-([^}]+))?}$')

def set_env_from_filename(notebook_filename: str):
    """
    Đọc toàn bộ tham số config từ tên file notebook và nạp vào os.environ
    để pipeline.yml có thể đọc qua cú pháp ${VAR:-default}.

    Tên notebook ablation phải khai báo đầy đủ các tham số sau:

    1. ALPHA / BETA  — hệ số belief:
       Cú pháp tên file: alpha1E_1_beta85E_2  hoặc  alpha1E-1_beta85E-2
       → ALPHA=0.1, BETA=0.85

    2. LOCAL_METHOD  — phương pháp tính belief local:
       Từ khóa trong tên file: _optical_  → optical_aware_belief
                               _naive_    → naive_distance_belief

    3. GLOBAL  — có hòa belief với xương lân cận không:
       Từ khóa trong tên file: _global_   → true
                               _local_    → false   (phân biệt bằng context sau optical/naive)

    4. KINEMATIC_CONSTRAINTS  — ràng buộc động học khi optimize:
       Từ khóa trong tên file: _kinematic_      → true
                               _unconstrained_  → false

    5. LOSS_TYPE  — hàm loss cho optimizer:
       Từ khóa trong tên file: _huber_  → huber
                               _mse_    → mse
    """
    name = Path(notebook_filename).name
    keys = ("ALPHA", "BETA", "LOCAL_METHOD", "GLOBAL", "KINEMATIC_CONSTRAINTS", "LOSS_TYPE")
    for key in keys:
        os.environ.pop(key, None)

    # ── 1. ALPHA & BETA ────────────────────────────────────────────────────────
    pattern_ab = r"alpha(\d+[Ee][_\-]?\d+)_beta(\d+[Ee][_\-]?\d+)"
    m = re.search(pattern_ab, name)
    if m:
        alpha_val = float(m.group(1).replace("_", "-"))
        beta_val  = float(m.group(2).replace("_", "-"))
        os.environ["ALPHA"] = str(alpha_val)
        os.environ["BETA"]  = str(beta_val)
        print(f"[ENV] ALPHA={alpha_val}  (raw: '{m.group(1)}')")
        print(f"[ENV] BETA={beta_val}   (raw: '{m.group(2)}')")
    else:
        print("[WARNING] Khong tim thay alpha/beta trong ten file — dung default trong YML.")

    # Helper: match từ khóa được ngăn cách bởi _ trong tên file
    def kw(keyword):
        return bool(re.search(r'(?:^|_)' + keyword + r'(?:_|\.|$)', name))

    # ── 2. LOCAL_METHOD ────────────────────────────────────────────────────────
    if kw('optical'):
        os.environ["LOCAL_METHOD"] = "optical_aware_belief"
        print("[ENV] LOCAL_METHOD=optical_aware_belief")
    elif kw('naive'):
        os.environ["LOCAL_METHOD"] = "naive_distance_belief"
        print("[ENV] LOCAL_METHOD=naive_distance_belief")

    # ── 3. GLOBAL belief ───────────────────────────────────────────────────────
    # Tìm _global_ hoặc _local_ xuất hiện SAU phần method (optical/naive)
    # Dùng lookbehind để tránh nhầm "local" trong các ngữ cảnh khác
    if re.search(r'(?:optical|naive)_global(?:_|\.|$)', name):
        os.environ["GLOBAL"] = "true"
        print("[ENV] GLOBAL=true")
    elif re.search(r'(?:optical|naive)_local(?:_|\.|$)', name):
        os.environ["GLOBAL"] = "false"
        print("[ENV] GLOBAL=false")

    # ── 4. KINEMATIC_CONSTRAINTS ───────────────────────────────────────────────
    if kw('kinematic'):
        os.environ["KINEMATIC_CONSTRAINTS"] = "true"
        print("[ENV] KINEMATIC_CONSTRAINTS=true")
    elif kw('unconstrained'):
        os.environ["KINEMATIC_CONSTRAINTS"] = "false"
        print("[ENV] KINEMATIC_CONSTRAINTS=false")

    # ── 5. LOSS_TYPE ───────────────────────────────────────────────────────────
    if kw('huber'):
        os.environ["LOSS_TYPE"] = "huber"
        print("[ENV] LOSS_TYPE=huber")
    elif kw('mse'):
        os.environ["LOSS_TYPE"] = "mse"
        print("[ENV] LOSS_TYPE=mse")

    missing = [key for key in keys if key not in os.environ]
    if missing:
        raise ValueError(f"Notebook filename does not define: {', '.join(missing)}")

def get_notebook_name() -> str | None:
    """
    Trả về tên file notebook Colab hiện tại (không kèm đường dẫn thư mục).
    Ưu tiên theo thứ tự:
      1. Biến môi trường NOTEBOOK_NAME (có thể set thủ công trong notebook).
      2. google.colab._message API (chỉ dùng được khi đang chạy trong Colab).
      3. None nếu không xác định được.
    """
    # 1. Env var thủ công — tiện khi chạy ngoài Colab hoặc khi test
    name = os.environ.get("NOTEBOOK_NAME")
    if name:
        return name

    # 2. Colab runtime API
    try:
        from google.colab import _message  # type: ignore
        nb = _message.blocking_request("get_ipynb", request="", timeout_sec=5)
        name = nb.get("metadata", {}).get("colab", {}).get("name") or nb.get("metadata", {}).get("name")
        if name:
            return name
    except Exception:
        pass

    return None


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
    for key in ("enabled", "occlusion", "ransac", "correction", "optimization"):
        if key not in fusion_cfg:
            raise ValueError("Missing config section: fusion.{}".format(key))
    if not isinstance(fusion_cfg["enabled"], bool):
        raise ValueError("fusion.enabled must be a boolean")
    fusion_method = fusion_cfg.get("method", "proposed")
    if fusion_method not in ("proposed", "aligned_averaging", "higher_belief_selection"):
        raise ValueError("fusion.method must be proposed, aligned_averaging, or higher_belief_selection")
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

    belief_cfg = fusion_cfg.get("belief", {})
    if fusion_method in ("proposed", "higher_belief_selection"):
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
    for key in ("enabled", "orientation_enabled", "reject_new_mismatches"):
        if not isinstance(correction_cfg.get(key), bool):
            raise ValueError(f"fusion.correction.{key} must be a boolean")
    belief_delta_cap = correction_cfg.get("belief_delta_cap", 0.05)
    if not isinstance(belief_delta_cap, (int, float)) or isinstance(belief_delta_cap, bool) or belief_delta_cap < 0:
        raise ValueError("fusion.correction.belief_delta_cap must be a non-negative number")
    if correction_cfg.get("blend_mode", "belief") not in ("hard", "belief"):
        raise ValueError("fusion.correction.blend_mode must be hard or belief")
    if correction_cfg.get("alignment_mode", "frame") not in ("frame", "sequence_root"):
        raise ValueError("fusion.correction.alignment_mode must be frame or sequence_root")
    if correction_cfg.get("selector", "belief") not in ("belief", "occlusion", "limb_winner"):
        raise ValueError("fusion.correction.selector must be belief, occlusion, or limb_winner")
    max_bone_angle_deg = correction_cfg.get("max_bone_angle_deg", 60.0)
    if not isinstance(max_bone_angle_deg, (int, float)) or isinstance(max_bone_angle_deg, bool) or max_bone_angle_deg <= 0:
        raise ValueError("fusion.correction.max_bone_angle_deg must be a positive number")

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

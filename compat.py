import re
import sys
import numpy as np
import joblib
import inspect

def patch_numpy_and_inspect():
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'object'): np.object = object
    if not hasattr(np, 'typeDict'): np.typeDict = np.sctypeDict
    if not hasattr(np, 'complex'): np.complex = complex
    if not hasattr(np, 'unicode'): np.unicode = str
    if not hasattr(np, 'str'): np.str = str
    if not hasattr(inspect, 'getargspec'): inspect.getargspec = inspect.getfullargspec

def configure_stdout_encoding():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path


def load_joblib_compat(filepath):
    p = Path(filepath)
    if p.suffix.lower() in ('.pt', '.pth'):
        try:
            import torch
            return torch.load(p, map_location='cpu', weights_only=False)
        except Exception:
            pass
    with open(filepath, 'rb') as f:
        return joblib.load(f)


_smpl_model_cache = None


def _gvhmr_to_payload(data, pkl_path=None, target_str=None):
    # ponytail: translate GVHMR HMR4D prediction into canonical SMPL payload
    incam = data.get("smpl_params_incam", {})
    glob = data.get("smpl_params_global", {})

    def _to_np(x):
        if x is None:
            return None
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    bp = _to_np(incam.get("body_pose"))
    if bp is not None and bp.shape[1] == 63:
        bp = np.pad(bp, ((0, 0), (0, 6)))
    go = _to_np(incam.get("global_orient"))
    pose = np.concatenate([go, bp], axis=-1) if (go is not None and bp is not None) else None
    trans = _to_np(incam.get("transl"))
    betas = _to_np(incam.get("betas"))

    bp_g = _to_np(glob.get("body_pose", bp))
    if bp_g is not None and bp_g.shape[1] == 63:
        bp_g = np.pad(bp_g, ((0, 0), (0, 6)))
    go_g = _to_np(glob.get("global_orient"))
    pose_world = np.concatenate([go_g, bp_g], axis=-1) if (go_g is not None and bp_g is not None) else None
    trans_world = _to_np(glob.get("transl"))

    n_frames = len(pose) if pose is not None else 0
    frame_ids = np.arange(n_frames)

    # ponytail: generate SMPL mesh vertices for 3D ray-casting occlusion
    verts = None
    try:
        global _smpl_model_cache
        if _smpl_model_cache is None:
            patch_numpy_and_inspect()
            from pose_pipeline.smpl_runner import create_smpl_model
            _smpl_model_cache = create_smpl_model("models/SMPL_NEUTRAL.pkl")
        import torch
        with torch.no_grad():
            out = _smpl_model_cache(
                body_pose=torch.tensor(pose[:, 3:], dtype=torch.float32),
                global_orient=torch.tensor(pose[:, :3], dtype=torch.float32),
                transl=torch.tensor(trans, dtype=torch.float32),
                betas=torch.tensor(betas, dtype=torch.float32),
            )
            verts = out.vertices.cpu().numpy()
    except Exception:
        pass

    tracking = None
    if pkl_path is not None:
        p = Path(pkl_path)
        pattern = f"preprocess/vitpose*{target_str}*.pt" if target_str else "preprocess/vitpose*.pt"
        vit_files = list(p.parent.glob(pattern)) or list(p.parent.glob("preprocess/vitpose*.pt"))
        if vit_files:
            try:
                import torch
                kpts = torch.load(vit_files[0], map_location="cpu", weights_only=False)
                kpts_np = kpts.cpu().numpy() if hasattr(kpts, "cpu") else np.asarray(kpts)
                tracking = {"keypoints": kpts_np, "frame_id": np.arange(len(kpts_np))}
            except Exception:
                pass

    return {
        "pose": pose,
        "trans": trans,
        "betas": betas,
        "pose_world": pose_world,
        "trans_world": trans_world,
        "frame_ids": frame_ids,
        "verts": verts,
        "verts_cam": verts,
        "tracking_results_for_reproj": tracking,
    }


def parse_person_mapping(pkl_path):
    if pkl_path is None:
        return {}
    pkl_path = Path(pkl_path)
    candidates = list(pkl_path.parent.glob("*person_mapping.txt"))
    if not candidates and pkl_path.parent.parent.exists():
        candidates = list(pkl_path.parent.parent.glob("**/*person_mapping.txt"))
    mapping = {}
    if candidates:
        for line in candidates[0].read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                mapping[k.strip()] = v.strip().lower()
    return mapping


def _normalize_target_person(target_person):
    s = str(target_person).strip().lower()
    return {
        "aria1": "aria01", "aria_1": "aria01", "aria-1": "aria01",
        "aria2": "aria02", "aria_2": "aria02", "aria-2": "aria02",
    }.get(s, s)


def resolve_target_pt_path(pkl_path, target_person):
    p = Path(pkl_path)
    if not target_person or p.suffix.lower() not in (".pt", ".pth"):
        return p
    target_str = _normalize_target_person(target_person)
    if target_str and target_str not in p.stem.lower() and p.parent.exists():
        # ponytail: resolve target sibling .pt once up front before callers log or load stale path
        direct = p.with_name(re.sub(r"aria[_-]?0?[12]", target_str, p.name, flags=re.IGNORECASE))
        if direct != p and direct.exists():
            return direct
        candidates = sorted(p.parent.glob(f"*{target_str}*{p.suffix}"))
        if candidates:
            return candidates[0]
    return p


def resolve_person_payload(wham_data, pkl_path=None, target_person=None):
    if target_person is not None and pkl_path is not None:
        resolved_pt = resolve_target_pt_path(pkl_path, target_person)
        if resolved_pt != Path(pkl_path):
            wham_data = load_joblib_compat(resolved_pt)
            pkl_path = resolved_pt

    # ponytail: handle GVHMR HMR4D prediction dict
    if isinstance(wham_data, dict) and "smpl_params_incam" in wham_data:
        target_str = _normalize_target_person(target_person) if target_person else None
        subj = str(wham_data.get("subject", "")).lower()
        if target_str and subj and target_str != subj:
            return None
        return _gvhmr_to_payload(wham_data, pkl_path=pkl_path, target_str=target_str)

    if not isinstance(wham_data, dict) and "defaultdict" not in str(type(wham_data)):
        if isinstance(wham_data, list):
            for v in wham_data:
                if isinstance(v, dict):
                    return v
        return None

    if target_person is not None:
        target_str = _normalize_target_person(target_person)

        if pkl_path is not None:
            mapping = parse_person_mapping(pkl_path)
            candidate_tracks = []
            for track_str, person_name in mapping.items():
                if person_name == target_str:
                    track_id = int(track_str) if track_str.isdigit() else track_str
                    if track_id in wham_data:
                        candidate_tracks.append(wham_data[track_id])
                    elif str(track_id) in wham_data:
                        candidate_tracks.append(wham_data[str(track_id)])
            if candidate_tracks:
                return max(candidate_tracks, key=lambda v: len(v.get("pose", [])) if isinstance(v, dict) else 0)
            if mapping:
                # ponytail: strict identity - mapping exists but target is missing, never substitute another person
                return None

        if target_person in wham_data:
            return wham_data[target_person]
        if target_str in wham_data:
            return wham_data[target_str]
        if target_str.isdigit() and int(target_str) in wham_data:
            return wham_data[int(target_str)]

        # ponytail: strict identity - do NOT guess or substitute another person when target_person is specified
        return None

    if 0 in wham_data:
        return wham_data[0]
    if "0" in wham_data:
        return wham_data["0"]
    best_person = None
    max_frames = -1
    for k, v in wham_data.items():
        if isinstance(v, dict) and "pose" in v:
            flen = len(v["pose"])
            if flen > max_frames:
                max_frames = flen
                best_person = v
    if best_person is not None:
        return best_person
    for v in wham_data.values():
        if isinstance(v, dict):
            return v
    return None


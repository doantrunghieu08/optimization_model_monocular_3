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
    with open(filepath, 'rb') as f:
        return joblib.load(f)


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


def resolve_person_payload(wham_data, pkl_path=None, target_person=None):
    if not isinstance(wham_data, dict) and "defaultdict" not in str(type(wham_data)):
        if isinstance(wham_data, list):
            for v in wham_data:
                if isinstance(v, dict):
                    return v
        return None

    if target_person is not None:
        target_str = str(target_person).strip().lower()
        if pkl_path is not None:
            mapping = parse_person_mapping(pkl_path)
            for track_str, person_name in mapping.items():
                if person_name == target_str:
                    track_id = int(track_str) if track_str.isdigit() else track_str
                    if track_id in wham_data:
                        return wham_data[track_id]
                    if str(track_id) in wham_data:
                        return wham_data[str(track_id)]
        if target_person in wham_data:
            return wham_data[target_person]
        if target_str in wham_data:
            return wham_data[target_str]
        if target_str.isdigit() and int(target_str) in wham_data:
            return wham_data[int(target_str)]

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


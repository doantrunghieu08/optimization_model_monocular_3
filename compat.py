import sys
import copy
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


def _payload_frame_ids(payload):
    frame_ids = payload.get("frame_ids")
    if frame_ids is None and isinstance(payload.get("tracking_results_for_reproj"), dict):
        frame_ids = payload["tracking_results_for_reproj"].get("frame_id")
    if frame_ids is None:
        return None
    frame_ids = np.asarray(frame_ids).astype(int).reshape(-1)
    return frame_ids if len(frame_ids) == len(payload.get("pose", [])) else None


def _merge_frame_value(values, lengths, rows):
    if any(value is None for value in values):
        return None
    if all(isinstance(value, dict) for value in values):
        merged = {}
        for key in values[0]:
            if all(key in value for value in values):
                item = _merge_frame_value([value[key] for value in values], lengths, rows)
                if item is not None:
                    merged[key] = item
        return merged or None
    if not all(hasattr(value, "__len__") and len(value) == length for value, length in zip(values, lengths)):
        return None
    selected = [values[candidate][row] for candidate, row in rows]
    if all(isinstance(value, np.ndarray) for value in values):
        return np.asarray(selected)
    if all(isinstance(value, list) for value in values):
        return selected
    try:
        import torch
        if all(torch.is_tensor(value) for value in values):
            return torch.stack(selected)
    except ImportError:
        pass
    return None


def _merge_person_tracks(candidates):
    """Join re-ID fragments by absolute frame id; the longest track wins overlaps."""
    candidates = sorted(candidates, key=lambda value: len(value.get("pose", [])), reverse=True)
    frame_ids = [_payload_frame_ids(value) for value in candidates]
    if len(candidates) == 1 or any(ids is None for ids in frame_ids):
        return candidates[0]

    chosen = {}
    for candidate, ids in enumerate(frame_ids):
        for row, frame_id in enumerate(ids):
            chosen.setdefault(int(frame_id), (candidate, row))
    ordered_ids = sorted(chosen)
    rows = [chosen[frame_id] for frame_id in ordered_ids]
    lengths = [len(ids) for ids in frame_ids]
    merged = copy.deepcopy(candidates[0])
    for key in set.intersection(*(set(value) for value in candidates)):
        value = _merge_frame_value([candidate[key] for candidate in candidates], lengths, rows)
        if value is not None:
            merged[key] = value
    merged["frame_ids"] = np.asarray(ordered_ids)
    return merged


def resolve_person_payload(wham_data, pkl_path=None, target_person=None):
    if not isinstance(wham_data, dict) and "defaultdict" not in str(type(wham_data)):
        if isinstance(wham_data, list):
            for v in wham_data:
                if isinstance(v, dict):
                    return v
        return None

    if target_person is not None:
        target_str = str(target_person).strip().lower()
        if target_str in ("aria1", "aria_1", "aria-1"):
            target_str = "aria01"
        elif target_str in ("aria2", "aria_2", "aria-2"):
            target_str = "aria02"

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
                return _merge_person_tracks(candidate_tracks)
            if mapping:
                # ponytail: strict identity - mapping exists but target is missing, never substitute another person
                return None

        if target_person in wham_data:
            return wham_data[target_person]
        if target_str in wham_data:
            return wham_data[target_str]
        if target_str.isdigit() and int(target_str) in wham_data:
            return wham_data[int(target_str)]
        if target_str.startswith("s") and target_str[1:].isdigit() and 0 in wham_data:
            return wham_data[0]

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


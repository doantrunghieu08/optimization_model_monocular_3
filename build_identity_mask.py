"""Build a per-frame identity-margin mask for multi-person WHAM inputs.

WHAM tracks can switch person mid-sequence while *_person_mapping.txt assigns one
person per track. For every camera/person this script regresses the joints of the
track the pipeline would resolve and stores, per GT frame id,

    margin = PA(pred, GT_own) - PA(pred, GT_other)   [mm]

A large positive margin means the track follows the other person. The evaluator
excludes frames with margin > evaluation.identity_mask.margin_mm. Uses GT, so it is
an evaluation-set filter only; it never feeds the fusion.

pkl / ground_truth_dir may be Google Drive File IDs: preprocessing_data.label_persons
then downloads them and writes pkl_local / ground_truth_dir_local (absolute paths) into
the same YAML, which are read here.

Usage: python build_identity_mask.py configs/brute_force.yml configs/identity_mask_hugging_002.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from compat import load_joblib_compat, resolve_person_payload
from evaluation_pipeline.executor import _compute_pa_mpjpe, _load_ground_truth, _parse_new_gt
from keypoints_map import load_keypoints3d_map

ROOT = Path(__file__).resolve().parent


def input_path(entry, key, brute_cfg_path):
    """<key> when it is a path on this machine (relative to the repo root), otherwise
    <key>_local, written by preprocessing_data.label_persons for a Drive File ID."""
    value = entry.get(key)
    if value and (ROOT / str(value)).exists():
        return ROOT / str(value)
    local = entry.get(f"{key}_local")
    if local and Path(local).exists():
        return Path(local)
    raise FileNotFoundError(f"{key}: '{value}' is not a local path and has no existing {key}_local; "
                            f"run: python -m preprocessing_data.label_persons {brute_cfg_path}")


def build(brute_cfg_path, persons=None):
    seg = yaml.safe_load(Path(brute_cfg_path).read_text(encoding="utf-8"))["segments"][0]
    kmap = load_keypoints3d_map(str(ROOT / "configs/keypoints3D_map.yml"))
    names = {k["name"] for k in kmap["keypoints"]}
    reg_idx = {k["name"]: k["regressor_index"] for k in kmap["keypoints"]}
    keys = [k for k in kmap["priority1"] if k != "neck"]
    J = np.load(ROOT / "models/J_regressor_body25_extended.npy")
    truth_dir = input_path(seg, "ground_truth_dir", brute_cfg_path)
    if persons is None:
        from preprocessing_data.label_persons import gt_persons
        persons = tuple(gt_persons(truth_dir) or ("aria01", "aria02"))
    gt = {p: {f: _parse_new_gt(it, names, kmap)
              for f, it in _load_ground_truth(truth_dir / "_.json", truth_dir=truth_dir, target_person=p).items()}
          for p in persons}
    out = {}
    for cam in seg["cameras"]:
        pkl = input_path(cam, "pkl", brute_cfg_path)
        raw = load_joblib_compat(pkl)
        stem = pkl.stem
        out[stem] = {}
        for person in persons:
            others = [p for p in persons if p != person]
            track = resolve_person_payload(raw, pkl_path=pkl, target_person=person)
            margins = {}
            if track is not None and others:
                for row, fid in enumerate(np.asarray(track["frame_ids"]).astype(int)):
                    if fid not in gt[person]:
                        continue
                    other_gts = [gt[o][fid] for o in others if fid in gt[o]]
                    if not other_gts:
                        continue
                    jt = J @ np.asarray(track["verts"][row])
                    pred = {n: jt[i] for n, i in reg_idx.items()}
                    own = _compute_pa_mpjpe(pred, gt[person][fid], keys)[0]
                    # ponytail: N-person margin = PA(own) - min_{other != own} PA(other)
                    oth = min(_compute_pa_mpjpe(pred, og, keys)[0] for og in other_gts)
                    margins[str(int(fid))] = round(own - oth, 2)
            out[stem][person] = margins
            bad = sum(m > 10 for m in margins.values())
            print(f"{cam['id']} {person}: {bad}/{len(margins)} frames margin>10mm")
    return {"description": "margin_mm = PA(own) - min_other PA(other); exclude frames with margin > margin_mm",
            "source": str(brute_cfg_path), "margins": out}


if __name__ == "__main__":
    result = build(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", sys.argv[2])

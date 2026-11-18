"""Label WHAM tracks with Harmony4D person ids (aria01 / aria02 / ...) when no
*_person_mapping.txt exists.

GT USE — read before using: Harmony4D sequences such as 043_Grappling2 ship no 2D
boxes/poses, so the only identity reference is the 3D GT (processed_data/poses3d).
This tool therefore uses 3D GT, but only to NAME tracks:

* every WHAM track gets ONE label (majority vote over confident frames), exactly the
  per-track format the pipeline already reads. Tracker errors (identity switches inside
  a track, fragments) are NOT repaired, so the fusion input stays realistic;
* frames where a labelled track follows the wrong person are reported (JSON) and are
  removed from evaluation by build_identity_mask.py, like the Hugging identity mask;
* the fusion stages never read GT.

Per frame:  pa_p = PA-MPJPE(track joints, GT_p) on the 12 evaluated joints.
A frame votes for p if pa_p is the smallest, pa_p <= max_pa_mm and it beats the
runner-up by more than margin_mm. A track is labelled p if p wins at least
min_vote_fraction of the track's frames with GT; otherwise it is left unlabelled
(bystander / too ambiguous).

Usage:
  python -m preprocessing_data.label_persons input/043_Grappling2 \
      --brute-config configs/brute_force_grappling2_043.yml [--overwrite]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from compat import load_joblib_compat
from evaluation_pipeline.executor import _compute_pa_mpjpe, _load_ground_truth, _parse_new_gt
from keypoints_map import load_keypoints3d_map

ROOT = Path(__file__).resolve().parents[1]


def _segments(labels):
    """[(label, first_fid, last_fid, n)] for consecutive equal labels."""
    out = []
    for fid, lab in labels:
        if out and out[-1][0] == lab:
            out[-1][2] = fid
            out[-1][3] += 1
        else:
            out.append([lab, fid, fid, 1])
    return [tuple(s) for s in out]


def label_track(track, gt, persons, joints_fn, keys, margin_mm, max_pa_mm):
    per_frame = []
    for row, fid in enumerate(np.asarray(track["frame_ids"]).astype(int)):
        if not all(fid in gt[p] for p in persons):
            continue
        pred = joints_fn(np.asarray(track["verts"][row]))
        pa = {p: _compute_pa_mpjpe(pred, gt[p][fid], keys)[0] for p in persons}
        order = sorted(persons, key=pa.get)
        best, second = order[0], order[1] if len(order) > 1 else None
        if pa[best] > max_pa_mm:
            vote = "none"
        elif second is not None and pa[second] - pa[best] <= margin_mm:
            vote = "ambiguous"
        else:
            vote = best
        per_frame.append((int(fid), vote, {p: round(float(v), 1) for p, v in pa.items()}))
    return per_frame


def summarize_track(per_frame, persons, min_vote_fraction):
    n = len(per_frame)
    counts = {lab: sum(v == lab for _, v, _ in per_frame) for lab in persons + ["ambiguous", "none"]}
    label = None
    if n:
        best = max(persons, key=lambda p: counts[p])
        if counts[best] / n >= min_vote_fraction:
            label = best
    wrong = [fid for fid, v, _ in per_frame if label and v in persons and v != label]
    return {"label": label, "frames_with_gt": n, "votes": counts,
            "frames_following_other_person": wrong,
            "segments": _segments([(fid, v) for fid, v, _ in per_frame])}


def discover_cameras(dataset_dir):
    cams = []
    for pkl in sorted(Path(dataset_dir).glob("exo/cam*/*_wham/wham_*.pkl")):
        cam = pkl.parts[-3]
        # Prefer exo/camXX/images/*.mp4; fall back to duplicated folders like "images (1)".
        videos = sorted((pkl.parents[1] / "images").glob("*.mp4")) or sorted(pkl.parents[1].glob("images*/*.mp4"))
        cams.append({"id": cam, "pkl": pkl, "video": videos[0] if videos else None})
    return cams


def run(dataset_dir, persons=("aria01", "aria02"), margin_mm=10.0, max_pa_mm=120.0,
        min_vote_fraction=0.5, overwrite=False, brute_config=None):
    dataset_dir = Path(dataset_dir).resolve()
    persons = list(persons)
    kmap = load_keypoints3d_map(str(ROOT / "configs/keypoints3D_map.yml"))
    names = {k["name"] for k in kmap["keypoints"]}
    reg_idx = {k["name"]: k["regressor_index"] for k in kmap["keypoints"]}
    keys = [k for k in kmap["priority1"] if k != "neck"]
    J = np.load(ROOT / "models/J_regressor_body25_extended.npy")

    def joints_fn(verts):
        jt = J @ verts
        return {n: jt[i] for n, i in reg_idx.items()}

    truth_dir = dataset_dir / "processed_data" / "poses3d"
    gt = {p: {f: _parse_new_gt(it, names, kmap)
              for f, it in _load_ground_truth(truth_dir / "_.json", truth_dir=truth_dir, target_person=p).items()}
          for p in persons}
    report = {"dataset": str(dataset_dir), "persons": persons, "gt_used": "processed_data/poses3d (naming only)",
              "params": {"margin_mm": margin_mm, "max_pa_mm": max_pa_mm, "min_vote_fraction": min_vote_fraction},
              "cameras": {}}
    cams = discover_cameras(dataset_dir)
    for cam in cams:
        raw = load_joblib_compat(cam["pkl"])
        tracks = {}
        for tid, track in raw.items():
            per_frame = label_track(track, gt, persons, joints_fn, keys, margin_mm, max_pa_mm)
            tracks[str(tid)] = summarize_track(per_frame, persons, min_vote_fraction)
        mapping = {tid: t["label"] for tid, t in tracks.items() if t["label"]}
        stem = cam["pkl"].stem.replace("wham_", "", 1).replace("_ffmpeg", "")
        map_path = cam["pkl"].parent / f"wham_{stem}_person_mapping.txt"
        existing = list(cam["pkl"].parent.glob("*person_mapping.txt"))
        written = False
        if existing and not overwrite:
            print(f"[label] {cam['id']}: keep existing {existing[0].name} (use --overwrite)")
        else:
            map_path.write_text("".join(f"{tid} = {lab}\n" for tid, lab in
                                        sorted(mapping.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])),
                                encoding="utf-8")
            written = True
        (cam["pkl"].parent / f"wham_{stem}_person_labels.json").write_text(json.dumps(tracks, indent=1), encoding="utf-8")
        coverage = {p: sorted({fid for tid, t in tracks.items() if t["label"] == p
                               for lab, a, b, n in t["segments"] for fid in range(a, b + 1)}) for p in persons}
        report["cameras"][cam["id"]] = {"mapping": mapping, "mapping_written": written,
                                        "frames_covered": {p: len(v) for p, v in coverage.items()},
                                        "switch_frames": {tid: len(t["frames_following_other_person"])
                                                          for tid, t in tracks.items() if t["label"]}}
        print(f"[label] {cam['id']}: {mapping}  covered={report['cameras'][cam['id']]['frames_covered']}")
    (dataset_dir / "person_labels_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if brute_config:
        seg_name = re.sub(r"[^0-9a-z]+", "_", dataset_dir.name.lower()).strip("_")
        cameras = [{"id": c["id"], "pkl": str(c["pkl"].relative_to(ROOT)).replace("\\", "/"),
                    "video": str(c["video"].relative_to(ROOT)).replace("\\", "/") if c["video"] else None}
                   for c in cams]
        # One segment per person so a single brute-force run covers every labelled subject.
        cfg = {"segments": [{
            "name": f"{seg_name}_{p}",
            "target_person": p,
            "ground_truth_dir": str(truth_dir.relative_to(ROOT)).replace("\\", "/"),
            "cameras": cameras,
        } for p in persons]}
        Path(brute_config).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"[label] wrote {brute_config}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset_dir")
    ap.add_argument("--persons", nargs="+", default=["aria01", "aria02"])
    ap.add_argument("--margin-mm", type=float, default=10.0)
    ap.add_argument("--max-pa-mm", type=float, default=120.0)
    ap.add_argument("--min-vote-fraction", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--brute-config", default=None, help="also write a brute-force segment config")
    a = ap.parse_args()
    run(a.dataset_dir, a.persons, a.margin_mm, a.max_pa_mm, a.min_vote_fraction, a.overwrite, a.brute_config)


if __name__ == "__main__":
    main()

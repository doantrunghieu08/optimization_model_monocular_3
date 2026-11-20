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

Input (first argument) is either

* a dataset folder: cameras are found at exo/cam*/*_wham/wham_*.pkl, GT at
  processed_data/poses3d, the mapping is written next to each pkl; or
* a brute-force YAML (configs/brute_force_*.yml). Each `ground_truth_dir`, `pkl` and
  `video` is a local path (relative to the repo root) OR a Google Drive File ID / link:
      ground_truth_dir: GT folder | local .zip | Drive ID of a zip holding poses3d/*.npy
      cameras[].pkl:    WHAM pkl  | Drive ID of the pkl
      cameras[].video:  video     | Drive ID of the video (optional; downloaded, not read)
  Segments sharing a ground_truth_dir are one sequence, labelled with their
  target_person values (--persons overrides). Drive files and GT zips are KEPT under
  <data-dir>/<sequence>/ in the dataset-folder layout, because build_identity_mask.py
  and the pipeline read the pkl and its mapping again later:
      exo/<cam id>/<stem>_wham/wham_<stem>_ffmpeg.pkl  (+ _person_mapping.txt, _person_labels.json)
      exo/<cam id>/images/<video>.mp4
      processed_data/poses3d/*.npy
  Downloads are one at a time and reused on a rerun (same Drive ID). Local files are
  only read. The mapping is always written next to the pkl. Finally, in the SAME YAML,
  `<key>_local: "<absolute path>"` is added right under each Drive ID / GT zip value
  (only those lines change; comments and anchors stay), e.g.
      pkl: 1-YLUt9UZNVuwUw3si1aLFENMIw-qAXY1
      pkl_local: "/.../inputs/drive/005_sword3/exo/cam01/..._wham/wham_..._ffmpeg.pkl"
  build_identity_mask.py reads <key>_local whenever <key> is not a local path.

Usage:
  python -m preprocessing_data.label_persons input/043_Grappling2 \
      --brute-config configs/brute_force_grappling2_043.yml [--overwrite]
  python -m preprocessing_data.label_persons configs/brute_force_005_sword3.yml \
      [--data-dir inputs/drive] [--overwrite]
  python build_identity_mask.py configs/brute_force_005_sword3.yml configs/identity_mask_005_sword3.json
"""
import argparse
import gc
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import yaml

from compat import load_joblib_compat
from evaluation_pipeline.executor import _compute_pa_mpjpe, _load_ground_truth, _parse_new_gt
from keypoints_map import load_keypoints3d_map

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSONS = ("aria01", "aria02")
DEFAULT_DATA_DIR = ROOT / "inputs" / "drive"
GT_USED = "processed_data/poses3d (naming only)"
DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{20,}")
DRIVE_LINK = re.compile(r"drive\.google\.com/\S*?(?:/d/|[?&]id=)([A-Za-z0-9_-]{20,})")
DRIVE_MARKER = ".drive_source.json"  # in exo/<cam id>/: Drive IDs of the kept pkl / video
GT_MARKER = ".gt_source"             # in processed_data/: which zip poses3d/ was extracted from
# YAML keys that may hold a Drive File ID; label_persons writes <key>_local (absolute path)
# next to them, which build_identity_mask.py reads.
LOCAL_KEYS = ("ground_truth_dir", "pkl", "video")


# ------------------------------------------------------------------ voting

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


class Labeller:
    """Joint regressor, GT parsing and voting parameters shared by both input modes."""

    def __init__(self, margin_mm=10.0, max_pa_mm=120.0, min_vote_fraction=0.5):
        self.kmap = load_keypoints3d_map(str(ROOT / "configs/keypoints3D_map.yml"))
        self.names = {k["name"] for k in self.kmap["keypoints"]}
        self.reg_idx = {k["name"]: k["regressor_index"] for k in self.kmap["keypoints"]}
        self.keys = [k for k in self.kmap["priority1"] if k != "neck"]
        self.J = np.load(ROOT / "models/J_regressor_body25_extended.npy")
        self.params = {"margin_mm": margin_mm, "max_pa_mm": max_pa_mm,
                       "min_vote_fraction": min_vote_fraction}

    def joints(self, verts):
        jt = self.J @ verts
        return {n: jt[i] for n, i in self.reg_idx.items()}

    def load_gt(self, truth_dir, persons):
        truth_dir = Path(truth_dir)
        gt = {p: {f: _parse_new_gt(it, self.names, self.kmap)
                  for f, it in _load_ground_truth(truth_dir / "_.json", truth_dir=truth_dir,
                                                  target_person=p).items()}
              for p in persons}
        for p in persons:
            if not gt[p]:
                print(f"[label] WARNING: no GT frame for '{p}' in {truth_dir} -> no track can be labelled")
        return gt

    def label(self, raw, gt, persons):
        """{track id: summary} for every track of a loaded WHAM pkl."""
        p = self.params
        return {str(tid): summarize_track(
                    label_track(track, gt, persons, self.joints, self.keys, p["margin_mm"], p["max_pa_mm"]),
                    persons, p["min_vote_fraction"])
                for tid, track in raw.items()}


# ------------------------------------------------------------------ outputs

def pkl_stem(pkl):
    """wham_harmony4D_043_grappling2_cam01_ffmpeg.pkl -> harmony4D_043_grappling2_cam01."""
    return Path(pkl).stem.replace("wham_", "", 1).replace("_ffmpeg", "")


def _track_order(tid):
    return (not tid.isdigit(), int(tid) if tid.isdigit() else 0, tid)


def write_camera_outputs(tracks, persons, out_dir, stem, overwrite):
    """Write wham_<stem>_person_mapping.txt (an existing mapping is kept unless
    overwrite) and wham_<stem>_person_labels.json; return the camera's report entry."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {tid: t["label"] for tid, t in tracks.items() if t["label"]}
    existing = sorted(out_dir.glob("*person_mapping.txt"))
    written = False
    if existing and not overwrite:
        print(f"[label]   keep existing {existing[0].name} (use --overwrite)")
    else:
        (out_dir / f"wham_{stem}_person_mapping.txt").write_text(
            "".join(f"{tid} = {lab}\n" for tid, lab in sorted(mapping.items(), key=lambda kv: _track_order(kv[0]))),
            encoding="utf-8")
        written = True
    (out_dir / f"wham_{stem}_person_labels.json").write_text(json.dumps(tracks, indent=1), encoding="utf-8")
    coverage = {p: sorted({fid for t in tracks.values() if t["label"] == p
                           for lab, a, b, n in t["segments"] for fid in range(a, b + 1)}) for p in persons}
    return {"mapping": mapping, "mapping_written": written,
            "frames_covered": {p: len(v) for p, v in coverage.items()},
            "switch_frames": {tid: len(t["frames_following_other_person"])
                              for tid, t in tracks.items() if t["label"]}}


# ------------------------------------------------------------------ dataset folder input

def discover_cameras(dataset_dir):
    cams = []
    for pkl in sorted(Path(dataset_dir).glob("exo/cam*/*_wham/wham_*.pkl")):
        cam = pkl.parts[-3]
        # Prefer exo/camXX/images/*.mp4; fall back to duplicated folders like "images (1)".
        videos = sorted((pkl.parents[1] / "images").glob("*.mp4")) or sorted(pkl.parents[1].glob("images*/*.mp4"))
        cams.append({"id": cam, "pkl": pkl, "video": videos[0] if videos else None})
    return cams


def run(dataset_dir, persons=DEFAULT_PERSONS, margin_mm=10.0, max_pa_mm=120.0,
        min_vote_fraction=0.5, overwrite=False, brute_config=None):
    dataset_dir = Path(dataset_dir).resolve()
    persons = list(persons)
    labeller = Labeller(margin_mm, max_pa_mm, min_vote_fraction)
    truth_dir = dataset_dir / "processed_data" / "poses3d"
    gt = labeller.load_gt(truth_dir, persons)
    report = {"dataset": str(dataset_dir), "persons": persons, "gt_used": GT_USED,
              "params": labeller.params, "cameras": {}}
    cams = discover_cameras(dataset_dir)
    for cam in cams:
        tracks = labeller.label(load_joblib_compat(cam["pkl"]), gt, persons)
        entry = write_camera_outputs(tracks, persons, cam["pkl"].parent, pkl_stem(cam["pkl"]), overwrite)
        report["cameras"][cam["id"]] = entry
        print(f"[label] {cam['id']}: {entry['mapping']}  covered={entry['frames_covered']}")
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


# ------------------------------------------------------------------ local path or Google Drive

def resolve_source(value, what):
    """('local', Path) for an existing path (relative paths: repo root), otherwise
    ('drive', file_id) for a Google Drive File ID or link."""
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{what}: empty value")
    path = Path(text).expanduser()
    path = path if path.is_absolute() else ROOT / path
    if path.exists():
        return "local", path.resolve()
    link = DRIVE_LINK.search(text)
    if link:
        return "drive", link.group(1)
    if DRIVE_ID.fullmatch(text):
        return "drive", text
    raise FileNotFoundError(f"{what}: '{text}' is neither an existing path nor a Google Drive File ID")


def config_path_str(path):
    """Path as written in a brute-force YAML: relative to the repo root when inside it."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def describe_source(source):
    kind, ref = source
    return f"drive:{ref}" if kind == "drive" else config_path_str(ref)


def download_drive_file(file_id, dest_dir, what):
    """Download a Drive file shared 'Anyone with the link' into dest_dir, keeping its
    Drive file name; return its path."""
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Google Drive File IDs need gdown: pip install gdown") from exc
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {what}: Drive ID {file_id}")
    out = gdown.download(id=file_id, output=str(dest_dir) + os.sep, quiet=False)
    path = Path(out).resolve() if out else None
    if path is None or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"cannot download {what} ({file_id}); check the ID and that the file "
                           "is shared 'Anyone with the link'")
    if dest_dir not in path.parents:
        raise RuntimeError(f"Drive file name leaves the download folder: {out}")
    print(f"[download] {path.name}: {path.stat().st_size / 2**20:.1f} MB")
    return path


def find_npy_dir(root):
    """Folder holding the GT *.npy: prefer one named poses3d, then the shallowest."""
    dirs = {p.parent for p in Path(root).rglob("*.npy") if "__MACOSX" not in p.parts}
    return min(dirs, key=lambda d: (d.name != "poses3d", len(d.parts), str(d))) if dirs else None


def extract_gt_zip(zip_path, dest):
    """Extract a GT zip (poses3d/*.npy, processed_data/poses3d/*.npy or *.npy at its
    root) into dest; return the folder holding the *.npy."""
    root = Path(dest).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.infolist() if not m.filename.startswith("__MACOSX/")]
            for m in members:
                target = (root / m.filename).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"unsafe path in GT zip: {m.filename}")
            zf.extractall(root, members)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"GT is not a valid zip: {Path(zip_path).name}") from exc
    truth_dir = find_npy_dir(root)
    if truth_dir is None:
        raise ValueError(f"no .npy file in GT zip {Path(zip_path).name}")
    return truth_dir


def fetch_ground_truth(source, seq_dir):
    """Folder of GT *.npy. A local folder is used in place. A zip (Drive ID or local
    file) is extracted to seq_dir/processed_data/poses3d and kept there for
    build_identity_mask.py; it is reused while processed_data/.gt_source names the
    same zip. The downloaded zip itself is deleted."""
    kind, ref = source
    if kind == "local" and ref.is_dir():
        return ref
    poses_dir = seq_dir / "processed_data" / "poses3d"
    marker = poses_dir.parent / GT_MARKER
    if kind == "drive":
        tag = f"drive:{ref}"
    else:
        st = ref.stat()
        tag = f"local:{ref}|{st.st_size}|{st.st_mtime_ns}"
    if poses_dir.exists():
        if marker.is_file() and marker.read_text(encoding="utf-8") == tag and any(poses_dir.glob("*.npy")):
            print(f"[cache] GT: reuse {config_path_str(poses_dir)}")
            return poses_dir
        if not marker.is_file():  # never replace a GT folder this tool did not create
            raise FileExistsError(f"{poses_dir} exists but was not extracted by label_persons; "
                                  "use another --data-dir")
    seq_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gt_", dir=seq_dir) as tmp:
        tmp = Path(tmp)
        zip_path = download_drive_file(ref, tmp / "zip", "GT zip") if kind == "drive" else ref
        print(f"[gt] extracting {zip_path.name}")
        npy_dir = extract_gt_zip(zip_path, tmp / "x")
        shutil.rmtree(poses_dir, ignore_errors=True)
        marker.unlink(missing_ok=True)
        poses_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(npy_dir), str(poses_dir))
    marker.write_text(tag, encoding="utf-8")
    print(f"[gt] kept {config_path_str(poses_dir)}")
    return poses_dir


def _read_drive_marker(cam_dir):
    """{'pkl' | 'video': {'id': Drive ID, 'path': path under cam_dir}} of the kept downloads."""
    try:
        info = json.loads((cam_dir / DRIVE_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in info.items() if isinstance(v, dict) and "id" in v and "path" in v} \
        if isinstance(info, dict) else {}


def fetch_drive_file(file_id, cam_dir, cam_id, kind):
    """Keep a camera's Drive file in the dataset-folder layout (kept, not deleted:
    build_identity_mask.py and the pipeline read it later):
        kind 'pkl':   cam_dir/<stem>_wham/<Drive file name>  (the mapping goes next to it)
        kind 'video': cam_dir/images/<Drive file name>
    Reused while cam_dir/.drive_source.json names the same ID. When the ID changed, the
    old file is removed first (for a pkl: its whole <stem>_wham folder, mapping included)."""
    markers = _read_drive_marker(cam_dir)
    old = markers.get(kind)
    if old is not None:
        old_path = (cam_dir / old["path"]).resolve()
        if old["id"] == file_id and old_path.is_file() and old_path.stat().st_size > 0:
            print(f"[cache] {cam_id} {kind}: reuse {config_path_str(old_path)}")
            return old_path
        if old["id"] != file_id:
            stale = old_path.parent if kind == "pkl" else old_path
            print(f"[cache] {cam_id} {kind}: Drive ID changed ({old['id']} -> {file_id}); "
                  f"removing {config_path_str(stale)}")
            if cam_dir.resolve() in stale.parents:  # only what this tool put in cam_dir
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                else:
                    stale.unlink(missing_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)
    # Download next to the final place, then move: a half-downloaded file is never kept.
    with tempfile.TemporaryDirectory(prefix=".download_", dir=cam_dir) as tmp:
        got = download_drive_file(file_id, Path(tmp), f"{cam_id} {'WHAM pkl' if kind == 'pkl' else kind}")
        dest = cam_dir / (f"{pkl_stem(got)}_wham" if kind == "pkl" else "images") / got.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        got.replace(dest)
    markers[kind] = {"id": file_id, "path": dest.relative_to(cam_dir).as_posix()}
    (cam_dir / DRIVE_MARKER).write_text(json.dumps(markers, indent=1), encoding="utf-8")
    print(f"[download] kept {config_path_str(dest)}")
    return dest


# ------------------------------------------------------------------ YAML input

def _sequence_name(segment):
    """'005_sword3_aria01' -> '005_sword3' (brute-force segments end with _<target_person>)."""
    name = str(segment.get("name") or "sequence")
    suffix = f"_{segment.get('target_person', '')}"
    if name.endswith(suffix) and len(name) > len(suffix):
        name = name[:-len(suffix)]
    return re.sub(r"[^\w.-]+", "_", name)


def optional_source(value):
    """resolve_source for an optional field (video): None when empty or when it is a
    local path missing on this machine (labelling does not need the video)."""
    if value is None or not str(value).strip():
        return None
    try:
        return resolve_source(value, "")
    except FileNotFoundError:
        return None


def load_config(config_path, persons=None):
    """Group the YAML segments by ground_truth_dir into sequences
    {name, gt, persons, cameras: [{id, pkl, video}]}; gt / pkl / video are
    ('local', Path) or ('drive', id) (video may be None). Every ground_truth_dir / pkl is
    checked here, before any download."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    segments = cfg.get("segments") or []
    if not segments:
        raise ValueError(f"{config_path}: no segments")
    groups = {}
    for i, seg in enumerate(segments):
        where = f"segments[{i}] ({seg.get('name', '?')})"
        gt = resolve_source(seg.get("ground_truth_dir"), f"{where}.ground_truth_dir")
        if gt[0] == "local" and gt[1].is_file() and not zipfile.is_zipfile(gt[1]):
            raise ValueError(f"{where}.ground_truth_dir: {gt[1]} is neither a folder nor a zip")
        seq = groups.setdefault(gt, {"name": _sequence_name(seg), "gt": gt, "persons": [], "cameras": {}})
        person = seg.get("target_person")
        if person and str(person) not in seq["persons"]:
            seq["persons"].append(str(person))
        for j, cam in enumerate(seg.get("cameras") or []):
            cam_id = str(cam.get("id") or f"cam{j}")
            pkl = resolve_source(cam.get("pkl"), f"{where}.cameras[{cam_id}].pkl")
            known = seq["cameras"].get(cam_id)
            if known is not None and known["pkl"] != pkl:
                raise ValueError(f"{where}: camera {cam_id} has another pkl than in an earlier "
                                 "segment with the same ground_truth_dir")
            seq["cameras"][cam_id] = {"id": cam_id, "pkl": pkl, "video": optional_source(cam.get("video"))}
    sequences, used = [], set()
    for seq in groups.values():
        name, k = seq["name"], 2
        while seq["name"] in used:  # two sequences with the same segment name -> keep data apart
            seq["name"], k = f"{name}_{k}", k + 1
        used.add(seq["name"])
        seq["persons"] = list(persons) if persons else (seq["persons"] or list(DEFAULT_PERSONS))
        seq["cameras"] = list(seq["cameras"].values())
        sequences.append(seq)
    return sequences


# ------------------------------------------------------------------ <key>_local in the YAML

def _strip_local_keys(obj):
    if isinstance(obj, dict):
        return {k: _strip_local_keys(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.endswith("_local") and k[:-len("_local")] in LOCAL_KEYS)}
    if isinstance(obj, list):
        return [_strip_local_keys(v) for v in obj]
    return obj


def write_local_keys(config_path, local):
    """In the YAML itself, put `<key>_local: "<absolute path>"` right under every
    ground_truth_dir / pkl / video whose value is a Drive File ID (or a GT zip) that this
    run fetched, and drop a stale <key>_local whose <key> is now a local path.
    build_identity_mask.py reads <key>_local when <key> is not a local path.
    Only those lines change: comments, anchors (&id001 / *id001), order are kept.
    `local` maps ('local'|'drive', ref) -> local path. Returns the changes made."""
    path = Path(config_path)
    with open(path, encoding="utf-8", newline="") as f:  # keep \r\n if the file has them
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    edits, changes, seen = [], [], set()

    def line_end(i):  # index just after the end of the line holding index i
        j = text.find("\n", i)
        return len(text) if j < 0 else j + 1

    def visit(node):
        if id(node) in seen:  # an alias (*id001) is the same node: edit it once
            return
        seen.add(id(node))
        if isinstance(node, yaml.SequenceNode):
            for child in node.value:
                visit(child)
            return
        if not isinstance(node, yaml.MappingNode):
            return
        items = {k.value: (k, v) for k, v in node.value if isinstance(k, yaml.ScalarNode)}
        for key in LOCAL_KEYS:
            if key not in items or not isinstance(items[key][1], yaml.ScalarNode):
                continue
            knode, vnode = items[key]
            src = optional_source(vnode.value)
            target = local.get(src) if src is not None else None
            old = items.get(f"{key}_local")
            if target is not None and (src[0] == "drive" or Path(target) != src[1]):
                value = json.dumps(str(target), ensure_ascii=False)
                if old is not None:
                    if old[1].value != str(target):
                        edits.append((old[1].start_mark.index, old[1].end_mark.index, value))
                        changes.append(f"~ {key}_local: {target}")
                elif node.flow_style:
                    print(f"[yaml] WARNING: line {knode.start_mark.line + 1} is a flow mapping; "
                          f"add {key}_local: {value} by hand")
                else:
                    at = line_end(vnode.end_mark.index)
                    head = "" if at == 0 or text[at - 1] == "\n" else nl
                    edits.append((at, at, f"{head}{' ' * knode.start_mark.column}{key}_local: {value}{nl}"))
                    changes.append(f"+ {key}_local: {target}")
            elif old is not None and target is not None and not node.flow_style:
                start = text.rfind("\n", 0, old[0].start_mark.index) + 1  # <key> is local now
                edits.append((start, line_end(old[1].end_mark.index), ""))
                changes.append(f"- {key}_local (stale: {key} is a local path)")
        for _, child in node.value:
            visit(child)

    root = yaml.compose(text)
    if root is not None:
        visit(root)
    if not edits:
        return []
    new = text
    for start, end, repl in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        new = new[:start] + repl + new[end:]
    # Safety net: apart from the *_local keys, the YAML must load exactly as before.
    if _strip_local_keys(yaml.safe_load(text)) != _strip_local_keys(yaml.safe_load(new)):
        raise RuntimeError(f"could not add the *_local keys to {path} safely; the file was not changed")
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(new)
    os.replace(tmp, path)
    return changes


# ------------------------------------------------------------------ YAML input

def run_config(config_path, persons=None, margin_mm=10.0, max_pa_mm=120.0, min_vote_fraction=0.5,
               overwrite=False, data_dir=DEFAULT_DATA_DIR, labeller=None):
    """Label every camera of a brute-force YAML, keep the Drive downloads, then write the
    <key>_local paths into the same YAML. -> (reports, failed)."""
    config_path = Path(config_path).resolve()
    sequences = load_config(config_path, persons)
    labeller = labeller or Labeller(margin_mm, max_pa_mm, min_vote_fraction)
    data_root = Path(data_dir).resolve()
    local = {}  # ('local'|'drive', ref) -> path on this machine
    reports, failed = [], []
    for seq in sequences:
        seq_dir = data_root / seq["name"]
        print("=" * 72)
        print(f"[label] {seq['name']}: GT {describe_source(seq['gt'])}, persons {seq['persons']}, "
              f"{len(seq['cameras'])} camera(s)")
        if len(seq["persons"]) < 2:
            print("[label] WARNING: only one person -> every track close to that GT is labelled with it; "
                  "pass --persons aria01 aria02 to label against all subjects")
        report = {"config": str(config_path), "sequence": seq["name"],
                  "ground_truth": describe_source(seq["gt"]), "persons": seq["persons"],
                  "gt_used": GT_USED, "params": labeller.params, "cameras": {}}
        try:
            truth_dir = fetch_ground_truth(seq["gt"], seq_dir)
            gt = labeller.load_gt(truth_dir, seq["persons"])
            local[seq["gt"]] = truth_dir
            report["local_ground_truth"] = config_path_str(truth_dir)
        except Exception as exc:  # GT download / extraction / loading
            failed.append(f"{seq['name']} (GT)")
            report["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[label] ERROR GT of {seq['name']}: {type(exc).__name__}: {exc}")
            gt = None
        for k, cam in enumerate(seq["cameras"] if gt is not None else [], 1):
            cam_dir = seq_dir / "exo" / cam["id"]
            print(f"[label] {cam['id']} ({k}/{len(seq['cameras'])}): {describe_source(cam['pkl'])}")
            entry = {"pkl": describe_source(cam["pkl"])}
            try:  # download / pkl error: record it, go on with the next camera
                kind, ref = cam["pkl"]
                pkl = ref if kind == "local" else fetch_drive_file(ref, cam_dir, cam["id"], "pkl")
                local[cam["pkl"]] = pkl
                raw = load_joblib_compat(pkl)
                print(f"[label]   {len(raw)} WHAM tracks in {pkl.name}")
                tracks = labeller.label(raw, gt, seq["persons"])
                del raw
                gc.collect()
                entry["local_pkl"] = config_path_str(pkl)
                entry.update(write_camera_outputs(tracks, seq["persons"], pkl.parent, pkl_stem(pkl), overwrite))
                print(f"[label]   {cam['id']}: {entry['mapping']}  covered={entry['frames_covered']}  "
                      f"-> {config_path_str(pkl.parent)}")
            except Exception as exc:
                failed.append(f"{seq['name']}/{cam['id']}")
                entry["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[label]   ERROR {cam['id']}: {type(exc).__name__}: {exc}")
            if cam["video"] is not None:
                try:
                    kind, ref = cam["video"]
                    local[cam["video"]] = ref if kind == "local" else fetch_drive_file(ref, cam_dir, cam["id"], "video")
                    entry["video"] = describe_source(cam["video"])
                    entry["local_video"] = config_path_str(local[cam["video"]])
                except Exception as exc:
                    failed.append(f"{seq['name']}/{cam['id']} (video)")
                    entry["video_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[label]   ERROR {cam['id']} video: {type(exc).__name__}: {exc}")
            report["cameras"][cam["id"]] = entry
        gt = None
        report_path = seq_dir / "person_labels_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"[label] report: {report_path}")
        reports.append(report)

    print("=" * 72)
    try:
        changes = write_local_keys(config_path, local)
    except Exception as exc:
        failed.append("*_local keys")
        print(f"[yaml] ERROR: {type(exc).__name__}: {exc}")
    else:
        print(f"[yaml] {config_path_str(config_path)}: "
              + ("unchanged" if not changes else f"{len(changes)} change(s)"))
        for change in changes:
            print(f"[yaml]   {change}")
    if failed:
        print(f"[label] FAILED: {', '.join(failed)}; rerun the same command (kept downloads are reused)")
    mask = f"configs/identity_mask_{sequences[0]['name']}.json"
    print(f"[label] next: python build_identity_mask.py {config_path_str(config_path)} {mask}")
    if len(sequences) > 1:
        print("[label] note: build_identity_mask.py reads only the first segment's sequence")
    return reports, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="dataset folder, or brute-force YAML whose pkl / video / ground_truth_dir "
                                   "are local paths or Google Drive File IDs")
    ap.add_argument("--persons", nargs="+", default=None,
                    help="default: aria01 aria02 (folder), the target_person values (YAML)")
    ap.add_argument("--margin-mm", type=float, default=10.0)
    ap.add_argument("--max-pa-mm", type=float, default=120.0)
    ap.add_argument("--min-vote-fraction", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--brute-config", default=None, help="folder input: also write a brute-force segment config")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="YAML input: Drive files and extracted GT zips are kept in <data-dir>/<sequence>/")
    a = ap.parse_args()
    source = Path(a.source)
    if source.suffix.lower() in (".yml", ".yaml") or source.is_file():
        if a.brute_config:
            ap.error("--brute-config applies to a dataset folder; a YAML gets <key>_local keys instead")
        try:  # config errors (missing local path, conflicting cameras) stop before any download
            _, failed = run_config(source, a.persons, a.margin_mm, a.max_pa_mm, a.min_vote_fraction,
                                   a.overwrite, a.data_dir)
        except (ValueError, FileNotFoundError) as exc:
            ap.error(str(exc))
        if failed:
            raise SystemExit(1)
    else:
        run(source, a.persons or DEFAULT_PERSONS, a.margin_mm, a.max_pa_mm, a.min_vote_fraction,
            a.overwrite, a.brute_config)


if __name__ == "__main__":
    main()

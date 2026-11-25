"""Supervised calibration of partwise weights from the v4 Sheets export.

GT is used for track/time association and training targets, never for inference.
Alignment reports must be reviewed before interpreting validation scores.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from compat import patch_numpy_and_inspect
from fusion_pipeline.aotr import _ordered_bones
from fusion_pipeline.partwise import (
    FEATURE_NAMES, blend_frame, eval_weight_model, extract_pair_features,
    fuse_partwise, slave_directions,
)
from keypoints_map import load_keypoints3d_map

patch_numpy_and_inspect()

ROOT = Path(__file__).resolve().parent
BODY = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle")


def validate_rows(rows):
    groups, seen, gt_splits = {}, set(), {}
    for row in rows:
        action, cam, split = (str(row.get(k, "")).strip() for k in ("action", "cam", "split"))
        if not action or not cam.startswith("cam") or not cam[3:].isdigit():
            raise ValueError(f"Invalid action/camera: {row}")
        if action.endswith("_" + cam) or split not in {"train", "val"}:
            raise ValueError(f"Normalize action and split first: {action}/{cam}/{split}")
        if (action, cam) in seen:
            raise ValueError(f"Duplicate camera: {action}/{cam}")
        seen.add((action, cam))
        if not row.get("file PKL") or not row.get("GT"):
            raise ValueError(f"Missing PKL/GT: {action}/{cam}")
        for key, mapping in ((action, groups), (str(row["GT"]), gt_splits)):
            previous = mapping.setdefault(key, split)
            if previous != split:
                raise ValueError(f"Train/val leakage: {key}")
    if set(groups.values()) != {"train", "val"}:
        raise ValueError("Both train and val sequences are required")
    for action in groups:
        if sum(r["action"] == action for r in rows) < 2:
            raise ValueError(f"Need two cameras: {action}")
    return groups


def file_id(value):
    import re
    value = str(value).strip()
    if "/" in value or "?" in value:
        match = re.search(r"/d/([\w-]+)|[?&]id=([\w-]+)", value)
        if not match:
            raise ValueError(f"Not a Drive file ID/URL: {value}")
        return next(x for x in match.groups() if x)
    if not re.fullmatch(r"[\w-]{10,}", value):
        raise ValueError(f"Invalid Drive file ID: {value}")
    return value


SKIP_SCAN_DIRS = frozenset({
    "images", "images (1)", "color", "depth", "colmap", "sparse", "dense",
    ".git", "__pycache__", ".ipynb_checkpoints", "node_modules", "Trash", ".Trash",
})


def _xattr_drive_id(path):
    if not hasattr(os, "getxattr"):
        return None
    try:
        return os.getxattr(path, "user.drive.id").decode("utf-8").strip()
    except OSError:
        return None


def index_mounted_files(wanted_ids, data_dir, exclude_dir=None):
    """Scan an explicit dataset folder via user.drive.id xattr (never crawls all of MyDrive)."""
    root = Path(data_dir)
    if not data_dir or not root.is_dir():
        return {}
    wanted = set(wanted_ids)
    found = {}
    excl = os.path.realpath(exclude_dir) if exclude_dir and Path(exclude_dir).exists() else None
    visited = set()
    for dirpath, dirs, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited or (excl and (real == excl or real.startswith(excl + os.sep))):
            dirs[:] = []
            continue
        visited.add(real)
        dirs[:] = [d for d in dirs if d not in SKIP_SCAN_DIRS and not d.startswith(".")]
        for fname in filenames:
            if not fname.endswith((".pkl", ".npz")):
                continue
            full = os.path.join(dirpath, fname)
            did = _xattr_drive_id(full)
            if did in wanted:
                found[did] = Path(full)
                if len(found) == len(wanted):
                    return found
    return found


def resolve_mounted(drive, value, mount, state=None):
    """Resolve a Drive file ID/URL to a local path on mounted Google Drive (0 media downloads)."""
    raw = str(value).strip()
    direct = Path(raw)
    if direct.is_file():
        return direct
    mount = Path(mount)
    if not mount.exists():
        raise FileNotFoundError(f"Google Drive chưa được mount tại {mount}; hãy chạy drive.mount('/content/drive') trước.")
    fid = file_id(raw)
    state = state if state is not None else {}
    files_map = state.setdefault("files", {})
    if fid in files_map and files_map[fid].is_file():
        return files_map[fid]
    # ponytail: never os.walk entire MyDrive over FUSE (hangs on thousands of dirs); scan only explicit data_dir
    if not state.get("indexed") and state.get("wanted_ids") and str(state.get("data_dir", "")).strip():
        files_map.update(index_mounted_files(
            state["wanted_ids"], state["data_dir"], state.get("exclude_dir")))
        state["indexed"] = True
        if fid in files_map and files_map[fid].is_file():
            return files_map[fid]
    if drive is None:
        by_id = mount / ".shortcut-targets-by-id" / fid
        if by_id.is_file():
            files_map[fid] = by_id
            return by_id
        raise FileNotFoundError(f"Không tìm thấy Drive ID {fid} trên {mount} (thiếu Drive metadata client).")

    # ponytail: metadata-only parent walk (0 bytes downloaded -> never hits Drive download quota)
    meta_cache = state.setdefault("meta", {})
    folders = state.setdefault("folders", {})
    if "root_id" not in state:
        state["root_id"] = drive.files().get(fileId="root", fields="id").execute(num_retries=5)["id"]
        folders[state["root_id"]] = mount / "MyDrive"

    def get_meta(item_id):
        if item_id not in meta_cache:
            meta_cache[item_id] = drive.files().get(
                fileId=item_id,
                fields="id,name,mimeType,parents,driveId,shortcutDetails,owners",
                supportsAllDrives=True,
            ).execute(num_retries=5)
        return meta_cache[item_id]

    chain, seen, cur, base = [], set(), fid, None
    top_folder_id, top_folder_name, leaf_meta, leaf_err = None, None, None, None
    while cur and cur not in seen:
        seen.add(cur)
        if cur in folders and folders[cur].is_dir():
            base = folders[cur]
            break
        if chain:
            sc_dir = mount / ".shortcut-targets-by-id" / cur
            if sc_dir.is_dir():
                base = folders[cur] = sc_dir
                break
        try:
            meta = get_meta(cur)
        except Exception as exc:
            if cur == fid:
                leaf_err = exc
            break
        if cur == fid:
            leaf_meta = meta
        target_id = (meta.get("shortcutDetails") or {}).get("targetId")
        if target_id:
            cur = target_id
            continue
        if meta.get("mimeType") == "application/vnd.google-apps.folder":
            top_folder_id, top_folder_name = cur, meta.get("name")
        parents = meta.get("parents") or []
        if not parents:
            drive_id = meta.get("driveId")
            if cur == state["root_id"]:
                base = mount / "MyDrive"
            elif drive_id and cur == drive_id:
                sd_root = mount / "Shareddrives"
                cand = sd_root / meta["name"]
                if cand.is_dir():
                    base = cand
                elif sd_root.is_dir():
                    base = next((p for p in sd_root.iterdir() if _xattr_drive_id(p) == drive_id), None)
            elif top_folder_id == cur:
                sc_dir = mount / ".shortcut-targets-by-id" / cur
                if sc_dir.is_dir():
                    base = sc_dir
            if base is not None:
                folders[cur] = base
            break
        chain.append((cur, meta["name"]))
        parent = parents[0]
        if top_folder_id is None:
            top_folder_id = parent
        if parent == state["root_id"]:
            base = folders[parent] = mount / "MyDrive"
            break
        cur = parent

    if base is not None:
        curr = base
        for item_id, item_name in reversed(chain):
            cand = curr / item_name
            if cand.exists() and (_xattr_drive_id(cand) in (None, item_id)):
                curr = cand
            elif curr.is_dir():
                matched = next((p for p in curr.iterdir() if _xattr_drive_id(p) == item_id), None)
                curr = matched if matched is not None else cand
            else:
                curr = cand
            if curr.is_dir():
                folders[item_id] = curr
        if curr.is_file():
            files_map[fid] = curr
            return curr

    if top_folder_id and state.get("unmounted_folders") is not None:
        state["unmounted_folders"][top_folder_id] = top_folder_name or top_folder_id
    fname = (leaf_meta or {}).get("name", fid)
    owner = (((leaf_meta or {}).get("owners") or [{}])[0]).get("emailAddress", "chủ sở hữu")
    if leaf_err is not None:
        raise FileNotFoundError(
            f"Không truy cập được file {fid} ({leaf_err}). Hãy kiểm tra quyền truy cập hoặc mount đúng tài khoản Google Drive sở hữu dữ liệu."
        )
    if top_folder_id:
        label = f"'{top_folder_name}' " if top_folder_name else ""
        raise FileNotFoundError(
            f"File '{fname}' ({fid}) chưa có trên Drive đã mount ({mount}). "
            f"Hãy mở thư mục gốc {label}https://drive.google.com/drive/folders/{top_folder_id} "
            f"-> bấm 'Thêm lối tắt vào Drive' (Add shortcut to Drive) vào MyDrive rồi chạy lại."
        )
    raise FileNotFoundError(
        f"File '{fname}' ({fid}, chủ sở hữu: {owner}) không nằm trong MyDrive của bạn và chỉ được chia sẻ lẻ link file "
        f"(không chia sẻ thư mục cha), nên Google Drive Mount không thể gắn file này vào {mount}. "
        f"Cách xử lý: (1) Khi chạy drive.mount('/content/drive'), hãy đăng nhập tài khoản Google sở hữu dữ liệu ({owner}), "
        f"HOẶC (2) nhờ {owner} chia sẻ THƯ MỤC GỐC chứa dữ liệu v4 rồi bấm 'Add shortcut to Drive' vào MyDrive."
    )


def ensure_folder_shortcuts(drive, folder_ids, shortcut_dir_name="_v4_shortcuts"):
    """Create folder shortcuts in MyDrive/<shortcut_dir_name> so DriveFS mounts shared root folders."""
    if not folder_ids:
        return 0
    q = f"mimeType='application/vnd.google-apps.folder' and name='{shortcut_dir_name}' and 'root' in parents and trashed=false"
    res = drive.files().list(q=q, fields="files(id)", pageSize=1).execute(num_retries=5)
    if res.get("files"):
        parent_id = res["files"][0]["id"]
    else:
        parent_id = drive.files().create(
            body={"name": shortcut_dir_name, "mimeType": "application/vnd.google-apps.folder", "parents": ["root"]},
            fields="id",
        ).execute(num_retries=5)["id"]
    existing = drive.files().list(
        q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.shortcut' and trashed=false",
        fields="files(id,shortcutDetails)", pageSize=200,
    ).execute(num_retries=5).get("files", [])
    have = {(f.get("shortcutDetails") or {}).get("targetId") for f in existing}
    created = 0
    for fid, label in folder_ids.items():
        if fid not in have:
            drive.files().create(
                body={
                    "name": f"{label}_{fid[:6]}",
                    "mimeType": "application/vnd.google-apps.shortcut",
                    "parents": [parent_id],
                    "shortcutDetails": {"targetId": fid},
                },
                fields="id",
                supportsAllDrives=True,
            ).execute(num_retries=5)
            created += 1
    return created


def joints_from_vertices(vertices, regressor):
    vertices = np.asarray(vertices)
    if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3):
        raise ValueError(f"Expected [T,6890,3] vertices, got {vertices.shape}")
    result = np.einsum("jv,tvc->tjc", regressor, vertices, optimize=True)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite joint coordinates")
    return result.astype(np.float32)


def centered(a, names):
    return a - a[:, [names.index("left_hip"), names.index("right_hip")]].mean(1)[:, None]


def rotation(source, target):
    u, _, vt = np.linalg.svd(source.reshape(-1, 3).T @ target.reshape(-1, 3))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def pa_errors(source, target):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source = source - source.mean(1, keepdims=True)
    target = target - target.mean(1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum("tji,tjk->tik", source, target))
    correction = np.broadcast_to(np.eye(3, dtype=np.float32), u.shape).copy()
    correction[:, -1, -1] = np.linalg.det(u @ vt)
    rot = u @ correction @ vt
    moved = source @ rot
    scale = (moved * target).sum((1, 2)) / np.maximum((moved * moved).sum((1, 2)), 1e-12)
    return np.linalg.norm(moved * scale[:, None, None] - target, axis=-1).mean(1)


class AlignmentError(ValueError):
    """A camera has insufficient overlap or pose agreement with GT."""


def associate(tracks, gt, names, ratios=(0.5, 1.0, 2.0)):
    """GT-assisted timeline search and track stitching with a switch penalty."""
    body = [names.index(n) for n in BODY]
    keys = sorted(tracks)
    all_ids = np.unique(np.concatenate([np.asarray(tracks[k][0], dtype=int) for k in keys]))
    if len(all_ids) < 16:
        raise AlignmentError("Need >=16 camera frames for alignment")
    # Precompute PA costs against every GT tick once, then search timelines cheaply.
    costs = np.full((len(all_ids), len(keys), len(gt)), np.inf, dtype=np.float32)
    joints_at = {}
    gt_body = np.asarray(gt[:, body], dtype=np.float32)
    for ki, key in enumerate(keys):
        ids, joints = tracks[key]
        ids = np.asarray(ids, dtype=int)
        if len(np.unique(ids)) != len(ids) or len(ids) != len(joints):
            raise ValueError(f"Invalid/duplicate frame ids for track {key}")
        positions = np.searchsorted(all_ids, ids)
        for s0 in range(0, len(ids), 128):
            chunk = joints[s0:s0 + 128, body]
            source = np.repeat(chunk[:, None], len(gt), axis=1)
            target = np.broadcast_to(gt_body[None], source.shape)
            costs[positions[s0:s0 + 128], ki] = pa_errors(
                source.reshape(-1, len(body), 3), target.reshape(-1, len(body), 3)
            ).reshape(len(chunk), len(gt))
        for pos, joint in zip(positions, joints):
            joints_at[(int(pos), ki)] = joint
    roots = np.zeros((len(all_ids), len(keys), 3))
    for (pos, ki), joints in joints_at.items():
        roots[pos, ki] = joints[[names.index("left_hip"), names.index("right_hip")]].mean(0)
    continuity = {}
    candidates = []
    best_lb = float("inf")
    for ratio in ratios:
        mapped = np.rint((all_ids - all_ids.min()) * ratio).astype(int)
        # ponytail: cap required camera frames by achievable GT window when video outlasts mocap clip
        min_keep = max(16, int(np.ceil(0.7 * min(len(all_ids), np.ceil(len(gt) / ratio)))))
        for offset in range(-int(mapped.max()), len(gt)):
            gi = mapped + offset
            keep = np.where((gi >= 0) & (gi < len(gt)))[0]
            if len(keep) < min_keep:
                continue
            unique_gt, unique = np.unique(gi[keep], return_index=True)
            positions = keep[unique]
            coverage = min(1.0, (float(len(unique_gt)) * max(ratio, 1.0)) / len(gt))
            if len(unique_gt) < 16 or coverage < 0.5:
                continue
            local = costs[positions, :, unique_gt]
            lb = float(local.min(axis=1).mean())
            best_lb = min(best_lb, lb)
            if lb > 0.15:
                continue
            previous = local[0].copy()
            parents = []
            for step, emission in enumerate(local[1:], 1):
                link = (int(positions[step - 1]), int(positions[step]))
                if link not in continuity:
                    distance = np.linalg.norm(roots[link[0], :, None] - roots[link[1], None, :], axis=-1)
                    continuity[link] = (1 - np.eye(len(keys))) * (0.025 + 0.1 * np.minimum(distance, 5))
                options = previous[:, None] + continuity[link]
                parent = np.argmin(options, axis=0)
                previous = emission + options[parent, np.arange(len(keys))]
                parents.append(parent)
            final = int(np.argmin(previous))
            if not np.isfinite(previous[final]):
                continue
            path = [final]
            for parent in reversed(parents):
                path.append(int(parent[path[-1]]))
            path = np.array(path[::-1])
            raw_error = float(np.mean(local[np.arange(len(path)), path]))
            score = float(previous[final] / len(path) + 0.03 * (1 - coverage))
            candidates.append((score, raw_error, ratio, offset, positions, unique_gt, path, coverage))
    if not candidates:
        if np.isfinite(best_lb):
            raise AlignmentError(f"Alignment PA error {best_lb * 1000:.1f}mm exceeds 150mm; inspect data")
        raise AlignmentError("No alignment covers >=70% achievable camera overlap and >=50% GT timeline")
    candidates.sort(key=lambda c: c[:4])
    score, error, ratio, offset, positions, gi, path, coverage = candidates[0]
    if error > 0.15:
        raise AlignmentError(f"Alignment PA error {error * 1000:.1f}mm exceeds 150mm; inspect data")
    joints = np.array([joints_at[(int(pos), int(ki))] for pos, ki in zip(positions, path)])
    segments = []
    for i, ki in enumerate(path):
        key = keys[ki]
        if not segments or segments[-1]["track"] != key:
            segments.append({"track": key, "gt_start": int(gi[i]), "gt_end": int(gi[i])})
        else:
            segments[-1]["gt_end"] = int(gi[i])
    report = dict(track=segments[0]["track"] if len(segments) == 1 else "stitched",
                  segments=segments, ratio=ratio, offset=offset, matched=len(gi),
                  coverage=coverage, pa_mm=error * 1000,
                  switches=len(segments)-1,
                  runner_up_score_mm=candidates[1][0] * 1000 if len(candidates) > 1 else None)
    return gi, joints, report


def synchronize(ids_m, jm, ids_s, js):
    """Interpolate cameras onto shared GT ticks; never bridge missing track gaps."""
    common = np.arange(max(ids_m.min(), ids_s.min()), min(ids_m.max(), ids_s.max()) + 1)
    arrays, valid = [], np.ones(len(common), dtype=bool)
    for ids, joints in ((ids_m, jm), (ids_s, js)):
        hi = np.minimum(np.searchsorted(ids, common), len(ids) - 1)
        lo = np.maximum(hi - 1, 0)
        step = max(1.0, float(np.median(np.diff(ids))))
        valid &= (common == ids[hi]) | ((ids[hi] - ids[lo]) <= 1.5 * step)
        flat = joints.reshape(len(ids), -1)
        values = np.stack([np.interp(common, ids, column) for column in flat.T], axis=1)
        arrays.append(values.reshape(len(common), *joints.shape[1:]).astype(np.float32))
    return common[valid], arrays[0][valid], arrays[1][valid]


def prepare(rows, drive, cache, config, alignment_path=None, skip_unaligned_actions=False,
            drive_mount="/content/drive", data_dir=""):
    validate_rows(rows)
    mapping = load_keypoints3d_map(str(ROOT / "configs/keypoints3D_map.yml"))
    names = [k["name"] for k in mapping["keypoints"]]
    reg = np.load(ROOT / "models/J_regressor_body25_extended.npy")
    reg = reg[[k["regressor_index"] for k in mapping["keypoints"]]]
    wanted_ids = {file_id(r["GT"]) for r in rows} | {file_id(r["file PKL"]) for r in rows}
    mount_state = {
        "wanted_ids": wanted_ids,
        "data_dir": data_dir,
        "exclude_dir": str(Path(alignment_path).parent) if alignment_path else None,
    }
    pairs, reports = [], []
    for action in dict.fromkeys(r["action"] for r in rows):
        cameras = []
        group = sorted((r for r in rows if r["action"] == action), key=lambda r: int(r["cam"][3:]))
        if len({file_id(r["GT"]) for r in group}) != 1:
            raise ValueError(f"Different GT files in one sequence: {action}")
        gt_ref = group[0].get("GT_path") or group[0]["GT"]
        gt_path = resolve_mounted(drive, gt_ref, drive_mount, mount_state)
        with np.load(gt_path, allow_pickle=False) as data:
            gt = joints_from_vertices(data["verts"], reg)
        for row in group:
            # ponytail: cache compact 21-joint tracks (~100KB) so mounted PKLs (~150MB) are read once
            jcache = Path(cache) / f"{file_id(row['file PKL'])}.joints.joblib" if (drive is not None or row.get("PKL_path")) else None
            if jcache is not None and jcache.exists():
                tracks = joblib.load(jcache)
            else:
                pkl_ref = row.get("PKL_path") or row["file PKL"]
                pkl_path = resolve_mounted(drive, pkl_ref, drive_mount, mount_state)
                raw = joblib.load(pkl_path)
                tracks = {}
                for key, track in raw.items():
                    if isinstance(track, dict) and "verts" in track and "frame_ids" in track:
                        tracks[str(key)] = (np.asarray(track["frame_ids"], dtype=int),
                                            joints_from_vertices(track["verts"], reg))
                if jcache is not None and tracks:
                    jcache.parent.mkdir(parents=True, exist_ok=True)
                    tmp_j = jcache.with_suffix(".tmp")
                    joblib.dump(tracks, tmp_j)
                    tmp_j.replace(jcache)
            if not tracks:
                raise ValueError(f"Expected WHAM tracks with camera-frame verts: {action}/{row['cam']}")
            print(f"Aligning {action}/{row['cam']}", flush=True)
            try:
                gi, joints, report = associate(tracks, gt, names)
            except AlignmentError as exc:
                frame_ids = np.unique(np.concatenate([track[0] for track in tracks.values()]))
                report = dict(status="rejected", reason=str(exc), coverage=None, pa_mm=None,
                              camera_frames=len(frame_ids), gt_frames=len(gt),
                              frame_start=int(frame_ids.min()) if len(frame_ids) else None,
                              frame_end=int(frame_ids.max()) if len(frame_ids) else None)
            else:
                report = dict(status="accepted", **report)
                cameras.append((row["cam"], gi, joints))
            reports.append(dict(action=action, cam=row["cam"], **report))
            print(json.dumps(reports[-1]), flush=True)
            if alignment_path is not None:
                atomic_json(Path(alignment_path), reports)
        for (cam_m, ids_m, jm), (cam_s, ids_s, js) in itertools.combinations(cameras, 2):
            common, synced_m, synced_s = synchronize(ids_m, jm, ids_s, js)
            # Temporal features/stability must never bridge missing GT ticks.
            stride = float(np.median(np.diff(common))) if len(common) > 1 else 1
            runs = np.split(np.arange(len(common)), np.where(np.diff(common) != stride)[0] + 1)
            for run in runs:
                if len(run) < 8:
                    continue
                for cm, cs, master, slave in ((cam_m, cam_s, synced_m[run], synced_s[run]),
                                              (cam_s, cam_m, synced_s[run], synced_m[run])):
                    target = centered(gt[common[run]], names)
                    master_c = centered(master, names)
                    body = [names.index(n) for n in BODY]
                    target = target @ rotation(target[:, body], master_c[:, body])
                    M = [dict(zip(names, f)) for f in master]
                    S = [dict(zip(names, f)) for f in slave]
                    feats, children = extract_pair_features(M, S)
                    directions = slave_directions(M, S, config.get("rotation_half_window", 4))
                    pairs.append(dict(action=action, split=group[0]["split"], cameras=[cm, cs],
                                      names=names, M=M, S=S, features=feats, children=children,
                                      directions=directions, target=target.astype(np.float32)))
        if not any(p["action"] == action for p in pairs):
            reason = (f"{action}: need at least two aligned cameras; got {len(cameras)}."
                      if len(cameras) < 2 else f"No synchronized segments for {action}.")
            if not skip_unaligned_actions:
                raise ValueError(reason + " Inspect rejected cameras in alignment.json and correct their data/timeline.")
            reports.append(dict(action=action, split=group[0]["split"], cam=None,
                                status="skipped_action", reason=reason, coverage=None, pa_mm=None))
            print(json.dumps(reports[-1]), flush=True)
            if alignment_path is not None:
                atomic_json(Path(alignment_path), reports)
            continue
        print(f"Prepared {action}: {len(cameras)} cameras", flush=True)
    if {p["split"] for p in pairs} != {"train", "val"}:
        raise ValueError("No usable pairs for one of the splits; inspect alignment/coverage")
    return pairs, reports


def export_model(weights, std):
    return dict(type="linear", feature_names=list(FEATURE_NAMES), mean=[0.0] * len(std),
                std=np.asarray(std).tolist(), weights=np.asarray(weights).tolist())


def torch_fuse(pair, weights, std):
    names = pair["names"]
    master = torch.as_tensor(np.array([[f[n] for n in names] for f in pair["M"]]), dtype=torch.float32)
    features = torch.as_tensor(pair["features"])
    w = torch.sigmoid((features / std) @ weights)
    result = {n: master[:, i] for i, n in enumerate(names)}
    for child, parent in _ordered_bones(pair["M"][0]):
        bi = pair["children"].index(child)
        direction = torch.tensor(np.array([d.get(child, np.zeros(3)) for d in pair["directions"]]), dtype=torch.float32)
        m = master[:, names.index(child)] - master[:, names.index(parent)]
        length = torch.linalg.vector_norm(m, dim=1, keepdim=True)
        unit = w[:, bi, None] * m / length.clamp_min(1e-9) + (1 - w[:, bi, None]) * direction
        norm = torch.linalg.vector_norm(unit, dim=1, keepdim=True)
        unit = torch.where(norm < 1e-9, m / length.clamp_min(1e-9), unit / norm.clamp_min(1e-9))
        delta = torch.where(length < 1e-9, m, unit * length)
        result[child] = result[parent] + delta
    output = torch.stack([result[n] for n in names], dim=1)
    root = output[:, [names.index("left_hip"), names.index("right_hip")]].mean(1, keepdim=True)
    return output - root


def torch_pa_loss(pred, target):
    # ponytail: Envelope-theorem Procrustes PA-MPJPE avoids global yaw drift in uncalibrated v4 GT & SVD backward NaNs
    pred_c = pred - pred.mean(1, keepdim=True)
    tgt_c = target - target.mean(1, keepdim=True)
    with torch.no_grad():
        u, _, vt = torch.linalg.svd(torch.einsum("tji,tjk->tik", pred_c, tgt_c))
        corr = torch.eye(3, dtype=pred.dtype, device=pred.device).expand_as(u).clone()
        corr[:, -1, -1] = torch.det(u @ vt)
        rot = u @ corr @ vt
        moved = pred_c @ rot
        scale = (moved * tgt_c).sum((1, 2)) / (moved * moved).sum((1, 2)).clamp_min(1e-12)
    return torch.linalg.vector_norm((pred_c @ rot) * scale[:, None, None] - tgt_c, dim=-1).mean()


def validation(pairs, model, config):
    # Equal sequence weight prevents sequences with more camera pairs dominating selection.
    scores, baselines = {}, {}
    for p in pairs:
        cfg = copy.deepcopy(config)
        cfg["weight_model"] = model
        fused, _ = fuse_partwise(p["M"], p["S"], cfg)
        names = p["names"]
        inds = [names.index(n) for n in BODY]
        output = np.array([[f[n] for n in names] for f in fused], dtype=np.float32)
        master = np.array([[f[n] for n in names] for f in p["M"]], dtype=np.float32)
        for values, pred in ((scores, output), (baselines, master)):
            values.setdefault(p["action"], []).append(
                float(pa_errors(pred[:, inds], p["target"][:, inds]).mean() * 1000))
    return (float(np.mean([np.mean(v) for v in scores.values()])),
            float(np.mean([np.mean(v) for v in baselines.values()])))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def fit(pairs, config, out, epochs=50, lr=0.01, resume=None, identity=""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train = [p for p in pairs if p["split"] == "train"]
    val = [p for p in pairs if p["split"] == "val"]
    if not train or not val:
        raise ValueError("Both splits need usable pairs")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # Antisymmetry requires zero mean; estimate scale on train only.
    flat = np.concatenate([p["features"].reshape(-1, len(FEATURE_NAMES)) for p in train])
    std = torch.tensor(np.maximum(np.sqrt(np.mean(flat ** 2, axis=0)), 1e-6))
    weights = torch.nn.Parameter(torch.zeros(len(FEATURE_NAMES)))
    optimizer = torch.optim.Adam([weights], lr=lr)
    signature = hashlib.sha256((identity + json.dumps(config, sort_keys=True) +
                               str(lr)).encode()).hexdigest()
    history, best, start = [], float("inf"), 1
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        if ckpt["signature"] != signature or not np.allclose(ckpt["std"], std.numpy()):
            raise ValueError("Resume dataset/config/lr differs")
        weights.data.copy_(ckpt["weights"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history, best, start = ckpt["history"], ckpt["best_val_mm"], ckpt["epoch"] + 1
        random.setstate(ckpt["random_state"])
        torch.set_rng_state(ckpt["torch_rng"])
        # Restore the selected model even when resuming into a new output folder.
        atomic_json(out / "best_weights.json", ckpt["best_model"])
    elif list(out.glob("epoch_*.pt")):
        raise ValueError("Output already has checkpoints; resume or choose a new folder")
    best_model = ckpt["best_model"] if resume else None
    equal_val, _ = validation(val, export_model(np.zeros(len(FEATURE_NAMES)), std.numpy()), config)
    if start > epochs:
        print(json.dumps({"status": "already_completed", "epochs": epochs,
                          "best_epoch": best_model["epoch"] if best_model else None,
                          "best_val_mpjpe_mm": best}), flush=True)
        return history
    actions = sorted({p["action"] for p in train})
    for epoch in range(start, epochs + 1):
        order = actions.copy()
        random.shuffle(order)
        losses = []
        for action in order:
            group = [p for p in train if p["action"] == action]
            optimizer.zero_grad()
            action_losses = []
            for p in group:
                inds = [p["names"].index(n) for n in BODY]
                target = torch.as_tensor(p["target"][:, inds])
                loss = torch_pa_loss(torch_fuse(p, weights, std)[:, inds], target)
                (loss / len(group)).backward()
                action_losses.append(float(loss.detach()) * 1000)
            optimizer.step()
            losses.append(float(np.mean(action_losses)))
        model = export_model(weights.detach().numpy(), std.numpy())
        score, baseline = validation(val, model, config)
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        improved = score < best
        if improved:
            best, best_model = score, dict(model, epoch=epoch, val_mpjpe_mm=score,
                                           equal_val_mpjpe_mm=equal_val,
                                           baseline_val_mpjpe_mm=baseline)
        history.append(dict(epoch=epoch, train_mpjpe_mm=float(np.mean(losses)),
                            val_mpjpe_mm=score, equal_val_mpjpe_mm=equal_val,
                            baseline_val_mpjpe_mm=baseline,
                            best_epoch=best_model["epoch"], best_val_mpjpe_mm=best))
        checkpoint = dict(epoch=epoch, weights=weights.detach().clone(), std=std.numpy(),
                          optimizer=optimizer.state_dict(), signature=signature, history=history,
                          best_val_mm=best, best_model=best_model,
                          random_state=random.getstate(), torch_rng=torch.get_rng_state())
        temp = out / "checkpoint.tmp"
        torch.save(checkpoint, temp)
        temp.replace(out / f"epoch_{epoch:04d}.pt")
        atomic_json(out / f"epoch_{epoch:04d}_weights.json", model)
        atomic_json(out / "history.json", history)
        atomic_json(out / "last.json", {"checkpoint": f"epoch_{epoch:04d}.pt", "signature": signature, "epoch": epoch})
        atomic_json(out / "best_weights.json", best_model)
        patch = copy.deepcopy(config)
        # ponytail: inline 8 linear weights directly in YAML so config works across Colab/local without broken paths
        patch["weight_model"] = {
            "enabled": True,
            **{k: best_model[k] for k in ("type", "feature_names", "mean", "std", "weights")},
        }
        (out / "best_partwise.yml").write_text(yaml.safe_dump(
            {"fusion": {"method": "partwise", "partwise": patch}}, sort_keys=False), encoding="utf-8")
        print(json.dumps(history[-1]), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--drive-mount", default="/content/drive",
                        help="Root directory where Google Drive is mounted (0 media downloads)")
    parser.add_argument("--data-dir", default="",
                        help="Optional dataset folder inside mounted Drive to scan first")
    parser.add_argument("--cache", default="inputs/v4_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-unaligned-actions", action="store_true",
                        help="Exclude sequences without usable camera pairs and record them in alignment.json")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_rows(rows)
    config = yaml.safe_load((ROOT / "configs/pipeline.yml").read_text(encoding="utf-8"))["fusion"]["partwise"]
    config.pop("weight_model", None)
    # Sheet lacks visibility/tracking results: match training and inference evidence.
    config["multi_person"] = {"enabled": False}
    need_drive = not all(r.get("GT_path") and r.get("PKL_path") for r in rows)
    drive = None
    if need_drive:
        from google.auth import default
        from googleapiclient.discovery import build
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/drive"])
        drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    stable_rows = [{k: v for k, v in r.items() if k not in ("GT_path", "PKL_path")} for r in rows]
    identity = hashlib.sha256(json.dumps(stable_rows, sort_keys=True, ensure_ascii=False).encode() +
        (ROOT / "models/J_regressor_body25_extended.npy").read_bytes() +
        Path(__file__).read_bytes() + (ROOT / "fusion_pipeline/partwise.py").read_bytes() +
        (ROOT / "configs/keypoints3D_map.yml").read_bytes() + json.dumps(config, sort_keys=True).encode() +
        str(args.skip_unaligned_actions).encode()).hexdigest()
    prepared_out = out / f"prepared_{identity}.joblib"
    prepared_cache = cache / f"prepared_{identity}.joblib"
    if prepared_out.exists():
        pairs, reports = joblib.load(prepared_out)
    elif prepared_cache.exists():
        pairs, reports = joblib.load(prepared_cache)
    else:
        pairs, reports = prepare(rows, drive, cache, config, out / "alignment.json",
                                 skip_unaligned_actions=args.skip_unaligned_actions,
                                 drive_mount=args.drive_mount, data_dir=args.data_dir)
        temp = prepared_out.with_suffix(".tmp")
        joblib.dump((pairs, reports), temp)
        temp.replace(prepared_out)
    atomic_json(out / "alignment.json", reports)
    atomic_json(out / "manifest.json", rows)
    atomic_json(out / "protocol.json", {
        "metric": "12-body-joint Procrustes PA-MPJPE (mm)",
        "selection": "mean of pair PA-MPJPE scores within each action, then mean across val actions",
        "association": "GT-assisted proper-rotation PA timeline search and penalized track stitching; ratio 0.5/1/2, integer offset",
        "limitations": "GT-assisted time/track association; review alignment.json. No independent test score.",
        "feature_names": list(FEATURE_NAMES), "config": config,
        "train_actions": sorted({p["action"] for p in pairs if p["split"] == "train"}),
        "val_actions": sorted({p["action"] for p in pairs if p["split"] == "val"}),
        "skip_unaligned_actions": args.skip_unaligned_actions,
        "skipped_actions": [r["action"] for r in reports if r.get("status") == "skipped_action"],
        "synchronization": "linear interpolation to shared GT ticks, excluding missing-track gaps",
        "directed_segments": len(pairs),
    })
    if args.prepare_only:
        print("Prepared data and alignment report:", out / "alignment.json")
        return
    fit(pairs, config, out, args.epochs, args.lr, args.resume, identity)


if __name__ == "__main__":
    main()

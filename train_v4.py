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
from fusion_pipeline.naive import fuse_naive
from fusion_pipeline.detector import (
    compute_visibility_from_mesh_vertices, load_torso_mask, require_ray_casting_backend,
)
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
    for cand in (mount / ".shortcut-targets-by-id" / fid,
                 mount / "MyDrive" / "_v4_shortcuts" / fid):
        if cand.is_file():
            files_map[fid] = cand
            return cand
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

    if leaf_err is None and state.get("unmounted_targets") is not None:
        target_id = top_folder_id or fid
        state["unmounted_targets"][target_id] = (top_folder_name or fid) if top_folder_id else fid
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
        f"File '{fname}' ({fid}, chủ sở hữu: {owner}) chưa gắn được vào {mount} dù đã tạo lối tắt. "
        f"Hãy mount đúng tài khoản Google sở hữu ({owner}) hoặc nhờ {owner} chia sẻ thư mục cha."
    )


def ensure_shortcuts(drive, target_ids, shortcut_dir_name="_v4_shortcuts"):
    """Create Drive shortcuts in MyDrive/<shortcut_dir_name> so DriveFS mounts shared folders/files."""
    if not target_ids:
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
    have, page_token = set(), None
    while True:
        resp = drive.files().list(
            q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.shortcut' and trashed=false",
            fields="nextPageToken,files(shortcutDetails)", pageSize=1000, pageToken=page_token,
        ).execute(num_retries=5)
        have.update((f.get("shortcutDetails") or {}).get("targetId") for f in resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    missing = [(tid, label) for tid, label in target_ids.items() if tid not in have]
    for i in range(0, len(missing), 50):
        batch = drive.new_batch_http_request()
        for tid, label in missing[i:i + 50]:
            name = tid if label == tid else f"{label}_{tid[:6]}"
            batch.add(drive.files().create(
                body={
                    "name": name,
                    "mimeType": "application/vnd.google-apps.shortcut",
                    "parents": [parent_id],
                    "shortcutDetails": {"targetId": tid},
                },
                fields="id",
                supportsAllDrives=True,
            ))
        batch.execute()
    return len(missing)


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


def associate(tracks, gt, names, ratios=(0.5, 1.0, 2.0), return_sources=False):
    """GT-assisted timeline search and track stitching with a switch penalty."""
    body = [names.index(n) for n in BODY]
    keys = sorted(tracks)
    all_ids = np.unique(np.concatenate([np.asarray(tracks[k][0], dtype=int) for k in keys]))
    if len(all_ids) < 16:
        raise AlignmentError("Need >=16 camera frames for alignment")
    # Precompute PA costs against every GT tick once, then search timelines cheaply.
    costs = np.full((len(all_ids), len(keys), len(gt)), np.inf, dtype=np.float32)
    joints_at = {}
    source_rows = {}
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
        for row, (pos, joint) in enumerate(zip(positions, joints)):
            joints_at[(int(pos), ki)] = joint
            if return_sources:
                source_rows[(int(pos), ki)] = row
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
    if return_sources:
        sources = [(keys[int(ki)], source_rows[(int(pos), int(ki))])
                   for pos, ki in zip(positions, path)]
        return gi, joints, report, sources
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


def synchronize_visibility(ids, visibility, ticks):
    """Interpolated poses are visible only if both source endpoints are visible."""
    hi = np.minimum(np.searchsorted(ids, ticks), len(ids) - 1)
    lo = np.maximum(hi - 1, 0)
    exact = ticks == ids[hi]
    return np.where(exact[:, None], visibility[hi], visibility[lo] & visibility[hi])


def tracks_from_wham(raw, reg, names, occlusion):
    """Keep compact joints/visibility; meshes are needed only during preparation."""
    tracks, meshes, by_frame = {}, {}, {}
    for key, track in raw.items():
        if not isinstance(track, dict) or "frame_ids" not in track:
            continue
        vertices = track.get("verts_cam", track.get("verts"))
        if vertices is None:
            continue
        vertices = np.asarray(vertices)
        ids = np.asarray(track["frame_ids"], dtype=int).reshape(-1)
        if len(ids) != len(vertices) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Invalid/duplicate frame ids for track {key}")
        key = str(key)
        tracks[key] = (ids, joints_from_vertices(vertices, reg))
        if occlusion.get("enabled", False):
            meshes[key] = vertices
            for fid, mesh in zip(ids, vertices):
                by_frame.setdefault(int(fid), []).append((key, mesh))
    visibility = {}
    for key, vertices in meshes.items():
        ids, joints = tracks[key]
        frames = []
        for fid, mesh, pose in zip(ids, vertices, joints):
            others = [v for k, v in by_frame[int(fid)] if k != key] if occlusion.get("multi_person", False) else None
            vis = compute_visibility_from_mesh_vertices(
                dict(zip(names, pose)), mesh,
                occlusion_tau=float(occlusion.get("tau", 0.05)), extra_verts=others,
                inter_person_radius=float(occlusion.get("inter_person_radius_m", 0.045)),
                inter_person_min_hits=int(occlusion.get("inter_person_min_hits", 3)),
                inter_person_tau=float(occlusion.get("inter_person_tau", 0.15)),
            )
            frames.append([vis[n] for n in names])
        visibility[key] = np.asarray(frames, dtype=bool)
    return tracks, visibility


def _download_drive_file(drive, fid, dest):
    """Stream a Drive file via OAuth API (or gdown if quota-limited) to a temporary path."""
    from googleapiclient.http import MediaIoBaseDownload
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".download")
    try:
        with temp.open("wb") as stream:
            reader = MediaIoBaseDownload(
                stream,
                drive.files().get_media(fileId=fid, supportsAllDrives=True),
                chunksize=32 * 1024 * 1024,
            )
            done = False
            while not done:
                _, done = reader.next_chunk(num_retries=5)
        temp.replace(dest)
        return dest
    except Exception:
        temp.unlink(missing_ok=True)
        import gdown
        out = gdown.download(id=fid, output=str(temp), quiet=True, fuzzy=True)
        if not out or not temp.is_file() or not temp.stat().st_size:
            raise
        temp.replace(dest)
        return dest


def _resolve_or_download(drive, ref, mount, temp_dir, mount_state):
    """Prefer mounted Drive (0% download); fall back to OAuth API only for unmounted orphan files."""
    try:
        return resolve_mounted(drive, ref, mount, mount_state), False
    except FileNotFoundError:
        if drive is None:
            raise
        fid = file_id(ref)
        print(f"  [Fallback API] Đang đọc file chia sẻ lẻ {fid}...", flush=True)
        return _download_drive_file(drive, fid, Path(temp_dir) / fid), True


def prepare(rows, drive, cache, config, alignment_path=None, skip_unaligned_actions=False,
            drive_mount="/content/drive", data_dir=""):
    validate_rows(rows)
    mapping = load_keypoints3d_map(str(ROOT / "configs/keypoints3D_map.yml"))
    names = [k["name"] for k in mapping["keypoints"]]
    reg = np.load(ROOT / "models/J_regressor_body25_extended.npy")
    reg = reg[[k["regressor_index"] for k in mapping["keypoints"]]]
    occlusion = config.get("occlusion", {})
    use_occ = bool(occlusion.get("enabled", False))
    cache_suffix = ".joints.joblib"
    if use_occ:
        require_ray_casting_backend()
        mask = load_torso_mask(str(ROOT / "models/smpl_partSegmentation_mapping.pkl"))
        if mask is None:
            raise ValueError("Occlusion training requires a valid SMPL torso mask")
        occ_hash = hashlib.sha256(json.dumps(occlusion, sort_keys=True).encode()
                                  + mask.tobytes() + reg.tobytes()
                                  + Path(__file__).read_bytes()
                                  + (ROOT / "fusion_pipeline/detector.py").read_bytes()
                                  + (ROOT / "fusion_pipeline/config.py").read_bytes()
                                  + (ROOT / "configs/keypoints3D_map.yml").read_bytes()).hexdigest()[:16]
        cache_suffix = f".occ_{occ_hash}.joblib"
    wanted_ids = {file_id(r["GT"]) for r in rows} | {file_id(r["file PKL"]) for r in rows}
    mount_state = {
        "wanted_ids": wanted_ids,
        "data_dir": data_dir,
        "exclude_dir": str(Path(alignment_path).parent) if alignment_path else None,
    }
    temp_dir = Path("/content/v4_tmp") if Path("/content").is_dir() else Path(cache) / "_tmp"
    pairs, reports = [], []
    for action in dict.fromkeys(r["action"] for r in rows):
        cameras = []
        group = sorted((r for r in rows if r["action"] == action), key=lambda r: int(r["cam"][3:]))
        if len({file_id(r["GT"]) for r in group}) != 1:
            raise ValueError(f"Different GT files in one sequence: {action}")
        gt_fid = file_id(group[0]["GT"])
        gt_jcache = Path(cache) / f"{gt_fid}.gt_joints.joblib" if drive is not None else None
        if gt_jcache is not None and gt_jcache.exists():
            gt = joblib.load(gt_jcache)
        else:
            gt_ref = group[0].get("GT_path") or group[0]["GT"]
            gt_path, gt_is_tmp = _resolve_or_download(drive, gt_ref, drive_mount, temp_dir, mount_state)
            try:
                with np.load(gt_path, allow_pickle=False) as data:
                    gt = joints_from_vertices(data["verts"], reg)
            finally:
                if gt_is_tmp:
                    gt_path.unlink(missing_ok=True)
            if gt_jcache is not None:
                gt_jcache.parent.mkdir(parents=True, exist_ok=True)
                tmp_g = gt_jcache.with_suffix(".tmp")
                joblib.dump(gt, tmp_g)
                tmp_g.replace(gt_jcache)
        for row in group:
            # Cache compact joints and visibility, never full meshes.
            jcache = Path(cache) / f"{file_id(row['file PKL'])}{cache_suffix}" if (drive is not None or row.get("PKL_path")) else None
            if jcache is not None and jcache.exists():
                cached = joblib.load(jcache)
                tracks, visibility = cached if use_occ else (cached, {})
            else:
                pkl_ref = row.get("PKL_path") or row["file PKL"]
                pkl_path, pkl_is_tmp = _resolve_or_download(drive, pkl_ref, drive_mount, temp_dir, mount_state)
                try:
                    raw = joblib.load(pkl_path)
                    tracks, visibility = tracks_from_wham(raw, reg, names, occlusion)
                    del raw
                finally:
                    if pkl_is_tmp:
                        pkl_path.unlink(missing_ok=True)
                if jcache is not None and tracks:
                    jcache.parent.mkdir(parents=True, exist_ok=True)
                    tmp_j = jcache.with_suffix(".tmp")
                    joblib.dump((tracks, visibility) if use_occ else tracks, tmp_j)
                    tmp_j.replace(jcache)
            if not tracks:
                raise ValueError(f"Expected WHAM tracks with camera-frame verts: {action}/{row['cam']}")
            print(f"Aligning {action}/{row['cam']}", flush=True)
            try:
                if use_occ:
                    gi, joints, report, sources = associate(tracks, gt, names, return_sources=True)
                    aligned_vis = np.asarray([visibility[key][i] for key, i in sources], dtype=bool)
                    report["occluded_joint_fraction"] = float(np.mean(~aligned_vis))
                else:
                    gi, joints, report = associate(tracks, gt, names)
            except AlignmentError as exc:
                frame_ids = np.unique(np.concatenate([track[0] for track in tracks.values()]))
                report = dict(status="rejected", reason=str(exc), coverage=None, pa_mm=None,
                              camera_frames=len(frame_ids), gt_frames=len(gt),
                              frame_start=int(frame_ids.min()) if len(frame_ids) else None,
                              frame_end=int(frame_ids.max()) if len(frame_ids) else None)
            else:
                report = dict(status="accepted", **report)
                cameras.append((row["cam"], gi, joints, aligned_vis if use_occ else None))
            reports.append(dict(action=action, split=row["split"], cam=row["cam"], **report))
            print(json.dumps(reports[-1]), flush=True)
            if alignment_path is not None:
                atomic_json(Path(alignment_path), reports)
        # One baseline rotation per sequence/camera, shared by every pair and run.
        body = [names.index(n) for n in BODY]
        target_rotations = {
            cam: rotation(centered(gt[ids], names)[:, body], centered(joints, names)[:, body])
            for cam, ids, joints, _ in cameras
        }
        for (cam_m, ids_m, jm, vm), (cam_s, ids_s, js, vs) in itertools.combinations(cameras, 2):
            common, synced_m, synced_s = synchronize(ids_m, jm, ids_s, js)
            synced_vm = synchronize_visibility(ids_m, vm, common) if use_occ else None
            synced_vs = synchronize_visibility(ids_s, vs, common) if use_occ else None
            # Temporal features/stability must never bridge missing GT ticks.
            stride = float(np.median(np.diff(common))) if len(common) > 1 else 1
            runs = np.split(np.arange(len(common)), np.where(np.diff(common) != stride)[0] + 1)
            for run in runs:
                if len(run) < 8:
                    continue
                for cm, cs, master, slave, vis_m, vis_s in (
                        (cam_m, cam_s, synced_m[run], synced_s[run], synced_vm, synced_vs),
                        (cam_s, cam_m, synced_s[run], synced_m[run], synced_vs, synced_vm)):
                    target = centered(gt[common[run]], names)
                    target = target @ target_rotations[cm]
                    M = [dict(zip(names, f)) for f in master]
                    S = [dict(zip(names, f)) for f in slave]
                    vis_master = [dict(zip(names, v)) for v in vis_m[run]] if use_occ else None
                    vis_slave = [dict(zip(names, v)) for v in vis_s[run]] if use_occ else None
                    feats, children = extract_pair_features(M, S, vis_master, vis_slave)
                    mp = config.get("multi_person", {})
                    use_vis_rot = mp.get("enabled", False) and mp.get("visibility_weighted_rotation", True)
                    directions = slave_directions(
                        M, S, config.get("rotation_half_window", 4),
                        vis_master=vis_master if use_vis_rot else None,
                        vis_slave=vis_slave if use_vis_rot else None)
                    pairs.append(dict(action=action, split=group[0]["split"], cameras=[cm, cs],
                                      names=names, M=M, S=S, features=feats, children=children,
                                      vis_master=vis_master, vis_slave=vis_slave,
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


def export_model(weights, std, active_features=None):
    model = dict(type="linear", feature_names=list(FEATURE_NAMES), mean=[0.0] * len(std),
                 std=np.asarray(std).tolist(), weights=np.asarray(weights).tolist())
    if active_features is not None:
        model["active_features"] = np.asarray(active_features, dtype=bool).tolist()
    return model


def training_options(config):
    opts = dict(mpjpe_weight=0.25, prior_l2=0.001, patience=10, min_delta_mm=0.01,
                naive_regret_weight=1.0, naive_margin_mm=0.0)
    opts.update(config.get("training", {}))
    for key in ("mpjpe_weight", "prior_l2", "min_delta_mm", "naive_regret_weight", "naive_margin_mm"):
        if not np.isfinite(opts[key]) or opts[key] < 0:
            raise ValueError(f"training.{key} must be finite and non-negative")
    if isinstance(opts["patience"], bool) or not isinstance(opts["patience"], int) or opts["patience"] < 1:
        raise ValueError("training.patience must be a positive integer")
    return opts


def training_config(pipeline):
    fusion = pipeline["fusion"]
    config = copy.deepcopy(fusion["partwise"])
    config.pop("weight_model", None)
    config["occlusion"] = copy.deepcopy(fusion.get("occlusion", {"enabled": False}))
    naive = copy.deepcopy(fusion.get("naive", {}))
    if naive.get("use_aotr_stability", False):
        naive["sequence_stability"] = copy.deepcopy(fusion["aotr"].get("sequence_stability", {}))
    config["naive_reference"] = naive
    return config


def fit_signature(identity, config, lr):
    return hashlib.sha256((identity + json.dumps(config, sort_keys=True) + str(lr) +
                           "pa+sequence-mpjpe-naive-regret-nonneg-v3").encode()).hexdigest()


def preparation_identity(rows, config, skip_unaligned_actions=False):
    stable_rows = [{k: v for k, v in row.items() if k not in ("GT_path", "PKL_path")} for row in rows]
    digest = hashlib.sha256(json.dumps(stable_rows, sort_keys=True, ensure_ascii=False).encode())
    for name in ("models/J_regressor_body25_extended.npy", "train_v4.py",
                 "fusion_pipeline/partwise.py", "fusion_pipeline/naive.py",
                 "fusion_pipeline/aotr_stability.py", "fusion_pipeline/detector.py",
                 "fusion_pipeline/config.py", "configs/keypoints3D_map.yml"):
        digest.update((ROOT / name).read_bytes())
    mask = ROOT / "models/torso_indices.npy"
    if config.get("occlusion", {}).get("enabled", False) and mask.exists():
        digest.update(mask.read_bytes())
    digest.update(json.dumps(config, sort_keys=True).encode())
    digest.update(str(skip_unaligned_actions).encode())
    return digest.hexdigest()


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


def torch_pa_loss(pred, target, per_frame=False):
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
    errors = torch.linalg.vector_norm((pred_c @ rot) * scale[:, None, None] - tgt_c, dim=-1).mean(-1)
    return errors if per_frame else errors.mean()


def naive_relative_loss(pa, mpjpe, naive_pa, naive_mpjpe, opts):
    margin = opts["naive_margin_mm"] / 1000.0
    regret = (torch.relu(pa - naive_pa + margin) +
              opts["mpjpe_weight"] * torch.relu(mpjpe - naive_mpjpe + margin)).mean()
    loss = (pa + opts["mpjpe_weight"] * mpjpe).mean() + opts["naive_regret_weight"] * regret
    return loss, regret


def selection_key(metrics, naive, opts):
    margin = opts["naive_margin_mm"]
    beats = (metrics["pa_mm"] < naive["pa_mm"] - margin and
             metrics["mpjpe_mm"] < naive["mpjpe_mm"] - margin)
    return (not beats, metrics["pa_mm"] + opts["mpjpe_weight"] * metrics["mpjpe_mm"])


def naive_prediction(pair, config):
    # Match the benchmark's naive smoothing settings; never fit a rotation to GT here.
    fused, _ = fuse_naive(pair["M"], pair["S"], config.get("naive_reference", {"master_weight": 0.5}))
    return np.array([[f[n] for n in pair["names"]] for f in fused], dtype=np.float32)


def validation(pairs, model, config, method="partwise"):
    # Equal sequence weight prevents sequences with more camera pairs dominating selection.
    scores, baselines, mpjpe, mono_mpjpe = {}, {}, {}, {}
    for p in pairs:
        cfg = copy.deepcopy(config)
        if model is None:
            cfg.pop("weight_model", None)
        else:
            cfg["weight_model"] = model
        names = p["names"]
        inds = [names.index(n) for n in BODY]
        if method == "naive":
            output = naive_prediction(p, config)
        elif method == "partwise":
            fused, _ = fuse_partwise(p["M"], p["S"], cfg,
                                     vis_master=p.get("vis_master"), vis_slave=p.get("vis_slave"))
            output = np.array([[f[n] for n in names] for f in fused], dtype=np.float32)
        else:
            raise ValueError(f"Unknown validation method: {method}")
        master = np.array([[f[n] for n in names] for f in p["M"]], dtype=np.float32)
        for values, pred in ((scores, output), (baselines, master)):
            values.setdefault(p["action"], []).append(
                float(pa_errors(pred[:, inds], p["target"][:, inds]).mean() * 1000))
        for values, pred in ((mpjpe, output), (mono_mpjpe, master)):
            errors = np.linalg.norm(centered(pred, names)[:, inds] - p["target"][:, inds], axis=-1)
            values.setdefault(p["action"], []).append(float(errors.mean() * 1000))
    return dict(zip(("pa_mm", "mono_pa_mm", "mpjpe_mm", "mono_mpjpe_mm"),
                    [float(np.mean([np.mean(v) for v in values.values()]))
                     for values in (scores, baselines, mpjpe, mono_mpjpe)]))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_best(out, config, model):
    atomic_json(out / "best_weights.json", model)
    patch = copy.deepcopy(config)
    occlusion = patch.pop("occlusion", None)
    patch["weight_model"] = {"enabled": True, **{
        k: model[k] for k in ("type", "feature_names", "mean", "std", "weights", "active_features")}}
    fusion = {"method": "partwise", "partwise": patch}
    if occlusion is not None:
        fusion["occlusion"] = occlusion
    (out / "best_partwise.yml").write_text(yaml.safe_dump(
        {"fusion": fusion}, sort_keys=False), encoding="utf-8")


def fit(pairs, config, out, epochs=50, lr=0.01, resume=None, identity=""):
    opts = training_options(config)
    if epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("epochs and lr must be positive and finite")
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
    if not np.isfinite(flat).all():
        raise ValueError("Non-finite training features")
    active = np.ptp(flat, axis=0) > 1e-6
    std = torch.tensor(np.where(active, np.maximum(np.sqrt(np.mean(flat ** 2, axis=0)), 1e-6), 1.0))
    # ponytail: anchor init & L2 prior at 0 (w=0.5 symmetric 50-50) so model only deviates when features help
    prior = torch.zeros(len(FEATURE_NAMES))
    weights = torch.nn.Parameter(prior.clone())
    optimizer = torch.optim.Adam([weights], lr=lr)
    signature = fit_signature(identity, config, lr)
    history, best, start, stale, stopping_best = [], float("inf"), 1, 0, (True, float("inf"))
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        if ckpt["signature"] != signature or not np.allclose(ckpt["std"], std.numpy()):
            raise ValueError("Resume dataset/config/lr differs")
        weights.data.copy_(ckpt["weights"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history, best, start = ckpt["history"], ckpt["best_val_mm"], ckpt["epoch"] + 1
        stale, stopping_best = ckpt["stale_epochs"], ckpt["stopping_best"]
        random.setstate(ckpt["random_state"])
        torch.set_rng_state(ckpt["torch_rng"])
        # Restore the selected model even when resuming into a new output folder.
        save_best(out, config, ckpt["best_model"])
        atomic_json(out / "history.json", history)
    elif list(out.glob("epoch_*.pt")):
        raise ValueError("Output already has checkpoints; resume or choose a new folder")
    best_model = ckpt["best_model"] if resume else None
    equal = validation(val, export_model(np.zeros(len(FEATURE_NAMES)), std.numpy(), active), config)
    depth = validation(val, None, config)
    naive = validation(val, None, config, method="naive")
    if start > epochs or stale >= opts["patience"]:
        print(json.dumps({"status": "already_completed", "epochs": epochs,
                          "best_epoch": best_model["epoch"] if best_model else None,
                          "best_val_objective_mm": best,
                          "val_beats_naive": best_model["val_beats_naive"]}), flush=True)
        return history
    # Fixed train-only references are computed once; no GT-dependent inference routing.
    naive_errors = {}
    for p in train:
        inds = [p["names"].index(n) for n in BODY]
        output = centered(naive_prediction(p, config), p["names"])[:, inds]
        target = p["target"][:, inds]
        naive_errors[id(p)] = (torch.as_tensor(pa_errors(output, target)),
                               torch.as_tensor(np.linalg.norm(output - target, axis=-1).mean(-1)))
    actions = sorted({p["action"] for p in train})
    for epoch in range(start, epochs + 1):
        order = actions.copy()
        random.shuffle(order)
        losses, train_pa, train_mpjpe, train_regret = [], [], [], []
        for action in order:
            group = [p for p in train if p["action"] == action]
            optimizer.zero_grad()
            action_losses = []
            action_pa, action_mpjpe, action_regret = [], [], []
            for p in group:
                inds = [p["names"].index(n) for n in BODY]
                target = torch.as_tensor(p["target"][:, inds])
                pred = torch_fuse(p, weights, std)[:, inds]
                pa = torch_pa_loss(pred, target, per_frame=True)
                mpjpe = torch.linalg.vector_norm(pred - target, dim=-1).mean(-1)
                loss, regret = naive_relative_loss(pa, mpjpe, *naive_errors[id(p)], opts)
                (loss / len(group)).backward()
                action_losses.append(float(loss.detach()) * 1000)
                action_pa.append(float(pa.detach().mean()) * 1000)
                action_mpjpe.append(float(mpjpe.detach().mean()) * 1000)
                action_regret.append(float(regret.detach()) * 1000)
            (opts["prior_l2"] * (weights - prior).square().sum()).backward()
            optimizer.step()
            with torch.no_grad():
                # ponytail: all 8 antisymmetric features are defined >0 = Master better; forbid inverted negative weights
                weights.clamp_(min=0.0)
                weights[~torch.as_tensor(active)] = 0.0
            losses.append(float(np.mean(action_losses)))
            train_pa.append(float(np.mean(action_pa)))
            train_mpjpe.append(float(np.mean(action_mpjpe)))
            train_regret.append(float(np.mean(action_regret)))
        model = export_model(weights.detach().numpy(), std.numpy(), active)
        metrics = validation(val, model, config)
        key = selection_key(metrics, naive, opts)
        beats_naive, score = not key[0], key[1]
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        improved = best_model is None or key < (not best_model["val_beats_naive"], best)
        if improved:
            best, best_model = score, dict(model, epoch=epoch, val_objective_mm=score,
                                           val_mpjpe_mm=metrics["pa_mm"],
                                           val_sequence_mpjpe_mm=metrics["mpjpe_mm"],
                                           equal_val_mpjpe_mm=equal["pa_mm"],
                                           baseline_val_mpjpe_mm=metrics["mono_pa_mm"],
                                           naive_val_mpjpe_mm=naive["pa_mm"],
                                           naive_val_sequence_mpjpe_mm=naive["mpjpe_mm"],
                                           val_pa_gain_vs_naive_mm=naive["pa_mm"] - metrics["pa_mm"],
                                           val_mpjpe_gain_vs_naive_mm=naive["mpjpe_mm"] - metrics["mpjpe_mm"],
                                           val_beats_naive=beats_naive)
        if key[0] < stopping_best[0] or (key[0] == stopping_best[0] and score < stopping_best[1] - opts["min_delta_mm"]):
            stopping_best, stale = key, 0
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_mpjpe_mm=float(np.mean(train_pa)),
                            train_sequence_mpjpe_mm=float(np.mean(train_mpjpe)),
                            train_naive_regret_mm=float(np.mean(train_regret)),
                            train_objective_mm=float(np.mean(losses)), val_objective_mm=score,
                            val_mpjpe_mm=metrics["pa_mm"], val_sequence_mpjpe_mm=metrics["mpjpe_mm"],
                            equal_val_mpjpe_mm=equal["pa_mm"], equal_val_sequence_mpjpe_mm=equal["mpjpe_mm"],
                            depth_val_mpjpe_mm=depth["pa_mm"], depth_val_sequence_mpjpe_mm=depth["mpjpe_mm"],
                            naive_val_mpjpe_mm=naive["pa_mm"], naive_val_sequence_mpjpe_mm=naive["mpjpe_mm"],
                            val_pa_gain_vs_naive_mm=naive["pa_mm"] - metrics["pa_mm"],
                            val_mpjpe_gain_vs_naive_mm=naive["mpjpe_mm"] - metrics["mpjpe_mm"],
                            val_beats_naive=beats_naive, best_val_beats_naive=best_model["val_beats_naive"],
                            baseline_val_mpjpe_mm=metrics["mono_pa_mm"],
                            baseline_val_sequence_mpjpe_mm=metrics["mono_mpjpe_mm"],
                            stale_epochs=stale, best_epoch=best_model["epoch"],
                            best_val_mpjpe_mm=best_model["val_mpjpe_mm"], best_val_objective_mm=best))
        checkpoint = dict(epoch=epoch, weights=weights.detach().clone(), std=std.numpy(),
                          optimizer=optimizer.state_dict(), signature=signature, history=history,
                          best_val_mm=best, best_model=best_model,
                          stale_epochs=stale, stopping_best=stopping_best,
                          random_state=random.getstate(), torch_rng=torch.get_rng_state())
        temp = out / "checkpoint.tmp"
        torch.save(checkpoint, temp)
        temp.replace(out / f"epoch_{epoch:04d}.pt")
        atomic_json(out / f"epoch_{epoch:04d}_weights.json", model)
        atomic_json(out / "history.json", history)
        atomic_json(out / "last.json", {"checkpoint": f"epoch_{epoch:04d}.pt", "signature": signature, "epoch": epoch})
        save_best(out, config, best_model)
        print(json.dumps(history[-1]), flush=True)
        if stale >= opts["patience"]:
            print(json.dumps({"status": "early_stopped", "epoch": epoch, "best_epoch": best_model["epoch"],
                              "val_beats_naive": best_model["val_beats_naive"]}), flush=True)
            break
    print(json.dumps({"status": "finished", "best_epoch": best_model["epoch"],
                      "val_beats_naive": best_model["val_beats_naive"],
                      "note": "Validation gate only; independent coco17 benchmark still required."}), flush=True)
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
    config = training_config(yaml.safe_load((ROOT / "configs/pipeline.yml").read_text(encoding="utf-8")))
    # Validation uses inference guards/stability; optimization remains a differentiable surrogate.
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
    identity = preparation_identity(rows, config, args.skip_unaligned_actions)
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
        "metric": "12-body-joint PA-MPJPE and baseline-sequence-aligned MPJPE (mm), not coco17",
        "selection": "Prefer checkpoints beating naive on BOTH PA and MPJPE by naive_margin_mm; then minimize PA + mpjpe_weight * MPJPE. Equal segment/action weight.",
        "loss": "PA + mpjpe_weight * MPJPE + naive_regret_weight * mean(relu(PA_frame - naive_PA_frame + margin) + mpjpe_weight * relu(MPJPE_frame - naive_MPJPE_frame + margin)) + zero-prior L2 (non-negative weights); train references only",
        "naive_reference": config["naive_reference"],
        "training": training_options(config),
        "target_alignment": "one mono-derived rotation per action/camera on all associated frames (12 body joints)",
        "surrogate": "gradient step excludes non-differentiable inference guards/stability; validation includes them",
        "association": "GT-assisted proper-rotation PA timeline search and penalized track stitching; ratio 0.5/1/2, integer offset",
        "limitations": "GT-assisted time/track association; review alignment.json. No independent test score.",
        "feature_names": list(FEATURE_NAMES), "config": config,
        "visibility": {"enabled": bool(config["occlusion"].get("enabled", False)),
                       "source": "3D mesh occlusion at original camera frames, aligned with selected tracks; interpolated visibility requires both endpoints visible. No confidence2d input."},
        "train_actions": sorted({p["action"] for p in pairs if p["split"] == "train"}),
        "val_actions": sorted({p["action"] for p in pairs if p["split"] == "val"}),
        "skip_unaligned_actions": args.skip_unaligned_actions,
        "skipped_actions": [r["action"] for r in reports if r.get("status") == "skipped_action"],
        "synchronization": "linear interpolation to shared GT ticks, excluding missing-track gaps",
        "directed_segments": len(pairs),
        "split_counts": {split: {
            "actions": len({p["action"] for p in pairs if p["split"] == split}),
            "directed_camera_pairs": len({(p["action"], *p["cameras"]) for p in pairs if p["split"] == split}),
            "directed_segments": sum(p["split"] == split for p in pairs),
            "segment_frames": sum(len(p["M"]) for p in pairs if p["split"] == split),
        } for split in ("train", "val")},
    })
    if args.prepare_only:
        print("Prepared data and alignment report:", out / "alignment.json")
        return
    fit(pairs, config, out, args.epochs, args.lr, args.resume, identity)


if __name__ == "__main__":
    main()

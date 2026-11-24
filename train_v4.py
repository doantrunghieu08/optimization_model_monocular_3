"""Supervised calibration of partwise weights from the v4 Sheets export.

GT is used for track/time association and training targets, never for inference.
Alignment reports must be reviewed before interpreting validation scores.
"""
import argparse
import copy
import hashlib
import itertools
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from fusion_pipeline.aotr import _ordered_bones
from fusion_pipeline.partwise import (
    FEATURE_NAMES, blend_frame, eval_weight_model, extract_pair_features,
    fuse_partwise, slave_directions,
)
from keypoints_map import load_keypoints3d_map

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


def download(drive, value, cache):
    from googleapiclient.http import MediaIoBaseDownload
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / file_id(value)
    if not path.exists():
        temp = path.with_suffix(".download")
        with temp.open("wb") as stream:
            reader = MediaIoBaseDownload(stream, drive.files().get_media(fileId=path.name))
            done = False
            while not done:
                _, done = reader.next_chunk()
        temp.replace(path)
    return path


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
    source = source - source.mean(1, keepdims=True)
    target = target - target.mean(1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum("tji,tjk->tik", source, target))
    correction = np.broadcast_to(np.eye(3), u.shape).copy()
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
    costs = np.full((len(all_ids), len(keys), len(gt)), np.inf)
    joints_at = {}
    for ki, key in enumerate(keys):
        ids, joints = tracks[key]
        ids = np.asarray(ids, dtype=int)
        if len(np.unique(ids)) != len(ids) or len(ids) != len(joints):
            raise ValueError(f"Invalid/duplicate frame ids for track {key}")
        positions = np.searchsorted(all_ids, ids)
        source = np.repeat(joints[:, None, body], len(gt), axis=1)
        target = np.broadcast_to(gt[None, :, body], source.shape)
        costs[positions, ki] = pa_errors(source.reshape(-1, len(body), 3),
                                        target.reshape(-1, len(body), 3)).reshape(len(ids), len(gt))
        for pos, joint in zip(positions, joints):
            joints_at[(int(pos), ki)] = joint
    roots = np.zeros((len(all_ids), len(keys), 3))
    for (pos, ki), joints in joints_at.items():
        roots[pos, ki] = joints[[names.index("left_hip"), names.index("right_hip")]].mean(0)
    continuity = {}
    candidates = []
    for ratio in ratios:
        mapped = np.rint((all_ids - all_ids.min()) * ratio).astype(int)
        for offset in range(-int(mapped.max()), len(gt)):
            gi = mapped + offset
            keep = np.where((gi >= 0) & (gi < len(gt)))[0]
            if len(keep) < max(16, int(np.ceil(0.7 * len(all_ids)))):
                continue
            unique_gt, unique = np.unique(gi[keep], return_index=True)
            positions = keep[unique]
            coverage = min(1.0, (float(np.ptp(unique_gt)) + ratio) / len(gt))
            if len(unique_gt) < 16 or coverage < 0.5:
                continue
            local = costs[positions, :, unique_gt]
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
        raise AlignmentError("No alignment covers >=70% camera frames and >=50% GT timeline")
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


def prepare(rows, drive, cache, config, alignment_path=None):
    validate_rows(rows)
    mapping = load_keypoints3d_map(str(ROOT / "configs/keypoints3D_map.yml"))
    names = [k["name"] for k in mapping["keypoints"]]
    reg = np.load(ROOT / "models/J_regressor_body25_extended.npy")
    reg = reg[[k["regressor_index"] for k in mapping["keypoints"]]]
    pairs, reports = [], []
    for action in dict.fromkeys(r["action"] for r in rows):
        cameras = []
        group = sorted((r for r in rows if r["action"] == action), key=lambda r: int(r["cam"][3:]))
        if len({file_id(r["GT"]) for r in group}) != 1:
            raise ValueError(f"Different GT files in one sequence: {action}")
        with np.load(download(drive, group[0]["GT"], cache), allow_pickle=False) as data:
            gt = joints_from_vertices(data["verts"], reg)
        for row in group:
            raw = joblib.load(download(drive, row["file PKL"], cache))
            tracks = {}
            for key, track in raw.items():
                if isinstance(track, dict) and "verts" in track and "frame_ids" in track:
                    tracks[str(key)] = (track["frame_ids"], joints_from_vertices(track["verts"], reg))
            if not tracks:
                raise ValueError(f"Expected WHAM tracks with camera-frame verts: {action}/{row['cam']}")
            print(f"Aligning {action}/{row['cam']}", flush=True)
            try:
                gi, joints, report = associate(tracks, gt, names)
            except AlignmentError as exc:
                report = dict(status="rejected", reason=str(exc), coverage=None, pa_mm=None)
            else:
                report = dict(status="accepted", **report)
                cameras.append((row["cam"], gi, joints))
            reports.append(dict(action=action, cam=row["cam"], **report))
            print(json.dumps(reports[-1]), flush=True)
            if alignment_path is not None:
                atomic_json(Path(alignment_path), reports)
        if len(cameras) < 2:
            raise ValueError(f"{action}: need at least two aligned cameras; got {len(cameras)}. "
                             "Inspect rejected cameras in alignment.json and correct their data/timeline.")
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
            raise ValueError(f"No synchronized segments for {action}; inspect camera alignment")
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


def validation(pairs, model, config):
    # Equal sequence weight prevents sequences with more camera pairs dominating selection.
    scores, baselines = {}, {}
    for p in pairs:
        cfg = copy.deepcopy(config)
        cfg["weight_model"] = model
        fused, _ = fuse_partwise(p["M"], p["S"], cfg)
        names = p["names"]
        inds = [names.index(n) for n in BODY]
        output = centered(np.array([[f[n] for n in names] for f in fused]), names)
        master = centered(np.array([[f[n] for n in names] for f in p["M"]]), names)
        for values, pred in ((scores, output), (baselines, master)):
            values.setdefault(p["action"], []).append(float(np.linalg.norm(
                pred[:, inds] - p["target"][:, inds], axis=-1).mean() * 1000))
    return (float(np.mean([np.mean(v) for v in scores.values()])),
            float(np.mean([np.mean(v) for v in baselines.values()])))


def atomic_json(path, value):
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
                target = torch.as_tensor(p["target"])
                loss = torch.linalg.vector_norm(torch_fuse(p, weights, std)[:, inds] - target[:, inds], dim=-1).mean()
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
                                           baseline_val_mpjpe_mm=baseline)
        history.append(dict(epoch=epoch, train_mpjpe_mm=float(np.mean(losses)),
                            val_mpjpe_mm=score, baseline_val_mpjpe_mm=baseline))
        checkpoint = dict(epoch=epoch, weights=weights.detach().clone(), std=std.numpy(),
                          optimizer=optimizer.state_dict(), signature=signature, history=history,
                          best_val_mm=best, best_model=best_model,
                          random_state=random.getstate(), torch_rng=torch.get_rng_state())
        temp = out / "checkpoint.tmp"
        torch.save(checkpoint, temp)
        temp.replace(out / f"epoch_{epoch:04d}.pt")
        atomic_json(out / f"epoch_{epoch:04d}_weights.json", model)
        atomic_json(out / "history.json", history)
        atomic_json(out / "last.json", {"checkpoint": f"epoch_{epoch:04d}.pt"})
        atomic_json(out / "best_weights.json", best_model)
        patch = copy.deepcopy(config)
        patch["weight_model"] = {"enabled": True, "path": str((out / "best_weights.json").resolve())}
        (out / "best_partwise.yml").write_text(yaml.safe_dump(
            {"fusion": {"method": "partwise", "partwise": patch}}, sort_keys=False), encoding="utf-8")
        print(json.dumps(history[-1]), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", default="inputs/v4_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_rows(rows)
    config = yaml.safe_load((ROOT / "configs/pipeline.yml").read_text(encoding="utf-8"))["fusion"]["partwise"]
    config.pop("weight_model", None)
    # Sheet lacks visibility/tracking results: match training and inference evidence.
    config["multi_person"] = {"enabled": False}
    from google.auth import default
    from googleapiclient.discovery import build
    credentials, _ = default()
    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    identity = hashlib.sha256(Path(args.manifest).read_bytes() +
        (ROOT / "models/J_regressor_body25_extended.npy").read_bytes() +
        Path(__file__).read_bytes() + (ROOT / "fusion_pipeline/partwise.py").read_bytes() +
        (ROOT / "configs/keypoints3D_map.yml").read_bytes() + json.dumps(config, sort_keys=True).encode()).hexdigest()
    prepared = cache / f"prepared_{identity}.joblib"
    if prepared.exists():
        pairs, reports = joblib.load(prepared)
    else:
        pairs, reports = prepare(rows, drive, cache, config, out / "alignment.json")
        cache.mkdir(parents=True, exist_ok=True)
        temp = prepared.with_suffix(".tmp")
        joblib.dump((pairs, reports), temp)
        temp.replace(prepared)
    atomic_json(out / "alignment.json", reports)
    atomic_json(out / "manifest.json", rows)
    atomic_json(out / "protocol.json", {
        "metric": "12-body-joint root-centered MPJPE, one GT rotation per camera-pair segment",
        "selection": "mean of pair scores within each action, then mean across val actions",
        "association": "GT-assisted proper-rotation PA timeline search and penalized track stitching; ratio 0.5/1/2, integer offset",
        "limitations": "GT-assisted time/track association; review alignment.json. No independent test score.",
        "feature_names": list(FEATURE_NAMES), "config": config,
        "train_actions": sorted({p["action"] for p in pairs if p["split"] == "train"}),
        "val_actions": sorted({p["action"] for p in pairs if p["split"] == "val"}),
        "synchronization": "linear interpolation to shared GT ticks, excluding missing-track gaps",
        "directed_segments": len(pairs),
    })
    if args.prepare_only:
        print("Prepared data and alignment report:", out / "alignment.json")
        return
    fit(pairs, config, out, args.epochs, args.lr, args.resume, identity)


if __name__ == "__main__":
    main()

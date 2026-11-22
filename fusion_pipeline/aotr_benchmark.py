"""Independent benchmark identity and AOTR activation checks.

Standard-library only: usable by the existing brute-force runner without
bringing neural models or evaluation labels into the fusion stages.
"""
import hashlib
import json
from pathlib import Path

LEGACY_PROTOCOL = "no_gt_v10_master_only_postopt"
# v3: sequence stability also covers the real two-view AOTR path.
# Invalidate results where enabled acceleration control was silently skipped.
AOTR_REVISION = "aotr_fk_trustregion_v3_all_evidence_stable"
PARTWISE_REVISION = "partwise_depth_logit_v1"
NAIVE_REVISION = "naive_kabsch_average_v1"
SEQUENCE_METHODS = ("partwise", "naive")  # replace camera1 after the per-frame AOTR pass


ROOT = Path(__file__).resolve().parents[1]  # relative mask paths: same base as evaluation_pipeline


def fusion_method(config):
    return str(config.get("fusion", {}).get("method", "aotr")).strip().lower()


def identity_mask_exclusions(id_mask):
    """Frames the evaluator drops: {pkl stem: {person: [frame ids]}}.

    Mirrors evaluation_pipeline._identity_excluded_ids: the mask files are merged by
    pkl stem (a later file wins) and a frame is dropped when margin > margin_mm. Only
    this set changes the scores, so rebuilding a mask with the same dropped frames (new
    "source" path, other margins below the threshold) keeps the protocol. A missing or
    unreadable mask file is returned in `problems` instead of raising.
    """
    paths = id_mask["path"] if isinstance(id_mask["path"], (list, tuple)) else [id_mask["path"]]
    margin = float(id_mask.get("margin_mm", 10.0))
    merged, problems = {}, []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        try:
            merged.update(json.loads(path.read_text(encoding="utf-8")).get("margins", {}))
        except (OSError, ValueError, AttributeError) as exc:
            problems.append(f"{raw}: {type(exc).__name__}")
    excluded = {}
    for stem, per_person in merged.items():
        for person, frames in (per_person or {}).items():
            try:
                ids = sorted(int(fid) for fid, value in frames.items() if float(value) > margin)
            except (AttributeError, TypeError, ValueError) as exc:
                problems.append(f"{stem}/{person}: {type(exc).__name__}")
                continue
            if ids:
                excluded.setdefault(str(stem), {})[str(person).lower()] = ids
    return excluded, problems


def aotr_enabled(config):
    return bool(config.get("fusion", {}).get("aotr", {}).get("enabled", False))


def evaluation_protocol(config):
    if not aotr_enabled(config):
        return LEGACY_PROTOCOL
    # A cached result is reusable only with exactly the same AOTR settings.
    inputs = config.get("inputs", {})
    person = str(inputs.get("target_person", "aria01")).strip().lower()
    if person in {"aria1", "aria_1", "aria-1"}:
        person = "aria01"
    elif person in {"aria2", "aria_2", "aria-2"}:
        person = "aria02"
    heights = inputs.get("person_heights", {})
    heights_lower = {str(k).strip().lower(): v for k, v in heights.items()} if isinstance(heights, dict) else {}
    p_val = heights_lower.get(person) if heights_lower else None
    height = p_val.get("height_m") if isinstance(p_val, dict) else p_val
    if height is None:
        height = inputs.get("person_height_m")
    settings = {
        "aotr": config["fusion"]["aotr"],
        "target_person": person,
        "person_height_m": height,
    }
    occ_cfg = config.get("fusion", {}).get("occlusion", {})
    if isinstance(occ_cfg, dict) and occ_cfg.get("multi_person", False):
        settings["occlusion"] = {
            "multi_person": True,
            "tau": float(occ_cfg.get("tau", 0.05)),
            "inter_person_tau": float(occ_cfg.get("inter_person_tau", 0.15)),
            "inter_person_radius_m": float(occ_cfg.get("inter_person_radius_m", 0.045)),
            "inter_person_min_hits": int(occ_cfg.get("inter_person_min_hits", 3)),
        }
    method = fusion_method(config)
    revision = AOTR_REVISION
    if method == "partwise":
        revision = PARTWISE_REVISION
        settings["partwise"] = config["fusion"].get("partwise", {})
    elif method == "naive":
        revision = NAIVE_REVISION
        settings["naive"] = {"master_weight": float(config["fusion"].get("naive", {}).get("master_weight", 0.5))}
    # Excluding identity-swapped frames changes the metrics: never reuse unmasked caches.
    id_mask = config.get("evaluation", {}).get("identity_mask")
    if id_mask and id_mask.get("path"):
        # The CONTENT counts too: a rebuilt mask that drops other frames scores differently,
        # even under the same file name, so results cached with the old mask are stale.
        excluded, problems = identity_mask_exclusions(id_mask)
        settings["identity_mask"] = {
            "path": str(id_mask["path"]), "margin_mm": id_mask.get("margin_mm", 10.0),
            "excluded_frames_sha256": hashlib.sha256(
                json.dumps(excluded, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "unreadable": problems,
        }
    # Scoring definition changes the numbers: joint set and MPJPE alignment are part of the identity.
    ev = config.get("evaluation", {})
    if ev.get("joint_set", "priority1") != "priority1" or ev.get("mpjpe_alignment", "frame") != "frame":
        settings["scoring"] = {"joint_set": ev.get("joint_set", "priority1"),
                               "mpjpe_alignment": ev.get("mpjpe_alignment", "frame")}
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256((revision + ":" + payload).encode("utf-8")).hexdigest()[:12]
    return f"{LEGACY_PROTOCOL}__{revision}__{digest}"


def report_csv_name(config):
    if aotr_enabled(config):
        method = fusion_method(config)
        prefix = method if method in SEQUENCE_METHODS else "aotr"
        return f"brute_force_report_{prefix}_" + evaluation_protocol(config).rsplit("__", 1)[-1] + ".csv"
    return "brute_force_report.csv"


def cache_compatible(record, expected_protocol):
    return bool(record) and record.get("protocol") == expected_protocol


def verify_aotr_execution(config):
    """Reject mislabeled results, missing fusion runs and all-fallback runs.

    3D ground truth is not accessed. Diagnostics must have been written
    by the active fusion stage for the CURRENT camera pair.
    """
    if not aotr_enabled(config):
        return None
    report = Path(config["paths"]["fused_output_dir"]) / "aotr_diagnostics.json"
    if not report.is_file():
        raise RuntimeError(
            "AOTR was enabled in config, but aotr_diagnostics.json is absent. "
            "Fusion did not execute AOTR; cannot label these metrics as AOTR."
        )
    summary = json.loads(report.read_text(encoding="utf-8"))
    if fusion_method(config) == "partwise":
        return _verify_partwise(config, summary)
    if fusion_method(config) == "naive":
        return _verify_naive(config, summary)
    frames = int(summary.get("frames", 0))
    corrected = int(summary.get("corrected_frames", 0))
    fallbacks = int(summary.get("fallback_frames", 0))
    if summary.get("mode") != "aotr" or frames <= 0:
        raise RuntimeError("AOTR output diagnostics are invalid: " + str(summary))
    if fallbacks >= frames:
        raise RuntimeError(
            "All AOTR frames fell back to Master; this is not an AOTR benchmark. "
            f"Reasons: {summary.get('reason_counts', {})}. "
            "Fix input/intrinsics/synchronization before benchmarking."
        )
    stability_enabled = bool(
        config["fusion"]["aotr"].get("sequence_stability", {}).get("enabled", False)
    )
    stability_status = summary.get("sequence_stability", {}).get("status")
    if stability_enabled and stability_status not in {"applied", "unchanged"}:
        raise RuntimeError(
            "AOTR sequence stability was enabled but did not run: "
            + str(summary.get("sequence_stability", {}))
        )
    return summary


def _verify_partwise(config, summary):
    """Partwise replaces camera1 for every frame; its own diagnostics must exist and match."""
    report = Path(config["paths"]["fused_output_dir"]) / "partwise_diagnostics.json"
    if summary.get("mode") != "partwise" or not report.is_file():
        raise RuntimeError("fusion.method=partwise but partwise diagnostics are missing: " + str(summary))
    diag = json.loads(report.read_text(encoding="utf-8"))
    if int(diag.get("frames", 0)) <= 0 or int(diag["frames"]) != int(summary.get("frames", -1)):
        raise RuntimeError("Partwise diagnostics do not cover the current pair: " + str(diag))
    stab_enabled = bool(config["fusion"].get("partwise", {}).get("sequence_stability", {}).get("enabled", True))
    status = diag.get("sequence_stability", {}).get("status")
    if stab_enabled and status not in {"applied", "unchanged"}:
        raise RuntimeError("Partwise sequence stability was enabled but did not run: " + str(diag))
    return summary


def _verify_naive(config, summary):
    """Naive baseline replaces camera1 for every frame; its diagnostics must cover the current pair."""
    report = Path(config["paths"]["fused_output_dir"]) / "naive_diagnostics.json"
    if summary.get("mode") != "naive" or not report.is_file():
        raise RuntimeError("fusion.method=naive but naive diagnostics are missing: " + str(summary))
    diag = json.loads(report.read_text(encoding="utf-8"))
    if int(diag.get("frames", 0)) <= 0 or int(diag["frames"]) != int(summary.get("frames", -1)):
        raise RuntimeError("Naive diagnostics do not cover the current pair: " + str(diag))
    return summary

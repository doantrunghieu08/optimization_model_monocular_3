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


def fusion_method(config):
    return str(config.get("fusion", {}).get("method", "aotr")).strip().lower()


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
    method = fusion_method(config)
    revision = AOTR_REVISION
    if method == "partwise":
        revision = PARTWISE_REVISION
        settings["partwise"] = config["fusion"].get("partwise", {})
    # Excluding identity-swapped frames changes the metrics: never reuse unmasked caches.
    id_mask = config.get("evaluation", {}).get("identity_mask")
    if id_mask and id_mask.get("path"):
        settings["identity_mask"] = {"path": str(id_mask["path"]), "margin_mm": id_mask.get("margin_mm", 10.0)}
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
        prefix = "partwise" if fusion_method(config) == "partwise" else "aotr"
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

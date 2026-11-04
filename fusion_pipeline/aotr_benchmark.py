"""Independent benchmark identity and AOTR activation checks.

Standard-library only: usable by the existing brute-force runner without
bringing neural models or evaluation labels into the fusion stages.
"""
import hashlib
import json
from pathlib import Path

LEGACY_PROTOCOL = "no_gt_v10_master_only_postopt"
# v2: AOTR now bypasses legacy smoothing and unconditional anatomy rewriting.
# Invalidate pre-fix cached CSVs and Google Sheet rows regardless of config.
AOTR_REVISION = "aotr_fk_trustregion_v2_output_isolated"


def aotr_enabled(config):
    return bool(config.get("fusion", {}).get("aotr", {}).get("enabled", False))


def evaluation_protocol(config):
    if not aotr_enabled(config):
        return LEGACY_PROTOCOL
    # A cached result is reusable only with exactly the same AOTR settings.
    settings = config["fusion"]["aotr"]
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256((AOTR_REVISION + ":" + payload).encode("utf-8")).hexdigest()[:12]
    return f"{LEGACY_PROTOCOL}__{AOTR_REVISION}__{digest}"


def report_csv_name(config):
    if aotr_enabled(config):
        return "brute_force_report_aotr_" + evaluation_protocol(config).rsplit("__", 1)[-1] + ".csv"
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
    frames = int(summary.get("frames", 0))
    corrected = int(summary.get("corrected_frames", 0))
    fallbacks = int(summary.get("fallback_frames", 0))
    if summary.get("mode") != "aotr" or frames <= 0:
        raise RuntimeError("AOTR output diagnostics are invalid: " + str(summary))
    if fallbacks >= frames:
        raise RuntimeError(
            "All AOTR frames fell back to Master. "
            f"Reasons: {summary.get('reason_counts', {})}. "
            "Fix input/intrinsics/synchronization before benchmarking."
        )
    return summary

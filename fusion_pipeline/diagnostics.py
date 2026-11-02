"""Explain why experimental selective fusion abstains.

Diagnostic counts are NOT evidence of 3D accuracy. Use a fixed held-out GT
benchmark to compare real gains and losses against the legacy fusion pipeline.
"""
from collections import Counter, defaultdict
from math import isfinite


def summarize_selective_frames(frame_results):
    """Aggregate frame/joint decisions without depending on GT or Torch."""
    decision_counts = Counter()
    rejection_reasons = Counter()
    by_joint = defaultdict(Counter)
    invalid_alignment = 0
    frame_fallbacks = 0
    residual_medians = []
    rejected_examples = []

    for frame_index, record in enumerate(frame_results):
        modes = record.get("joint_selection")
        if not isinstance(modes, dict) or not modes:
            frame_fallbacks += 1
            continue
        selection = record.get("selection_summary") or {}
        if not selection.get("alignment_valid", False):
            invalid_alignment += 1
        residual = selection.get("alignment_median_m")
        if isinstance(residual, (int, float)) and isfinite(residual):
            residual_medians.append(float(residual))
        details = record.get("joint_selection_details") or {}
        for joint, mode in modes.items():
            decision_counts[str(mode)] += 1
            by_joint[joint][str(mode)] += 1
            if mode != "KEEP":
                continue
            reason_info = details.get(joint, "no_independent_advantage")
            reason = (
                reason_info.get("reason", "unspecified")
                if isinstance(reason_info, dict) else str(reason_info)
            )
            rejection_reasons[reason] += 1
            by_joint[joint]["reject_" + reason] += 1
            if len(rejected_examples) < 20:
                rejected_examples.append({
                    "frame_position": frame_index,
                    "joint": joint,
                    "reason": reason,
                })

    accepted = decision_counts["BLEND"] + decision_counts["REPLACE"]
    total = sum(decision_counts.values())
    summary = {
        "frames": len(frame_results),
        "frames_with_missing_decisions_or_fallback": frame_fallbacks,
        "frames_with_invalid_alignment": invalid_alignment,
        "joint_frames_evaluated": total,
        "joint_frames_accepted": accepted,
        "joint_acceptance_rate": accepted / total if total else 0.0,
        "decisions": dict(sorted(decision_counts.items())),
        "keep_reasons": dict(sorted(rejection_reasons.items(), key=lambda v: -v[1])),
        "alignment_median_residual_m": (
            sorted(residual_medians)[len(residual_medians) // 2]
            if residual_medians else None
        ),
        "per_joint": {j: dict(sorted(v.items())) for j, v in sorted(by_joint.items())},
        "rejected_examples": rejected_examples,
        "warning": (
            "These are acceptance/rejection statistics only. They do not "
            "measure PA-MPJPE, MPJPE, MBLE or positive/negative transfer."
        ),
    }
    return summary

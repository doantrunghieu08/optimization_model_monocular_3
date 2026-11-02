# Adaptive joint/frame camera protection (zero-GT)

The default fusion path now **protects each joint on each frame** instead of
classifying entire camera pairs into protected/unprotected modes.

For a target joint at a given frame, the transfer reliability uses (a) detector
belief, (b) actual ray-cast / 2D visibility, (c) excess frame-to-frame joint
acceleration relative to the other camera, and (d) rigid-view registration
quality and kinematic bone-vector disagreement:

    score_c = confidence_c * joint_quality_c * visibility_factor_c
    agreement = 1 / (1 + (disagreement / scale)^2)
    transfer_weight = clamp(
        score_source / (score_source + score_target) *
        agreement * transform_reliability, 0, max_transfer)

The transfer weight decays *continuously* with disagreement, rather than
switching off beyond legacy 5cm / 8cm / 15cm thresholds. The method gives more
authority to a visible, consistent camera when its counterpart is occluded.
Pelvis and neck remain structural anchors. Bone lengths are projected from
the corresponding input view, with checks only for hinges affected by a move.
Temporal excess acceleration is not measured across discontinuous source frames.

The sequence GPU pass remains truly multi-view even for equal-quality cameras.
It uses reciprocal, nonzero per-joint mobility factors (more reliable view moves
less), and gates the view-consistency loss by accepted joint agreement.
Legacy final-stage master reset is disabled for the adaptive path.

## Configuration

Under \`fusion.adaptive_protection\` (see \`configs/pipeline.yml\`):
- \`enabled: true\`: default adaptive path; set false for legacy ablation.
- \`consensus_scale_m: 0.18\`, \`occluded_scale_m: 0.55\`: soft geometric decay.
- \`transform_scale_m: 0.20\`: soft registration penalty.
- \`max_visible_transfer: 0.55\`, \`max_occluded_transfer: 0.90\`: max blend.
- \`min_source_reliability: 0.12\`: source-quality safety floor.
- \`jitter_ratio: 1.35\`, \`jitter_floor_m: 0.025\`, \`jitter_scale_m: 0.08\`:
  view-specific excess-jitter checks (m/frame²).
- \`min_gpu_mobility: 0.05\`: keeps every joint potentially optimizable.

Legacy per-pair modes (including explicit \`posed_fallback\`) remain available.
Legacy pair-quality decision is retained for diagnostics only when adaptive mode
is enabled; it never disables adaptive GPU multiview refinement.

## Verification

    python -m unittest tests.test_adaptive_protection
    python -m pytest tests/

Run a controlled Aria02 ablation on **identical synchronized frames**:
A0: adaptive disabled; A1: adaptive enabled, optimization disabled;
A2: adaptive enabled, optimization enabled. Set
\`inputs.target_person: "aria02"\` before running those experiments.
Compare PA-MPJPE, MPJPE, acceleration error, MBLE, per-joint occlusion accuracy
and the count of camera pairs with negative transfer. Read
\`joint_quality\`, \`joint_transfer\`, and \`adaptive_stats\` in frame metadata
for debugging. None of these values is inferred from evaluation ground truth.

These defaults are initial engineering choices, not calibrated Aria02
hyperparameters. No claim of exceeding the reported 14% improvement is made
without rerunning benchmark data.


## M2 + M4 + M5: Zero-GT bone-length reliability and risk-aware kinematics

**Status:** implementation added after the A0–A5/L0 Aria02 report.
Main-branch's zero-GT camera-pair quality diagnostics and poses3d GT path
are preserved in the merge; global quality fallback is legacy-only when
adaptive joint-level gating is active.
**Not yet benchmarked:** do not ascribe those earlier metrics to this revision.

### M2 — View-local bone-length quality

For each child-parent bone, estimate a reference length by the **median of
valid frames in that camera**, requiring at least three observations.
There is no shared camera scale assumption and ground truth is never used.

At frame t:

    e_len = |L_t - median(L)| / median(L)
    q_len = 1 / (1 + (max(0, e_len - tolerance_ratio) / residual_scale_ratio)^2)
    e_bilat = |L_left - L_right| / ((L_left + L_right)/2)
    q_bilat = 1 / (1 + (e_bilat / bilateral_scale_ratio)^2)
    q_bone = q_len * (1 - bilateral_weight + bilateral_weight * q_bilat)
    q_joint = q_temporal * q_bone

A missing or statistically insufficient reference gives a neutral score (1.0,
"unknown", not a certification that a joint is accurate). Nonfinite or
degenerate segments score 0. Bilateral consistency is a weak heuristic:
physiological asymmetry, occlusion and erroneous but constant segment lengths
can still fool it. No GT-driven tuning is permitted.

### M4 — Occluded transfer safety

An occluded target is rescued only when the source is actually visible,
the source's bone quality exceeds min_occluded_source_bone_quality,
the estimated cross-camera alignment quality exceeds
min_occluded_alignment_reliability, and at least min_alignment_anchors RANSAC
anchors are present. Additionally, the transfer cap depends smoothly on
source bone quality and alignment reliability, and the candidate must pass
a local hinge sanity check. Both directions are treated symmetrically.
No camera IDs or GT annotations enter these decisions.

### M5 — Kinematic refinement

Candidate joints are processed in kinematic parent-first order and projected
onto *temporal median* bone lengths of the **target camera**; only if no
reference exists does projection fall back to the current raw-frame length.
A large projection relative to nominal length is rejected, as are local
hinge-angle degradations relative to the current partially corrected skeleton.

The downstream initial fusion save, Savitzky-Golay smoothing,
GPU length-constraint bounds and final post-GPU projection now all receive the
**same robust per-camera bone references**. Missing references fall back to
original-frame bone lengths. Legacy mode is unchanged. Compare full M5 on
identical Aria02 inputs; the report supplied before this change predates M5.

### Suggested ablation after implementing these modules

Compare on identical synchronized Aria02 frames and untouched GT:
- A3: preceding adaptive implementation, commit 942273f;
- B0: this implementation with bone_quality.enabled=true (M2+M4+M5);
- B1: bone_quality.enabled=false (M4/M5 without calibrated references);
- B2: no occluded transfer (set max_occluded_transfer=0, with
  min_occluded_source_bone_quality=1.01 to force zero rescues);
- B3: adjust only min_occluded_source_bone_quality on a *validation* split.

Primary diagnostics: PA-MPJPE macro mean and worst-case, count of negative
pairs versus posed baseline, PA-Occ/PA-Vis on comparable visible/occluded
sets and per-joint counts. Inspect metadata bone_quality, joint_quality,
joint_transfer and adaptive_stats for cam19->cam04, cam19->cam05,
cam20->cam04, cam20->cam05.

**Note:** Setting max_occluded_transfer=0 suffices to disable
occluded transfers in the current implementation (the soft blend cap
becomes zero); setting the source-bone threshold above 1 provides an
additional diagnostic guard.

## Full Model v2: M4 + M6 + M7

The default path adds risk-aware protection for **visible as well as occluded**
joints on a per-frame, per-joint basis. For a reliable visible target,
the gate either rejects a sufficiently weak/geometrically inconsistent
source or smoothly attenuates its transfer weight. Both cameras are treated
symmetrically; the pipeline does not set a global master protection flag.

Motion-aware smoothing (M6) builds Savitzky-Golay candidates, then blends
them according to joint reliability and **motion shared by both cameras**.
An isolated noisy spike does not count as shared motion. Nonconsecutive
output frame IDs are treated as separate temporal segments.

The GPU objective (M7) introduces confidence-weighted original-observation
preservation and motion-aware temporal penalties, in addition to the
existing bone, hinge, proximity, and cross-view losses. Raw WHAM poses
serve as observation anchors; GT remains evaluation-only.

Default options under fusion:
- adaptive_protection.risk_aware_enabled: true
- adaptive_protection.visible_risk_enabled: true
- smoothing.motion_aware.enabled: true
- optimization.adaptive_objectives_enabled: true
- optimization.observation_lambda: 2.0

### New ablations

Historical A1-A5 explicitly disable M2/M4/M6/M7 to avoid inheriting v2
defaults; exact historical reproduction should still compare frozen commits.
- B0: Full Model v2 enabled (M2, M4, M5, M6, M7)
- B1: B0 with bone quality disabled
- B2: B0 without occlusion rescue
- B3: B0 without visible-target risk gate
- B4: B0 without motion-aware smoothing
- B5: B0 without adaptive GPU objectives

All hyperparameters are provisional. No Aria02 benchmark has been run on
Full Model v2; historical A0-A5 metrics must not be attributed to v2.

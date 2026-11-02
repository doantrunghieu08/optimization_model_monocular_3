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

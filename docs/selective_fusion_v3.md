# V3 Selective Risk-Aware Joint Fusion (Zero-GT inference)

## Status

This is a **conservative GT-free heuristic**, NOT a learned or calibrated estimator
of the probability that a slave joint improves 3D MPJPE. Do not claim a benchmark
gain without running Harmony4D. The implementation is explicitly designed to
fail closed and preserve Master on ambiguity.

## Enable / rollback

Default in configs/pipeline.yml:

    fusion:
      selective_joint_fusion:
        enabled: true
        quality_margin_visible: 0.10
        quality_margin_occluded: 0.06
        max_displacement_visible: 0.12
        max_displacement_occluded: 0.35
        blend_strength: 0.35
        optimize_selected: false

Set enabled=false to restore the previous pipeline, including original
master protection, global smoothing and sequence optimizer.

## Mechanism

1. Calculate subject-specific median bone lengths from original Master poses
   across the time sequence (never per-frame lengths as the only reference).
2. Detect per-view occlusion and read available 2D keypoints.
3. Fit a camera-to-camera rigid transform from jointly visible torso anchors.
   Abort selective transfer if fewer than three non-collinear anchors survive
   or the median/q75 residual exceeds configured thresholds.
4. Score each Slave and Master joint separately with:
   - visibility (0.40);
   - independent 2D detection confidence & camera reprojection residual (0.30);
   - subject-specific bone length consistency (0.15);
   - temporal constant-velocity consistency (0.15).
5. Require a sufficient quality advantage and conservative displacement
   limits. If both views are visible and there is no independent 2D support,
   require a substantial local anatomical or temporal advantage. The
   distance-derived harmonic belief only provides a minimum support floor.
6. Generate a joint-relative Slave candidate (parent-centered kinematic
   rotation), project to the subject reference bone length when known, and
   reject candidate local geometry if the bone errors or elbow/knee angles
   degrade.
7. Replace an occluded Master joint only with strong Slave support. Otherwise
   soft-blend at strength <=0.5; leave all other Master coordinates untouched.
8. Record each decision in fused metadata:
   joint_selection: {<name>: KEEP | BLEND | REPLACE}
   joint_selection_details: {<name>: reason or scored evidence}
   selection_summary: {accepted, kept, alignment_valid, alignment_median_m}
9. Bypass legacy global smoothing and optimizer by default. They may move
   KEEP joints even when the incoming frame rejected the Slave candidate.
   Optional selective optimizer reads joint_selection and freezes KEEP
   coordinates; enabling it must first pass held-out tests.

**Important limitations**:
- A rigid transform estimated from 3D monocular poses can still be wrong.
  Independently calibrated extrinsics and synchronization would improve this.
- Self-occlusion from predicted SMPL mesh can be wrong. For hugging scenes,
  separately test occlusion.multi_person and compare against real 2D masks.
- Visibility and 2D confidence are not ground-truth 3D accuracy. Some cases
  are inherently ambiguous: the correct behavior is to abstain.
- Distant poor views may miss frame/keypoint tracks and 2D evidence can be
  absent. Those cases should be reported as low selection coverage, not
  "zero error".
- Master untouched by selected-joint fusion means unchanged from exported
  pose, which already uses subject shape fixed from the Master.

## Required benchmark / ablation protocol

Evaluate on held-out subjects and camera pairs. Preserve the exact same
evaluation frame set and GT for all versions. Track per-joint:
- oracle gain: GT-only decision comparing candidate against Master;
- precision and coverage of accepted joint/frame decisions;
- harmful transfer count and severity (including 90th/99th percentile);
- PA-MPJPE and root-aligned MPJPE;
- MBLE and acceleration error;
- number of affected Master joints that are not in accepted mask (must be 0
  before any explicitly opted-in single-view smoothing).
- pair-level negative-transfer rate, including difficult cam19/cam17/cam02
  and high-gain cam03/cam19 examples.

Ablation: legacy/aggressive; legacy/protected; selective (visible+bone);
selective (+2D); selective (+temporal); selective (+strict alignment);
selective (+optional masked optimizer).

Thresholds are initial conservative defaults, not tuned on the test set.

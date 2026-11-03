# AOTR — optimization-based observability-aware trust-region experiment

> Status: **experimental, not benchmark-validated**. Implemented only on
> `experiment/observability-aware-trust-region`. Does not use AI/ML or 3D
> ground truth during inference. **Reducing the surrogate objective is not a
> guarantee of reducing MPJPE or PA-MPJPE.**

## Run

```bash
git fetch origin
git switch experiment/observability-aware-trust-region
git pull --ff-only
python run.py --config configs/pipeline.yml
```

For a rollback **without changing branches**, set `fusion.aotr.enabled: false`
in `configs/pipeline.yml`. This restores legacy fusion (the V3 selector also
remains disabled). Never enable `fusion.aotr` and
`fusion.selective_joint_fusion` simultaneously.

This is an **offline, per-frame first prototype**. It keeps WHAM as the pose
source but does not train or introduce any new neural network.

## Input requirements

- Two synchronized WHAM 3D poses, in their respective camera coordinates.
- Camera intrinsics for both views (via existing calibration resolver).
- The **actual** tracked 2D keypoints for both views. Missing companion
  tracking files cause **fallback to raw Master**, not silent synthetic
  confidence. Copied hand/foot detector aliases are not independent evidence.
- At least three visible, non-collinear torso alignment anchors; median
  and 75th-percentile rigid-fit residual checks.
- At least four 2D correspondences shared between the views.
- Correct identity matching and frame alignment are prerequisites.

**Uncalibrated relative rotation/translation:** this prototype estimates a
rigid transform from WHAM 3D torso anchors. That is a *surrogate*, not
independent calibrated extrinsics. If torso estimates have systematic depth
error, the apparent cross-view observability can be overestimated. Reprojection
must therefore NOT be treated as a certification of 3D improvement.

## Algorithm

1. Validate each camera's 2D points, intrinsics and torso geometry; reject
   unreliable alignment before any corrections.
2. Estimate rigid camera-frame alignment from visible torso anchors
   (RANSAC + unit-scale rigid fit). **Never** fit the alignment using the
   limb currently being corrected.
3. Construct Slave bone orientation candidates in Master coordinates.
   Correct *orientations*, not independent joint positions; parent-to-child
   lengths remain equal to the Master frame via forward kinematics.
4. Build a **weighted two-view projection Jacobian** at Master and derive
   the null-space projector with SVD. Use stronger Master regularization for
   poorly observable directions:
   ```
   E_prior = lambda_null * ||P_null (X-M)||²
           + lambda_obs * ||(I-P_null) (X-M)||²
   ```
5. Define robust two-view geometric merit:
   ```
   E = sum_{camera,joint} confidence * Huber(||(pi_camera(X)-kp2D)/f||)
       + E_prior
   ```
   Only direct detector joints contribute 2D terms. Note this is a surrogate
   and can be minimized while true 3D error increases.
6. Solve **one-dimensional bounded trust-region subproblems** for each bone
   orientation interpolation coefficient alpha in [0,1]. Use actual/predicted
   cost decrease to accept steps and adapt the radius. Multiple starts
   at 0, 0.5, 1 prevent the conservative KEEP initialization from completely
   eliminating large but geometrically supported updates.
7. Reject updates without a positive surrogate reduction. Preserve untouched
   bone orientations. Downstream joints follow FK, are marked PROPAGATE in
   metadata if their positions move; unchanged joints are labelled KEEP.
8. If any prerequisite is missing or unsafe, return the raw Master pose and a
   structured fallback reason.

### Why this bypasses legacy postprocessing

Legacy global Savitzky-Golay smoothing, GPU sequence optimization, bone
reprojection, and hard-coded anatomical limb-ratio rewriting can move joints
that AOTR marked KEEP. When AOTR is enabled all those postprocessors are
**bypassed**. Legacy behavior is unchanged when AOTR is disabled.

### Output and diagnostics

- `output/fused_results/keypoints3d/fused_data_*.json`
- `output/fused_results/metadata/fused_data_*.json`: `aotr`, `joint_selection`,
  accepted bone alpha, normalized objective before/after, alignment stats.
- `output/fused_results/aotr_diagnostics.json`: frame-level accepted/fallback
  counts and reasons.

### Limitations / known risks

- Does **not** correct global root position or camera-pair scale. Thus it may
  improve local shape/PA-MPJPE while leaving root MPJPE unchanged.
- Does **not** jointly optimize temporal trajectories. It avoids the previous
  temporal smoothing's motion suppression, but may leave jitter.
- Skeleton joint updates are coordinate-descent, not a global pose optimum.
- SVD rank depends on the quality of relative extrinsics and 2D tracks.
- **AOTR does not guarantee a non-negative oracle 3D gain.** A candidate with
  lower 2D+prior cost may have worse 3D joint error.
- The Master per-frame bone length is preserved, including pre-existing
  incorrect or time-varying lengths; keeping it constant across frames is
  future work.
- If a joint has no direct Slave observation, AOTR does not modify its own
  bone orientation. Descendants may move by inherited parent changes.
- Master-occluded, Slave-visible joints are eligible (under the same alignment
  and geometric evidence prerequisites), but their depth may remain ambiguous.

## Validation protocol

Run isolated synthetic tests:

```bash
python -m unittest discover -s tests -p 'test_aotr.py' -v
```

Run the project regression suite using the dependencies in the existing
requirements file before attempting real benchmarks.

For Harmony4D **keep the same person, time alignment, frame sets and GT only
in the evaluation stage**. Compare AOTR with:
- baseline single-view WHAM (Master);
- original legacy pipeline / V1;
- V2 and selective V3 (explicitly enabled for its own ablation);
- AOTR with null-space prior disabled vs enabled;
- AOTR with trust expansion disabled vs enabled;
- AOTR without 2D from one view (expect fallback, not an improvement score).

Report PA-MPJPE, MPJPE, root-aligned MPJPE, MBLE, acceleration error,
pair-level negative transfer, p95/p99 harmful deltas, fusion coverage and
gain relative to an **offline, GT-only oracle**. Do not tune on the final
benchmark test set.

**Do not conclude performance improvement without real held-out results.**

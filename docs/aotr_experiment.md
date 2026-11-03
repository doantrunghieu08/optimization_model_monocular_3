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
python main.py --config configs/pipeline.yml
```

For a rollback **without changing branches**, set `fusion.aotr.enabled: false`
in `configs/pipeline.yml`. This restores legacy fusion (the V3 selector also
remains disabled). Never enable `fusion.aotr` and
`fusion.selective_joint_fusion` simultaneously.

This is an **offline, per-frame first prototype**. It keeps WHAM as the pose
source but does not train or introduce any new neural network.

## Missing 2D: experimental WHAM-3D-only optimization

The earlier all-frame `insufficient_2d_correspondences` error is not repaired
by inventing 2D keypoints or setting `min_correspondences: 0`.

The AOTR experiment configuration now uses:

```yaml
fusion:
  aotr:
    enabled: true
    missing_2d_mode: 3d_surrogate
```

When real 2D keypoints are available with valid intrinsics, the original
two-view observability-aware 2D objective is used. When they are missing,
`3d_surrogate` invokes **a different optimization-only branch** using:
- an anchor-derived rigid Master↔Slave alignment (estimated from 3D WHAM
  torso joints, *not* independently calibrated camera extrinsics);
- each camera's limb orientation candidate in its own 3D coordinates;
- mesh visibility and, when available, the angular acceleration of local
  bones over three consecutive WHAM 3D frames;
- a soft asymmetric Master prior and a bounded adaptive one-dimensional
  trust-region search over each bone's interpolation coefficient;
- forward kinematics to preserve the original Master bone length.

The 3D-only surrogate has **NO 2D observability certificate** and **NO
guarantee of lower MPJPE**. Stable but biased Slave poses can still harm
Master; this mode is for measured ablation only. An experimental
`three_d_consensus_weight` also permits small corrections even when
both WHAM cameras have indistinguishable temporal consistency. Do not
interpret this as independent evidence that Slave is better.

To require real 2D evidence and return the original explicit fallback:

```yaml
fusion:
  aotr:
    missing_2d_mode: fail_closed
```

The output `aotr_diagnostics.json` contains `evidence_counts`, e.g.
`{"3d_surrogate": 301}` or `{"two_view_2d": 301}`, plus
`corrected_frames`, `accepted_bone_frames`, and `reason_counts`.
Each frame's metadata records `aotr.evidence_mode`.

**Do not compare a 3D-only run with 2D-enabled AOTR without reporting
the evidence mode and the coverage.** For meaningful physics-vs-accuracy
ablation, compare the current 3D-only optimizer with legacy V1 and Master
using exactly the same camera pair, individual and evaluation frames.

## Input requirements

- Two synchronized WHAM 3D poses, in their respective camera coordinates.
- Camera intrinsics for both views (via existing calibration resolver).
- The **actual** tracked 2D keypoints for both views are required for the
  two-view 2D **observability** objective. If those are unavailable,
  `missing_2d_mode: 3d_surrogate` uses a different 3D-only objective,
  explicitly tagged as such. Copied hand/foot detector aliases are not
  independent evidence.
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
8. In 2D mode, if trustworthy observations are missing, use the explicitly
   configured 3D-only surrogate or fail closed. Unsafe torso alignment still
   falls back to raw Master with a structured reason.

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

## Brute-force provenance and cache isolation

Run all configured ordered camera pairs:

```bash
git fetch origin
git switch experiment/observability-aware-trust-region
git pull --ff-only
python brute_force_runner.py
```

The runner now prints `[BruteForce] AOTR enabled: True` and the full
`[BruteForce] Active protocol` in the startup logs. The protocol contains
`__aotr_fk_trustregion_v1__` and a 12-character settings fingerprint.

- Legacy benchmark output remains `output/brute_force_report.csv`.
- AOTR output is separate: `output/brute_force_report_aotr_<hash>.csv`.
- Cached legacy rows will NOT be loaded or reused by AOTR; changing AOTR
  parameters yields a new fingerprint and independent CSV.
- For each pair the runner prints `[AOTR] camXX->camYY:
  frames=..., corrected=..., accepted_bones=..., fallback=...`.
- The report has 3 AOTR-specific columns: corrected frames, accepted
  bones and fallback frames.
- If `aotr_diagnostics.json` does not exist, has zero processed frames,
  or all frames fallback (even in the 3D-only surrogate), that camera
  pair fails explicitly rather than being recorded as a valid metric.

**Important:** If the report still shows
`no_gt_v10_master_only_postopt` and populated Belief Master / Belief Slave
arrays for all 21 joints, you are viewing the **legacy** path or stale cached
results, not a validated AOTR run. Confirm the active Git branch and config
before drawing any performance conclusions.

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

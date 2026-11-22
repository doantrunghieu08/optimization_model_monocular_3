# Two-View Monocular 3D Human Pose Fusion

Fuses two independent monocular 3D pose estimates of the same person (WHAM outputs from
two uncalibrated cameras) into one improved skeleton for the first camera (the *Master*).
There are no camera extrinsics; the two views are related through the body itself.

This README describes what the code does today and only reports results that were
re-checked. See [MODEL_CHANGES.md](MODEL_CHANGES.md) for the full change log, the
experiments behind each decision, and the negative results.

---

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Preprocess | `preprocess_pipeline/` | Frame-offset estimation between the two WHAM tracks, 2D keypoint profiles (when `tracking_results*.pth` exists), intrinsics estimated from the image size |
| Pose export | `pose_pipeline/` | SMPL → 21 joints per frame (camera frame) for the target person; tracks are chosen through `*_person_mapping.txt` |
| Fusion | `fusion_pipeline/` | `fusion.method: aotr` (default) or `partwise`; only camera1 is modified |
| Evaluation | `evaluation_pipeline/` | PA-MPJPE, MPJPE, MBLE and Accel against GT; this is the only stage that reads GT |
| Dataset prep (offline) | `preprocessing_data/`, `build_identity_mask.py` | Person labelling for datasets without a mapping file, and the identity mask used by evaluation |

Metric definitions (in `evaluation_pipeline/executor.py`):

* **Joint set** (`evaluation.joint_set`). The default `coco17` uses the Harmony4D COCO-17
  keypoints: 12 body joints plus nose, eyes and ears. Face points come from SMPL mesh
  vertices and are stored as `<cam>_face` outside the 21-joint skeleton; after fusion they
  follow the fused neck→head bone. The legacy `priority1` uses the 12 body joints only.
* **MPJPE** (`evaluation.mpjpe_alignment`):
  * default `sequence`: root-centred at the hip midpoint, with **one rotation per camera for
    the whole sequence**. This stands in for the static-camera extrinsics, which are not
    available locally, so per-frame orientation errors are counted.
  * legacy `frame`: re-rotates every frame. This hides those errors and gives much lower
    numbers.
* **PA-MPJPE** uses full Procrustes alignment.
* **MBLE** is the mean absolute bone-length error.
* **Accel** is the second-difference error over consecutive frames, in the camera frame.

Every module is scored on the same joints in each frame.

### Fusion methods

**`aotr` (default): Observability-Aware Trust-Region.** The Master skeleton is kept. For
each bone, a blend weight α ∈ [0, 1] between the Master direction and the aligned Slave
direction is found by a 1D trust-region search. The skeleton is rebuilt by forward
kinematics with the Master bone lengths.

* When both views have 2D keypoints, the objective is two-view 2D reprojection.
* Otherwise a 3D surrogate is used (`missing_2d_mode: 3d_surrogate`), driven by temporal
  jitter and mesh visibility.
* A sequence-stability pass then damps the acceleration of the correction residual.

**`partwise`: part-wise symmetric fusion** (`fusion_pipeline/partwise.py`). For every bone,

    u = normalize(w · u_Master + (1 − w) · R_part · u_Slave)

and the skeleton is rebuilt with the Master bone lengths, so MBLE equals the baseline.

* `R_part = B_M · B_Sᵀ` aligns the torso part the bone hangs from:
  * hip axis + spine for the legs;
  * shoulder axis + spine for the arms and head.
* This part-wise anchoring replaced a single whole-body rotation, which twisted the upper
  body whenever one view was poor.
* `w` is a per-frame Master weight: a logistic function of the root-depth difference
  (the farther view is weighted more).
* The same sequence-stability pass is applied afterwards.

**`naive`: comparison baseline** (`fusion_pipeline/naive.py`). For each frame, the Slave is
rigidly aligned to the Master with one Kabsch rotation about the pelvis, and the joint
positions are averaged (`fusion.naive.master_weight`, fixed at 0.5). It has no learning and
does not preserve bone lengths. It is the minimum that any proposed fusion must beat.
With `fusion.naive.use_aotr_stability: true` (the default in `configs/pipeline.yml`) the
result goes through AOTR's sequence-stability pass (savgol + FK onto the Master bone
lengths), so a comparison with AOTR does not mix in the smoothing gain. Set it to `false`
for the raw baseline.

**`adaptive_naive`: control for AOTR** (`fusion_pipeline/adaptive_naive.py`). It uses the
same inputs and steps as AOTR (rigid torso RANSAC with the `fusion.aotr` thresholds and gates,
FK with the Master bone lengths, the same sequence-stability pass), but replaces the
optimisation (2D reprojection objective, observability prior, trust region) with a
closed-form per-bone weight:

    q = c^p · exp(−j / τ),   w_Slave = q_S / (q_M + q_S),
    u = normalize((1 − w) · u_Master + w · R21 · u_Slave)

* `c` is the bone's 2D detector confidence: the minimum over its endpoints that have a direct
  2D keypoint, or `missing_confidence` when there is none.
* `j` is the bone-direction jitter in that view: the second difference of the unit vector,
  averaged over ±`jitter_half_window` frames and divided by the sequence median for that bone.
* `p = confidence_power` and `τ = jitter_scale` are the only free numbers. `p = 0` with a very
  large `τ` gives the fixed 0.5 blend. Fit them on a train split (see the grid example below),
  then freeze them.

### Ground-truth use

| Where | GT used? |
|---|---|
| Fusion at inference (both methods) | **No** |
| `partwise` depth-logit (4 numbers in `configs/pipeline.yml`) | **Yes, fitted once** on GT of a design split (Hugging, aria01, cam02/04/13/19) and then frozen. `partwise` is therefore *GT-free at inference, calibrated on design GT*, **not** zero-GT. |
| `adaptive_naive` `confidence_power`, `jitter_scale` | Defaults 1 and 1 are set a priori, not fitted. If they are fitted (grid on a train split), report it like `partwise`: *GT-free at inference, calibrated on train GT*. |
| `preprocessing_data.label_persons` | **Yes, 3D GT names tracks** (aria01 / aria02) when a dataset has no mapping file and no 2D boxes. One label per track, so tracker errors are kept. |
| Identity mask (evaluation) | **Yes.** Frames where the input track follows the wrong person are excluded from scoring, identically for all methods. |

---

## Results (Hugging 002, WHAM)

> **Scoring note.** The tables below were produced with the **legacy scoring**
> (`joint_set: priority1`, `mpjpe_alignment: frame`). Under the current default (COCO-17 +
> sequence-aligned MPJPE), one held-out pair (cam03→cam05) gives the following PA / MPJPE:
>
> | Method | PA-MPJPE | MPJPE |
> |---|---|---|
> | mono | 35.2 | 113.4 |
> | aotr | 33.3 | 113.2 |
> | partwise | 30.0 | 112.9 |
>
> So fusion improves PA-MPJPE but barely changes MPJPE. Both methods keep the Master's
> global torso orientation, and WHAM's per-frame orientation error (median 4–12° on clean
> frames, front/back flips on some cameras) dominates the standard MPJPE. A full re-run
> under the new scoring is pending.

Protocol: design split = aria01, cam02/04/13/19; held-out split = aria02, cam03/05/17/20.
The two splits share no subject and no camera. Configs were frozen and selections locked
before held-out runs. 12 ordered pairs per split, scored on camera1, identity mask T = 10 mm.

**Held-out**

| Method | PA-MPJPE | MPJPE | MBLE | Accel | PA wins vs AOTR |
|---|---|---|---|---|---|
| WHAM mono | 51.47 | 64.91 | 22.43 | 22.11 | — |
| `aotr` | 46.16 | 58.54 | 22.43 | 13.16 | — |
| `partwise` | **39.30** | **47.98** | 22.43 | **12.85** | 9/12 (MPJPE 11/12) |

Without the identity mask: mono 56.14 / 69.00, `aotr` 51.29 / 63.08, `partwise` 45.33 / 53.92
(PA / MPJPE).

**Design:** mono 55.90 / 73.08, `aotr` 49.12 / 64.15, `partwise` 47.07 / 63.75.

Caveats, all verified:

* Most of the `partwise` gain over `aotr` comes from the 3 held-out pairs with cam20 as
  Master (MPJPE 80.9 → 48.3). The part-wise alignment was designed after diagnosing exactly
  those pairs, so held-out is **not clean** for that part. On the other 9 pairs, `partwise`
  ≈ the pre-registered whole-body variant, and both beat `aotr` by about 3 mm PA.
* The gain over `aotr` depends on the learned depth sign. With w = 0.5 the result is ≈
  `aotr` (45.10); with the sign flipped it is worse than mono (56.33). Scaling the slope
  ×0.5 or ×2 changes PA by < 1 mm.
* There is only one scene. Generalization to other scenes, actions and HMR models is
  **not yet established**. Evaluation on 043_Grappling2 (a new scene) is in progress and
  not reported here.

---

## Known data issues

* **Hugging identity swaps.** Some WHAM tracks switch person mid-sequence, while
  `*_person_mapping.txt` gives one person per track. 15–43% of frames are affected on
  cam02/03/17/20. These frames are handled by the identity mask
  (`configs/identity_mask_hugging_002.json`).
* **2D keypoints missing** for Hugging cam02/03/17/20 and all of 043. AOTR then runs its
  3D surrogate.
* **MPI-INF-3DHP:** WHAM segments are not frame-aligned with the GT JSONs, so naive scores
  are meaningless.
* **043_Grappling2:**
  * no mapping files and no 2D boxes; tracks were labelled with `label_persons`;
  * 29 duplicate GT files (`00059(1).npy` …), which the loader now ignores;
  * cam07's video lives in `images (1)/`.

---

## Usage

```bash
# 0) once per new dataset without person mapping (writes *_person_mapping.txt next to the pkl)
python -m preprocessing_data.label_persons input/043_Grappling2 --brute-config configs/brute_force_grappling2_043.yml
python build_identity_mask.py configs/brute_force_grappling2_043.yml configs/identity_mask_grappling2_043.json
#    or from a brute-force YAML whose ground_truth_dir / pkl / video are local paths OR Google Drive
#    File IDs (GT: zip of poses3d/*.npy). Drive files and the extracted GT are kept in
#    inputs/drive/<sequence>/ (--data-dir; same layout as a dataset folder, reused on rerun), the
#    mapping is written next to each pkl, and `<key>_local: "<absolute path>"` is added under each
#    Drive ID in the same YAML (comments kept); build_identity_mask.py reads it.
#    A Drive pkl is stored as exo/<cam>/<sequence>_<cam>_wham/wham_<sequence>_<cam>.pkl: the identity
#    mask is keyed by the pkl stem, and Drive names (wham_output.pkl) repeat across cameras.
#    Tracks are labelled against every person of the GT; add --overwrite to replace mappings made
#    before that (they only know aria01).
#    cameras[].tracking_result (WHAM tracking_results.pth, path or Drive ID) is put next to the pkl,
#    where the 2D preprocessing reads keypoints + confidences; its coverage per person is printed.
#    --persons N improves the first N GT persons: --persons 2 adds a <sequence>_aria02 segment
#    (same cameras, `added_by: label_persons --persons`), so the runner does 2 x 380 pairs;
#    --persons 1 removes it again. The runner keeps a cache per target person.
#    Add the mask to evaluation.identity_mask.path in configs/pipeline.yml.
python -m preprocessing_data.label_persons configs/brute_force_005_sword3.yml --persons 2
python build_identity_mask.py configs/brute_force_005_sword3.yml configs/identity_mask_005_sword3.json

# 1) one camera pair: set inputs.* in configs/pipeline.yml, then
#    runtime.stage: visualization  + visualization.enabled: false  -> full pipeline, no video render
#    runtime.stage: evaluation                                     -> re-score existing outputs only
python main.py --config configs/pipeline.yml

# 2) all ordered pairs of a dataset
python brute_force_runner.py --config configs/pipeline.yml --brute-config configs/brute_force.yml               # Hugging
python brute_force_runner.py --config configs/pipeline.yml --brute-config configs/brute_force_grappling2_043.yml  # 043
#    --workers N runs N pairs at once (one process, one BLAS/torch thread each; a pair is
#    single-threaded Python, the GPU is not used when AOTR is on). Same pairs, cache and CSV as
#    the sequential run (identical numbers); each worker uses its own
#    output/brute_force_workers/wNN/*_results and every pair logs to
#    output/brute_force_workers/logs/<segment>__<master>-<slave>.log. About one worker per core,
#    minus one or two (M1, 8 cores: --workers 6). Ctrl+C keeps finished pairs; rerun to resume.
#    Never start two runners on the same report at once: the last one to write the CSV wins.
python brute_force_runner.py --config configs/pipeline.yml --brute-config configs/001_mma5.yml --workers 6

# 3) tests (185; test_method_comparison needs a working pandas)
python -m pytest -q

# 4) check a new (Harmony4D) sequence before running anything; prints the remaining steps
python -m preprocessing_data.check_dataset input/<sequence> --gt-align

# 5) mono vs aotr vs partwise vs naive on the same pairs, current scoring, masked + unmasked
#    (see docs/PREREGISTRATION_harmony4d.md); e.g. Hugging design / held-out:
python run_method_comparison.py run --brute configs/brute_force.yml \
    --mask configs/identity_mask_hugging_002.json \
    --split design=aria01:cam02,cam04,cam13,cam19 --split heldout=aria02:cam03,cam05,cam17,cam20 \
    --out output/method_comparison/hugging_002 --worker 0 --n-workers 1
python run_method_comparison.py summarize --out output/method_comparison/hugging_002

# 6) AOTR vs naive vs adaptive_naive (no brute_force_runner run needed: each pair is exported,
#    fused and scored from the local inputs; pkl_local / ground_truth_dir_local are used)
python run_method_comparison.py run --brute configs/001_mma5.yml --segments 001_mma5_aria02 \
    --split try=aria02:cam07,cam11,cam14 --methods aotr naive adaptive_naive \
    --mask configs/identity_mask_001_mma5.json --out output/method_comparison/001_mma5_try
#    grid for adaptive_naive on a train split: each variant is its own row and protocol
python run_method_comparison.py run --brute configs/002_mma5.yml \
    --methods adaptive_naive@confidence_power=0 adaptive_naive adaptive_naive@confidence_power=2 \
              adaptive_naive@jitter_scale=0.5 adaptive_naive@jitter_scale=1000 \
    --out output/method_comparison/002_mma5_grid
```

`run_method_comparison.py` needs pandas; it must import with the pipeline's numpy (pandas 3.x
needs numpy >= 2.0.2).

Choose the method with `fusion.method: aotr | partwise | naive | adaptive_naive` in `configs/pipeline.yml`. Each
method writes its own report (`output/brute_force_report_<method>_<hash>.csv`). The hash
covers the method, its parameters and the identity mask, so cached results are never mixed.
For the mask it covers the frames actually excluded (all mask files merged, cut at
`margin_mm`), not only the file names: rebuilding a mask that drops other frames starts a
new report, while rebuilding one with the same excluded frames reuses the cache.

Evaluation outputs (`output/evaluation_results/*_cam1.csv`): `posed_*` columns are WHAM
mono, `fused_*` columns are the fusion result.

---

## Project structure

```
configs/
  pipeline.yml                     main config (fusion.method, aotr, partwise, evaluation.identity_mask)
  pipeline_mpi.yml                 MPI variant
  brute_force*.yml                 dataset segment lists (Hugging, MPI, 043)
  identity_mask_*.json             per-frame identity margins (evaluation only)
preprocess_pipeline/               offset, 2D profiles, intrinsics
pose_pipeline/                     SMPL joint export
fusion_pipeline/
  executor.py                      fusion stage, method dispatch
  aotr.py, aotr_3d.py              AOTR (2D path, 3D surrogate)
  aotr_stability.py                sequence stability pass
  partwise.py                      part-wise symmetric fusion
  aotr_benchmark.py                benchmark protocol identity / execution checks
evaluation_pipeline/executor.py    metrics, identity mask, GT loading
preprocessing_data/label_persons.py   track -> person labelling
build_identity_mask.py             identity mask builder
brute_force_runner.py              all-pairs benchmark (--workers N: pairs in parallel)
tests/                             unit and regression tests
```

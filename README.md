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

Metric definitions (in `evaluation_pipeline/executor.py`) are computed on the 12 GT joints
(shoulders, elbows, wrists, hips, knees, ankles):

* **MPJPE** is root-centred at the hip midpoint, with an optimal rotation and no scale.
* **PA-MPJPE** uses full Procrustes alignment.
* **MBLE** is the mean absolute bone-length error.
* **Accel** is the second-difference error over consecutive frames, in the camera frame.

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

### Ground-truth use

| Where | GT used? |
|---|---|
| Fusion at inference (both methods) | **No** |
| `partwise` depth-logit (4 numbers in `configs/pipeline.yml`) | **Yes, fitted once** on GT of a design split (Hugging, aria01, cam02/04/13/19) and then frozen. `partwise` is therefore *GT-free at inference, calibrated on design GT*, **not** zero-GT. |
| `preprocessing_data.label_persons` | **Yes, 3D GT names tracks** (aria01 / aria02) when a dataset has no mapping file and no 2D boxes. One label per track, so tracker errors are kept. |
| Identity mask (evaluation) | **Yes.** Frames where the input track follows the wrong person are excluded from scoring, identically for all methods. |

---

## Results (Hugging 002, WHAM)

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

# 1) one camera pair: set inputs.* in configs/pipeline.yml, then
#    runtime.stage: visualization  + visualization.enabled: false  -> full pipeline, no video render
#    runtime.stage: evaluation                                     -> re-score existing outputs only
python main.py --config configs/pipeline.yml

# 2) all ordered pairs of a dataset
python brute_force_runner.py --config configs/pipeline.yml --brute-config configs/brute_force.yml               # Hugging
python brute_force_runner.py --config configs/pipeline.yml --brute-config configs/brute_force_grappling2_043.yml  # 043

# 3) tests (108)
python -m pytest -q
```

Choose the method with `fusion.method: aotr | partwise` in `configs/pipeline.yml`. Each
method writes its own report (`output/brute_force_report_<method>_<hash>.csv`). The hash
covers the method, its parameters and the identity mask, so cached results are never mixed.

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
brute_force_runner.py              all-pairs benchmark
tests/                             unit and regression tests
```

# Harmony4D generalization test — pre-registration

Written 2026-11-19, **before** any new Harmony4D sequence has been received or scored.
This file is locked by the commit that adds it; fill in §0 and commit again **before the
first fusion run**. Any later change must be appended to §9 with a date and a reason.

## 0. Lock (fill in before the first run)

| Field | Value |
|---|---|
| Commit of frozen code + config | `________` (`git rev-parse --short HEAD`) |
| Held-out sequences (official Harmony4D test split) | `________` (7 sequences, names from the official release) |
| Development sequences (optional, non-test) | `________` |
| Date data received | `________` |

## 1. Question

Does two-view, calibration-free late fusion of monocular WHAM estimates (`aotr`, `partwise`)
improve over (a) monocular WHAM and (b) a naive two-view fusion baseline, on scenes, actions,
subjects and camera layouts never used for design?

## 2. What is frozen

* `configs/pipeline.yml` at the commit in §0, including the AOTR parameters, the `partwise`
  `depth_logit` (fitted once on the Hugging design split), the stability settings,
  `anatomical_calibration: false`, and `naive.master_weight: 0.5`.
* Per pair, only `inputs` (pkl, video, GT dir, target_person) and output paths change.
* `fusion.master_protection.pair_modes` is **off** (`{}`, `pair_mode: aggressive`): those
  pairs were hand-picked on Hugging. `run_method_comparison.py` does this by default.
* HMR model: WHAM, the same version and settings as for Hugging.
* Nothing is tuned on held-out data. If any parameter is changed, every method is re-run
  and both results are reported (§9).

## 3. Data, units and exclusions

* Held-out = the 7 official test sequences (§0). Already-seen data is reported separately
  and never pooled with the held-out data:
  * Hugging 002 is the design data.
  * Grappling 043 has already been scored (2026-11-18).
* Development sequences may only be used to make the pipeline *run* (crashes, file layout).
  They are not used to choose parameters, methods or exclusions.
* Unit = (sequence, person ∈ {aria01, aria02}, ordered camera pair Master→Slave).
* Units are included or excluded **only by mechanical rules, decided before scoring**:
  1. Both cameras have a WHAM pkl and a person mapping for that person.
  2. The two cameras share WHAM frame ids (otherwise the runner records `skip`).
  3. The sequence has no unresolved ERROR in `preprocessing_data.check_dataset --gt-align`.
     A sequence whose WHAM frames are shifted against GT is excluded as a whole, or repaired
     before any fusion run. The repair is described in §9.
* Errors during a run are reported as errors, never silently dropped.

## 4. Allowed data preparation (may read GT; done before any fusion run, never after)

1. `python -m preprocessing_data.check_dataset <seq> --gt-align --json <seq>/dataset_check.json`
2. `python -m preprocessing_data.label_persons <seq> --brute-config configs/brute_force_<seq>.yml`
   This names tracks only and does not repair tracker errors.
3. `python build_identity_mask.py configs/brute_force_<seq>.yml configs/identity_mask_<seq>.json`
   (T = 10 mm).
4. `processed_data/poses2d` and `processed_data/bbox` are derived from 3D GT and are **never**
   used as pipeline inputs.

## 5. Scoring

* Official evaluator, camera1 (Master) only.
* `joint_set: coco17`, `mpjpe_alignment: sequence`.
* **Primary:** identity mask applied inside the evaluator. **Secondary:** unmasked.
* Metrics: PA-MPJPE, MPJPE, MBLE, Accel. Mono = the `posed_*` columns of the same run, on
  the same frames.
* If calibration with extrinsics (`colmap/workplace/images.txt`) is available and a
  world-frame MPJPE is implemented **before the first run**, it is reported as a secondary
  metric. Otherwise it is not reported at all.

## 6. Methods

`mono` (WHAM camera1), `naive` (Kabsch-align the Slave to the Master and average joints,
w = 0.5), `aotr` (default), `partwise`. All methods run on the same pose export.

## 7. Analysis and decision rules

Aggregation: the primary statistic is the **macro mean**. For each sequence, average over
its units, then average over sequences, so that large sequences do not dominate. The
secondary statistic is the micro mean over all units.

Uncertainty and dependence: A→B and B→A share data, and pairs share cameras. So, besides
the per-unit paired Wilcoxon test and the bootstrap CI that `run_method_comparison.py
summarize` prints, report **in how many of the 7 sequences** the method wins. Conclusions
rest on this sequence-level count and the macro mean.

Δ = method − reference; negative is better. A method **passes** against a reference if all
of the following hold on the masked macro mean:

| | Criterion |
|---|---|
| C1 | ΔPA-MPJPE < 0, with the 95% bootstrap CI over sequences entirely below 0 |
| C2 | ΔMPJPE ≤ +1 mm |
| C3 | ΔAccel ≤ +2 mm/frame² |
| C4 | Wins (ΔPA < 0) in at least 5 of 7 sequences |

* **H1 (vs mono):** `aotr`, `partwise` and `naive` are each tested against mono.
* **H2 (vs naive) — the main claim:** `aotr` and `partwise` against `naive`. If neither
  passes H2, the paper may not claim that the proposed fusion is better than simple averaging.
* **H3:** `partwise` vs `aotr`, the same criteria as in the 043 pre-registration.
* MBLE is reported. `aotr` and `partwise` keep the Master bone lengths by design, while
  `naive` does not.

**Predicted beforehand** (from the Hugging re-scoring): the PA-MPJPE gains will be larger
than the MPJPE gains, because WHAM's global orientation error dominates MPJPE and none of
the methods corrects it. A small ΔMPJPE is therefore expected and is not a failure as long
as C2 holds.

Every table cell is reported, including negative results and per-sequence breakdowns.

Analysis-only breakdowns (not criteria): per person; pairs where the Master is better vs
pairs where the Slave is better (by mono PA); with vs without 2D input; the AUC of the
depth signal.

## 8. Commands (run by the author)

```bash
# per held-out sequence, after §4
python run_method_comparison.py run --brute configs/brute_force_<seq>.yml \
    --mask configs/identity_mask_<seq>.json \
    --out output/method_comparison/<seq> --worker 0 --n-workers 4   # start workers 1..3 too
python run_method_comparison.py summarize --out output/method_comparison/<seq>
```

`run_meta.json` in each output directory records the commit, config, mask and methods.
Every result row records the commit and the protocol hash.

## 9. Deviations log

| Date | Change | Reason | Effect on results |
|---|---|---|---|
| | | | |

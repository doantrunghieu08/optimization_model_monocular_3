"""Compare fusion methods on the same camera pairs with the CURRENT scoring.

Mono (WHAM camera1, the `posed_*` columns) comes for free; every listed fusion method
(default: aotr, partwise, naive) runs on the same pose export and frames. Each method is
scored twice with the official evaluator:
  * masked   (primary): evaluation.identity_mask applied INSIDE the evaluator, so the
                        sequence-aligned MPJPE rotation is fitted without swapped frames;
  * unmasked (secondary).
Only `inputs` and output paths change per pair; everything else comes from --config.

Run (split into workers, resumable). Resuming skips pairs already 'ok' in results_w*.csv, and is
refused if configs/pipeline.yml changed since (config_sha in run_meta.json): then pass --fresh to
rerun everything (old CSVs go to <out>/old/), or --redo-masters camXX to recompute some masters.
  python run_method_comparison.py run --brute configs/brute_force.yml \
      --mask configs/identity_mask_hugging_002.json \
      --split design=aria01:cam02,cam04,cam13,cam19 \
      --split heldout=aria02:cam03,cam05,cam17,cam20 \
      --out output/method_comparison/hugging_002 --worker 0 --n-workers 4

Without --split, every segment x every ordered camera pair is used, with the person taken
from the segment's target_person (or --persons).

Summarize (means, win counts, paired Wilcoxon, bootstrap CI):
  python run_method_comparison.py summarize --out output/method_comparison/hugging_002

A method may carry parameter overrides for fusion.<method>, which is how a grid is run
on a train split (each variant is its own row label and its own protocol hash):
  --methods aotr adaptive_naive adaptive_naive@confidence_power=2,jitter_scale=0.5

Cameras and GT are read from pkl_local / video_local / ground_truth_dir_local when those
exist on disk (current brute configs keep Drive file IDs in pkl / video / ground_truth_dir);
nothing is downloaded. --segments restricts a brute config with one segment per person.
"""
import argparse
import copy
import itertools
import json
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
METRICS = ("pa", "mpjpe", "mble", "accel")


# ----------------------------------------------------------------------------- units
def parse_split(text):
    """'heldout=aria02:cam03,cam05' -> ('heldout', 'aria02', ['cam03', 'cam05'])."""
    name, rest = text.split("=", 1)
    person, cams = rest.split(":", 1)
    return name.strip(), person.strip().lower(), [c.strip() for c in cams.split(",") if c.strip()]


def parse_method(spec):
    """'adaptive_naive@confidence_power=2,jitter_scale=0.5' -> ('adaptive_naive', {...})."""
    base, _, rest = spec.partition("@")
    overrides = {}
    for item in filter(None, (x.strip() for x in rest.split(","))):
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"bad method override {item!r} in {spec!r}; use key=value")
        overrides[key.strip()] = yaml.safe_load(value.strip())
    return base.strip().lower(), overrides


def _local_or(entry, key):
    """entry[key + '_local'] when that path exists, else entry[key]."""
    local = entry.get(f"{key}_local")
    if local and Path(str(local)).expanduser().exists():
        return str(Path(str(local)).expanduser())
    return entry.get(key)


def _local_video(entry):
    """Existing local video file or None. Videos are optional: they are never downloaded and
    a missing/commented video only means the frame size falls back (calib.resolve_intrinsics)."""
    for key in ("video_local", "video"):
        value = entry.get(key)
        if value and Path(str(value)).expanduser().is_file():
            return str(Path(str(value)).expanduser())
    return None


def build_units(brute_cfg, splits=None, persons=None, segments=None):
    units = []
    for seg in brute_cfg["segments"]:
        if segments and seg["name"] not in segments:
            continue
        gt_dir = _local_or(seg, "ground_truth_dir")
        cams = {c["id"]: c for c in seg["cameras"]}
        if splits:
            plan = [(name, person, ids) for name, person, ids in splits]
        else:
            seg_persons = persons or [seg.get("target_person", "aria01")]
            plan = [("all", p, sorted(cams)) for p in seg_persons]
        for split, person, ids in plan:
            missing = [c for c in ids if c not in cams]
            if missing:
                raise ValueError(f"split {split}: cameras {missing} not in segment {seg['name']}")
            for a, b in itertools.permutations(ids, 2):
                units.append({"segment": seg["name"], "sequence": sequence_name(gt_dir),
                              "split": split, "person": person,
                              "gt_dir": gt_dir, "A": cams[a], "B": cams[b]})
    if segments and not units:
        raise ValueError(f"no units: segments {segments} not in the brute config")
    return units


def sequence_name(gt_dir):
    """input/043_Grappling2/processed_data/poses3d -> 043_Grappling2 (segments may be per person)."""
    parts = Path(gt_dir).parts
    return parts[parts.index("processed_data") - 1] if "processed_data" in parts[1:] else Path(gt_dir).name


def unit_key(u):
    return (u["segment"], u["split"], u["person"], u["A"]["id"], u["B"]["id"])


# ----------------------------------------------------------------------------- scoring
def _numeric(d):
    d = d[pd.to_numeric(d.Frame, errors="coerce").notna()].copy()
    d["Frame"] = d.Frame.astype(int)
    return d


def score_eval_dir(eval_dir):
    """Means over the evaluator's per-frame rows: {metric: (mono, fused)} plus frame count."""
    e = Path(eval_dir)
    out = {}
    for met, key in (("PA-MPJPE", "pa"), ("MPJPE", "mpjpe")):
        d = _numeric(pd.read_csv(e / f"{met}_cam1.csv", na_values="N/A"))
        out[key] = (d.posed_priority1_mm.mean(), d.fused_priority1_mm.mean())
        out["frames"] = len(d)
    for fname, col, key in (("MBLE_cam1.csv", "Frame_MBLE_mm", "mble"),
                            ("Accel_cam1.csv", "Frame_Accel_Error_mm_frame2", "accel")):
        d = _numeric(pd.read_csv(e / fname))
        per = {m: d[d.Module == m].groupby("Frame")[col].first().mean() for m in ("posed", "fused")}
        out[key] = (per["posed"], per["fused"])
    return out


# ----------------------------------------------------------------------------- run
def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def cmd_run(a):
    import sys
    sys.path.insert(0, str(ROOT))
    from config_loader import load_config, absolutize_config_paths
    from pipeline import check_cameras_synchronized
    from preprocess_pipeline.executor import run_preprocess
    from pose_pipeline.executor import run_pose_export
    from fusion_pipeline.executor import run_fusion
    from fusion_pipeline.aotr_benchmark import evaluation_protocol, verify_aotr_execution
    from evaluation_pipeline.executor import run_evaluation

    brute = yaml.safe_load(Path(a.brute).read_text(encoding="utf-8"))
    splits = [parse_split(s) for s in a.split] if a.split else None
    units = build_units(brute, splits, a.persons, a.segments)
    methods = [(spec, *parse_method(spec)) for spec in a.methods]
    mine = units[a.worker::a.n_workers]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_w{a.worker}.csv"
    if a.fresh and out_csv.exists():  # start over, but never delete: move the old CSV aside
        backup = out_dir / "old" / f"results_w{a.worker}.{time.strftime('%Y%m%d-%H%M%S')}.csv"
        backup.parent.mkdir(parents=True, exist_ok=True)
        out_csv.rename(backup)
        print(f"[compare w{a.worker}] --fresh: old results moved to {backup}")
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    if a.redo_masters:  # e.g. after fixing one camera's extrinsic: recompute only its master pairs
        n0 = len(rows)
        rows = [r for r in rows if r["master"] not in set(a.redo_masters)]
        print(f"[compare w{a.worker}] --redo-masters {a.redo_masters}: {n0 - len(rows)} rows dropped")
    done = {(r["segment"], r["split"], r["person"], r["master"], r["slave"]) for r in rows
            if str(r.get("status", "")).startswith(("ok", "skip"))}
    rows = [r for r in rows if (r["segment"], r["split"], r["person"], r["master"], r["slave"]) in done]
    base_cfg = load_config(Path(a.config))
    # Fingerprint of the whole pipeline config: resuming with different parameters would silently mix
    # two parameter sets in one table (this happened once), so it is refused unless --fresh / new --out.
    import hashlib
    config_sha = hashlib.sha1(json.dumps(base_cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    mask = {"path": a.mask, "margin_mm": a.mask_margin} if a.mask else None
    commit = _git_commit()
    from evaluation_pipeline.extrinsics import ext_enabled
    ext = ext_enabled()
    meta_path = out_dir / "run_meta.json"
    if rows and meta_path.exists():
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        prev_ext = bool(prev.get("ext", False))
        if prev_ext != ext:
            raise SystemExit(f"{out_dir} holds results with EXT={int(prev_ext)}, this run has EXT={int(ext)}: "
                             "use another --out so the two scorings are not mixed")
        prev_sha = prev.get("config_sha")
        if prev_sha is None:
            print(f"[compare w{a.worker}] WARNING: {out_dir} predates config fingerprints; resuming cannot "
                  f"check that {a.config} is unchanged. Use --fresh to rerun everything with the current config.")
        elif prev_sha != config_sha:
            raise SystemExit(f"{out_dir} was run with a different {a.config} (config_sha {prev_sha} != {config_sha}). "
                             "Resuming would mix two parameter sets in one table: pass --fresh to rerun all "
                             "pairs (old CSVs are moved to <out>/old/), or use another --out.")
    meta = {"commit": commit, "ext": ext, "config": a.config, "config_sha": config_sha,
            "brute": a.brute, "mask": a.mask,
            "mask_margin_mm": a.mask_margin, "splits": a.split, "methods": a.methods,
            "use_pair_modes": a.use_pair_modes, "units_total": len(units)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[compare w{a.worker}] {len(mine)} of {len(units)} units, methods={a.methods}, commit={commit}")

    for u in mine:
        if unit_key(u) in done:
            continue
        t0 = time.time()
        A, B = u["A"], u["B"]
        work = out_dir / "runs" / "_".join(unit_key(u))
        cfg = copy.deepcopy(base_cfg)
        cfg["inputs"].update({"cam1_pkl": _local_or(A, "pkl"), "camera1_video": _local_video(A),
                              "cam2_pkl": _local_or(B, "pkl"), "camera2_video": _local_video(B),
                              "ground_truth_dir": u["gt_dir"], "target_person": u["person"]})
        cfg["inputs"].pop("person_heights", None)
        for cam, entry in (("camera1", A), ("camera2", B)):  # YAML extrinsic (used only under EXT=1)
            cfg["inputs"].pop(f"{cam}_extrinsic", None)
            if entry.get("extrinsic"):
                cfg["inputs"][f"{cam}_extrinsic"] = entry["extrinsic"]
        cfg["runtime"].update({"clean_output": True, "stage": "visualization"})
        cfg["visualization"]["enabled"] = False
        if a.use_pair_modes:  # reproduce brute_force_runner (hand-picked Hugging pairs)
            from brute_force_runner import _resolve_pair_mode
            cfg["fusion"]["pair_mode"] = _resolve_pair_mode(cfg["fusion"]["master_protection"], A["id"], B["id"])
        else:
            cfg["fusion"]["master_protection"]["pair_modes"] = {}
            cfg["fusion"]["pair_mode"] = "aggressive"
        for k in ("preprocess_output_dir", "pose_output_dir"):
            cfg["paths"][k] = str(work / k)
        cfg = absolutize_config_paths(cfg, ROOT)
        rec_base = {"segment": u["segment"], "sequence": u["sequence"], "split": u["split"], "person": u["person"],
                    "master": A["id"], "slave": B["id"], "commit": commit, "config_sha": config_sha, "ext": ext}
        new_rows = []
        try:
            synced, reason = check_cameras_synchronized(cfg)
            if not synced:
                rows.append({**rec_base, "status": f"skip: {reason}"})
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                continue
            run_preprocess(cfg, extract_frames=False)
            run_pose_export(cfg)
            for method, base_method, overrides in methods:
                vc = copy.deepcopy(cfg)
                vc["fusion"]["method"] = base_method
                if overrides:
                    vc["fusion"].setdefault(base_method, {}).update(overrides)
                method_dir = work / re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
                vc["paths"]["fused_output_dir"] = str(method_dir / "fused")
                run_fusion(vc)
                verify_aotr_execution(vc)
                for tag, m in (("masked", mask), ("unmasked", None)):
                    if tag == "masked" and m is None:
                        continue
                    ec = copy.deepcopy(vc)
                    ec["evaluation"]["identity_mask"] = copy.deepcopy(m)
                    ec["paths"]["evaluation_output_dir"] = str(method_dir / f"eval_{tag}")
                    run_evaluation(ec)
                    s = score_eval_dir(ec["paths"]["evaluation_output_dir"])
                    rec = {**rec_base, "method": method, "mask": tag, "frames": s["frames"],
                           "protocol": evaluation_protocol(vc)}
                    for k in METRICS:
                        rec[f"{k}_mono"], rec[k] = s[k]
                    new_rows.append(rec)
                    if a.keep_per_frame:
                        keep = out_dir / "per_frame" / f"{'_'.join(unit_key(u))}_{method_dir.name}_{tag}"
                        keep.mkdir(parents=True, exist_ok=True)
                        for f in Path(ec["paths"]["evaluation_output_dir"]).glob("*_cam1.csv"):
                            shutil.copy(f, keep / f.name)
                shutil.rmtree(method_dir, ignore_errors=True)
            secs = round(time.time() - t0, 1)
            for r in new_rows:
                r.update({"status": "ok", "seconds": secs})
            rows.extend(new_rows)
        except Exception as exc:  # record and continue with the next pair
            rows.append({**rec_base, "status": f"error: {type(exc).__name__}: {exc}"[:300]})
            with (out_dir / "errors.log").open("a", encoding="utf-8") as f:
                f.write(f"{rec_base}\n{traceback.format_exc()}\n")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        brief = " ".join(f"{r['method']} {r['pa']:.1f}" for r in new_rows if r["mask"] == new_rows[0]["mask"]) \
            if new_rows else rows[-1].get("status")
        mono = f"mono {new_rows[0]['pa_mono']:.1f} " if new_rows else ""
        print(f"[compare w{a.worker}] {u['split']} {u['person']} {A['id']}->{B['id']}: PA {mono}{brief} "
              f"({round(time.time() - t0, 1)}s)", flush=True)


# ----------------------------------------------------------------------------- summarize
def _bootstrap_ci(diff, n=10000, seed=0):
    diff = np.asarray(diff, float)
    if len(diff) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def _wilcoxon_p(diff):
    from scipy.stats import wilcoxon
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 5 or np.allclose(diff, 0):
        return np.nan
    return float(wilcoxon(diff).pvalue)


def _macro_lines(g, methods, refs):
    """
    Sequence-level statistical analysis.

    Each sequence is one statistical observation:
      - Macro mean: equal weight for each sequence.
      - Bootstrap CI: resampling sequences.
      - Wilcoxon: paired test over sequence-level differences.
      - Holm: multiple-comparison correction across methods
              within each reference and primary metric (PA).
      - Wins: number of sequences improved.
    """
    metrics = list(METRICS)
    mono_cols = [f"{k}_mono" for k in metrics]

    per = (
        g.groupby(["sequence", "method"])[metrics + mono_cols]
         .mean()
    )

    # Only retain sequences available for every compared method.
    available = [
        set(per.xs(m, level="method").index)
        for m in methods
    ]
    common = sorted(set.intersection(*available))

    if len(common) == 0:
        return [
            "\nMacro analysis skipped: no common sequences across methods.\n"
        ]

    per = per.loc[
        per.index.get_level_values("sequence").isin(common)
    ]

    # Mono is identical across methods because all methods use
    # the same pose export. Use the first method's mono measurements.
    mono = per.xs(methods[0], level="method").loc[common]

    method_data = {
        m: per.xs(m, level="method").loc[common]
        for m in methods
    }

    out = [
        f"\n## Sequence-level analysis "
        f"(n={len(common)} sequences)\n",
        "Each sequence contributes equally, regardless of its "
        "number of frames or camera pairs.\n",
        "| Method | PA-MPJPE | MPJPE | MBLE | Accel |",
        "|---|---:|---:|---:|---:|",
    ]

    out.append(
        "| mono | "
        + " | ".join(f"{mono[k].mean():.2f}" for k in metrics)
        + " |"
    )

    for m in methods:
        pm = method_data[m]
        out.append(
            f"| {m} | "
            + " | ".join(f"{pm[k].mean():.2f}" for k in metrics)
            + " |"
        )

    out += [
        "\n### Paired sequence-level comparisons",
        "",
        "Difference = method - reference; negative is better.",
        "Bootstrap CI: 95%, resampling sequences (10,000 iterations).",
        "Wilcoxon: paired signed-rank test on sequence-level PA differences.",
        "Holm p: Holm-Bonferroni correction across methods "
        "within each reference comparison.",
        "",
        "| Method | vs | ΔPA [95% CI] | ΔMPJPE | ΔAccel "
        "| PA wins | p raw | p Holm |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for ref in refs + ["mono"]:

        if ref == "mono":
            reference = mono
            ref_suffix = "_mono"
        else:
            if ref not in method_data:
                continue
            reference = method_data[ref]
            ref_suffix = ""

        comparisons = []

        for m in methods:
            if m == ref:
                continue

            pm = method_data[m]

            differences = {}

            for k in ("pa", "mpjpe", "accel"):
                x = pm[k].to_numpy(dtype=float)
                y = reference[f"{k}{ref_suffix}"].to_numpy(dtype=float)

                valid = np.isfinite(x) & np.isfinite(y)
                differences[k] = x[valid] - y[valid]

            dpa = differences["pa"]

            if len(dpa) >= 2:
                lo, hi = _bootstrap_ci(dpa)
            else:
                lo, hi = np.nan, np.nan

            p = _wilcoxon_p(dpa)

            wins = int((dpa < 0).sum())
            n = len(dpa)

            comparisons.append({
                "method": m,
                "dpa": dpa,
                "dmpjpe": differences["mpjpe"],
                "daccel": differences["accel"],
                "lo": lo,
                "hi": hi,
                "wins": wins,
                "n": n,
                "p": p,
            })

        # Holm correction for PA-MPJPE comparisons against this reference.
        adjusted = _holm_adjust([c["p"] for c in comparisons])

        for c, p_holm in zip(comparisons, adjusted):

            if c["n"] > 0:
                pa_text = (
                    f"{c['dpa'].mean():+.2f} "
                    f"[{c['lo']:+.2f}, {c['hi']:+.2f}]"
                )
                win_text = f"{c['wins']}/{c['n']}"
            else:
                pa_text = "N/A"
                win_text = "0/0"

            def mean_text(values):
                return (
                    f"{values.mean():+.2f}"
                    if len(values) else "N/A"
                )

            p_text = (
                f"{c['p']:.4g}"
                if np.isfinite(c["p"]) else "N/A"
            )
            holm_text = (
                f"{p_holm:.4g}"
                if np.isfinite(p_holm) else "N/A"
            )

            out.append(
                f"| {c['method']} | {ref} | {pa_text} "
                f"| {mean_text(c['dmpjpe'])} "
                f"| {mean_text(c['daccel'])} "
                f"| {win_text} | {p_text} | {holm_text} |"
            )

    out += [
        "",
        "### Interpretation",
        "",
        "- Negative ΔPA indicates improvement over the reference.",
        "- A 95% CI excluding zero indicates evidence of a "
        "non-zero mean difference under the sequence-level bootstrap.",
        "- Holm-adjusted p < 0.05 indicates statistical significance "
        "after correcting for multiple method comparisons.",
        "- PA wins measures how many sequences improved, "
        "not how many camera pairs improved.",
        "- Results are conditional on the evaluated sequences; "
        "they do not establish generalization to unseen datasets.",
    ]

    return out

def summarize(df):
    """Markdown tables per (split, mask): one row per method incl. mono; paired comparisons."""
    lines = []
    ok = df[df.status.astype(str).str.startswith("ok")].copy()
    key = ["segment", "split", "person", "master", "slave"]
    for (split, tag), g in ok.groupby(["split", "mask"], sort=False):
        methods = list(dict.fromkeys(g.method))
        # Mono must be identical across methods for a run (same pose export, same frames).
        mono = g.groupby(key)[[f"{k}_mono" for k in METRICS] + ["frames"]].agg(["min", "max"])
        spread = max(float((mono[(f"{k}_mono", "max")] - mono[(f"{k}_mono", "min")]).abs().max()) for k in METRICS)
        wide = {m: g[g.method == m].set_index(key) for m in methods}
        common = sorted(set.intersection(*(set(w.index) for w in wide.values())))
        lines.append(f"\n### split={split}, {tag}  (runs={len(common)})\n")
        if spread > 0.05:
            lines.append(f"> WARNING: mono differs across methods by up to {spread:.2f} mm (frames not identical).\n")
        lines.append("| Method | PA-MPJPE | MPJPE | MBLE | Accel | PA wins vs mono |")
        lines.append("|---|---|---|---|---|---|")
        first = wide[methods[0]].loc[common]
        lines.append("| mono | " + " | ".join(f"{first[f'{k}_mono'].mean():.2f}" for k in METRICS) + " | — |")
        for m in methods:
            w = wide[m].loc[common]
            wins = int((w.pa < w.pa_mono).sum())
            lines.append(f"| {m} | " + " | ".join(f"{w[k].mean():.2f}" for k in METRICS)
                         + f" | {wins}/{len(common)} |")
        refs = [r for r in ("naive", "aotr") if r in methods]
        if refs:
            lines.append("\nPaired differences (method − reference; negative = better), "
                         "mean [bootstrap 95% CI], wins, Wilcoxon p:\n")
            lines.append("| Method | vs | ΔPA | ΔMPJPE | ΔAccel | PA wins | p(PA) |")
            lines.append("|---|---|---|---|---|---|---|")
            for ref in refs + ["mono"]:
                for m in methods:
                    if m == ref:
                        continue
                    w = wide[m].loc[common]
                    r = w if ref == "mono" else wide[ref].loc[common]
                    sfx = "_mono" if ref == "mono" else ""
                    cells = []
                    for k in ("pa", "mpjpe", "accel"):
                        d = (w[k] - r[f"{k}{sfx}"]).to_numpy()
                        lo, hi = _bootstrap_ci(d)
                        cells.append(f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
                    dpa = (w.pa - r[f"pa{sfx}"]).to_numpy()
                    p = _wilcoxon_p(dpa)
                    lines.append(f"| {m} | {ref} | " + " | ".join(cells)
                                 + f" | {int((dpa < 0).sum())}/{len(dpa)} | {p:.3g} |")
        if "sequence" in g.columns and g.sequence.nunique() > 1:
            lines.extend(_macro_lines(g, methods, refs))
    status = df.drop_duplicates(key).status.astype(str)
    lines.append(f"\nUnits: {int(status.str.startswith('ok').sum())} ok, "
                 f"{int(status.str.startswith('skip').sum())} skipped, "
                 f"{int(status.str.startswith('error').sum())} errors.")
    return "\n".join(lines)


def cmd_summarize(a):
    out_dir = Path(a.out)
    files = sorted(out_dir.glob("results_w*.csv"))
    if not files:
        raise SystemExit(f"no results_w*.csv in {out_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    text = summarize(df)
    meta = out_dir / "run_meta.json"
    header = f"# Method comparison: {out_dir.name}\n"
    if meta.exists():
        header += "\n```json\n" + meta.read_text(encoding="utf-8") + "\n```\n"
    (out_dir / "summary.md").write_text(header + text + "\n", encoding="utf-8")
    print(header + text)


def _holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values. NaN values remain NaN."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))

    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    m = len(order)
    running_max = 0.0

    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--brute", required=True, help="brute-force segment config (cameras, GT dir)")
    r.add_argument("--config", default="configs/pipeline.yml")
    r.add_argument("--out", required=True)
    r.add_argument("--split", action="append", help="NAME=PERSON:camA,camB,... (repeatable)")
    r.add_argument("--persons", nargs="+", help="persons when no --split (default: segment target_person)")
    r.add_argument("--segments", nargs="+", help="only these segment names from --brute")
    r.add_argument("--methods", nargs="+", default=["aotr", "partwise", "naive"])
    r.add_argument("--mask", nargs="+", help="identity-mask JSON(s); omit for unmasked only")
    r.add_argument("--mask-margin", type=float, default=10.0)
    r.add_argument("--use-pair-modes", action="store_true",
                   help="apply fusion.master_protection.pair_modes (hand-picked on Hugging); default off")
    r.add_argument("--keep-per-frame", action="store_true", help="keep per-frame evaluator CSVs")
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--n-workers", type=int, default=1)
    r.add_argument("--fresh", action="store_true",
                   help="rerun every pair: this worker's old results CSV is moved to <out>/old/ (not deleted)")
    r.add_argument("--redo-masters", nargs="+", metavar="CAM",
                   help="drop and recompute the pairs whose master is one of these cameras (e.g. cam15)")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summarize(a)


if __name__ == "__main__":
    main()

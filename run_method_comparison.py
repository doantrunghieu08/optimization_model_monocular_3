"""Compare fusion methods on the same camera pairs with the CURRENT scoring.

Mono (WHAM camera1, the `posed_*` columns) comes for free; every listed fusion method
(default: aotr, partwise, naive) runs on the same pose export and frames. Each method is
scored twice with the official evaluator:
  * masked   (primary): evaluation.identity_mask applied INSIDE the evaluator, so the
                        sequence-aligned MPJPE rotation is fitted without swapped frames;
  * unmasked (secondary).
Only `inputs` and output paths change per pair; everything else comes from --config.

Run (split into workers, resumable):
  python run_method_comparison.py run --brute configs/brute_force.yml \
      --mask configs/identity_mask_hugging_002.json \
      --split design=aria01:cam02,cam04,cam13,cam19 \
      --split heldout=aria02:cam03,cam05,cam17,cam20 \
      --out output/method_comparison/hugging_002 --worker 0 --n-workers 4

Without --split, every segment x every ordered camera pair is used, with the person taken
from the segment's target_person (or --persons).

Summarize (means, win counts, paired Wilcoxon, bootstrap CI):
  python run_method_comparison.py summarize --out output/method_comparison/hugging_002
"""
import argparse
import copy
import itertools
import json
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


def build_units(brute_cfg, splits=None, persons=None):
    units = []
    for seg in brute_cfg["segments"]:
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
                units.append({"segment": seg["name"], "sequence": sequence_name(seg["ground_truth_dir"]),
                              "split": split, "person": person,
                              "gt_dir": seg["ground_truth_dir"], "A": cams[a], "B": cams[b]})
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
    units = build_units(brute, splits, a.persons)
    mine = units[a.worker::a.n_workers]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_w{a.worker}.csv"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {(r["segment"], r["split"], r["person"], r["master"], r["slave"]) for r in rows
            if str(r.get("status", "")).startswith(("ok", "skip"))}
    rows = [r for r in rows if (r["segment"], r["split"], r["person"], r["master"], r["slave"]) in done]
    base_cfg = load_config(Path(a.config))
    mask = {"path": a.mask, "margin_mm": a.mask_margin} if a.mask else None
    commit = _git_commit()
    meta = {"commit": commit, "config": a.config, "brute": a.brute, "mask": a.mask,
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
        cfg["inputs"].update({"cam1_pkl": A["pkl"], "camera1_video": A.get("video"),
                              "cam2_pkl": B["pkl"], "camera2_video": B.get("video"),
                              "ground_truth_dir": u["gt_dir"], "target_person": u["person"]})
        cfg["inputs"].pop("person_heights", None)
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
                    "master": A["id"], "slave": B["id"], "commit": commit}
        new_rows = []
        try:
            synced, reason = check_cameras_synchronized(cfg)
            if not synced:
                rows.append({**rec_base, "status": f"skip: {reason}"})
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                continue
            run_preprocess(cfg, extract_frames=False)
            run_pose_export(cfg)
            for method in a.methods:
                vc = copy.deepcopy(cfg)
                vc["fusion"]["method"] = method
                vc["paths"]["fused_output_dir"] = str(work / method / "fused")
                run_fusion(vc)
                verify_aotr_execution(vc)
                for tag, m in (("masked", mask), ("unmasked", None)):
                    if tag == "masked" and m is None:
                        continue
                    ec = copy.deepcopy(vc)
                    ec["evaluation"]["identity_mask"] = copy.deepcopy(m)
                    ec["paths"]["evaluation_output_dir"] = str(work / method / f"eval_{tag}")
                    run_evaluation(ec)
                    s = score_eval_dir(ec["paths"]["evaluation_output_dir"])
                    rec = {**rec_base, "method": method, "mask": tag, "frames": s["frames"],
                           "protocol": evaluation_protocol(vc)}
                    for k in METRICS:
                        rec[f"{k}_mono"], rec[k] = s[k]
                    new_rows.append(rec)
                    if a.keep_per_frame:
                        keep = out_dir / "per_frame" / f"{'_'.join(unit_key(u))}_{method}_{tag}"
                        keep.mkdir(parents=True, exist_ok=True)
                        for f in Path(ec["paths"]["evaluation_output_dir"]).glob("*_cam1.csv"):
                            shutil.copy(f, keep / f.name)
                shutil.rmtree(work / method, ignore_errors=True)
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
    """Sequence-level analysis (pre-registration primary): per-sequence means, macro mean,
    bootstrap CI over sequences, number of sequences won."""
    per = g.groupby(["sequence", "method"])[[*METRICS, *(f"{k}_mono" for k in METRICS)]].mean()
    seqs = sorted(g.sequence.unique())
    out = [f"\nMacro (mean over {len(seqs)} sequences; each sequence = mean over its runs):\n",
           "| Method | PA-MPJPE | MPJPE | MBLE | Accel |", "|---|---|---|---|---|"]
    m0 = per.xs(methods[0], level="method")
    out.append("| mono | " + " | ".join(f"{m0[f'{k}_mono'].mean():.2f}" for k in METRICS) + " |")
    for m in methods:
        pm = per.xs(m, level="method")
        out.append(f"| {m} | " + " | ".join(f"{pm[k].mean():.2f}" for k in METRICS) + " |")
    out += ["\n| Method | vs | ΔPA macro [CI over sequences] | ΔMPJPE | ΔAccel | sequences won (PA) |",
            "|---|---|---|---|---|---|"]
    for ref in refs + ["mono"]:
        for m in methods:
            if m == ref:
                continue
            pm = per.xs(m, level="method")
            pr = m0 if ref == "mono" else per.xs(ref, level="method")
            sfx = "_mono" if ref == "mono" else ""
            common = pm.index.intersection(pr.index)
            d = {k: (pm.loc[common, k] - pr.loc[common, f"{k}{sfx}"]).to_numpy() for k in ("pa", "mpjpe", "accel")}
            lo, hi = _bootstrap_ci(d["pa"])
            out.append(f"| {m} | {ref} | {d['pa'].mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {d['mpjpe'].mean():+.2f} "
                       f"| {d['accel'].mean():+.2f} | {int((d['pa'] < 0).sum())}/{len(common)} |")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--brute", required=True, help="brute-force segment config (cameras, GT dir)")
    r.add_argument("--config", default="configs/pipeline.yml")
    r.add_argument("--out", required=True)
    r.add_argument("--split", action="append", help="NAME=PERSON:camA,camB,... (repeatable)")
    r.add_argument("--persons", nargs="+", help="persons when no --split (default: segment target_person)")
    r.add_argument("--methods", nargs="+", default=["aotr", "partwise", "naive"])
    r.add_argument("--mask", nargs="+", help="identity-mask JSON(s); omit for unmasked only")
    r.add_argument("--mask-margin", type=float, default=10.0)
    r.add_argument("--use-pair-modes", action="store_true",
                   help="apply fusion.master_protection.pair_modes (hand-picked on Hugging); default off")
    r.add_argument("--keep-per-frame", action="store_true", help="keep per-frame evaluator CSVs")
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--n-workers", type=int, default=1)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summarize(a)


if __name__ == "__main__":
    main()

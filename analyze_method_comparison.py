"""Phân tích kết quả run_method_comparison.py (bổ sung cho lệnh `summarize`).

    venv311/bin/python analyze_method_comparison.py output/method_comparison/<tên> [--mask masked|unmasked]

So với `summarize`:
- Khoảng tin cậy bootstrap theo cụm camera (Master, rồi Slave): các cặp dùng chung camera
  không độc lập, nên bootstrap theo từng cặp cho khoảng quá hẹp.
- Hiệu theo độ chênh chất lượng Slave − Master (PA mono của camera đó khi làm Master; dùng GT,
  chỉ để phân tích), oracle chọn phương pháp tốt nhất cho từng cặp, các cặp thua mono.
- Chẩn đoán của adaptive_naive (trọng số Slave) đọc từ w*.log.

Mỗi người (aria01, aria02, …) là một nhóm riêng; không gộp hai người vào cùng một cặp camera.
Ghi bảng theo cặp ra <out>/analysis_pairs_<mask>.csv.
"""
import argparse
import ast
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

KEY = ["person", "master", "slave"]


def load(out, mask):
    files = sorted(glob.glob(f"{out}/results_w*.csv"))
    if not files:
        raise SystemExit(f"không có results_w*.csv trong {out}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.status.astype(str).str.startswith("ok") & (df["mask"] == mask)]
    if df.empty:
        raise SystemExit(f"không có dòng ok với mask={mask}")
    dup = df.duplicated(KEY + ["method"]).sum()
    if dup:
        print(f"! {dup} dòng trùng (person, master, slave, method): giữ dòng cuối")
        df = df.drop_duplicates(KEY + ["method"], keep="last")
    methods = sorted(df.method.unique())
    pa = df.pivot(index=KEY, columns="method", values="pa")
    mp = df.pivot(index=KEY, columns="method", values="mpjpe").add_prefix("mp_")
    mono = df.groupby(KEY)[["pa_mono", "mpjpe_mono", "frames"]].first()
    P = pd.concat([mono, pa, mp], axis=1).dropna(subset=methods).reset_index()
    return P.rename(columns={"pa_mono": "mono", "mpjpe_mono": "mp_mono"}), methods


def cluster_ci(values, groups, n=5000, seed=0):
    """Khoảng 95% bootstrap theo cụm (lấy lại cả cụm)."""
    rng = np.random.default_rng(seed)
    by = {g: values[groups == g] for g in np.unique(groups)}
    names = list(by)
    stats = [np.concatenate([by[g] for g in rng.choice(names, len(names))]).mean() for _ in range(n)]
    return np.percentile(stats, [2.5, 97.5])


def compare(P, a, b, prefix=""):
    d = (P[prefix + a] - P[prefix + b]).to_numpy()
    lo_m, hi_m = cluster_ci(d, P.master.to_numpy())
    lo_s, hi_s = cluster_ci(d, P.slave.to_numpy())
    per_master = pd.Series(d).groupby(P.master.to_numpy()).mean()
    p = wilcoxon(per_master).pvalue if len(per_master) > 1 and per_master.abs().sum() > 0 else float("nan")
    return (f"{a} − {b}: {d.mean():+.2f} | cụm Master [{lo_m:+.2f}, {hi_m:+.2f}] | cụm Slave [{lo_s:+.2f}, {hi_s:+.2f}]"
            f" | Master tốt hơn {int((per_master < 0).sum())}/{len(per_master)} | Wilcoxon trên trung bình Master p={p:.3g}")


def parse_logs(out):
    """{(person, master, slave): chẩn đoán adaptive_naive} từ w*.log."""
    diag = {}
    for f in sorted(glob.glob(f"{out}/w*.log")):
        last = None
        for line in open(f, encoding="utf-8", errors="replace"):
            if line.startswith("[Fusion] adaptive_naive diagnostics:"):
                try:
                    last = ast.literal_eval(line.split(":", 1)[1].strip())
                except (ValueError, SyntaxError):
                    last = None
            m = re.match(r"\[compare w\d+\] \S+ (\S+) (cam\d+)->(cam\d+): PA", line)
            if m and last is not None:
                diag[m.groups()] = last
                last = None
    rows = [{"person": k[0], "master": k[1], "slave": k[2], "w_mean": v["slave_weight_mean"],
             "w_p10": v["slave_weight_p10"], "w_p90": v["slave_weight_p90"], "fallback": v["fallback_frames"]}
            for k, v in diag.items()]
    return pd.DataFrame(rows, columns=["person", "master", "slave", "w_mean", "w_p10", "w_p90", "fallback"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--mask", default="masked", choices=["masked", "unmasked"])
    a = ap.parse_args()
    P, methods = load(a.out, a.mask)
    print(f"{len(P)} cặp ({a.mask}), phương pháp: {', '.join(methods)}; theo người: {P.person.value_counts().to_dict()}")
    print("Master:", P.master.value_counts().sort_index().to_dict())

    print("\n== Trung bình PA-MPJPE / MPJPE (mm)")
    cols = ["mono"] + methods
    tab = P.groupby("person")[cols].mean()
    tab.loc["tất cả"] = P[cols].mean()
    tab.loc["mỗi Master như nhau"] = P.groupby("master")[cols].mean().mean()
    print(tab.round(2).to_string())
    mcols = ["mp_mono"] + [f"mp_{m}" for m in methods]
    print(P[mcols].mean().round(2).to_string())

    ref = "aotr" if "aotr" in methods else methods[0]
    print(f"\n== Hiệu PA theo cặp (âm = tốt hơn), bootstrap theo cụm camera")
    for m in methods:
        if m != ref:
            print(compare(P, m, ref))
    if {"adaptive_naive", "naive"} <= set(methods):
        print(compare(P, "adaptive_naive", "naive"))
    print("MPJPE:")
    for m in methods:
        if m != ref:
            print("  " + compare(P, m, ref, prefix="mp_"))

    print("\n== Cặp thua mono (ΔPA > 0)")
    for m in methods:
        lost = P[P[m] > P.mono]
        print(f"{m}: {len(lost)}/{len(P)}" + (f", ví dụ {', '.join((lost.person + ' ' + lost.master + '→' + lost.slave).head(6))}"
                                             if len(lost) else ""))

    cam_mono = P.groupby(["person", "master"]).mono.median()
    P["mono_slave"] = [cam_mono.get((p, s), np.nan) for p, s in zip(P.person, P.slave)]
    P["gap"] = P.mono - P.mono_slave  # > 0: Slave tốt hơn Master (dùng GT, chỉ để phân tích)
    Q = P.dropna(subset=["gap"]).copy()
    if len(Q) > 10:
        print(f"\n== Theo độ chênh chất lượng Slave − Master (n={len(Q)}; dương = Slave tốt hơn)")
        Q["bin"] = pd.cut(Q.gap, [-np.inf, -5, -1, 1, 5, np.inf],
                          labels=["Slave kém >5", "kém 1–5", "±1", "tốt 1–5", "tốt >5"])
        g = Q.assign(**{f"{m}−mono": Q[m] - Q.mono for m in methods})
        cols_b = [f"{m}−mono" for m in methods]
        if "adaptive_naive" in methods and ref != "adaptive_naive":
            g["adaptive−" + ref] = Q["adaptive_naive"] - Q[ref]
            cols_b.append("adaptive−" + ref)
        out = g.groupby("bin", observed=True)[cols_b].mean()
        out.insert(0, "n", g.groupby("bin", observed=True).size())
        print(out.round(2).to_string())
        for m in methods:
            if m != ref:
                r, p = spearmanr(Q.gap, Q[m] - Q[ref])
                print(f"spearman(gap, {m} − {ref}) = {r:+.2f} (p={p:.2g})")

    print("\n== Oracle: chọn phương pháp tốt nhất cho từng cặp")
    best = P[methods].min(axis=1)
    print(" | ".join(f"{m} {P[m].mean():.2f}" for m in methods) + f" | oracle {best.mean():.2f}")
    print("thắng:", P[methods].idxmin(axis=1).value_counts().to_dict())

    D = parse_logs(a.out)
    for f in sorted(glob.glob(f"{a.out}/adaptive_diag_*.csv")):  # bản sao lưu khi log cũ bị ghi đè
        B = pd.read_csv(f)
        if "person" not in B and P.person.nunique() == 1:
            B["person"] = P.person.iloc[0]
        if "person" not in B:
            print(f"! bỏ qua {f}: không có cột person mà kết quả có nhiều người")
            continue
        B = B[[c for c in D.columns if c in B]]
        have = set(zip(D.person, D.master, D.slave))
        D = pd.concat([D, B[[k not in have for k in zip(B.person, B.master, B.slave)]]], ignore_index=True)
    if len(D):
        P = P.merge(D, on=KEY, how="left")
        R = P.dropna(subset=["w_mean"])
        print(f"\n== Chẩn đoán adaptive_naive (n={len(R)} cặp có log)")
        print(R[["w_mean", "w_p10", "w_p90", "fallback"]].describe().round(3).to_string())
        Rg = R.dropna(subset=["gap"])
        if len(Rg) > 10:
            r, p = spearmanr(Rg.gap, Rg.w_mean)
            print(f"spearman(gap, trọng số Slave trung bình) = {r:+.2f} (p={p:.2g}): trọng số có tăng khi Slave tốt hơn không")
    path = os.path.join(a.out, f"analysis_pairs_{a.mask}.csv")
    P.to_csv(path, index=False)
    print(f"\nBảng theo cặp: {path}")


if __name__ == "__main__":
    main()

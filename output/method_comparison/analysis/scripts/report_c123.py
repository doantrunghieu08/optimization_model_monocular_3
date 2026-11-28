"""Bảng thực nghiệm cho 3 đóng góp (C1, C2, C3) từ các thư mục run_method_comparison.

  C1  AOTR an toàn hơn naive / adaptive_naive: MG, WG10, HR@0, HR@1, HBR cho PA/MPJPE/MBLE/ACCEL,
      và hiệu AOTR - baseline với CI95 bootstrap theo cụm (person, master).
  C2  Chọn slave TRƯỚC khi dung hợp, không cần GT: theo độ rung (jitter) 2D->3D của slave,
      so với chọn ngẫu nhiên và oracle; luật tỉ lệ jitter slave/master.
  C3  Giao thức đánh giá: (a) chỉ số rủi ro đổi thứ hạng so với MG;
      (b) HOE = MPJPE(khung camera, extrinsic thật, EXT=1) - MPJPE(xoay khớp theo GT, EXT=0), cùng cặp.

Chỉ dùng hàng status=ok, mask=masked, method in (aotr, naive, adaptive_naive).
Các master có MPJPE mono > 3 x trung vị của sequence bị loại khỏi MPJPE (dấu hiệu extrinsic sai),
PA vẫn giữ (PA căn Procrustes nên không phụ thuộc extrinsic).

Không cần đủ mọi cặp camera: một cặp (person, master, slave) được dùng khi có đủ 3 phương pháp
(status=ok); cặp thiếu/lỗi chỉ bị bỏ, và bảng "Độ phủ" cho biết thiếu bao nhiêu, thiếu ở camera nào.

Tổng hợp mọi sequence (tự dò output/method_comparison/*: run_meta.json ext=true -> EXT=1, ext=false
-> EXT=0 cho C3b, analysis/prefusion_features_<seq>.csv cho C2; nhiều thư mục EXT=1 cùng sequence
-> lấy thư mục mới nhất, in ra để kiểm tra; --seq/--noext/--feats ghi đè):
  venv311/bin/python output/method_comparison/analysis/scripts/report_c123.py --auto \
    --txt output/method_comparison/analysis/report_c123_all.txt
  (--only 001_mma5 003_mma5 ... để giới hạn; --skip-dir 001_mma5_ext ... để bỏ thư mục)

C1-J: MPJPE / PA-MPJPE theo 13 khớp (neck = trung điểm 2 vai, 2 vai, 2 khuỷu, 2 cổ tay, 2 hông,
2 gối, 2 cổ chân) cho từng sequence, GỘP và MACRO; cần results_w*.csv do runner bản mới ghi.

ABLATION: mọi nhãn phương pháp khác 3 pp chính trong results (vd. aotr@fixed_alpha=0.5,
naive@use_aotr_stability=false) được so với AOTR đầy đủ trên cùng cặp, theo sequence, GỘP và MACRO.

Phần GỘP: C1, C2, C3a, C3b trên mọi cặp (cụm bootstrap = (sequence, person, master)), bảng sai số
tuyệt đối, và trung bình MACRO (mỗi sequence trọng số bằng nhau) để sequence nhiều cặp không chi phối.

Ví dụ chỉ định tay:
  venv311/bin/python output/method_comparison/analysis/scripts/report_c123.py \
    --seq 001_mma5 output/method_comparison/001_mma5_ext \
    --seq 003_mma5 output/method_comparison/003_mma5_ext \
    --seq 025_grappling2 output/method_comparison/025_grappling2_ext \
    --noext 001_mma5 output/method_comparison/001_mma5 \
    --feats 001_mma5 output/method_comparison/analysis/prefusion_features_001_mma5.csv \
    --feats 003_mma5 output/method_comparison/analysis/prefusion_features_003_mma5.csv \
    --feats 025_grappling2 output/method_comparison/analysis/prefusion_features_025_grappling2.csv \
    --txt output/method_comparison/analysis/report_c123.txt
"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np

METHODS = ("aotr", "naive", "adaptive_naive")
BASES = ("naive", "adaptive_naive")
METRICS = ("pa", "mpjpe", "mble", "accel")
TAU = 1.0
OUTLIER_FACTOR = 3.0
PERSON = {"a1": "aria01", "a2": "aria02"}
# Theo khớp (cột <met>_mono_<joint>, <met>_<joint> do run_method_comparison ghi; neck = trung điểm 2 vai).
JOINTS = ("neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle")
JMETRICS = ("pa", "mpjpe")


# ----------------------------------------------------------------------------- load
def load_dir(d):
    """{(person, master, slave): {method: {metric: gain, metric_mono: value}}} + info."""
    data, tags = collections.defaultdict(dict), collections.defaultdict(set)
    status = collections.Counter()
    for f in sorted(glob.glob(f"{d}/results_w*.csv")):
        for r in csv.DictReader(open(f)):
            status[str(r.get("status", ""))[:150]] += 1
            # Mọi nhãn phương pháp được đọc: 3 pp chính + biến thể ablation (vd. aotr@fixed_alpha=0.5).
            if r.get("status") != "ok" or r.get("mask") != "masked" or not r.get("method"):
                continue
            k = (r["person"], r["master"], r["slave"])
            if r["method"] in data[k]:
                continue  # same pair from another split: keep the first
            rec = {}
            for m in METRICS:
                rec[m + "_mono"] = float(r[m + "_mono"])
                rec[m] = rec[m + "_mono"] - float(r[m])  # gain: > 0 = fusion better than mono
            for m in JMETRICS:  # theo khớp: chỉ có ở kết quả chạy bằng runner bản mới
                for j in JOINTS:
                    try:
                        mono, fused = float(r[f"{m}_mono_{j}"]), float(r[f"{m}_{j}"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if np.isfinite(mono) and np.isfinite(fused):
                        rec[f"{m}@{j}_mono"], rec[f"{m}@{j}"] = mono, mono - fused
            data[k][r["method"]] = rec
            tags[r["method"]].add(r.get("config_sha") or f"commit:{r.get('commit')}")
    data = {k: v for k, v in data.items() if all(m in v for m in METHODS)}
    bad = {s: n for s, n in status.items() if s != "ok" and not s.startswith("skip")}
    if bad:
        print(f"  [{d}] hàng lỗi: " + "; ".join(f"{n} x {s}" for s, n in sorted(bad.items(), key=lambda x: -x[1])[:3]))
    return data, tags


def mpjpe_outlier_masters(data):
    per = collections.defaultdict(list)
    for k, v in data.items():
        per[k[:2]].append(v["aotr"]["mpjpe_mono"])
    med = {c: float(np.median(x)) for c, x in per.items()}
    seq_med = float(np.median(list(med.values())))
    return {c: m for c, m in med.items() if m > OUTLIER_FACTOR * seq_med}, seq_med


# ----------------------------------------------------------------------------- metrics
def metrics(g):
    g = np.asarray(g, float); n = len(g); k = max(1, n // 10)
    ben, harm = np.maximum(g, 0).sum(), np.maximum(-g, 0).sum()
    # HBR = harm/benefit. 0/0 (không lợi, không hại) -> nan (hiển thị N/A); x/0 với x>0 -> inf (chỉ có hại).
    hbr = harm / ben if ben > 0 else (np.inf if harm > 0 else np.nan)
    return {"MG": g.mean(), "WG10": np.sort(g)[:k].mean(), "HR0": 100 * (g < 0).mean(),
            "HR1": 100 * (g < -TAU).mean(), "HBR": hbr}


def fmt_hbr(v, w=8):
    """N/A cho 0/0 (không lợi, không hại); ∞ khi có hại mà lợi = 0; còn lại là số."""
    if np.isnan(v):
        return f"{'N/A':>{w}s}"
    if np.isinf(v):
        return f"{'∞':>{w}s}"
    return f"{v:{w}.3f}"


def cluster_bootstrap(clusters, fn, B=2000, seed=0):
    """clusters: list of arrays (rows = pairs). fn(rows) -> vector. Returns observed, lo, hi."""
    rng = np.random.default_rng(seed)
    obs = np.asarray(fn(np.concatenate(clusters)))
    bs = np.array([fn(np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]))
                   for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return obs, lo, hi


def sig(lo, hi, better_positive=True):
    if better_positive:
        return "*" if lo > 0 else ("(-)" if hi < 0 else "")
    return "*" if hi < 0 else ("(+)" if lo > 0 else "")


def coverage(d, data):
    """(units_total, pairs đủ 3 pp, {camera: số unit lỗi}) — unit = (person, master, slave)."""
    try:
        total = int(json.load(open(f"{d}/run_meta.json")).get("units_total") or 0)
    except (OSError, ValueError):
        total = 0
    bad = collections.Counter()
    seen_bad = set()
    for f in sorted(glob.glob(f"{d}/results_w*.csv")):
        for r in csv.DictReader(open(f)):
            st = str(r.get("status", ""))
            k = (r["person"], r["master"], r["slave"])
            if st != "ok" and not st.startswith("skip") and k not in data and k not in seen_bad:
                seen_bad.add(k); bad[r["master"]] += 1; bad[r["slave"]] += 1
    return total, len(data), bad


def discover(root, only=None, skip=()):
    """-> ({seq: ext_dir}, {seq: noext_dir}, {seq: feats_csv}, notes)."""
    ext, noext, notes = collections.defaultdict(list), {}, []
    for meta in sorted(glob.glob(f"{root}/*/run_meta.json")):
        d = os.path.dirname(meta)
        if os.path.basename(d) in skip:
            continue
        try:
            m = json.load(open(meta))
        except ValueError:
            continue
        if sorted(m.get("methods") or []) != sorted(METHODS):
            continue  # ablation / other method sets
        seq = os.path.splitext(os.path.basename(str(m.get("brute") or d)))[0]
        if only and seq not in only:
            continue
        if not glob.glob(f"{d}/results_w*.csv"):
            continue
        if m.get("ext") is True:
            ext[seq].append(d)
        elif m.get("ext") is False:
            noext.setdefault(seq, d)
    chosen = {}
    for seq, ds in ext.items():
        ds.sort(key=lambda d: os.path.getmtime(f"{d}/run_meta.json"))
        chosen[seq] = ds[-1]
        if len(ds) > 1:
            notes.append(f"{seq}: {len(ds)} thư mục EXT=1 {[os.path.basename(x) for x in ds]} -> dùng "
                         f"{os.path.basename(ds[-1])} (mới nhất; đổi bằng --seq hoặc --skip-dir)")
    feats = {s: f for s in chosen if os.path.isfile(f := f"{root}/analysis/prefusion_features_{s}.csv")}
    return chosen, {s: d for s, d in noext.items() if s in chosen}, feats, notes


def abs_table(title, datasets, out):
    """datasets: [(name, data, excl)]. Sai số tuyệt đối (mm) + MACRO."""
    out(f"\n### {title}: sai số tuyệt đối (mm), thấp hơn = tốt hơn")
    out(f"  {'sequence':18s}{'n':>5s} |" + "|".join(f"{m.upper():^36s}" for m in METRICS))
    out(f"  {'':18s}{'':5s} |" + "|".join("".join(f"{x:>9s}" for x in ("mono", "aotr", "naive", "adapt"))
                                           for _ in METRICS))
    per = {}
    for name, data, excl in datasets:
        cells, row = [], {}
        for met in METRICS:
            ks = [k for k in data if not (met == "mpjpe" and k[:2] in excl)]
            mono = np.mean([data[k]["aotr"][met + "_mono"] for k in ks])
            vals = [mono] + [np.mean([data[k][m][met + "_mono"] - data[k][m][met] for k in ks]) for m in METHODS]
            row[met] = vals
            cells.append("".join(f"{v:9.2f}" for v in vals))
        per[name] = row
        out(f"  {name:18s}{len(data):5d} |" + "|".join(cells))
    if len(per) > 1:
        out(f"  {'MACRO (TB sequence)':18s}{'':5s} |" + "|".join(
            "".join(f"{np.mean([per[n][met][i] for n in per]):9.2f}" for i in range(4)) for met in METRICS))


def macro_c1(datasets, out):
    """Mỗi sequence một phiếu: MG/WG10/HR1 của từng pp, trung bình không trọng số + số sequence AOTR thắng."""
    out("\n### C1 MACRO: mỗi sequence trọng số bằng nhau")
    for met in ("pa", "mpjpe"):
        res = {}
        for name, data, excl in datasets:
            ks = [k for k in data if not (met == "mpjpe" and k[:2] in excl)]
            res[name] = {m: metrics([data[k][m][met] for k in ks]) for m in METHODS}
        out(f"  {met.upper()}")
        for crit, hi_better in (("MG", True), ("WG10", True), ("HR1", False)):
            avg = {m: np.mean([res[n][m][crit] for n in res]) for m in METHODS}
            wins = {b: sum((res[n]["aotr"][crit] > res[n][b][crit]) if hi_better else
                           (res[n]["aotr"][crit] < res[n][b][crit]) for n in res) for b in BASES}
            out(f"    {crit:5s} " + "  ".join(f"{m} {avg[m]:+7.2f}" for m in METHODS) + "   | AOTR tốt hơn "
                + ", ".join(f"{b} ở {wins[b]}/{len(res)} seq" for b in BASES))


# ----------------------------------------------------------------------------- C1
def c1(name, data, excl, out):
    out(f"\n### C1 — {name}")
    for met in METRICS:
        keys = [k for k in data if not (met == "mpjpe" and k[:2] in excl)]
        mono = np.mean([data[k]["aotr"][met + "_mono"] for k in keys])
        out(f"  {met.upper():6s} n={len(keys)} cặp, mono {mono:.2f}")
        out(f"    {'method':15s}{'MG':>8s}{'WG10':>8s}{'HR@0':>8s}{'HR@1':>8s}{'HBR':>8s}")
        for m in METHODS:
            d = metrics([data[k][m][met] for k in keys])
            out(f"    {m:15s}{d['MG']:+8.2f}{d['WG10']:+8.2f}{d['HR0']:7.1f}%{d['HR1']:7.1f}%{fmt_hbr(d['HBR'])}")
        if met not in ("pa", "mpjpe"):
            continue
        by = collections.defaultdict(list)
        for k in keys:
            by[k[:2]].append([data[k][m][met] for m in METHODS])
        cl = [np.array(v) for v in by.values()]
        for bi, b in enumerate(BASES, start=1):
            def delta(rows, bi=bi):
                a, c = metrics(rows[:, 0]), metrics(rows[:, bi])
                return [a["MG"] - c["MG"], a["WG10"] - c["WG10"], a["HR1"] - c["HR1"]]
            o, lo, hi = cluster_bootstrap(cl, delta)
            out(f"    AOTR − {b:15s} ΔMG {o[0]:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}]{sig(lo[0], hi[0]):3s}"
                f" ΔWG10 {o[1]:+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]{sig(lo[1], hi[1]):3s}"
                f" ΔHR@1 {o[2]:+.1f} [{lo[2]:+.1f},{hi[2]:+.1f}]{sig(lo[2], hi[2], False)}")
        if met == "pa":
            for b in BASES:
                a_only = sum(data[k]["aotr"]["pa"] < 0 <= data[k][b]["pa"] for k in keys)
                b_only = sum(data[k][b]["pa"] < 0 <= data[k]["aotr"]["pa"] for k in keys)
                out(f"    cặp PA hỏng: chỉ AOTR {a_only}, chỉ {b} {b_only}")
    out("    (* = có ý nghĩa, CI95 bootstrap theo cụm (person, master) không chứa 0, cùng chiều có lợi cho AOTR;"
        " (-)/(+) = có ý nghĩa nhưng ngược chiều)")


# ----------------------------------------------------------------------------- theo khớp
def _jkeys(data, excl, met, j):
    """Cặp có số liệu khớp j cho cả 3 pp (MPJPE: bỏ master ngoại lai như C1)."""
    return [k for k in data if not (met == "mpjpe" and k[:2] in excl)
            and all(f"{met}@{j}" in data[k][m] for m in METHODS)]


def joint_stats(data, excl, met, j):
    """{mono, n, <m>: gain MG, HR1_<m>} cho một khớp, hoặc None nếu không có số liệu."""
    keys = _jkeys(data, excl, met, j)
    if not keys:
        return None
    r = {"n": len(keys), "mono": float(np.mean([data[k]["aotr"][f"{met}@{j}_mono"] for k in keys]))}
    for m in METHODS:
        g = np.array([data[k][m][f"{met}@{j}"] for k in keys])
        r[m], r["HR1_" + m] = float(g.mean()), float(100 * (g < -TAU).mean())
    return r


def joint_table(name, data, excl, out, boot=True):
    """MPJPE / PA theo từng khớp: mono (mm), gain MG của từng pp, HR@1, Δ AOTR−baseline (CI cụm)."""
    if not any(f"pa@{j}" in v["aotr"] for v in data.values() for j in JOINTS):
        out(f"\n### C1-J — {name}: CHƯA có số liệu theo khớp (kết quả chạy trước khi runner ghi cột "
            f"<met>_<joint>; chạy lại run_method_comparison bản mới)")
        return
    out(f"\n### C1-J — {name}: theo khớp (mono = sai số WHAM mm; gain = mono − fused, > 0 là tốt hơn; "
        f"neck = trung điểm 2 vai, không tính vào trung bình chính)")
    for met in JMETRICS:
        out(f"  {met.upper()}")
        out(f"    {'joint':15s}{'n':>5s}{'mono':>8s}{'aotr':>8s}{'naive':>8s}{'adapt':>8s}"
            f"{'HR1 a/n/ad %':>17s}   {'Δ vs naive':>14s}  {'Δ vs adapt':>14s}")
        for j in JOINTS:
            st = joint_stats(data, excl, met, j)
            if st is None:
                out(f"    {j:15s}  (không có)"); continue
            line = (f"    {j:15s}{st['n']:5d}{st['mono']:8.1f}{st['aotr']:+8.2f}{st['naive']:+8.2f}"
                    f"{st['adaptive_naive']:+8.2f}  {st['HR1_aotr']:4.1f}/{st['HR1_naive']:4.1f}/{st['HR1_adaptive_naive']:4.1f}")
            if boot:
                by = collections.defaultdict(list)
                for k in _jkeys(data, excl, met, j):
                    by[k[:2]].append([data[k][m][f"{met}@{j}"] for m in METHODS])
                cl = [np.array(v) for v in by.values()]
                for bi in (1, 2):
                    o, lo, hi = cluster_bootstrap(cl, lambda rows, bi=bi: [rows[:, 0].mean() - rows[:, bi].mean()], B=1000)
                    line += f"   {o[0]:+6.2f}{sig(lo[0], hi[0]):3s}    "
            out(line)
    if boot:
        out("    (Δ = gain AOTR − gain baseline, mm; * = CI95 bootstrap theo cụm (person, master) > 0, (-) = < 0)")


def joint_macro(datasets, out):
    """Theo khớp, mỗi sequence trọng số bằng nhau + số sequence AOTR có gain lớn hơn từng baseline."""
    res = {}
    for name, data, excl in datasets:
        for met in JMETRICS:
            for j in JOINTS:
                st = joint_stats(data, excl, met, j)
                if st is not None:
                    res.setdefault((met, j), {})[name] = st
    if not res:
        return
    out("\n### C1-J MACRO: theo khớp, mỗi sequence trọng số bằng nhau")
    for met in JMETRICS:
        out(f"  {met.upper()}")
        out(f"    {'joint':15s}{'seq':>4s}{'mono':>8s}{'aotr':>8s}{'naive':>8s}{'adapt':>8s}   AOTR thắng naive / adapt")
        for j in JOINTS:
            per = res.get((met, j))
            if not per:
                continue
            avg = {c: float(np.mean([v[c] for v in per.values()])) for c in ("mono",) + METHODS}
            wins = [sum(v["aotr"] > v[b] for v in per.values()) for b in BASES]
            out(f"    {j:15s}{len(per):4d}{avg['mono']:8.1f}{avg['aotr']:+8.2f}{avg['naive']:+8.2f}"
                f"{avg['adaptive_naive']:+8.2f}   {wins[0]}/{len(per)} , {wins[1]}/{len(per)}")


# ----------------------------------------------------------------------------- ablation
ABLATION_NOTES = {
    "naive@use_aotr_stability=false": "naive thô (không làm mượt, không FK)",
    "adaptive_naive@use_aotr_stability=false": "adaptive không làm mượt",
    "adaptive_naive@torso_gate=false": "adaptive không RANSAC thân/ngưỡng",
    "adaptive_naive@torso_gate=false,use_aotr_stability=false,min_bone_direction_cos=-1.0":
        "adaptive thô (không ngưỡng, không lọc xương, không làm mượt; chỉ còn trộn hướng + FK)",
    "aotr@temporal_alignment_fallback=false": "− bước 1b (fallback căn chỉnh)",
    "aotr@max_cross_view_bone_angle_deg=null": "− bước 2 (ngưỡng góc 2 view)",
    "aotr@min_bone_direction_cos=-1.0": "− bước 3 (lọc xương ngược hướng)",
    "aotr@min_correspondences=999": "− bước 4 (bằng chứng 2D; luôn 3d_surrogate)",
    "aotr@unobserved_prior_weight=0.0001": "− prior observability (prior đẳng hướng)",
    "aotr@huber_delta_normalized=1000000": "− Huber (bình phương tối thiểu)",
    "aotr@fixed_alpha=0.5": "− bước 5 (không tối ưu, α=0.5, không kiểm tra cost)",
    "aotr@sequence_stability={enabled: false}": "− bước 7 (làm mượt chuỗi)",
}


def variants(data):
    return sorted({m for v in data.values() for m in v} - set(METHODS))


def ablation_table(name, data, excl, out):
    """Mỗi biến thể so với AOTR đầy đủ trên cùng các cặp: Δ = biến thể − AOTR (gain, mm).
    Δ MG < 0 nghĩa là bỏ thành phần đó làm kết quả kém đi, tức thành phần có ích."""
    labels = variants(data)
    if not labels:
        return
    out(f"\n### ABLATION — {name}: biến thể so với AOTR đầy đủ (Δ = biến thể − AOTR; Δ MG < 0 = thành phần bị bỏ có ích)")
    for met in ("pa", "mpjpe"):
        out(f"  {met.upper()}")
        out(f"    {'biến thể':44s}{'n':>5s}{'MG':>8s}{'WG10':>8s}{'HR@1':>7s}   {'ΔMG [CI95]':>24s}  {'ΔWG10':>7s} {'ΔHR@1':>7s}")
        for lab in labels:
            keys = [k for k in data if lab in data[k] and not (met == "mpjpe" and k[:2] in excl)]
            if not keys:
                continue
            g_v = np.array([data[k][lab][met] for k in keys]); g_a = np.array([data[k]["aotr"][met] for k in keys])
            dv, da = metrics(g_v), metrics(g_a)
            by = collections.defaultdict(list)
            for k, a, v in zip(keys, g_a, g_v):
                by[k[:2]].append([a, v])
            o, lo, hi = cluster_bootstrap([np.array(x) for x in by.values()],
                                          lambda rows: [rows[:, 1].mean() - rows[:, 0].mean()], B=1000)
            out(f"    {lab[:44]:44s}{len(keys):5d}{dv['MG']:+8.2f}{dv['WG10']:+8.2f}{dv['HR1']:6.1f}%"
                f"   {o[0]:+7.2f} [{lo[0]:+6.2f},{hi[0]:+6.2f}]{_abl_sig(lo[0], hi[0]):3s}"
                f"  {dv['WG10'] - da['WG10']:+7.2f} {dv['HR1'] - da['HR1']:+6.1f}%")
    out("    (* = CI95 bootstrap cụm (person, master) < 0: bỏ thành phần làm kém đi có ý nghĩa; (+) = > 0: bỏ đi lại tốt hơn)")
    for l in labels:
        if l in ABLATION_NOTES:
            out(f"      {l}: {ABLATION_NOTES[l]}")


def _abl_sig(lo, hi):
    return "*" if hi < 0 else ("(+)" if lo > 0 else "")


def ablation_macro(datasets, out):
    """Mỗi sequence một phiếu: ΔMG (biến thể − AOTR) trung bình + số sequence biến thể kém hơn AOTR."""
    labels = sorted({l for _, d, _ in datasets for l in variants(d)})
    if not labels:
        return
    out("\n### ABLATION MACRO: mỗi sequence trọng số bằng nhau (ΔMG = biến thể − AOTR, mm)")
    for met in ("pa", "mpjpe"):
        out(f"  {met.upper()}")
        for lab in labels:
            per = []
            for name, d, excl in datasets:
                keys = [k for k in d if lab in d[k] and not (met == "mpjpe" and k[:2] in excl)]
                if keys:
                    per.append(np.mean([d[k][lab][met] - d[k]["aotr"][met] for k in keys]))
            if per:
                out(f"    {lab[:44]:44s} seq={len(per):2d}  ΔMG {np.mean(per):+7.2f}   kém hơn AOTR ở {sum(x < 0 for x in per)}/{len(per)} seq")


# ----------------------------------------------------------------------------- C2
def load_feats(path):
    F = {}
    for r in csv.DictReader(open(path)):
        F[(PERSON.get(r["split"], r["split"]), r["master"], r["slave"])] = {
            k: float(v) for k, v in r.items() if k not in ("split", "master", "slave")}
    return F


def spearman(x, y):
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


def c2(name, data, F, out):
    keys = [k for k in data if k in F and np.isfinite(F[k]["slave_jitter"]) and F[k]["master_jitter"] > 0]
    groups = collections.defaultdict(list)
    for k in keys:
        groups[k[:2]].append(k)
    groups = {c: ks for c, ks in groups.items() if len(ks) > 3}
    out(f"\n### C2 — {name}: {len(keys)} cặp có đặc trưng, {len(groups)} master")
    rules = {"-jitter slave": lambda k: -F[k]["slave_jitter"],
             "bất đồng mono": lambda k: F[k]["disagree_pa"],
             "conf 2D slave": lambda k: F[k]["slave_conf"]}
    for met in ("pa", "mpjpe"):
        out(f"  {met.upper()}: chọn 1 slave cho mỗi master (gain trung bình trên các master)")
        out(f"    {'method':15s}{'ngẫu nhiên':>11s}{'oracle':>8s}  " + "  ".join(f"{r:>24s}" for r in rules))
        for m in METHODS:
            rnd, orc, pick = [], [], collections.defaultdict(list)
            for ks in groups.values():
                g = np.array([data[k][m][met] for k in ks])
                rnd.append(g.mean()); orc.append(g.max())
                for rn, f in rules.items():
                    pick[rn].append(g[int(np.argmax([f(k) for k in ks]))])
            R, O = np.mean(rnd), np.mean(orc)
            cells = []
            for rn in rules:
                p = np.array(pick[rn])
                cells.append(f"{p.mean():+.2f} ({100 * (p.mean() - R) / (O - R):+.0f}%, hỏng {int((p < 0).sum())}/{len(p)})")
            out(f"    {m:15s}{R:+11.2f}{O:+8.2f}  " + "  ".join(f"{c:>24s}" for c in cells))
        rho = np.nanmean([spearman([-F[k]["slave_jitter"] for k in ks], [data[k]["aotr"][met] for k in ks])
                          for ks in groups.values()])
        out(f"    Spearman trong cùng master (-jitter slave, gain AOTR) = {rho:+.2f}")
    out("  Luật tỉ lệ jitter r = jitter_slave / jitter_master  (tỉ lệ cặp PA hỏng)")
    for lab, cond in (("r < 0.8", lambda r: r < 0.8), ("0.8 ≤ r < 1.2", lambda r: 0.8 <= r < 1.2),
                      ("r ≥ 1.2", lambda r: r >= 1.2)):
        ks = [k for k in keys if cond(F[k]["slave_jitter"] / F[k]["master_jitter"])]
        if ks:
            out(f"    {lab:14s} n={len(ks):4d}  " + "  ".join(
                f"{m} hỏng {100 * np.mean([data[k][m]['pa'] < 0 for k in ks]):5.1f}% gain {np.mean([data[k][m]['pa'] for k in ks]):+.2f}"
                for m in METHODS))


# ----------------------------------------------------------------------------- C3
def c3_ranking(name, data, excl, out):
    out(f"\n### C3a — {name}: thứ hạng theo từng chỉ số (1 = tốt nhất)")
    for met in ("pa", "mpjpe"):
        keys = [k for k in data if not (met == "mpjpe" and k[:2] in excl)]
        d = {m: metrics([data[k][m][met] for k in keys]) for m in METHODS}
        line = []
        for crit, best_high in (("MG", True), ("WG10", True), ("HR1", False), ("HBR", False)):
            ok = [m for m in METHODS if not np.isnan(d[m][crit])]   # HBR 0/0 -> N/A, không xếp hạng
            order = sorted(ok, key=lambda m: -d[m][crit] if best_high else d[m][crit])
            na = [m for m in METHODS if m not in ok]
            line.append(f"{crit}: " + " > ".join(order) + (f" (N/A: {', '.join(na)})" if na else ""))
        flip = len({" > ".join(sorted(METHODS, key=lambda m: -d[m]["MG"])),
                    " > ".join(sorted(METHODS, key=lambda m: -d[m]["WG10"]))}) > 1
        out(f"  {met.upper():6s} " + " | ".join(line) + ("   <== MG và WG10 xếp hạng KHÁC nhau" if flip else ""))


def c3_hoe(name, ext, noext, excl, out):
    common = [k for k in ext if k in noext and k[:2] not in excl]
    out(f"\n### C3b — {name}: HOE trên {len(common)} cặp chung (EXT=1 − EXT=0)")
    if not common:
        out("  (không có cặp chung)"); return
    h = np.array([ext[k]["aotr"]["mpjpe_mono"] - noext[k]["aotr"]["mpjpe_mono"] for k in common])
    out(f"  mono: HOE TB {h.mean():+.1f} mm, trung vị {np.median(h):+.1f}, "
        f"MPJPE EXT=0 {np.mean([noext[k]['aotr']['mpjpe_mono'] for k in common]):.1f} -> "
        f"EXT=1 {np.mean([ext[k]['aotr']['mpjpe_mono'] for k in common]):.1f}")
    for tag, D in (("EXT=0", noext), ("EXT=1", ext)):
        d = {m: metrics([D[k][m]["mpjpe"] for k in common]) for m in METHODS}
        out(f"  {tag} MPJPE  " + "  ".join(f"{m} MG {d[m]['MG']:+.2f} HR1 {d[m]['HR1']:.1f}%" for m in METHODS)
            + "  | thứ hạng MG: " + " > ".join(sorted(METHODS, key=lambda m: -d[m]["MG"])))
    out("  (Thứ hạng khác nhau giữa EXT=0 và EXT=1 = giao thức xoay-theo-GT che mất lỗi hướng toàn cục.)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Bảng C1/C2/C3 (từng sequence + gộp)")
    ap.add_argument("--auto", nargs="?", const="output/method_comparison", metavar="ROOT",
                    help="tự dò mọi sequence trong ROOT (mặc định output/method_comparison)")
    ap.add_argument("--only", nargs="+", default=None, metavar="SEQ")
    ap.add_argument("--skip-dir", nargs="+", default=[], metavar="DIRNAME")
    ap.add_argument("--seq", nargs=2, action="append", default=[], metavar=("NAME", "EXT_DIR"))
    ap.add_argument("--noext", nargs=2, action="append", default=[], metavar=("NAME", "DIR"))
    ap.add_argument("--feats", nargs=2, action="append", default=[], metavar=("NAME", "CSV"))
    ap.add_argument("--txt", help="ghi thêm ra tệp này")
    a = ap.parse_args()
    lines = []
    def out(s=""):
        print(s); lines.append(s)
    seqs, noext, feats = {}, {}, {}
    if a.auto:
        seqs, noext, feats, notes = discover(a.auto, a.only, set(a.skip_dir))
        out(f"[auto] {a.auto}: {len(seqs)} sequence EXT=1, {len(noext)} có EXT=0 (C3b), {len(feats)} có đặc trưng (C2)")
        for n in notes:
            out(f"[auto] {n}")
    seqs.update(dict(a.seq)); noext.update(dict(a.noext)); feats.update(dict(a.feats))
    if not seqs:
        ap.error("cần --seq hoặc --auto")

    done, cover, all_tags = [], [], collections.defaultdict(set)
    P, PF, PN, PX = {}, {}, {}, set()   # pooled data / features / EXT=0 data / excluded (seq/person, master)
    for name, d in sorted(seqs.items()):
        data, tags = load_dir(d)
        total, n_ok, bad = coverage(d, data)
        cover.append((name, d, total, n_ok, bad))
        out(f"\n================ {name}  ({d}): {len(data)} cặp đủ 3 phương pháp")
        if not data:
            out("  BỎ QUA: không có cặp nào status=ok, mask=masked đủ aotr/naive/adaptive_naive (xem hàng lỗi ở trên).")
            continue
        for m, t in tags.items():
            all_tags[m] |= t
            if len(t) > 1:
                out(f"  CẢNH BÁO: {m} có kết quả từ {len(t)} cấu hình/commit khác nhau {sorted(t)} — "
                    "bảng đang trộn tham số; chạy lại với --fresh.")
        excl, med = mpjpe_outlier_masters(data)
        for c, m in excl.items():
            out(f"  LOẠI khỏi MPJPE: master {c} có MPJPE mono {m:.0f} mm (> {OUTLIER_FACTOR:g} x trung vị {med:.0f})"
                " -> kiểm tra extrinsic camera này")
        c1(name, data, excl, out)
        F = None
        if name in feats and not os.path.isfile(feats[name]):
            out(f"\n### C2 — {name}: BỎ QUA, không có {feats[name]}. Tạo bằng:\n"
                f"  venv311/bin/python output/method_comparison/analysis/scripts/prefusion_feats_any.py "
                f"{feats[name]} configs/{name}.yml")
        elif name in feats:
            F = load_feats(feats[name]); c2(name, data, F, out)
        c3_ranking(name, data, excl, out)
        joint_table(name, data, excl, out)
        ablation_table(name, data, excl, out)
        N = load_dir(noext[name])[0] if name in noext else None
        if N is not None:
            c3_hoe(name, data, N, excl, out)
        done.append((name, data, excl))
        pk = lambda k: (f"{name}/{k[0]}", k[1], k[2])   # cụm gộp = (sequence/person, master)
        for k, v in data.items():
            P[pk(k)] = v
        PX |= {(f"{name}/{c[0]}", c[1]) for c in excl}
        if F:
            PF.update({pk(k): v for k, v in F.items()})
        if N:
            PN.update({pk(k): v for k, v in N.items()})

    out("\n================ ĐỘ PHỦ (cặp thiếu chỉ bị bỏ, không làm hỏng sequence)")
    for name, d, total, n_ok, bad in cover:
        pct = f"{100 * n_ok / total:.0f}%" if total else "?"
        worst = ", ".join(f"{c}:{n}" for c, n in bad.most_common(3) if n)
        out(f"  {name:18s} {n_ok:4d}/{total or '?':>4} cặp ({pct})" + (f"   camera có unit lỗi: {worst}" if worst else ""))
    multi = {m: t for m, t in all_tags.items() if len(t) > 1}
    if multi:
        out("  CẢNH BÁO: các sequence dùng cấu hình khác nhau: " + "; ".join(f"{m}: {sorted(t)}" for m, t in multi.items())
            + " — chỉ gộp khi đó là cùng một bộ tham số (config_sha None = chạy trước khi có mã cấu hình).")

    abs_table("Theo sequence", done, out)
    if len(done) > 1:
        out(f"\n================ GỘP {len(done)} sequence: {len(P)} cặp "
            f"(MPJPE bỏ {len(PX)} master ngoại lai, các độ đo khác giữ)")
        c1("gộp", P, PX, out)
        if PF:
            c2(f"gộp ({len({k[0].split('/')[0] for k in PF})} sequence có đặc trưng)", P, PF, out)
        c3_ranking("gộp", P, PX, out)
        joint_table("gộp", P, PX, out)
        ablation_table("gộp", P, PX, out)
        if PN:
            c3_hoe(f"gộp ({len({k[0].split('/')[0] for k in PN})} sequence có EXT=0)", P, PN, PX, out)
        macro_c1(done, out)
        joint_macro(done, out)
        ablation_macro(done, out)
    if a.txt:
        open(a.txt, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"\n-> {a.txt}")


if __name__ == "__main__":
    main()

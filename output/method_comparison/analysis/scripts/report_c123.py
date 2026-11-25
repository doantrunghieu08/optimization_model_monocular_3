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

Ví dụ:
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
import argparse, collections, csv, glob, sys
import numpy as np

METHODS = ("aotr", "naive", "adaptive_naive")
BASES = ("naive", "adaptive_naive")
METRICS = ("pa", "mpjpe", "mble", "accel")
TAU = 1.0
OUTLIER_FACTOR = 3.0
PERSON = {"a1": "aria01", "a2": "aria02"}


# ----------------------------------------------------------------------------- load
def load_dir(d):
    """{(person, master, slave): {method: {metric: gain, metric_mono: value}}} + info."""
    data, tags = collections.defaultdict(dict), collections.defaultdict(set)
    status = collections.Counter()
    for f in sorted(glob.glob(f"{d}/results_w*.csv")):
        for r in csv.DictReader(open(f)):
            status[str(r.get("status", ""))[:150]] += 1
            if r.get("status") != "ok" or r.get("mask") != "masked" or r.get("method") not in METHODS:
                continue
            k = (r["person"], r["master"], r["slave"])
            if r["method"] in data[k]:
                continue  # same pair from another split: keep the first
            rec = {}
            for m in METRICS:
                rec[m + "_mono"] = float(r[m + "_mono"])
                rec[m] = rec[m + "_mono"] - float(r[m])  # gain: > 0 = fusion better than mono
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
    ben = np.maximum(g, 0).sum()
    return {"MG": g.mean(), "WG10": np.sort(g)[:k].mean(), "HR0": 100 * (g < 0).mean(),
            "HR1": 100 * (g < -TAU).mean(), "HBR": np.maximum(-g, 0).sum() / ben if ben > 0 else np.inf}


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
            out(f"    {m:15s}{d['MG']:+8.2f}{d['WG10']:+8.2f}{d['HR0']:7.1f}%{d['HR1']:7.1f}%{d['HBR']:8.3f}")
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
            order = sorted(METHODS, key=lambda m: -d[m][crit] if best_high else d[m][crit])
            line.append(f"{crit}: " + " > ".join(order))
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
    ap = argparse.ArgumentParser(description="Bảng C1/C2/C3")
    ap.add_argument("--seq", nargs=2, action="append", metavar=("NAME", "EXT_DIR"), required=True)
    ap.add_argument("--noext", nargs=2, action="append", default=[], metavar=("NAME", "DIR"))
    ap.add_argument("--feats", nargs=2, action="append", default=[], metavar=("NAME", "CSV"))
    ap.add_argument("--txt", help="ghi thêm ra tệp này")
    a = ap.parse_args()
    lines = []
    def out(s=""):
        print(s); lines.append(s)
    noext, feats = dict(a.noext), dict(a.feats)
    pooled = {}
    for name, d in a.seq:
        data, tags = load_dir(d)
        out(f"\n================ {name}  ({d}): {len(data)} cặp đủ 3 phương pháp")
        if not data:
            out("  BỎ QUA: không có cặp nào status=ok, mask=masked đủ aotr/naive/adaptive_naive (xem hàng lỗi ở trên).")
            continue
        for m, t in tags.items():
            if len(t) > 1:
                out(f"  CẢNH BÁO: {m} có kết quả từ {len(t)} cấu hình/commit khác nhau {sorted(t)} — "
                    "bảng đang trộn tham số; chạy lại với --fresh.")
        excl, med = mpjpe_outlier_masters(data)
        for c, m in excl.items():
            out(f"  LOẠI khỏi MPJPE: master {c} có MPJPE mono {m:.0f} mm (> {OUTLIER_FACTOR:g} x trung vị {med:.0f})"
                " -> kiểm tra extrinsic camera này")
        c1(name, data, excl, out)
        if name in feats:
            c2(name, data, load_feats(feats[name]), out)
        c3_ranking(name, data, excl, out)
        if name in noext:
            c3_hoe(name, data, load_dir(noext[name])[0], excl, out)
        for k, v in data.items():
            if not (k[:2] in excl):
                pooled[(name,) + k] = v
    if len(a.seq) > 1:
        # pooled: clusters become (sequence, person, master)
        P = {(k[0] + "/" + k[1], k[2], k[3]): v for k, v in pooled.items()}
        out(f"\n================ GỘP {len(a.seq)} sequence: {len(P)} cặp")
        c1("gộp", P, set(), out)
        c3_ranking("gộp", P, set(), out)
    if a.txt:
        open(a.txt, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"\n-> {a.txt}")


if __name__ == "__main__":
    main()

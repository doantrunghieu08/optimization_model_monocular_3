"""Xếp hạng biến thể so với 'aotr' gốc, ghép cặp, bootstrap theo master. usage: rank.py run_dir [run_dir...]
Mục tiêu J = MG + WG10 trên PA và MPJPE (mm, cao = tốt): thưởng cải thiện trung bình VÀ phạt đuôi xấu (C3).
Chỉ chấp nhận biến thể nếu CI95 của ΔJ > 0 (không đuổi theo nhiễu)."""
import csv,glob,sys,collections,random,statistics as st
T=collections.defaultdict(dict)
for d in sys.argv[1:]:
    for f in glob.glob(d+"/results_w*.csv"):
        for r in csv.DictReader(open(f)):
            if r["status"]=="ok" and r["mask"]=="masked": T[(r["person"],r["master"],r["slave"])][r["method"]]=r
M=sorted({m for v in T.values() for m in v})
def stats(rows,m):
    o={}
    for k in ("pa","mpjpe"):
        g=sorted(float(v[m][k+"_mono"])-float(v[m][k]) for v in rows); n=len(g)
        o[k+"_MG"]=st.mean(g); o[k+"_WG10"]=st.mean(g[:max(1,n//10)]); o[k+"_HR1"]=100*sum(x<-1 for x in g)/n
    o["J"]=o["pa_MG"]+o["pa_WG10"]+o["mpjpe_MG"]+o["mpjpe_WG10"]; return o
out=[]
for m in M:
    P=[v for v in T.values() if m in v and "aotr" in v]
    if not P: continue
    by=collections.defaultdict(list)
    for v in P: by[v[m]["master"]].append(v)
    rnd=random.Random(0); ks=list(by); bs=[]
    for _ in range(500):
        s=[v for c in (rnd.choice(ks) for _ in ks) for v in by[c]]; bs.append(stats(s,m)["J"]-stats(s,"aotr")["J"])
    bs.sort(); a,b=stats(P,m),stats(P,"aotr")
    out.append((a["J"]-b["J"],m,len(P),a,bs[12],bs[487]))
print(f"{'variant':60s} n   ΔJ [CI95]          PA MG/WG10/HR1      MPJPE MG/WG10/HR1")
for dj,m,n,a,lo,hi in sorted(out,reverse=True):
    flag="  <== tốt hơn có ý nghĩa" if lo>0 and m!="aotr" else ""
    print(f"{m:60s}{n:3d} {dj:+5.2f} [{lo:+.2f},{hi:+.2f}]  {a['pa_MG']:+.2f}/{a['pa_WG10']:+.2f}/{a['pa_HR1']:.1f}%   {a['mpjpe_MG']:+.2f}/{a['mpjpe_WG10']:+.2f}/{a['mpjpe_HR1']:.1f}%{flag}")

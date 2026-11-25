"""Pre-fusion (no GT) features per (person, master, slave) (args: out.csv config.yml; tracking_results.pth next to the pkl if not in yaml)."""
import sys,json,csv,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,".")
from compat import patch_numpy_and_inspect,load_joblib_compat,resolve_person_payload; patch_numpy_and_inspect()
import numpy as np, yaml
from evaluation_pipeline.extrinsics import extrinsic_from_config
cfg=yaml.safe_load(open(sys.argv[2]))
cams={c["id"]:c for c in cfg["segments"][0]["cameras"]}
J=np.load("models/smpl/J_regressor_body25.npy",allow_pickle=True); J=np.asarray(J.todense() if hasattr(J,"todense") else J)
BODY=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]  # neck..ankles
def load(cam,person):
    c=cams[cam]; w=load_joblib_compat(c["pkl_local"]); import os; t=load_joblib_compat(c.get("tracking_result_local") or os.path.join(os.path.dirname(c["pkl_local"]),"tracking_results.pth"))
    pw=resolve_person_payload(w,c["pkl_local"],person); pt=resolve_person_payload(t,c["pkl_local"],person)
    fid=np.asarray(pw["frame_ids"]).astype(int)
    X=np.einsum("jv,fvc->fjc",J,np.asarray(pw["verts"]))[:,BODY]          # camera frame, m
    tf=np.asarray(pt["frame_id"]).astype(int); kp=np.asarray(pt["keypoints"]); bb=np.asarray(pt["bbox"])
    conf={int(f):float(kp[i,:,2].mean()) for i,f in enumerate(tf)}; size={int(f):float(bb[i,2]) for i,f in enumerate(tf)}
    R=extrinsic_from_config(c["extrinsic"],cam)[0] if c.get("extrinsic") else np.full((3,3),np.nan)
    return dict(fid=fid,X=X,conf=conf,size=size,R=R,pos=np.median(np.asarray(pw["trans"]),0))
def pa(a,b):
    a=a-a.mean(0); b=b-b.mean(0); U,S,Vt=np.linalg.svd(a.T@b); d=np.sign(np.linalg.det(U@Vt))
    D=np.diag([1,1,d]); R=U@D@Vt; s=(S*np.diag(D)).sum()/(a**2).sum()
    return np.linalg.norm(s*a@R-b,axis=1).mean()*1000
def accel(X): return np.linalg.norm(X[2:]-2*X[1:-1]+X[:-2],axis=2).mean()*1000 if len(X)>2 else np.nan
splits={seg["target_person"]:(seg["target_person"],",".join(c["id"] for c in seg["cameras"])) for seg in cfg["segments"]}
out=[]
for sp,(person,cl) in splits.items():
    D={}
    for c in cl.split(","):
        try: D[c]=load(c,person)
        except Exception as e: print("skip",person,c,e,file=sys.stderr)
    for m in D:
        for s in D:
            if m==s: continue
            A,B=D[m],D[s]; common=sorted(set(A["fid"])&set(B["fid"]))
            ia=[list(A["fid"]).index(f) for f in common]; ib=[list(B["fid"]).index(f) for f in common]
            # disagreement master vs slave 3D (rotation-free shape compare, per frame PA)
            dis=np.mean([pa(B["X"][j],A["X"][i]) for i,j in zip(ia,ib)]) if common else np.nan
            # global orientation agreement given known extrinsics: slave pose mapped to master frame
            Rms=A["R"]@B["R"].T
            orient=[]
            for i,j in zip(ia,ib):
                a=A["X"][i]-A["X"][i].mean(0); b=(B["X"][j]-B["X"][j].mean(0))@Rms.T
                orient.append(np.linalg.norm(a-b,axis=1).mean()*1000)
            za=A["R"][2]; zb=B["R"][2]; ang=np.degrees(np.arccos(np.clip(za@zb,-1,1)))
            out.append(dict(split=person,master=m,slave=s,n_common=len(common),
                coverage_slave=len(B["fid"]),slave_conf=np.mean([B["conf"].get(f,np.nan) for f in common]) if common else np.nan,
                master_conf=np.mean([A["conf"].get(f,np.nan) for f in common]) if common else np.nan,
                slave_bbox=np.mean([B["size"][f] for f in common if f in B["size"]]) if common else np.nan,
                slave_dist=float(np.linalg.norm(B["pos"])), view_angle=ang,
                disagree_pa=dis, disagree_ext=np.mean(orient) if orient else np.nan,
                slave_jitter=accel(B["X"]), master_jitter=accel(A["X"])))
    print(sp,"done",file=sys.stderr)
w=csv.DictWriter(open(sys.argv[1],"w"),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

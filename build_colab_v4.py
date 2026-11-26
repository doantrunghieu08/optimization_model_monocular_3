import json
from pathlib import Path

root = Path(__file__).resolve().parent
snapshots = {
    name: (root / name).read_text(encoding="utf-8")
    for name in ["compat.py", "train_v4.py", "fusion_pipeline/partwise.py", "fusion_pipeline/naive.py",
                 "fusion_pipeline/aotr_stability.py", "tests/test_train_v4.py",
                 "configs/pipeline.yml"]
}
cells = []


def md(s):
    cells.append({
        "cell_type": "markdown",
        "id": f"cell-{len(cells)}",
        "metadata": {},
        "source": s.splitlines(True),
    })


def code(s, form=False):
    meta = {"cellView": "form"} if form else {}
    cells.append({
        "cell_type": "code",
        "id": f"cell-{len(cells)}",
        "metadata": meta,
        "execution_count": None,
        "outputs": [],
        "source": s.splitlines(True),
    })


md("""# Train `partwise` — Sheet `v4 22/11/26` (Colab — Google Drive Mount)
Mở notebook trong Google Colab và bấm **Run all (Chạy tất cả)** hoặc chạy lần lượt từ trên xuống.
- **Ưu tiên đọc từ Drive đã mount:** ánh xạ File ID trên sheet sang `/content/drive`; file không tìm thấy trên mount có thể được tải qua API/gdown. Với dữ liệu trong **Được chia sẻ với tôi**, thêm lối tắt thư mục vào `MyDrive` trước khi chạy.
- Loss có phạt theo frame khi PA hoặc MPJPE thua naive. **Best Epoch** ưu tiên checkpoint thắng naive ở cả PA và MPJPE validation (12 khớp thân), rồi mới xét PA + λ·MPJPE. Nếu chưa có checkpoint đạt, vẫn lưu ứng viên tốt nhất và ghi `val_beats_naive: false`.
- Lưu **mọi epoch** trên **Google Drive**, có giới hạn epoch và dừng sớm khi không cải thiện; xuất `best_weights.json`, `best_partwise.yml`, `pipeline_best.yml`. Thắng validation chưa thay thế benchmark độc lập; không dùng test để ép train đến khi thắng.

**Lưu ý phạm vi:** GT chỉ dùng khi chuẩn bị dữ liệu giám sát (ghép track/thời gian và tính hàm mục tiêu), **0% GT lúc suy luận**. Kiểm tra bảng `alignment.json` ở Bước 4a trước khi dùng bộ tham số.
""")

code("""#@title 1. Cài đặt môi trường & clone repo
import json, os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/doantrunghieu08/optimization_model_monocular_3.git"  #@param {type:"string"}
BRANCH = "experiment/observability-aware-trust-region"  #@param {type:"string"}
REPO_DIR = Path("/content/optimization_model_monocular_3")

if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "checkout", "-f", "-B", BRANCH, f"origin/{BRANCH}"], cwd=REPO_DIR, check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "chumpy"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy<2.3", "scipy", "torch", "joblib", "pyyaml", "trimesh",
                "gspread", "google-api-python-client", "gdown", "matplotlib", "pandas"], check=True)
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def run_live(cmd):
    # ponytail: stream child stdout live into Colab cell so progress & tracebacks are never hidden
    lines = []
    with subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"Command failed (exit {rc}): {' '.join(cmd)}\\n" + "".join(lines[-40:]))
    return lines

print("Ready | repo:", REPO_DIR, "| branch:", BRANCH)
""", form=True)

code("#@title 2. Đồng bộ snapshot trainer & chạy self-test\nSNAPSHOTS = " + repr(snapshots) + """
for name, text in SNAPSHOTS.items():
    path = REPO_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
import importlib
if "train_v4" in sys.modules:
    importlib.reload(sys.modules["train_v4"])
run_live([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_train_v4.py", "-v"])
""", form=True)

code("""#@title 3. Mount Google Drive, đọc Google Sheet & cấu hình lưu checkpoint
import logging, re
from google.colab import auth, drive
import google.auth, gspread
from train_v4 import validate_rows

logging.getLogger("google_auth_httplib2").setLevel(logging.ERROR)

DRIVE_MOUNT = "/content/drive"  #@param {type:"string"}
DRIVE_DATA_DIR = ""  #@param {type:"string"}
SPREADSHEET_ID = "11OipPLik-aXw0GmrTfEwg6wsAQYEmvGEV8VRlyry4Rg"  #@param {type:"string"}
TAB = "v4 22/11/26"  #@param {type:"string"}
OUTPUT_DIR = "/content/drive/MyDrive/partwise_v4/run_004"  #@param {type:"string"}
EPOCHS = 50  #@param {type:"integer"}
LEARNING_RATE = 0.01  #@param {type:"number"}
RESUME = True  #@param {type:"boolean"}
SKIP_UNALIGNED_ACTIONS = True  #@param {type:"boolean"}

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]
auth.authenticate_user()
drive.mount(DRIVE_MOUNT, force_remount=False)
credentials, _ = google.auth.default(scopes=SCOPES)
client = gspread.authorize(credentials)

OUTPUT = Path(OUTPUT_DIR.strip())
CACHE = OUTPUT / "joints_cache"
OUTPUT.mkdir(parents=True, exist_ok=True)

sh = client.open_by_key(SPREADSHEET_ID.strip())
try:
    sheet = sh.worksheet(TAB.strip())
except gspread.WorksheetNotFound:
    sheet = sh.worksheet(TAB.strip().replace("-", " "))
values = sheet.get(f"A1:O{sheet.row_count}")
headers = [h.strip() for h in values[0]]
rows = []
for raw_row in values[1:]:
    if not any(str(c).strip() for c in raw_row):
        continue
    d = dict(zip(headers, [str(c).strip() for c in raw_row] + [""] * (len(headers) - len(raw_row))))
    d["action"] = re.sub(r"_cam\\d+$", "", d.get("action", ""))
    rows.append(d)

groups = validate_rows(rows)
print(f"Loaded tab '{sheet.title}' | Rows: {len(rows)} | Train sequences: {list(groups.values()).count('train')} | Val sequences: {list(groups.values()).count('val')}", flush=True)

MANIFEST = OUTPUT / "manifest_input.json"
if (OUTPUT / "last.json").exists() and MANIFEST.exists():
    old_clean = [{k: v for k, v in r.items() if k not in ("GT_path", "PKL_path")}
                 for r in json.loads(MANIFEST.read_text(encoding="utf-8"))]
    new_clean = [{k: v for k, v in r.items() if k not in ("GT_path", "PKL_path")} for r in rows]
    if old_clean != new_clean:
        raise ValueError("Dữ liệu Sheet đã thay đổi so với checkpoint cũ; hãy đổi OUTPUT_DIR sang thư mục mới.")
MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("Output Drive folder:", OUTPUT)
""", form=True)

code("""#@title 4a. Chuẩn bị dữ liệu & kiểm tra căn chỉnh (Alignment)
import pandas as pd

prepare_cmd = [sys.executable, "-u", "train_v4.py", "--manifest", str(MANIFEST),
               "--drive-mount", DRIVE_MOUNT, "--data-dir", DRIVE_DATA_DIR.strip(),
               "--cache", str(CACHE), "--output", str(OUTPUT), "--prepare-only"]
if SKIP_UNALIGNED_ACTIONS:
    prepare_cmd.append("--skip-unaligned-actions")
run_live(prepare_cmd)

alignment = pd.DataFrame(json.loads((OUTPUT / "alignment.json").read_text(encoding="utf-8")))
print("Tổng hợp trạng thái camera/sequence:", alignment["status"].value_counts().to_dict())
flagged = alignment[
    alignment["status"].isin(["rejected", "skipped_action"])
    | (alignment["coverage"].fillna(1.0) < 0.8)
    | (alignment["pa_mm"].fillna(0.0) > 80.0)
    | (alignment.get("switches", pd.Series(0, index=alignment.index)).fillna(0) > 0)
]
if not flagged.empty:
    print("Các camera/sequence cần xem lại (bị loại, phủ <80% GT, PA >80mm, hoặc chuyển track):")
    display(flagged)
else:
    print("Không có cờ cảnh báo tự động; vẫn cần kiểm tra alignment và danh tính track.")
""")

code("""#@title 4b. Chạy Train & lưu checkpoint từng epoch trên Google Drive
import hashlib, shutil, torch, yaml
from train_v4 import fit_signature, training_config

# Tính đúng chữ ký cấu hình của train_v4.py để kiểm tra tương thích trước khi resume
cfg_pw = training_config(yaml.safe_load((REPO_DIR / "configs/pipeline.yml").read_text(encoding="utf-8")))
stable_rows = [{k: v for k, v in r.items() if k not in ("GT_path", "PKL_path")}
               for r in json.loads(MANIFEST.read_text(encoding="utf-8"))]
cur_identity = hashlib.sha256(
    json.dumps(stable_rows, sort_keys=True, ensure_ascii=False).encode()
    + (REPO_DIR / "models/J_regressor_body25_extended.npy").read_bytes()
    + (REPO_DIR / "train_v4.py").read_bytes()
    + (REPO_DIR / "fusion_pipeline/partwise.py").read_bytes()
    + (REPO_DIR / "fusion_pipeline/naive.py").read_bytes()
    + (REPO_DIR / "fusion_pipeline/aotr_stability.py").read_bytes()
    + (REPO_DIR / "configs/keypoints3D_map.yml").read_bytes()
    + json.dumps(cfg_pw, sort_keys=True).encode()
    + str(bool(SKIP_UNALIGNED_ACTIONS)).encode()
).hexdigest()
cur_sig = fit_signature(cur_identity, cfg_pw, float(LEARNING_RATE))


def _probe_ckpt(folder):
    lp = folder / "last.json"
    if not lp.exists():
        return None, None, None
    info = json.loads(lp.read_text(encoding="utf-8"))
    pt = folder / info.get("checkpoint", "")
    if not pt.exists():
        return None, None, None
    sig = info.get("signature") or torch.load(pt, map_location="cpu", weights_only=False).get("signature")
    return pt, sig, info.get("epoch")


pt, sig, ep = _probe_ckpt(OUTPUT)
has_pts = any(OUTPUT.glob("epoch_*.pt"))
if not has_pts:
    ACTIVE_OUT, resume_pt = OUTPUT, None
elif RESUME and pt is not None and sig == cur_sig:
    ACTIVE_OUT, resume_pt = OUTPUT, pt
else:
    # Khác lr/config hoặc tắt RESUME: tự tách sang thư mục con và tái sử dụng dữ liệu đã căn chỉnh
    sub = OUTPUT / f"lr_{float(LEARNING_RATE):g}"
    pt2, sig2, ep2 = _probe_ckpt(sub)
    if RESUME and pt2 is not None and sig2 == cur_sig:
        ACTIVE_OUT, resume_pt, ep = sub, pt2, ep2
    elif not any(sub.glob("epoch_*.pt")):
        ACTIVE_OUT, resume_pt = sub, None
    else:
        k = 2
        while any((OUTPUT / f"lr_{float(LEARNING_RATE):g}_run{k}").glob("epoch_*.pt")):
            k += 1
        ACTIVE_OUT, resume_pt = OUTPUT / f"lr_{float(LEARNING_RATE):g}_run{k}", None
    ACTIVE_OUT.mkdir(parents=True, exist_ok=True)
    prep_src = OUTPUT / f"prepared_{cur_identity}.joblib"
    prep_dst = ACTIVE_OUT / prep_src.name
    if prep_src.exists() and not prep_dst.exists():
        shutil.copy2(prep_src, prep_dst)
    for aux in ("alignment.json", "manifest.json", "protocol.json"):
        if (OUTPUT / aux).exists() and not (ACTIVE_OUT / aux).exists():
            shutil.copy2(OUTPUT / aux, ACTIVE_OUT / aux)
    print(f"Khác lr/cấu hình so với checkpoint gốc -> lưu tại: {ACTIVE_OUT} (tái sử dụng dữ liệu đã căn chỉnh)")

train_cmd = [sys.executable, "-u", "train_v4.py", "--manifest", str(MANIFEST),
             "--drive-mount", DRIVE_MOUNT, "--cache", str(CACHE),
             "--output", str(ACTIVE_OUT), "--epochs", str(EPOCHS), "--lr", str(LEARNING_RATE)]
if SKIP_UNALIGNED_ACTIONS:
    train_cmd.append("--skip-unaligned-actions")
if resume_pt is not None:
    print(f"Resume từ checkpoint: {resume_pt} (epoch {ep})")
    train_cmd += ["--resume", str(resume_pt)]

run_live(train_cmd)
""")

code("""#@title 5. Chọn Best Epoch, xem bộ tham số tốt nhất & vẽ đường học
import matplotlib.pyplot as plt
import pandas as pd

RUN_OUT = globals().get("ACTIVE_OUT", OUTPUT)
history = json.loads((RUN_OUT / "history.json").read_text(encoding="utf-8"))
best = json.loads((RUN_OUT / "best_weights.json").read_text(encoding="utf-8"))
saved_pts = sorted(RUN_OUT.glob("epoch_*.pt"))

print(f"Đã lưu {len(saved_pts)} epoch checkpoints tại: {RUN_OUT}")
print(f"★ Best Epoch          : {best['epoch']} / {len(history)}")
print(f"  Val MPJPE sequence  : {best['val_sequence_mpjpe_mm']:.2f} mm")
print(f"  Val objective       : {best['val_objective_mm']:.2f} mm")
print(f"  Val PA-MPJPE (Best) : {best['val_mpjpe_mm']:.2f} mm")
if "equal_val_mpjpe_mm" in best:
    print(f"  Val PA-MPJPE (50-50): {best['equal_val_mpjpe_mm']:.2f} mm")
print(f"  Val PA-MPJPE (Mono) : {best['baseline_val_mpjpe_mm']:.2f} mm")
print(f"  Naive PA / MPJPE    : {best['naive_val_mpjpe_mm']:.2f} / {best['naive_val_sequence_mpjpe_mm']:.2f} mm")
print(f"  Gain vs naive PA/MP : {best['val_pa_gain_vs_naive_mm']:.2f} / {best['val_mpjpe_gain_vs_naive_mm']:.2f} mm (dương = tốt hơn)")
print(f"  Thắng cả PA và MP   : {best['val_beats_naive']}")
if not best["val_beats_naive"]:
    print("Chưa có checkpoint thắng naive ở cả hai chỉ số validation; file best là ứng viên để kiểm tra.")
print(f"  File tham số tốt nhất: {RUN_OUT / 'best_weights.json'}")
print(f"  File patch YAML     : {RUN_OUT / 'best_partwise.yml'}")

param_df = pd.DataFrame({
    "feature": best["feature_names"],
    "weight": [round(w, 6) for w in best["weights"]],
    "train_std": [round(s, 6) for s in best["std"]],
})
display(param_df)

epochs_x = [h["epoch"] for h in history]
plt.figure(figsize=(7, 4))
plt.plot(epochs_x, [h["train_mpjpe_mm"] for h in history], label="Train PA-MPJPE")
plt.plot(epochs_x, [h["val_mpjpe_mm"] for h in history], label="Val PA-MPJPE (partwise learned)")
if "equal_val_mpjpe_mm" in history[0]:
    plt.axhline(history[0]["equal_val_mpjpe_mm"], color="gray", linestyle="--", label="Val 50-50 (w=0.5)")
plt.axhline(history[0]["depth_val_mpjpe_mm"], color="green", linestyle=":", label="Val depth_logit")
plt.axhline(history[0]["naive_val_mpjpe_mm"], color="orange", linestyle="--", label="Val naive")
plt.scatter([best["epoch"]], [best["val_mpjpe_mm"]], color="red", zorder=5, label=f"Selected checkpoint (epoch {best['epoch']})")
plt.xlabel("Epoch")
plt.ylabel("PA-MPJPE (mm)")
plt.title("Quá trình huấn luyện bộ trọng số partwise")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.figure(figsize=(7, 4))
plt.plot(epochs_x, [h["val_sequence_mpjpe_mm"] for h in history], label="Val MPJPE learned")
plt.axhline(history[0]["depth_val_sequence_mpjpe_mm"], label="Val MPJPE depth_logit", linestyle="--")
plt.axhline(history[0]["naive_val_sequence_mpjpe_mm"], label="Val MPJPE naive", linestyle="-.")
plt.axhline(history[0]["baseline_val_sequence_mpjpe_mm"], label="Val MPJPE mono", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Sequence MPJPE (mm), 12 body joints")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
""")

code("""#@title 6. Ghép bộ tham số tốt nhất vào pipeline_best.yml để suy luận / benchmark
import yaml

RUN_OUT = globals().get("ACTIVE_OUT", OUTPUT)
base = yaml.safe_load((REPO_DIR / "configs/pipeline.yml").read_text(encoding="utf-8"))
patch = yaml.safe_load((RUN_OUT / "best_partwise.yml").read_text(encoding="utf-8"))
base["fusion"]["method"] = "partwise"
base["fusion"]["partwise"] = patch["fusion"]["partwise"]
out_cfg = RUN_OUT / "pipeline_best.yml"
out_cfg.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("Cấu hình thử nghiệm: cần benchmark coco17 trước khi dùng chính thức.")
print("Đã xuất cấu hình đầy đủ (nhúng sẵn trọng số, dùng được ngay cả trên Colab lẫn máy local):")
print(" ->", out_cfg)
print("\\nĐoạn cấu hình weight_model tốt nhất:")
print(yaml.safe_dump({"weight_model": patch["fusion"]["partwise"]["weight_model"]}, sort_keys=False))
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "colab_train_v4.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = root / "colab_train_v4.ipynb"
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
for cell in cells:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), str(path), "exec")
print("Created notebook; all code cells compile:", path)

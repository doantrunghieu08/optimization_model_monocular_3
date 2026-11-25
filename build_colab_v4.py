import json
from pathlib import Path

root = Path(__file__).resolve().parent
snapshots = {
    name: (root / name).read_text(encoding="utf-8")
    for name in ["compat.py", "train_v4.py", "fusion_pipeline/partwise.py", "tests/test_train_v4.py"]
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
- **Không tải file (`0% download` / `0% gdown`):** mount trực tiếp Google Drive tại `/content/drive` và ánh xạ từng File ID trên sheet `v4 22/11/26` sang đường dẫn mount nội bộ, tránh hoàn toàn việc Google Drive chặn quota tải. *(Nếu thư mục dữ liệu gốc thuộc mục **"Được chia sẻ với tôi"**, hãy bấm **"Thêm lối tắt vào Drive" (Add shortcut to Drive)** vào `MyDrive` trước khi chạy.)*
- Lưu **mọi epoch** (`epoch_0001.pt`, `epoch_0001_weights.json`, ...) trên **Google Drive**, tự động chọn **Best Epoch** theo validation PA-MPJPE và xuất bộ tham số tốt nhất ra `best_weights.json`, `best_partwise.yml`, `pipeline_best.yml`.

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
                "gspread", "google-api-python-client", "matplotlib", "pandas"], check=True)
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
run_live([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_train_v4.py", "-v"])
""", form=True)

code("""#@title 3. Mount Google Drive, đọc Google Sheet & cấu hình lưu checkpoint
import re
from google.colab import auth, drive
import google.auth, gspread

DRIVE_MOUNT = "/content/drive"  #@param {type:"string"}
SPREADSHEET_ID = "11OipPLik-aXw0GmrTfEwg6wsAQYEmvGEV8VRlyry4Rg"  #@param {type:"string"}
TAB = "v4 22/11/26"  #@param {type:"string"}
OUTPUT_DIR = "/content/drive/MyDrive/partwise_v4/run_002"  #@param {type:"string"}
EPOCHS = 50  #@param {type:"integer"}
LEARNING_RATE = 0.01  #@param {type:"number"}
RESUME = True  #@param {type:"boolean"}
SKIP_UNALIGNED_ACTIONS = True  #@param {type:"boolean"}

auth.authenticate_user()
drive.mount(DRIVE_MOUNT, force_remount=False)
credentials, _ = google.auth.default()
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

from train_v4 import validate_rows
groups = validate_rows(rows)
print(f"Loaded tab '{sheet.title}' | Rows: {len(rows)} | Train sequences: {list(groups.values()).count('train')} | Val sequences: {list(groups.values()).count('val')}")

MANIFEST = OUTPUT / "manifest_input.json"
if (OUTPUT / "last.json").exists() and MANIFEST.exists():
    if json.loads(MANIFEST.read_text(encoding="utf-8")) != rows:
        raise ValueError("Dữ liệu Sheet đã thay đổi so với checkpoint cũ; hãy đổi OUTPUT_DIR sang thư mục mới.")
MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("Output Drive folder:", OUTPUT)
""", form=True)

code("""#@title 4a. Chuẩn bị dữ liệu & kiểm tra căn chỉnh (Alignment)
import pandas as pd

prepare_cmd = [sys.executable, "-u", "train_v4.py", "--manifest", str(MANIFEST),
               "--drive-mount", DRIVE_MOUNT, "--cache", str(CACHE),
               "--output", str(OUTPUT), "--prepare-only"]
if SKIP_UNALIGNED_ACTIONS:
    prepare_cmd.append("--skip-unaligned-actions")
run_live(prepare_cmd)

alignment = pd.DataFrame(json.loads((OUTPUT / "alignment.json").read_text(encoding="utf-8")))
print("Tổng hợp trạng thái camera/sequence:", alignment["status"].value_counts().to_dict())
flagged = alignment[
    alignment["status"].isin(["rejected", "skipped_action"])
    | (alignment["coverage"].fillna(1.0) < 0.8)
    | (alignment["pa_mm"].fillna(0.0) > 80.0)
]
if not flagged.empty:
    print("Các camera/sequence cần lưu ý (bị loại, phủ <80% GT, hoặc PA >80mm):")
    display(flagged)
else:
    print("Tất cả camera đều khớp tốt với GT (coverage >= 80%, PA <= 80mm).")
""")

code("""#@title 4b. Chạy Train & lưu checkpoint từng epoch trên Google Drive
train_cmd = [sys.executable, "-u", "train_v4.py", "--manifest", str(MANIFEST),
             "--drive-mount", DRIVE_MOUNT, "--cache", str(CACHE),
             "--output", str(OUTPUT), "--epochs", str(EPOCHS), "--lr", str(LEARNING_RATE)]
if SKIP_UNALIGNED_ACTIONS:
    train_cmd.append("--skip-unaligned-actions")

last_path = OUTPUT / "last.json"
if RESUME and last_path.exists():
    last = json.loads(last_path.read_text(encoding="utf-8"))
    ckpt_file = OUTPUT / last["checkpoint"]
    if ckpt_file.exists() and "signature" in last:
        print(f"Resume từ checkpoint: {ckpt_file.name} (epoch {last.get('epoch')})")
        train_cmd += ["--resume", str(ckpt_file)]
    else:
        raise RuntimeError(f"Thư mục {OUTPUT} chứa checkpoint phiên bản cũ không tương thích; hãy đổi OUTPUT_DIR ở Bước 3.")

run_live(train_cmd)
""")

code("""#@title 5. Chọn Best Epoch, xem bộ tham số tốt nhất & vẽ đường học
import matplotlib.pyplot as plt
import pandas as pd

history = json.loads((OUTPUT / "history.json").read_text(encoding="utf-8"))
best = json.loads((OUTPUT / "best_weights.json").read_text(encoding="utf-8"))
saved_pts = sorted(OUTPUT.glob("epoch_*.pt"))

print(f"Đã lưu {len(saved_pts)} epoch checkpoints tại: {OUTPUT}")
print(f"★ Best Epoch          : {best['epoch']} / {len(history)}")
print(f"  Val PA-MPJPE (Best) : {best['val_mpjpe_mm']:.2f} mm")
if "equal_val_mpjpe_mm" in best:
    print(f"  Val PA-MPJPE (50-50): {best['equal_val_mpjpe_mm']:.2f} mm")
print(f"  Val PA-MPJPE (Mono) : {best['baseline_val_mpjpe_mm']:.2f} mm")
print(f"  File tham số tốt nhất: {OUTPUT / 'best_weights.json'}")
print(f"  File patch YAML     : {OUTPUT / 'best_partwise.yml'}")

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
plt.scatter([best["epoch"]], [best["val_mpjpe_mm"]], color="red", zorder=5, label=f"Best Epoch ({best['epoch']})")
plt.xlabel("Epoch")
plt.ylabel("PA-MPJPE (mm)")
plt.title("Quá trình huấn luyện bộ trọng số partwise")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
""")

code("""#@title 6. Ghép bộ tham số tốt nhất vào pipeline_best.yml để suy luận / benchmark
import yaml

base = yaml.safe_load((REPO_DIR / "configs/pipeline.yml").read_text(encoding="utf-8"))
patch = yaml.safe_load((OUTPUT / "best_partwise.yml").read_text(encoding="utf-8"))
base["fusion"]["method"] = "partwise"
base["fusion"].setdefault("partwise", {})["weight_model"] = patch["fusion"]["partwise"]["weight_model"]
out_cfg = OUTPUT / "pipeline_best.yml"
out_cfg.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")
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

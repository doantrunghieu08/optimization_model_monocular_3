import json
from pathlib import Path

root = Path(__file__).resolve().parent
snapshots = {name: (root/name).read_text(encoding="utf-8") for name in
             ["train_v4.py", "fusion_pipeline/partwise.py", "tests/test_train_v4.py"]}
cells = []
def md(s):
    cells.append({"cell_type":"markdown","id":f"cell-{len(cells)}","metadata":{},"source":s.splitlines(True)})
def code(s):
    cells.append({"cell_type":"code","id":f"cell-{len(cells)}","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(True)})

md("""# Train partwise — v4 22/11/26
Mở notebook trong Google Colab, chạy các ô từ trên xuống. Có thể dùng CPU.
Notebook đọc trực tiếp sheet, tải PKL/GT bằng tài khoản Google của bạn, học trọng số linear cho mô hình partwise hiện có.
Checkpoint từng epoch và bộ tham số tốt nhất lưu trên **Google Drive**.

**Phạm vi:** GT được dùng để chọn track và ghép thời gian khi chuẩn bị dữ liệu có giám sát.
Đây không phải benchmark đồng bộ/nhận dạng độc lập với GT.
Validation là 12 body joints, root-centered MPJPE với một rotation GT cho mỗi đoạn camera;
điểm này không tương đương MPJPE camera có extrinsics.
Không có test độc lập. Kiểm tra alignment.json trước khi tin kết quả.
""")
code("""# 1. Môi trường và phiên bản code
import os, sys, subprocess, json
from pathlib import Path
REPO_URL = "https://github.com/doantrunghieu08/optimization_model_monocular_3.git"
COMMIT = "817dbb0c01ce0354d4a2c98a7fc4b50fc9720bb0"
REPO_DIR = Path("/content/optimization_model_monocular_3")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=REPO_DIR, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy<2.3", "scipy", "torch", "joblib", "pyyaml", "trimesh",
                "gspread", "google-api-python-client", "matplotlib"], check=True)
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
""")
md("Trainer và module partwise được nhúng ngay trong notebook, nên không cần push các file đang làm dở lên GitHub.")
code("# 2. Cài snapshot trainer đã kiểm tra\nSNAPSHOTS = " + repr(snapshots) + """
for name, text in SNAPSHOTS.items():
    path = REPO_DIR/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
test_result = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests",
     "-p", "test_train_v4.py", "-v"], cwd=REPO_DIR,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    encoding="utf-8", errors="replace")
print(test_result.stdout, flush=True)
if test_result.returncode:
    raise RuntimeError("Trainer tests failed:\\n" + test_result.stdout)
""")
code("""# 3. Đăng nhập và nơi lưu checkpoint
from google.colab import auth, drive
auth.authenticate_user()
drive.mount("/content/drive")
import google.auth, gspread
credentials, _ = google.auth.default()
client = gspread.authorize(credentials)
SPREADSHEET_ID = "11OipPLik-aXw0GmrTfEwg6wsAQYEmvGEV8VRlyry4Rg"
TAB = "v4 22/11/26"
OUTPUT = Path("/content/drive/MyDrive/partwise_v4/run_001")
CACHE = Path("/content/v4_cache")
EPOCHS = 50
LEARNING_RATE = 0.01
RESUME = True
SKIP_UNALIGNED_ACTIONS = True  # Report and exclude sequences without valid camera pairs.
OUTPUT.mkdir(parents=True, exist_ok=True)

sheet = client.open_by_key(SPREADSHEET_ID).worksheet(TAB)
values = sheet.get(f"A1:O{sheet.row_count}")
headers = values[0]
rows = [dict(zip(headers, row + [""] * (len(headers)-len(row))))
        for row in values[1:] if any(row)]
from train_v4 import validate_rows
groups = validate_rows(rows)
print("Rows:", len(rows), "Train actions:", list(groups.values()).count("train"),
      "Val actions:", list(groups.values()).count("val"))
MANIFEST = OUTPUT/"manifest_input.json"
text = json.dumps(rows, ensure_ascii=False, indent=2)
# Keep the original input snapshot on resume; don't silently replace it.
if (OUTPUT/"last.json").exists() and MANIFEST.exists():
    assert json.loads(MANIFEST.read_text()) == rows, "Sheet changed: use a new OUTPUT folder"
MANIFEST.write_text(text, encoding="utf-8")
""")
code("""# 4a. Prepare data and review alignment before training
prepare_command = [sys.executable, "-u", "train_v4.py", "--manifest", str(MANIFEST),
                   "--cache", str(CACHE), "--output", str(OUTPUT), "--prepare-only"]
if SKIP_UNALIGNED_ACTIONS:
    prepare_command.append("--skip-unaligned-actions")
prepare_log = []
with subprocess.Popen(prepare_command, cwd=REPO_DIR, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                      errors="replace", bufsize=1) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
        prepare_log.append(line)
    returncode = process.wait()
if returncode:
    raise RuntimeError("Data preparation failed:\\n" + "".join(prepare_log))
import pandas as pd
alignment = pd.DataFrame(json.loads((OUTPUT/"alignment.json").read_text()))
display(alignment)
print("GT-assisted matching: review identity, timeline and coverage before training.")
display(alignment[alignment["status"].isin(["rejected", "skipped_action"]) |
                  (alignment["coverage"] < 0.8) | (alignment["pa_mm"] > 80)])
""")
code("""# 4. Train / tiếp tục từ epoch cuối
command = [sys.executable, "-u", "train_v4.py", "--manifest", str(MANIFEST),
           "--cache", str(CACHE), "--output", str(OUTPUT),
           "--epochs", str(EPOCHS), "--lr", str(LEARNING_RATE)]
if SKIP_UNALIGNED_ACTIONS:
    command.append("--skip-unaligned-actions")
if RESUME and (OUTPUT/"last.json").exists():
    last = json.loads((OUTPUT/"last.json").read_text())
    command += ["--resume", str(OUTPUT/last["checkpoint"])]
print("Checkpoint output:", OUTPUT)
subprocess.run(command, cwd=REPO_DIR, check=True)
""")
code("""# 5. Xem best epoch, đường học và chất lượng alignment
import matplotlib.pyplot as plt
history = json.loads((OUTPUT/"history.json").read_text())
best = json.loads((OUTPUT/"best_weights.json").read_text())
print("Best epoch:", best["epoch"])
print("Validation MPJPE:", best["val_mpjpe_mm"], "mm")
print("Master baseline:", best["baseline_val_mpjpe_mm"], "mm")
print("Improved vs master:", best["val_mpjpe_mm"] < best["baseline_val_mpjpe_mm"])
print("Weights:", OUTPUT/"best_weights.json")
print("Inference config patch:", OUTPUT/"best_partwise.yml")
plt.plot([h["epoch"] for h in history], [h["train_mpjpe_mm"] for h in history], label="train")
plt.plot([h["epoch"] for h in history], [h["val_mpjpe_mm"] for h in history], label="val")
plt.xlabel("Epoch"); plt.ylabel("MPJPE (mm)"); plt.legend(); plt.show()

import pandas as pd
alignment = pd.DataFrame(json.loads((OUTPUT/"alignment.json").read_text()))
display(alignment)
display(alignment[alignment["status"].isin(["rejected", "skipped_action"]) |
                  (alignment["coverage"] < 0.5) | (alignment["pa_mm"] > 80)])
""")
md("""## Các file kết quả
- epoch_0001.pt, epoch_0002.pt, ...: trọng số, optimizer, trạng thái RNG để tiếp tục train.
- epoch_0001_weights.json, ...: tham số từng epoch dùng trực tiếp cho partwise.
- best_weights.json: epoch có validation MPJPE thấp nhất, kèm epoch và baseline.
- best_partwise.yml: cấu hình inference khớp với train (visibility không có trong sheet).
- history.json, manifest.json, protocol.json, alignment.json: kết quả và nguồn dữ liệu để kiểm tra.

Đường dẫn trong best_partwise.yml trỏ đến Drive của Colab. Nếu tải kết quả về máy, sửa đường dẫn weight_model.path.
Ghép phần fusion trong file patch vào cấu hình pipeline hiện có; giữ các phần inputs/paths khác.
Notebook không train SMPL/WHAM backbone và không thay đổi sheet khi chạy.
""")
code("""# 6. Ghép cấu hình tốt nhất vào pipeline để dùng inference
import copy, yaml
base = yaml.safe_load((REPO_DIR/"configs/pipeline.yml").read_text())
patch = yaml.safe_load((OUTPUT/"best_partwise.yml").read_text())
base["fusion"]["method"] = patch["fusion"]["method"]
base["fusion"]["partwise"] = patch["fusion"]["partwise"]
(OUTPUT/"pipeline_best.yml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
print("Đã xuất:", OUTPUT/"pipeline_best.yml")
print("Điền inputs/paths đúng dữ liệu inference trước khi chạy pipeline.")
""")
nb = {"cells":cells,"metadata":{"colab":{"name":"colab_train_v4.ipynb"},
      "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
path = root/"colab_train_v4.ipynb"
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
for cell in cells:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), str(path), "exec")
print("Created notebook; all code cells compile:", path)

import os
from pathlib import Path

files = [
    "visualization_pipeline/executor.py",
    "preprocess_pipeline/executor.py",
    "pose_pipeline/executor.py",
    "learnable_pipeline/executor.py",
    "evaluation_pipeline/executor.py",
    "fusion_pipeline/executor.py"
]

for f in files:
    path = Path(f)
    if not path.exists(): continue
    text = path.read_text(encoding="utf-8")
    new_text = text.replace(".unlink()", ".unlink(missing_ok=True)")
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"Fixed {f}")

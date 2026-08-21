import os
from pathlib import Path

def rename_pkls():
    base_dir = Path(r"d:\optimization_model_monocular_3\input\Seq1\imageSequence\Segments")
    for pkl_file in base_dir.rglob("*.pkl"):
        parent_name = pkl_file.parent.name
        if parent_name.startswith("wham_output_"):
            new_name = parent_name.replace("wham_output_", "") + ".pkl"
            if pkl_file.name != new_name:
                new_path = pkl_file.parent / new_name
                print(f"Renaming {pkl_file} to {new_path}")
                pkl_file.rename(new_path)

if __name__ == "__main__":
    rename_pkls()

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Set environment variables for best config
# os.environ["NOTEBOOK_NAME"] = "ablation_hieuDT_belief_fusion_H260912RayCasting_optical_global_kinematic_huber_alpha1E_2_beta85E_2.ipynb"
os.environ["NOTEBOOK_NAME"] = "ablation_hieuDT_belief_fusion_H260912RayCasting_optical_global_kinematic_huber_alpha1E_2_beta85E_2.ipynb"
os.environ["ALPHA"] = "0.01"
os.environ["BETA"] = "0.85"
os.environ["LOCAL_METHOD"] = "optical_aware_belief"
os.environ["GLOBAL"] = "true"
os.environ["KINEMATIC_CONSTRAINTS"] = "true"
os.environ["LOSS_TYPE"] = "huber"

from brute_force_runner import run_brute_force

if __name__ == "__main__":
    run_brute_force()

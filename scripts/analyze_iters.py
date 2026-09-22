import csv
import os
import numpy as np

base = r'd:\optimization_model_monocular_3\output\max_iter_cmp'


def read_metric_csv(fpath):
    rows = []
    with open(fpath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row and row[0] != 'AVERAGE':
                vals = []
                for v in row[2:]:
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
                if vals:
                    rows.append(np.mean(vals))
    return np.mean(rows) if rows else None


print("iter | MPJPE_cam1 | MPJPE_cam2 | PA_MPJPE_cam1 | PA_MPJPE_cam2")
for itr in [20, 40, 60, 80, 100, 150]:
    d = os.path.join(base, f"iter_{itr}", "evaluation")
    if not os.path.exists(d):
        print(f"{itr:4d} | no eval dir")
        continue
    results = {}
    for m in ["MPJPE_cam1", "MPJPE_cam2", "PA-MPJPE_cam1", "PA-MPJPE_cam2"]:
        fp = os.path.join(d, m + ".csv")
        results[m] = read_metric_csv(fp) if os.path.exists(fp) else None
    print(
        f"{itr:4d} | "
        f"{results['MPJPE_cam1']:.2f}       | "
        f"{results['MPJPE_cam2']:.2f}       | "
        f"{results['PA-MPJPE_cam1']:.2f}        | "
        f"{results['PA-MPJPE_cam2']:.2f}"
    )

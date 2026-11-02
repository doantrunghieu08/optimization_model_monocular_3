import csv

with open('output/brute_force_report.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

cam_baselines = {}
for r in rows:
    c = r['Cam Master']
    try:
        b = float(r['pa-mpjpe.all_baseline'])
        cam_baselines.setdefault(c, []).append(b)
    except (ValueError, TypeError):
        pass

for c, vals in sorted(cam_baselines.items()):
    import numpy as np
    print(f"{c}: mean baseline PA-MPJPE = {np.mean(vals):.2f} mm (min={min(vals):.2f}, max={max(vals):.2f})")


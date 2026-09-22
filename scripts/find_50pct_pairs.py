"""
Analyze evaluation CSV files to find pairs where error differs by more than 50%.
"Pair" here means comparing metrics across the pipeline stages: posed vs fused vs only_learnable.
Also comparing cam1 vs cam2 per joint.
"""
import csv
import os
import sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

EVAL_DIR = Path(r'd:\optimization_model_monocular_3\output\evaluation_results')
THRESHOLD_PCT = 50.0  # percent


def load_csv(fpath):
    rows = []
    with open(fpath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_val(s):
    try:
        return float(s)
    except Exception:
        return None


# ── 1. Load all CSVs ──────────────────────────────────────────────────────────
files = {
    'MPJPE_cam1':    load_csv(EVAL_DIR / 'MPJPE_cam1.csv'),
    'MPJPE_cam2':    load_csv(EVAL_DIR / 'MPJPE_cam2.csv'),
    'PA-MPJPE_cam1': load_csv(EVAL_DIR / 'PA-MPJPE_cam1.csv'),
    'PA-MPJPE_cam2': load_csv(EVAL_DIR / 'PA-MPJPE_cam2.csv'),
}

# Joint columns for each stage
JOINTS = [
    'priority1', 'priority2', 'neck',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip',  'left_knee',  'left_ankle',
]
STAGES = ['posed', 'fused', 'only_learnable']


def col(stage, joint):
    return f'{stage}_{joint}_mm'


# ── 2. For each metric × file, find stage-pairs with >50% relative difference ─
print("=" * 110)
print("PAIRS WITH ERROR DIFFERENCE > 50%  (relative to the smaller value)")
print("=" * 110)

all_findings = []

for metric_name, rows in files.items():
    data_rows = [r for r in rows if r.get('Frame', '').strip().upper() != 'AVERAGE']
    avg_rows  = [r for r in rows if r.get('Frame', '').strip().upper() == 'AVERAGE']

    # ── 2a. Stage-vs-stage comparison on AVERAGE row ─────────────────────────
    for avg_row in avg_rows:
        seg = avg_row.get('Frame', '?')
        for joint in JOINTS:
            vals = {}
            for stage in STAGES:
                c = col(stage, joint)
                v = parse_val(avg_row.get(c))
                if v is not None:
                    vals[stage] = v

            stage_list = list(vals.items())
            for i in range(len(stage_list)):
                for j in range(i + 1, len(stage_list)):
                    s1, v1 = stage_list[i]
                    s2, v2 = stage_list[j]
                    if min(v1, v2) < 1e-9:
                        continue
                    rel_diff = abs(v1 - v2) / min(v1, v2) * 100
                    if rel_diff >= THRESHOLD_PCT:
                        all_findings.append({
                            'metric': metric_name,
                            'level': 'AVERAGE',
                            'frame': seg,
                            'joint': joint,
                            'A': s1, 'A_val': v1,
                            'B': s2, 'B_val': v2,
                            'rel_diff_pct': rel_diff,
                        })

    # ── 2b. Per-frame scan ────────────────────────────────────────────────────
    for row in data_rows:
        frame = row.get('Frame', '?')
        for joint in JOINTS:
            vals = {}
            for stage in STAGES:
                c = col(stage, joint)
                v = parse_val(row.get(c))
                if v is not None:
                    vals[stage] = v

            stage_list = list(vals.items())
            for i in range(len(stage_list)):
                for j in range(i + 1, len(stage_list)):
                    s1, v1 = stage_list[i]
                    s2, v2 = stage_list[j]
                    if min(v1, v2) < 1e-9:
                        continue
                    rel_diff = abs(v1 - v2) / min(v1, v2) * 100
                    if rel_diff >= THRESHOLD_PCT:
                        all_findings.append({
                            'metric': metric_name,
                            'level': 'frame',
                            'frame': frame,
                            'joint': joint,
                            'A': s1, 'A_val': v1,
                            'B': s2, 'B_val': v2,
                            'rel_diff_pct': rel_diff,
                        })

# ── 3. Summary by (metric, joint, stage_pair) ─────────────────────────────────
from collections import defaultdict

frame_findings = [f for f in all_findings if f['level'] == 'frame']
avg_findings   = [f for f in all_findings if f['level'] == 'AVERAGE']

print(f"\nTotal per-frame violations: {len(frame_findings)}")
print(f"Average-row violations:     {len(avg_findings)}\n")

# Group by joint + stage pair
group = defaultdict(list)
for f in frame_findings:
    key = (f['metric'], f['joint'], f['A'] + ' vs ' + f['B'])
    group[key].append(f['rel_diff_pct'])

print(f"{'Metric':<18} {'Joint':<18} {'Stage Pair':<32} {'Frames':>7} {'Max %':>8} {'Mean %':>8}")
print("-" * 100)
sorted_groups = sorted(group.items(), key=lambda x: -np.max(x[1]))
for (metric, joint, pair), diffs in sorted_groups[:40]:
    print(f"  {metric:<16} {joint:<18} {pair:<32} {len(diffs):>7} {np.max(diffs):>7.1f}% {np.mean(diffs):>7.1f}%")

# ── 4. Show AVERAGE row violations ────────────────────────────────────────────
if avg_findings:
    print(f"\n{'─'*110}")
    print("AVERAGE ROW VIOLATIONS:")
    print(f"{'─'*110}")
    print(f"{'Metric':<18} {'Joint':<18} {'Stage A':<20} {'Val A':>8} {'Stage B':<20} {'Val B':>8} {'Diff%':>8}")
    print("-" * 110)
    for f in sorted(avg_findings, key=lambda x: -x['rel_diff_pct']):
        print(f"  {f['metric']:<16} {f['joint']:<18} {f['A']:<20} {f['A_val']:>8.2f} {f['B']:<20} {f['B_val']:>8.2f} {f['rel_diff_pct']:>7.1f}%")

# ── 5. Worst individual frames ────────────────────────────────────────────────
print(f"\n{'─'*110}")
print("TOP 20 WORST INDIVIDUAL FRAMES (highest relative diff):")
print(f"{'─'*110}")
print(f"{'Metric':<18} {'Frame':>6} {'Joint':<18} {'Stage A':<20} {'Val A':>8} {'Stage B':<20} {'Val B':>8} {'Diff%':>8}")
print("-" * 110)
top20 = sorted(frame_findings, key=lambda x: -x['rel_diff_pct'])[:20]
for f in top20:
    print(f"  {f['metric']:<16} {f['frame']:>6} {f['joint']:<18} {f['A']:<20} {f['A_val']:>8.2f} {f['B']:<20} {f['B_val']:>8.2f} {f['rel_diff_pct']:>7.1f}%")

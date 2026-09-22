import os
import json
import glob
import numpy as np
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

base = r'd:\optimization_model_monocular_3\output\max_iter_cmp'


def load_frames(itr, max_frames=None):
    meta_dir = os.path.join(base, f'iter_{itr}', 'fused', 'metadata')
    files = sorted(glob.glob(os.path.join(meta_dir, '*.json')))
    if max_frames:
        files = files[:max_frames]
    frames = []
    for fp in files:
        with open(fp, encoding='utf-8') as f:
            frames.append(json.load(f))
    return frames


def extract_optimized_coords(itr, max_frames=None):
    kp_dir = os.path.join(base, f'iter_{itr}', 'fused', 'keypoints3d')
    files = sorted(glob.glob(os.path.join(kp_dir, '*.json')))
    if max_frames:
        files = files[:max_frames]
    all_joints = {}
    for fp in files:
        with open(fp, encoding='utf-8') as f:
            data = json.load(f)
        for cam in ['camera1', 'camera2']:
            for joint, coords in data.get(cam, {}).items():
                key = f'{cam}_{joint}'
                if key not in all_joints:
                    all_joints[key] = []
                all_joints[key].append(np.array(coords))
    return all_joints


print("=== ANALYSIS: Why does before_loss differ between iter_20 and iter_100? ===\n")

frames_20 = load_frames(20, max_frames=10)
frames_100 = load_frames(100, max_frames=10)

print("Frame | before_loss_20 | before_loss_100 | after_loss_20 | after_loss_100 | diff_before")
print("-" * 100)
for i, (f20, f100) in enumerate(zip(frames_20, frames_100)):
    bs20 = f20.get('before_stats', [None]*5)
    bs100 = f100.get('before_stats', [None]*5)
    as20 = f20.get('after_stats', [None]*5)
    as100 = f100.get('after_stats', [None]*5)

    bl20 = bs20[4] if bs20 and len(bs20) > 4 else None
    bl100 = bs100[4] if bs100 and len(bs100) > 4 else None
    al20 = as20[4] if as20 and len(as20) > 4 else None
    al100 = as100[4] if as100 and len(as100) > 4 else None

    diff = (bl100 - bl20) if (bl100 is not None and bl20 is not None) else None
    print(f"  {i+1:3d} | {bl20:.7f}  | {bl100:.7f}   | {al20:.7f} | {al100:.7f}  | {diff:+.7f}")

print("\n=== F_optimized set check ===")
f20_sets = [set(f.get('F_optimized', [])) for f in frames_20]
f100_sets = [set(f.get('F_optimized', [])) for f in frames_100]
same_f = all(a == b for a, b in zip(f20_sets, f100_sets))
print(f"F_optimized identical across iter_20 vs iter_100: {same_f}")
if not same_f:
    for i, (a, b) in enumerate(zip(f20_sets, f100_sets)):
        if a != b:
            print(f"  Frame {i+1}: only_in_20={sorted(a-b)}, only_in_100={sorted(b-a)}")

print("\n=== A_new (anchors) check ===")
a20_list = [f.get('A_new', []) for f in frames_20]
a100_list = [f.get('A_new', []) for f in frames_100]
same_a = all(a == b for a, b in zip(a20_list, a100_list))
print(f"A_new identical: {same_a}")

print("\n=== K1/K2 applied check ===")
all_same = True
for i, (f20, f100) in enumerate(zip(frames_20, frames_100)):
    k1_20 = set(f20.get('K1_applied', []))
    k1_100 = set(f100.get('K1_applied', []))
    k2_20 = set(f20.get('K2_applied', []))
    k2_100 = set(f100.get('K2_applied', []))
    if k1_20 != k1_100 or k2_20 != k2_100:
        all_same = False
        print(f"  Frame {i+1}: K1_applied diff={k1_20.symmetric_difference(k1_100)}, K2 diff={k2_20.symmetric_difference(k2_100)}")
if all_same:
    print("  All identical")

print("\n=== Keypoint displacement: iter_20 vs iter_100 (first 50 frames) ===")
joints_20 = extract_optimized_coords(20, max_frames=50)
joints_100 = extract_optimized_coords(100, max_frames=50)

common_keys = set(joints_20) & set(joints_100)
diffs = {}
for key in sorted(common_keys):
    c20 = np.array(joints_20[key])
    c100 = np.array(joints_100[key])
    min_len = min(len(c20), len(c100))
    diff = np.mean(np.linalg.norm(c20[:min_len] - c100[:min_len], axis=1)) * 1000  # mm
    diffs[key] = diff

top = sorted(diffs.items(), key=lambda x: -x[1])[:14]
print("  Joint                          | Mean displacement (mm)")
for k, v in top:
    bar = '#' * min(int(v / 1), 40)
    print(f"  {k:34s} | {v:8.2f}  {bar}")

overall = np.mean(list(diffs.values()))
print(f"\n  Overall mean displacement: {overall:.2f} mm")

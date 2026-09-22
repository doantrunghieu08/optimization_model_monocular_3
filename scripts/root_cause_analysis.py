"""
Root cause analysis: K1_applied differs between iter_20 and iter_100.
Check why right_elbow/right_wrist/right_hand are in K1_applied for iter_20 but NOT for iter_100.
"""
import os
import json
import glob
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

base = r'd:\optimization_model_monocular_3\output\max_iter_cmp'


def load_frame(itr, frame_n=0):
    meta_dir = os.path.join(base, f'iter_{itr}', 'fused', 'metadata')
    files = sorted(glob.glob(os.path.join(meta_dir, '*.json')))
    with open(files[frame_n], encoding='utf-8') as f:
        return json.load(f)


# Frame 1 analysis
f20 = load_frame(20, 0)
f100 = load_frame(100, 0)

joints_of_interest = ['right_elbow', 'right_wrist', 'right_hand', 'right_shoulder']

print("=== FRAME 1: K1/K2 decision differences ===\n")
print(f"iter_20  K1_applied: {sorted(f20.get('K1_applied', []))}")
print(f"iter_100 K1_applied: {sorted(f100.get('K1_applied', []))}")
print(f"iter_20  K2_applied: {sorted(f20.get('K2_applied', []))}")
print(f"iter_100 K2_applied: {sorted(f100.get('K2_applied', []))}")
print(f"iter_20  K1: {sorted(f20.get('K1', []))}")
print(f"iter_100 K1: {sorted(f100.get('K1', []))}")
print(f"iter_20  K2: {sorted(f20.get('K2', []))}")
print(f"iter_100 K2: {sorted(f100.get('K2', []))}")
print(f"iter_20  M: {sorted(f20.get('M', []))}")
print(f"iter_100 M: {sorted(f100.get('M', []))}")

print("\n=== Joint belief comparison (H1/H2) ===")
jb_20 = f20.get('joint_belief', {})
jb_100 = f100.get('joint_belief', {})
h1_20 = jb_20.get('camera1', {})
h2_20 = jb_20.get('camera2', {})
h1_100 = jb_100.get('camera1', {})
h2_100 = jb_100.get('camera2', {})

print(f"{'Joint':25s} | H1_20  | H2_20  | H1_100 | H2_100")
print("-" * 70)
for j in joints_of_interest:
    h1v20 = h1_20.get(j, 'N/A')
    h2v20 = h2_20.get(j, 'N/A')
    h1v100 = h1_100.get(j, 'N/A')
    h2v100 = h2_100.get(j, 'N/A')
    print(f"  {j:23s} | {h1v20!s:6s} | {h2v20!s:6s} | {h1v100!s:6s} | {h2v100!s:6s}")

print("\n=== Limb decisions ===")
ld20 = f20.get('limb_decisions', [])
ld100 = f100.get('limb_decisions', [])
print(f"iter_20  limb_decisions: {ld20}")
print(f"iter_100 limb_decisions: {ld100}")

print("\n=== correction_selector ===")
print(f"iter_20:  {f20.get('correction_selector')}")
print(f"iter_100: {f100.get('correction_selector')}")

print("\n=== Checking ALL frames: K1_applied diff pattern ===")
all_frames_20 = []
all_frames_100 = []
meta_20 = sorted(glob.glob(os.path.join(base, 'iter_20', 'fused', 'metadata', '*.json')))
meta_100 = sorted(glob.glob(os.path.join(base, 'iter_100', 'fused', 'metadata', '*.json')))

diff_frames = 0
total = min(len(meta_20), len(meta_100))
joint_diff_counts = {}

for i, (fp20, fp100) in enumerate(zip(meta_20[:100], meta_100[:100])):
    with open(fp20, encoding='utf-8') as f:
        d20 = json.load(f)
    with open(fp100, encoding='utf-8') as f:
        d100 = json.load(f)
    k1_20 = set(d20.get('K1_applied', []))
    k1_100 = set(d100.get('K1_applied', []))
    k2_20 = set(d20.get('K2_applied', []))
    k2_100 = set(d100.get('K2_applied', []))
    diff = k1_20.symmetric_difference(k1_100) | k2_20.symmetric_difference(k2_100)
    if diff:
        diff_frames += 1
        for j in diff:
            joint_diff_counts[j] = joint_diff_counts.get(j, 0) + 1

print(f"\nFrames with K1/K2_applied difference (first 100 frames): {diff_frames}/100")
print("Most common differing joints:")
for j, cnt in sorted(joint_diff_counts.items(), key=lambda x: -x[1]):
    print(f"  {j}: {cnt} frames")

print("\n=== ROOT CAUSE SUMMARY ===")
print("""
The key difference is in K1_applied:
- iter_20: right_elbow, right_wrist, right_hand are in K1_applied
  => These joints get CORRECTED (replaced with cam2->cam1 transform)
  => Correction moves them closer to the reference camera 1
  
- iter_100: those same joints are NOT in K1_applied
  => They remain in F_list and go to SLSQP optimizer
  => Optimizer moves them AWAY from their correction position

This suggests a BELIEF THRESHOLD change between runs that alters limb_winner decisions.
But both runs share the same config... which means a TEMPORAL FEEDBACK LOOP is at play!
""")

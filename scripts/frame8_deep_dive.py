"""
Deep dive: Frame 8 - right_elbow PA-MPJPE drops from 38.69 (posed) to 3.54 (fused) then bounces back to 38.45 (only_learnable).
Use iter_20 data (the best-performing run) as the reference fused data.
"""
import os, json, glob, sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = r'd:\optimization_model_monocular_3'

# Use iter_20 as the main fused reference
FUSED_META = os.path.join(WORKSPACE, 'output', 'max_iter_cmp', 'iter_20', 'fused', 'metadata')
FUSED_KP   = os.path.join(WORKSPACE, 'output', 'max_iter_cmp', 'iter_20', 'fused', 'keypoints3d')
POSE_KP    = os.path.join(WORKSPACE, 'output', 'pose_results', 'keypoints3d')


def find_frame(directory, frame_num, pattern_template='*_{}.json'):
    files = glob.glob(os.path.join(directory, '*.json'))
    # Try different naming conventions
    for f in files:
        base = os.path.basename(f)
        # Extract number
        import re
        m = re.search(r'(\d+)', base)
        if m and int(m.group(1)) == frame_num:
            return f
    return None


# ── Load frame 8 fused metadata ───────────────────────────────────────────────
meta_path = find_frame(FUSED_META, 8)
if not meta_path:
    # list a few
    avail = sorted(os.listdir(FUSED_META))[:5]
    print(f"Frame 8 metadata not found. Available: {avail}")
else:
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    print("=== FRAME 8: Fusion metadata (iter_20 run) ===")
    print(f"  File: {os.path.basename(meta_path)}")
    print(f"  K1_applied (joints replaced from cam2->cam1): {sorted(meta.get('K1_applied', []))}")
    print(f"  K2_applied (joints replaced from cam1->cam2): {sorted(meta.get('K2_applied', []))}")
    print(f"  K1 (candidates): {sorted(meta.get('K1', []))}")
    print(f"  K2 (candidates): {sorted(meta.get('K2', []))}")
    print(f"  M  (orientation mismatch): {sorted(meta.get('M', []))}")
    print(f"  F_optimized: {sorted(meta.get('F_optimized', []))}")
    print(f"  A_new (anchors): {sorted(meta.get('A_new', []))}")
    print(f"  correction_selector: {meta.get('correction_selector')}")
    print(f"  belief_correction_enabled: {meta.get('belief_correction_enabled')}")

    jb = meta.get('joint_belief', {})
    h1 = jb.get('camera1', {})
    h2 = jb.get('camera2', {})
    joints_of_interest = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hand', 'left_elbow']
    print(f"\n  Belief values:")
    print(f"  {'Joint':22s} H1 (cam1)   H2 (cam2)   vis_cam1  vis_cam2")
    vis1 = meta.get('vis1', {})
    vis2 = meta.get('vis2', {})
    for j in joints_of_interest:
        print(f"  {j:22s} {h1.get(j, 0):.4f}      {h2.get(j, 0):.4f}      {vis1.get(j)}     {vis2.get(j)}")

    ld = meta.get('limb_decisions', [])
    print(f"\n  Limb decisions:")
    for d in ld:
        extra = f", angle_deg={d.get('angle_deg', 0):.1f}" if 'angle_deg' in d else ""
        print(f"    {d['limb']:12s}: winner={str(d.get('winner')):8s}, reason={d.get('reason')}, "
              f"H_cam1={d.get('belief_camera1', 0):.3f}, H_cam2={d.get('belief_camera2', 0):.3f}{extra}")

    bs = meta.get('before_stats')
    as_ = meta.get('after_stats')
    print(f"\n  before_stats (q1,q3,mean,median,loss): {bs}")
    print(f"  after_stats  (q1,q3,mean,median,loss): {as_}")

    changes = meta.get('changes', {})
    for cam in ['camera1', 'camera2']:
        c = changes.get(cam, {})
        print(f"\n  Changes {cam}: {c.get('changed_joint_count')} joints moved, "
              f"mean={c.get('mean_displacement_mm', 0):.2f} mm, "
              f"max={c.get('max_displacement_mm', 0):.2f} mm")
        print(f"    Changed joints: {c.get('changed_joints', [])}")

# ── Fused keypoints ────────────────────────────────────────────────────────────
kp_path = find_frame(FUSED_KP, 8)
if kp_path:
    with open(kp_path, encoding='utf-8') as f:
        fused_kp = json.load(f)
    print("\n=== FRAME 8: Fused keypoints (right arm) ===")
    for joint in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hand']:
        c1 = fused_kp.get('camera1', {}).get(joint, [])
        c2 = fused_kp.get('camera2', {}).get(joint, [])
        c1_s = [f'{x:.4f}' for x in c1] if c1 else 'N/A'
        c2_s = [f'{x:.4f}' for x in c2] if c2 else 'N/A'
        diff = np.linalg.norm(np.array(c1) - np.array(c2)) * 1000 if c1 and c2 else None
        print(f"  {joint:20s} cam1={c1_s}  cam2={c2_s}  diff={diff:.1f}mm" if diff else f"  {joint:20s} cam1={c1_s}  cam2={c2_s}")

# ── Posed keypoints ────────────────────────────────────────────────────────────
pose_path = find_frame(POSE_KP, 8)
if pose_path and kp_path:
    with open(pose_path, encoding='utf-8') as f:
        pose_kp = json.load(f)
    print("\n=== FRAME 8: Displacement posed -> fused ===")
    for joint in ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hand', 'left_elbow']:
        for cam in ['camera1', 'camera2']:
            pose_cam = pose_kp.get(cam, pose_kp.get('camera1', {})) if cam == 'camera1' else pose_kp.get(cam, pose_kp.get('camera2', {}))
            p = np.array(pose_cam.get(joint, [0, 0, 0]))
            fk = np.array(fused_kp.get(cam, {}).get(joint, [0, 0, 0]))
            disp = np.linalg.norm(p - fk) * 1000
            print(f"  {cam} {joint:22s}: displacement = {disp:.2f} mm")

# ── Check posed data structure ────────────────────────────────────────────────
if pose_path:
    print(f"\n=== Posed data keys (frame 8) ===")
    print(f"  Top-level keys: {list(pose_kp.keys())[:10]}")
    if 'camera1' in pose_kp:
        print(f"  camera1 joints: {sorted(pose_kp['camera1'].keys())[:8]}")
    if 'camera2' in pose_kp:
        print(f"  camera2 joints: {sorted(pose_kp['camera2'].keys())[:8]}")

print("\n=== PATTERN EXPLANATION ===")
print("""
From the evaluation CSV:
  Frame 8, right_elbow:
    posed          = 38.69 mm (PA-MPJPE cam2), 38.45 mm (PA-MPJPE cam1)
    fused          =  3.54 mm (cam2),  5.18 mm (cam1)    <-- HUGE improvement by fusion!
    only_learnable = 38.45 mm (cam2), 38.27 mm (cam1)    <-- back to almost posed error

The fusion pipeline correctly moved right_elbow ~10x closer to ground truth on frame 8.
The learnable stage then reverted this improvement.

Two possible explanations:
  1. The learnable model was trained on posed data and overfits to the posed pose distribution
  2. The learnable stage uses a DIFFERENT input (e.g., directly from posed, not fused)
""")

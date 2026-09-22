"""
Check SLSQP failure rates and optimization statistics across different max_iter settings.
"""
import os
import json
import glob
import numpy as np

base = r'd:\optimization_model_monocular_3\output\max_iter_cmp'

for itr in [20, 40, 60, 80, 100, 150]:
    meta_dir = os.path.join(base, f'iter_{itr}', 'fused', 'metadata')
    if not os.path.exists(meta_dir):
        print(f'iter_{itr}: no metadata dir')
        continue

    files = sorted(glob.glob(os.path.join(meta_dir, '*.json')))
    total = len(files)

    # Sample metrics
    before_losses = []
    after_losses = []
    f_counts = []
    opt_count = 0

    for fp in files:
        with open(fp, encoding='utf-8') as f:
            data = json.load(f)

        f_opt = data.get('F_optimized', [])
        f_counts.append(len(f_opt))
        if f_opt:
            opt_count += 1

        bs = data.get('before_stats')
        as_ = data.get('after_stats')
        # before_stats / after_stats = (q1, q3, mean_val, median_val, loss)
        if isinstance(bs, (list, tuple)) and len(bs) >= 5:
            before_losses.append(bs[4])  # mean loss
        if isinstance(as_, (list, tuple)) and len(as_) >= 5:
            after_losses.append(as_[4])

    print(f"\n=== iter_{itr} ({total} frames) ===")
    print(f"  Frames with optimized joints: {opt_count}/{total}")
    print(f"  Avg F-list size: {np.mean(f_counts):.1f}")
    if before_losses:
        print(f"  Before loss (mean): {np.mean(before_losses):.5f}")
    if after_losses:
        print(f"  After  loss (mean): {np.mean(after_losses):.5f}")
    if before_losses and after_losses:
        improvement = np.mean(before_losses) - np.mean(after_losses)
        print(f"  Loss improvement:   {improvement:.5f}")

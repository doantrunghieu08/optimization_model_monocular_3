#!/bin/bash
SEQ=${1:?usage: run_ablation.sh <seq>   vd. 016_mma4}
MASK=configs/identity_mask_${SEQ}.json
OUT=output/method_comparison/${SEQ}_ext
[ -f "$MASK" ] || { echo "Thiếu $MASK (chạy build_identity_mask.py trước)"; exit 1; }
mkdir -p "$OUT"
for w in $(seq 0 15); do
  EXT=1 python run_method_comparison.py run \
    --brute "configs/${SEQ}.yml" \
    --methods aotr naive adaptive_naive \
      naive@use_aotr_stability=false \
      "adaptive_naive@torso_gate=false,use_aotr_stability=false,min_bone_direction_cos=-1.0" \
      aotr@temporal_alignment_fallback=false \
      aotr@max_cross_view_bone_angle_deg=null \
      aotr@min_bone_direction_cos=-1.0 \
      aotr@min_correspondences=999 \
      aotr@unobserved_prior_weight=0.0001 \
      aotr@huber_delta_normalized=1000000 \
      aotr@fixed_alpha=0.5 \
      "aotr@sequence_stability={enabled: false}" \
    --mask "$MASK" \
    --out "$OUT" \
    --worker "$w" --n-workers 16 \
    > "$OUT/w$w.log" 2>&1 &
done
wait
echo "Xong EXT=1: $OUT"
python run_method_comparison.py summarize --out "$OUT"

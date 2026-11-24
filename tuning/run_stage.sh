#!/bin/zsh
# Một vòng tinh chỉnh AOTR trên tập DEV (003 aria01, 8 camera = 56 cặp), EXT=1, masked.
# usage: tuning/run_stage.sh <out_dir> <variant> [<variant> ...]
# variant = "aotr@key=value,key=value" (ghi đè fusion.aotr.*); luôn kèm "aotr" gốc để so ghép cặp.
# Chạy lại cùng lệnh sẽ tiếp tục từ chỗ dừng (resumable).
cd "$(dirname $0)/.." || exit 1
OUT=$1; shift
N=${N_WORKERS:-6}
mkdir -p $OUT
for w in $(seq 0 $((N-1))); do
  EXT=1 OMP_NUM_THREADS=1 nohup venv311/bin/python -W ignore run_method_comparison.py run \
    --brute ${BRUTE:-configs/003_mma5.yml} --config configs/pipeline.yml \
    --mask configs/identity_mask_003_mma5.json --segments 003_mma5_aria01 \
    --split dev=aria01:cam01,cam03,cam05,cam07,cam12,cam14,cam17,cam20 \
    --methods aotr "$@" --out $OUT --worker $w --n-workers $N >> $OUT/w$w.log 2>&1 &
  sleep 2
done
wait
venv311/bin/python output/method_comparison/analysis/scripts/ablation.py $OUT > $OUT/summary_ablation.txt

#!/usr/bin/env bash
# Links data from the RunPod Network Volume into /app, optionally runs the
# pipeline, then hands off to CMD (default: RunPod /start.sh).
#
# Env vars:
#   DATA_ROOT        volume mount point (default /workspace)
#   RUN_ON_START=1   run the pipeline in the background on boot
#   PIPELINE_CONFIG  config path relative to /app (default configs/pipeline.yml)
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
DATA_ROOT="${DATA_ROOT:-/workspace}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/pipeline.yml}"

link_dir() {
    # Replace /app/<name> with a symlink to $DATA_ROOT/<name>.
    local name="$1"
    mkdir -p "$DATA_ROOT/$name"
    if [ ! -L "$APP_DIR/$name" ]; then
        rm -rf "$APP_DIR/$name"
        ln -s "$DATA_ROOT/$name" "$APP_DIR/$name"
    fi
}

if [ -d "$DATA_ROOT" ]; then
    link_dir input
    link_dir output

    # models/: the image already holds the small tracked files (*.npy, ckpt).
    # Overlay every file from the volume (SMPL *.pkl etc.) on top of them.
    if [ -d "$DATA_ROOT/models" ]; then
        (cd "$DATA_ROOT/models" && find . -type f) | while read -r f; do
            mkdir -p "$APP_DIR/models/$(dirname "$f")"
            ln -sf "$DATA_ROOT/models/$f" "$APP_DIR/models/$f"
        done
    fi
fi

if [ ! -e "$APP_DIR/models/SMPL_NEUTRAL.pkl" ]; then
    echo "[entrypoint] WARNING: models/SMPL_NEUTRAL.pkl not found." \
         "Copy it to $DATA_ROOT/models/ on the network volume." >&2
fi

python - <<'EOF' || true
import torch
ok = torch.cuda.is_available()
print(f"[entrypoint] torch {torch.__version__} | CUDA available: {ok}"
      + (f" | {torch.cuda.get_device_name(0)}" if ok else ""))
EOF

if [ "${RUN_ON_START:-0}" = "1" ]; then
    LOG="$DATA_ROOT/output/run_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$LOG")"
    echo "[entrypoint] Running pipeline with $PIPELINE_CONFIG -> $LOG"
    (cd "$APP_DIR" && python main.py --config "$PIPELINE_CONFIG") >"$LOG" 2>&1 &
fi

exec "$@"

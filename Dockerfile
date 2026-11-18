# RunPod image for the monocular two-camera fusion pipeline.
#
# Base image: RunPod's PyTorch image (CUDA 12.4, Python 3.11, torch 2.4 already
# installed). It also ships /start.sh, which starts SSH + JupyterLab so the Pod
# stays up and you can connect to it.
#
# Large/licensed data is NOT baked into the image (SMPL *.pkl, input/, output/).
# Put it on a RunPod Network Volume mounted at /workspace:
#   /workspace/models/SMPL_NEUTRAL.pkl, /workspace/models/smpl/SMPL_*.pkl
#   /workspace/input/<dataset>/...
# The entrypoint links those into /app at startup; outputs go to /workspace/output.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    APP_DIR=/app \
    DATA_ROOT=/workspace

# System libs: ffmpeg (video conversion in main.py), libGL/glib (opencv, ultralytics),
# libspatialindex (rtree fallback for trimesh).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libspatialindex-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (cached layer). torch is already in the base image; pip keeps
# it because nothing in requirements.txt pins a different version.
COPY requirements.txt .
RUN python -m pip install --upgrade "pip<25" "setuptools<70" wheel \
    && python -m pip install "numpy<2" \
    # chumpy's setup.py imports pip, so it must build without isolation.
    && python -m pip install --no-build-isolation chumpy \
    && python -m pip install -r requirements.txt \
    # Imported by the code but missing from requirements.txt.
    && python -m pip install pyyaml easydict pandas scikit-learn \
    && python -c "import torch, numpy, smplx, chumpy, cv2, trimesh; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', numpy.__version__)"

# Project code (see .dockerignore for what is excluded).
COPY . .

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && python -m compileall -q fusion_pipeline evaluation_pipeline pose_pipeline \
        preprocess_pipeline learnable_pipeline visualization_pipeline main.py pipeline.py

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Default: RunPod's start script (SSH + Jupyter), keeps the Pod alive.
CMD ["/start.sh"]

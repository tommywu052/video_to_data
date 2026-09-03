FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=egl
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg \
    libgl1 libegl1 libglib2.0-0 libxrender1 libsm6 libxext6 libglfw3 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Vendor HaMeR at a pinned commit. We only need its hand-reconstruction code
# (models/, datasets/, utils/) — the demo-side detection chain (vitdet,
# vitpose) is intentionally NOT installed.
ARG HAMER_REPO=https://github.com/geopavlakos/hamer.git
ARG HAMER_COMMIT=3a01849f4148352e9260b69bf28b65d1671a4905
RUN git clone --recursive ${HAMER_REPO} /opt/hamer && \
    cd /opt/hamer && git checkout ${HAMER_COMMIT} && \
    git submodule update --init --recursive

RUN pip install --no-cache-dir \
    "numpy<2" \
    Pillow opencv-python tqdm \
    pyrender trimesh \
    smplx \
    yacs hydra-core omegaconf \
    pytorch-lightning \
    timm einops \
    webdataset braceexpand dill av joblib loguru \
    scikit-image

# manotorch and chumpy aren't on PyPI for Python 3.11 — pull from git.
RUN pip install --no-cache-dir \
    git+https://github.com/lixiny/manotorch.git \
    git+https://github.com/mattloper/chumpy.git

# Install HaMeR (skip its detector deps — we drive crops from SAM2 masks).
RUN pip install --no-deps -e /opt/hamer

ENV PYTHONPATH=/opt/hamer

COPY . /workspace
RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_hamer/lib

# Sanity-check imports at build time so a broken install fails the build.
RUN python -c "from hamer.models import load_hamer; from hamer.datasets.utils import generate_image_patch_cv2; print('hamer imports OK')"

# Verify the torch in the finished image can actually run on the architectures
# this image was built for, and repair it if some dependency replaced it.
RUN P="$(command -v python || command -v python3)"; \
    "$P" -c "import importlib.util as u, sys; \
sys.exit(0) if u.find_spec('torch') is None else None; \
import torch; \
sys.exit(3 if 'sm_120' not in torch.cuda.get_arch_list() else 0)" \
    || "$P" -m pip install --no-cache-dir --force-reinstall \
        torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

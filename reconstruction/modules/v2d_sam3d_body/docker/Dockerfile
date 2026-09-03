FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV MAX_JOBS=8 MAKEFLAGS=-j8
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    libosmesa6-dev \
    libegl1-mesa-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Heavy Python dependencies
RUN pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill \
    pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
    networkx==3.2.1 roma joblib cython jsonlines loguru optree fvcore pycocotools \
    tensorboard huggingface_hub webdataset rtree

# Pytorch3D (requires CUDA build; no GPU visible during docker build so we
# must specify architectures explicitly)
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0" \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

RUN pip install py-soma-x

# Install v2d packages (cached unless pyproject.toml or v2d/ source changes)
COPY v2d_common /workspace/v2d_common
RUN pip install -e "/workspace/v2d_common[io]"

COPY v2d_mv /workspace/v2d_mv
RUN pip install -e "/workspace/v2d_mv[math,vis]"

COPY v2d_sam3d_body /workspace/v2d_sam3d_body
RUN pip install -e /workspace/v2d_sam3d_body/lib

# sam_3d_body and tools are accessed via PYTHONPATH (legacy package structure)
ENV PYTHONPATH=/workspace/v2d_sam3d_body/lib
ENV PYOPENGL_PLATFORM=egl
ENV TORCH_HOME=/tmp/torch_cache

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

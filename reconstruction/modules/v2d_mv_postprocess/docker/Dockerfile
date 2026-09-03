FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV MAX_JOBS=8 MAKEFLAGS=-j8
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    libosmesa6-dev \
    libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install open3d

COPY v2d_common /workspace/v2d_common
RUN pip install -e "/workspace/v2d_common[io]"

COPY v2d_mv /workspace/v2d_mv
RUN pip install -e "/workspace/v2d_mv[vis,math]"

COPY v2d_mv_postprocess/lib /workspace/v2d_mv_postprocess/lib
RUN pip install -e /workspace/v2d_mv_postprocess/lib

ENV PYOPENGL_PLATFORM=egl

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

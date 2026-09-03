FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps: osmesa for pyrender offscreen rendering, libgl for trimesh
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    libegl1-mesa \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install pytest -e /workspace/v2d_common -e /workspace/v2d_mesh/lib

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

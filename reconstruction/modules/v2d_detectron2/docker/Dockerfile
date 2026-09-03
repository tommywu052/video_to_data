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
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install Detectron2 (requires --no-build-isolation due to fvcore/iopath)
RUN pip install 'detectron2[all] @ git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation

COPY . /workspace

RUN pip install -e "/workspace/v2d_common[io]" \
                -e /workspace/v2d_mv \
                -e /workspace/v2d_detectron2/lib

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

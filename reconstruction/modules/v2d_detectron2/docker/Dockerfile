FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
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

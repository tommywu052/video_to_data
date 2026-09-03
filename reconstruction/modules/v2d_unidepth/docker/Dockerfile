FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    zip \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install -e /workspace/v2d_common -e /workspace/v2d_unidepth/lib

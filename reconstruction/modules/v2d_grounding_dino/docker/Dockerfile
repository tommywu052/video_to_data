FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV MAX_JOBS=8 MAKEFLAGS=-j8
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO && \
    cd /tmp/GroundingDINO && \
    sed -i 's/.type().is_cuda()/.is_cuda()/g; s/value.type(), "ms_deform_attn/value.scalar_type(), "ms_deform_attn/g' groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;12.0" \
    pip install --no-build-isolation . && \
    rm -rf /tmp/GroundingDINO

RUN pip install "transformers==4.27.4"
RUN pip install huggingface-hub requests Pillow opencv-python

COPY . /workspace

RUN pip install -e "/workspace/v2d_common[io]" \
    -e /workspace/v2d_mv \
    -e /workspace/v2d_grounding_dino/lib

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

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV MAX_JOBS=8 MAKEFLAGS=-j8
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

RUN pip install --no-cache-dir \
    "numpy<2" \
    Pillow opencv-python tqdm \
    pyrender trimesh \
    smplx \
    yacs hydra-core omegaconf einops timm \
    huggingface_hub transformers safetensors \
    ultralytics dill

# manotorch + chumpy (same as hamer; needed by the MANO layer used in render).
RUN pip install --no-cache-dir \
    git+https://github.com/lixiny/manotorch.git \
    git+https://github.com/mattloper/chumpy.git

# WiLoR mini pipeline — provides WiLorHandPose3dEstimationPipeline (detector + reconstructor).
RUN pip install --no-cache-dir git+https://github.com/warmshao/WiLoR-mini.git

COPY . /workspace
RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_wilor/lib

# Sanity-check at build time so a broken install fails the build, not the run.
RUN python -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; print('wilor_mini import OK')"

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

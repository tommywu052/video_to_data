FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/*

# Build native CUDA extensions for the same arches as the rest of the repo.
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"
ENV FORCE_CUDA=1

RUN pip install --no-cache-dir \
    "numpy<2" \
    Pillow opencv-python tqdm \
    trimesh \
    imageio imageio-ffmpeg \
    scipy \
    roma

# manotorch (and chumpy, its transitive runtime dep) aren't on PyPI for 3.11.
RUN pip install --no-cache-dir \
    git+https://github.com/lixiny/manotorch.git \
    git+https://github.com/mattloper/chumpy.git

# gsplat builds CUDA kernels at install time; pinning a known-good tag keeps
# Dockerfile reproducibility independent of upstream HEAD.
RUN pip install --no-cache-dir --no-binary gsplat gsplat==1.4.0

RUN pip install rtree

COPY . /workspace
RUN pip install -e /workspace/v2d_common \
    -e /workspace/v2d_gsplat_refinement/lib

# Sanity-check imports at build time.
RUN python -c "import gsplat; from gsplat.rendering import rasterization; \
               from manotorch.manolayer import ManoLayer; \
               from v2d.gsplat_refinement.lib import refine; \
               print('gsplat_refinement imports OK')"

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

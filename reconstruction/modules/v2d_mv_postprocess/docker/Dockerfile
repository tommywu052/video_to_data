FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
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

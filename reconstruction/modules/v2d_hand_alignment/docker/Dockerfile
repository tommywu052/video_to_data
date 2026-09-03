FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libegl1-mesa \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace
RUN pip install opencv-python-headless \
                git+https://github.com/mattloper/chumpy \
                git+https://github.com/lixiny/manotorch
RUN pip install -e /workspace/v2d_common -e /workspace/v2d_hand_alignment/lib

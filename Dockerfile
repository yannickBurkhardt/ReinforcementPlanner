FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
# # updating the CUDA Linux GPG Repository Key
COPY ./cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb
# RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list && dpkg -i cuda-keyring_1.0-1_all.deb

# Install basic packages
ENV TZ=Europe/London
ENV DEBIAN_FRONTEND=noninteractive
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    libxrender-dev \
    wget \
    git \
    unzip \
    curl \
    nano \
    libopenexr-dev \
    openexr \
    python3-pip \
    python3-setuptools \
    libopenblas-dev \
    software-properties-common \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install stable baselines
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y cmake \
    libopenmpi-dev \
    zlib1g-dev \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
RUN pip install yacs
ENV PYTHONPATH=/usr/src/app/nav2D-envs/:/usr/src/app/rlkit/:/usr/src/app/rl/
COPY .bashrc /root/.bashrc
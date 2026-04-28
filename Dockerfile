FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake ninja-build \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    # Python 3.10 is the default on Ubuntu 22.04
    python3.10 python3.10-dev python3-pip python3-tk && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    echo "alias pip='pip3'" >> /root/.bashrc

# Science libraries
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    scipy \
    pyyaml \
    cffi \
    matplotlib \
    Cython \
    requests \
    opencv-python \
    pillow

# PyTorch 2.x for CUDA 11.8
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Expose port for TensorBoard
EXPOSE 6006

# cd to home on login
RUN echo "cd /root/dense_fusion" >> /root/.bashrc
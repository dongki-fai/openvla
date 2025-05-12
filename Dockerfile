FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.2.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace

# Copy all requirements files
COPY openvla/requirements-min.txt ./requirements-min.txt
COPY openvla/experiments/robot/libero/libero_requirements.txt ./libero_requirements.txt
COPY LIBERO/requirements.txt ./libero_main_requirements.txt

# Install all requirements
RUN pip install -r requirements-min.txt

RUN pip install -r libero_requirements.txt

# RUN pip install -r libero_main_requirements.txt

# System packages for later steps
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libegl1

# Install Python tools
RUN pip install packaging ninja && \
    ninja --version && \
    pip install "flash-attn==2.5.5" --no-build-isolation

# Continue pip installs
RUN pip install \
    "optimum[gptq]" \
    auto-gptq \
    accelerate \
    tensorflow \
    tensorflow_graphics \
    draccus \
    wandb \
    matplotlib

# Copy source repos for editable installs
COPY LIBERO /workspace/LIBERO
COPY openvla /workspace/openvla

# Install LIBERO in editable mode
RUN cd /workspace/LIBERO && pip install -e .

# Install openvla in editable mode (add dummy setup.py)
RUN cd /workspace/openvla && \
    echo "from setuptools import setup; setup()" > setup.py && \
    pip install -e .

# Clone and install dlimp
RUN git clone https://github.com/kvablack/dlimp /workspace/dlimp && \
    cd /workspace/dlimp && \
    pip install -e .
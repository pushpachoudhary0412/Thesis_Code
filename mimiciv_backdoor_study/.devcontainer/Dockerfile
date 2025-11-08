# Simple devcontainer Dockerfile for Python GPU-based experiments
FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11

# Install CUDA tools if available (assumes host has nvidia runtime)
# Keep image small - rely on host NVIDIA drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /workspace

# Copy environment spec and requirements
COPY environment.yml /workspace/environment.yml
COPY requirements.txt /workspace/requirements.txt

# Install micromamba (lightweight conda installer) and create the conda env from environment.yml
# This avoids building binary wheels like pyarrow at container build time when possible.
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xJ -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba create -y -n mimiciv_env -f /workspace/environment.yml -p /opt/conda/envs/mimiciv_env \
    && micromamba clean --all -y

# Ensure the conda env is first on PATH for subsequent commands and for the devcontainer's default interpreter
ENV PATH=/opt/conda/envs/mimiciv_env/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Install any remaining pip-only requirements inside the created env
RUN /opt/conda/envs/mimiciv_env/bin/python -m pip install --upgrade pip \
    && /opt/conda/envs/mimiciv_env/bin/pip install --no-cache-dir -r /workspace/requirements.txt

# Create a non-root user 'vscode' exists in base image; ensure permissions
USER vscode

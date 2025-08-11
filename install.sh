#!/usr/bin/env bash
set -e

# Ensure CUDA 12.7 or higher is installed
CUDA_VERSION=$(nvcc --version | grep "V[0-9]\+\.[0-9]\+" -o | cut -c2-)
if ! awk -v ver="$CUDA_VERSION" 'BEGIN {if (ver < 12.7) exit 1}' </dev/null; then
  echo "CUDA version must be 12.7 or higher. Current version: $CUDA_VERSION"
  exit 1
fi

# Ensure Python 3.13 is installed
if ! command -v python3.13 &> /dev/null; then
  echo "Python 3.13 is not installed. Please install Python 3.13 and try again."
  exit 1
fi

# Set up a virtual environment using uv
uv init --python 3.13
uv venv

# Update pip and install required packages
uv pip install --upgrade pip
uv pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja

# Install Axolotl with dependencies
uv pip install "axolotl[llmcompressor,ring-flash-attn] @ git+https://github.com/OpenAccess-AI-Collective/axolotl.git"

# Fetch example configurations
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL

# Install lm-eval with Hugging Face support
uv pip install lm-eval[hf]

# Verify installation
axolotl --version
uv pip list | grep axolotl
uv pip list | grep transformers
uv pip list | grep torch

echo "Installation complete. Axolotl and dependencies are ready to use."

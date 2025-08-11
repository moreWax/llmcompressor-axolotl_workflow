#!/usr/bin/env bash
set -e
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install "axolotl[llmcompressor,ring-flash-attn] @ git+https://github.com/OpenAccess-AI-Collective/axolotl.git"
pip install lm-eval[hf]

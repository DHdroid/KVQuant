#!/bin/bash
set -e

# Create and activate grad environment
cd /workspace/KVQuant/gradients
conda create -n grad python=3.9 -y
source activate grad
pip install -e .
pip install -r requirements.txt
pip install accelerate -U
pip install "numpy<2.0.0"
conda deactivate

# Create and activate kvquant environment
cd /workspace/KVQuant/quant
conda create -n kvquant python=3.10 -y
source activate kvquant
pip install -e .
pip install flash-attn --no-build-isolation
conda deactivate

#!/bin/bash
set -e

# Create and activate kvquant environment
cd /workspace/KVQuant/quant
conda create -n kvquant python=3.10 -y
source activate kvquant
pip install -e .
pip uninstall -y torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install google
pip install google-api-core
cd ../
ls
python setup.py install
conda deactivate

#!/bin/bash
set -e

# Create and activate kvquant environment
cd /workspace/KVQuant/quant
pip install transformers==4.51.0
pip install accelerate
pip install sentencepiece
pip install tokenizers>=0.12.1
pip install datasets
pip install scikit-learn
pip install matplotlib
pip install google
pip install google-api-core
cd ../
python setup.py install

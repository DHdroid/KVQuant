FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# Install dependencies
RUN apt-get update && apt-get install -y wget git curl bash vim tmux
RUN conda init
RUN wget -qO- cli.runpod.net | bash

ENV PATH="/opt/conda/bin:$PATH"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Clone KVQuant repository
RUN git clone https://github.com/DHdroid/KVQuant /workspace/KVQuant
WORKDIR /workspace/KVQuant

# Copy and run the environment setup script
COPY setup_env.sh .
RUN bash setup_env.sh

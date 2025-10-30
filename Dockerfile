FROM nvcr.io/nvidia/pytorch:24.10-py3
WORKDIR /app

ENV MAX_JOBS=8

ARG CAUSAL_CONV1D_WHL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
ARG MAMBA_SSM_WHL_URL="https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
ARG FLASH_ATTN_WHL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

RUN apt-get update && apt-get -y install sudo
RUN pip install -U pip setuptools wheel packaging ninja
RUN pip install --user azureml-mlflow tensorboard
RUN pip install packaging lightning
RUN pip install "jsonargparse[signatures]" tokenizers sentencepiece wandb torchmetrics
RUN pip install tensorboard zstandard pandas pyarrow huggingface_hub
RUN pip install --no-cache-dir "$FLASH_ATTN_WHL_URL"
RUN git clone https://github.com/Dao-AILab/flash-attention
WORKDIR flash-attention
# WORKDIR csrc/layer_norm
# RUN pip install . --no-build-isolation
# WORKDIR ../
RUN git checkout 413d07e
WORKDIR csrc/xentropy
RUN python3 -m pip install . --no-build-isolation
WORKDIR ../
RUN git checkout main
WORKDIR /app
RUN pip install transformers==4.46.1 numpy accelerate

# torch2.5 cp310 linux_x86_64 cu12 (ou cu126) cxx11abiTRUE
RUN pip install --no-cache-dir "$CAUSAL_CONV1D_WHL_URL"


RUN pip install --no-cache-dir "$MAMBA_SSM_WHL_URL"


# RUN pip install torchao --extra-index-url https://download.pytorch.org/whl/cu128
# RUN pip install triton
RUN pip install einops
RUN pip install opt_einsum
RUN pip install git+https://github.com/renll/flash-linear-attention.git
RUN pip install lm-eval["ruler"]
RUN pip install azureml-core
RUN pip install liger_kernel==0.4.0

# docker build -t archscale:latest .
# docker run -it --rm --gpus all \
#   -v /teamspace/studios/this_studio/ArchScale:/app \
#   -v /teamspace/studios/this_studio/persist:/home/jovyan/persist \
#   -e HF_HOME=/home/jovyan/persist/hf_cache \
#   -w /app archscale:latest
# python pretrain.py \
#   --train_data_dir /home/jovyan/persist/datasets/slimpajama_packed \
#   --val_data_dir /home/jovyan/persist/datasets/slimpajama_packed \
#   --train_model sambay --depth 8 --train_name scaling_mup_tie \
#   --ctx_len 2048 --max_tokens 1e7 --fsdp_save_mem true
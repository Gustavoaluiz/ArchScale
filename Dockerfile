FROM nvcr.io/nvidia/pytorch:24.10-py3

ARG FLASH_ATTN_VERSION=2.4.2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

RUN python -m pip install \
        packaging \
        lightning \
        jsonargparse[signatures] \
        tokenizers \
        sentencepiece \
        wandb \
        torchmetrics \
        tensorboard \
        zstandard \
        pandas \
        pyarrow \
        huggingface_hub

RUN python -m pip install \
        azureml-mlflow \
        azureml-core \
        azure-identity

RUN python -m pip install \
        flash-attn==${FLASH_ATTN_VERSION} \
        --no-build-isolation

RUN git clone --depth 1 --branch v${FLASH_ATTN_VERSION} \
        https://github.com/Dao-AILab/flash-attention \
        flash-attention

WORKDIR flash-attention/csrc/rotary
RUN python -m pip install .

WORKDIR ../layer_norm
RUN python -m pip install .

WORKDIR ../xentropy
RUN python -m pip install .

WORKDIR /app

RUN python -m pip install transformers==4.46.1 numpy

RUN python -m pip install causal-conv1d

RUN python -m pip install mamba-ssm --no-build-isolation

RUN python -m pip install torchao --extra-index-url https://download.pytorch.org/whl/cu126

RUN python -m pip install triton==3.1.0

RUN python -m pip install einops opt_einsum

RUN python -m pip install git+https://github.com/renll/flash-linear-attention.git

RUN python -m pip install "lm-eval[ruler]"

RUN python -m pip install liger_kernel==0.4.0

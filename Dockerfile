FROM nvcr.io/nvidia/pytorch:24.10-py3
WORKDIR /app

ENV MAX_JOBS=8

RUN apt-get update && apt-get -y install sudo
RUN pip install -U pip setuptools wheel packaging ninja
RUN pip install --user azureml-mlflow tensorboard
RUN pip install packaging lightning
RUN pip install "jsonargparse[signatures]" tokenizers sentencepiece wandb torchmetrics
RUN pip install tensorboard zstandard pandas pyarrow huggingface_hub
RUN pip install -U flash-attn --no-build-isolation
RUN git clone https://github.com/Dao-AILab/flash-attention
WORKDIR flash-attention
WORKDIR csrc/layer_norm
RUN pip install . --no-build-isolation
WORKDIR ../
RUN git checkout 413d07e
WORKDIR xentropy
RUN pip install .
WORKDIR ../
RUN git checkout main
WORKDIR /app
RUN pip install transformers==4.46.1 numpy accelerate
# demora papo de 15min causa-conv1d
RUN pip install causal-conv1d --no-build-isolation
RUN pip install mamba-ssm --no-build-isolation
# RUN pip install torchao --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install triton==3.2.0
RUN pip install einops
RUN pip install opt_einsum
RUN pip install git+https://github.com/renll/flash-linear-attention.git
RUN pip install lm-eval["ruler"]
RUN pip install azureml-core
RUN pip install liger_kernel==0.4.0

# docker build -t archscale:latest .
# docker run -it --rm --name archscale_dev \
#   -v /teamspace/studios/this_studio:/app -w /app \
#   --gpus all \
#   archscale:latest bash
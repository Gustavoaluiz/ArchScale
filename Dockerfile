FROM nvcr.io/nvidia/pytorch:24.06-py3
WORKDIR /app
RUN apt-get update && apt-get -y install sudo
RUN pip install --upgrade pip
RUN pip install --user azureml-mlflow tensorboard
RUN pip install packaging lightning
RUN pip install jsonargparse[signatures] tokenizers sentencepiece wandb torchmetrics
RUN pip install tensorboard zstandard pandas pyarrow huggingface_hub
RUN pip install -U flash-attn --no-build-isolation
RUN git clone https://github.com/Dao-AILab/flash-attention
WORKDIR flash-attention
WORKDIR csrc/layer_norm
RUN pip install .

# WORKDIR ../xentropy
# RUN pip install .
WORKDIR /app
RUN pip install transformers==4.46.1 numpy
RUN pip install causal-conv1d
RUN pip install mamba-ssm --no-build-isolation
RUN pip install torchao --extra-index-url https://download.pytorch.org/whl/cu124
RUN pip install triton==3.2.0
RUN pip install einops
RUN pip install opt_einsum
RUN pip install git+https://github.com/renll/flash-linear-attention.git
RUN pip install lm-eval["ruler"]
RUN pip install azureml-core
RUN pip install liger_kernel==0.4.0
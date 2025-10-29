set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Este script deve ser executado como root (use sudo)." >&2
  exit 1
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1


# Pip
python3 -m pip install --upgrade pip
PIP="python3 -m pip"

# Instalações de Python (na ordem do Dockerfile)
$PIP install azureml-mlflow tensorboard
$PIP install packaging lightning
$PIP install 'jsonargparse[signatures]' tokenizers sentencepiece wandb torchmetrics
$PIP install tensorboard zstandard pandas pyarrow huggingface_hub
$PIP install -U flash-attn --no-build-isolation

# flash-attention (componentes específicos)
if [ ! -d flash-attention ]; then
  git clone https://github.com/Dao-AILab/flash-attention
fi
cd flash-attention

cd csrc/layer_norm
$PIP install .
cd ..

git checkout 413d07e

cd xentropy
$PIP install .
cd ..

git checkout main
cd /app

# Demais libs
$PIP install transformers==4.46.1 numpy
$PIP install causal-conv1d
$PIP install mamba-ssm --no-build-isolation
$PIP install torchao --extra-index-url https://download.pytorch.org/whl/cu124
$PIP install triton==3.2.0
$PIP install einops
$PIP install opt_einsum
$PIP install git+https://github.com/renll/flash-linear-attention.git
$PIP install 'lm-eval["ruler"]'
$PIP install azureml-core
$PIP install liger_kernel==0.4.0

echo "Ambiente configurado com sucesso."
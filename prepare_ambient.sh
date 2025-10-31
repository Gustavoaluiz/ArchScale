export PIP_DISABLE_PIP_VERSION_CHECK=1


# Pip
python3 -m pip install --upgrade pip
PIP="python3 -m pip"

conda install -y gcc_linux-64 gxx_linux-64 &&
conda install cuda -c nvidia 

# Instalações de Python (na ordem do Dockerfile)
$PIP install azureml-mlflow tensorboard
$PIP install -U packaging setuptools wheel ninja cmake
$PIP install 'jsonargparse[signatures]' tokenizers sentencepiece wandb torchmetrics lightning
$PIP install tensorboard zstandard pandas pyarrow huggingface_hub
$PIP -U --no-build-isolation --no-cache-dir \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'

# flash-attention (componentes específicos)
if [ ! -d flash-attention ]; then
  git clone https://github.com/Dao-AILab/flash-attention
fi
cd flash-attention

cd csrc/layer_norm
$PIP install .
cd ..
# python -m pip install --no-build-isolation --no-cache-dir -v .

git checkout 413d07e

cd xentropy
$PIP install .
cd ..

git checkout main
cd /app

# Demais libs
$PIP install transformers==4.46.1 numpy accelerate
$PIP install --no-cache-dir "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
$PIP install --no-cache-dir "https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# $PIP install torchao --extra-index-url https://download.pytorch.org/whl/cu128
# $PIP install triton==3.2.0
# $PIP install einops
# $PIP install opt_einsum
$PIP install git+https://github.com/renll/flash-linear-attention.git
$PIP install 'lm-eval[ruler]'
$PIP install azureml-core
$PIP install liger_kernel==0.4.0

echo "Ambiente configurado com sucesso."
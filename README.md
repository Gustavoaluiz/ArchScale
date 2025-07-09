
<h1 align="left" style="display: flex; align-items: center;">
  ArchScale
  <img src="assets/shooting_stars.png" alt="ArchScale Logo" width="100" style="margin-left: 16px; vertical-align: middle;"/>
</h1>


**Simple & Scalable Pretraining for Neural Architecture Research**

ArchScale is a comprehensive toolkit for training and evaluating neural language models with a focus on architecture and scaling laws. It provides implementations of various state-of-the-art architectures, scaling techniques, training optimizations and evaluation tools in a unified codebase.


## Updates
- [July 9] Released the code for training [Decoder-Hybrid-Decoder Architectures](https://aka.ms/flashreasoning-paper) with μP++, and the model checkpoint for [Phi-4-mini-flash-reasoning](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning) ⚡

## Features

- **Architectures**: Transformers, various SSM/attention/hybrid architectures, [YOCO](https://arxiv.org/abs/2405.05254), [Differential Attention](https://arxiv.org/pdf/2410.05258).
- **Scaling Laws**: μP++, μP, Chinchilla FLOPs scaling, and various experimental scaling laws for batch size, weight decay, etc.
- **Optimizers**: Muon, AdamW, Hybrid Optimizers.
- **Research-Friendly**: Easy adding/modifying architectures/scaling-laws/optimizers/scheduling/initialization, [WYSIWYG](https://en.wikipedia.org/wiki/WYSIWYG) philosophy for experiments logging. 
- **Performance**: End2end torch.compile training, clean & correct [Lightning Fabric](https://github.com/Lightning-AI/pytorch-lightning) package for FSDP distributed training, mixed precision, tensor parallelism and experimental fp8 support.
- **Training**: Simple data mixture support, packed dataset with pre-tokenization, variable-length training for long-context.
- **Evaluation**: Simple support for likelihood/generation based evaluation, long-context evaluation on Phonebook and RULER, scaling curve fitting and comparisons.

## Pretraining

We provide the [`Dockerfile`](Dockerfile) for setting up the training and evaluation environments. One can refer to the [Samba](https://github.com/microsoft/Samba/?tab=readme-ov-file#data-preparation) codebase for Slimpajama data tokenization. Training across a scale from 110M to 3.3B-parameter SambaY model with μP++ and Chinchilla token scaling on 8 GPUs is as simple as:

```bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export LIGHTNING_ARTIFACTS_DIR='path/to/output_dir'
for depth in 8 12 16 20 24; do
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain.py \
        --train_data_dir path/to/slim_pajama/data  --val_data_dir path/to/slim_pajama/data \
        --train_model sambay --depth ${depth} \
        --train_name scaling_mup
done
```

In the backend, a dataclass [`BaseHyperparameters`](pretrain.py#44) defines the optimization related HyperParameters (HPs) for a d16 (depth=16) model, and the scaling laws defined in [`setup`](pretrain.py#129) function will transfer these HPs to the actual HPs used at the target depth such d8, d12 or d24. One can easily sweep the base HPs with the following scripts.

```bash
for lr in 4e-4 1e-4 1e-3; do
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain.py \
        --train_data_dir path/to/slim_pajama/data  --val_data_dir path/to/slim_pajama/data \
        --train_model transformer --depth 8 --base_hps.eta0=${lr} \
        --train_name scaling_mup
done
```
Note that in this case, the learning rate is tuned for the d16, 1.0B model with 100B training tokens, but the actual training is conducted at a d8 model with around 12B tokens, thanks to μP++ for scaling down the computation cost of HPs sweeping. Models are defined in [`lit_gpt/config.py`](lit_gpt/config.py) with architecture-specific HPs.


## Long-Context Training

After shuffling and pre-tokenizing the [ProLong-64K](https://huggingface.co/datasets/princeton-nlp/prolong-data-64K) data, we can train a d16 model with 32K sequence length and 40B tokens on 8 GPUs using the following script:  
```bash
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain.py \
    --train_data_dir path/to/prolong/data  --val_data_dir path/to/prolong/data \
    --train_model transformer --depth 16 --ctx_len 32768 --max_tokens 4e10 \
    --train_name scaling_mup_rbase_varlen
```
where the symbol `rbase` will trigger the model use a larger RoPE base for long-context training and `varlen` will applies variable length training that seperates documents based on the EOS tokens.

## Evaluation

ArchScale provides comprehensive evaluation support for trained models across multiple domains:

### Standard NLP Benchmarks

Evaluate trained models on common language understanding tasks for SambaY architecture:

```bash
export CUDA_VISIBLE_DEVICES=0
python eval.py --model ArchScale \
    --model_args pretrained=path/to/checkpoint.pth,config="sambay_d16" \
    --tasks wikitext,lambada_openai,arc_easy,arc_challenge,winogrande,hellaswag,piqa,social_iqa \
    --device cuda --batch_size 16 --trust_remote_code
```
The script will infer the μP++ and architecture modification based on name of ckpt path.

### Long-Context Evaluation

#### RULER Benchmark
Evaluate long-context capabilities using the RULER benchmark:

```bash
python eval.py --model ArchScale \
    --model_args pretrained=path/to/checkpoint.pth,config="sambay_d16",max_length=32768,tokenizer=Orkhan/llama-2-7b-absa \
    --metadata='{"max_seq_lengths":[32768]}' \
    --tasks niah_single_1 --device cuda --batch_size 8 --trust_remote_code
```

This runs a simple needle-in-a-haystack task at 32K context length.

#### Phonebook Evaluation
Test long-context retrieval using the Phonebook benchmark with 32K context length:

```bash
python eval_phonebook.py \
    --checkpoint_path path/to/checkpoint.pth \
    --config "model_config" \
    --min_eval_len 1850 \
    --max_eval_len 1850 \
    --output_dir results_dir \
    --eval_batch_size 4
```

### Reasoning Evaluation

Evaluate reasoning capabilities on mathematical and scientific tasks using `eval_reason.sh`:

```bash
./eval_reason.sh  microsoft/Phi-4-mini-flash-reasoning aime24 output_dir
```

The reasoning evaluation uses vLLM backend with configurable generation parameters and supports multi-GPU evaluation.

## Citation

If you find our work useful, please consider citing:

```bibtex
@software{archscale2025,
  title={ArchScale: Simple and Scalable Pretraining for Neural Architecture Research},
  author={Liliang Ren and Zichong Li and Yelong Shen},
  year={2025},
  url={https://github.com/microsoft/ArchScale}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Samba](https://github.com/microsoft/Samba/)
- [LitGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

---

**Happy scaling! 🚀**

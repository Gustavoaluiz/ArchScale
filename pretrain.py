# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Literal
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
import torch.nn as nn
from lightning.fabric.strategies import ModelParallelStrategy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
import json
from dataclasses import dataclass, field

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import chunked_cross_entropy, num_parameters
from lit_gpt.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from lightning.pytorch.loggers import MLFlowLogger,WandbLogger

from lit_gpt.optim import Muon_fsdp2
from lit_gpt import FusedCrossEntropyLoss
import random
import os

@dataclass
class BaseHyperparameters:
    """Base optimization hyperparameters for MuP (mu-parametrization) transformation"""
    d0: int = 16                   # Base depth/layers
    eta0: float = 4e-4             # Base learning rate
    b0: int = 2**21                # Base batch size (2M tokens)
    t0: int = int(1e11)            # Base tokens (100B)
    hd0: int = 128                 # Base head dimension
    w_init0: float = 1.0           # Base weight initialization multiplier
    n0_mult: float = 237568        # Base parameter count multiplier (n0 = n0_mult * d0^3)
    weight_decay0: float = 1e-1    # Base weight decay
    eps0: float = 1e-8             # Base optimizer epsilon
    min_lr0: float = 0.0           # Base minimum learning rate
    warmup_tokens0: int = int(1e9) # Base warmup tokens
    beta1_0: float = 0.9           # Base Adam beta1
    beta2_0: float = 0.95          # Base Adam beta2
    muon_beta_0: float = 0.95      # Base Muon beta
    weight_lr_scale_0: float = 1  # scale up learning rate for weight parameters for hybrid optimizers 
    
    def __post_init__(self):
        """Calculate derived base number of parameters"""
        self.n0 = self.n0_mult * (self.d0 ** 3)  # Base parameter count

# Global base hyperparameters object
base_hps = BaseHyperparameters()


model_name = "transformer_d16"
train_config = "scaling_mup"
name = train_config +"_" + model_name

out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name
ckpt_dir = None
devices = torch.cuda.device_count() or 1

label_smoothing = 0.0

mup = False
original_mup = False
use_cu_seqlen = False

nodes= int(os.getenv("NODE_COUNT", 1))

# Default Hyperparameters, will be overriden by setup()
train_tokens = int(1e11) # 100 billion
#4k
global_batch_size = 512 // nodes
micro_batch_size = 8 

depth_global = base_hps.d0 # record depth in global scope
seq_len = 4096
local_window = None
learning_rate = base_hps.eta0
beta1 = base_hps.beta1_0
beta2 = base_hps.beta2_0
weight_decay = base_hps.weight_decay0
warmup_tokens = base_hps.warmup_tokens0
muon_beta = base_hps.muon_beta_0
weight_lr_scale = base_hps.weight_lr_scale_0
eps = base_hps.eps0
min_lr = base_hps.min_lr0
grad_clip = 1.0
decay_lr = True
total_evals = 400

log_step_interval = 10
eval_iters = total_evals // micro_batch_size # 50 # 25 # eval is invariant to microbatch size
save_step_interval = 1000
eval_step_interval = 1000

num_extrapol = 4

batch_size = global_batch_size // devices
gradient_accumulation_steps = max(batch_size // micro_batch_size, 1)

log_iter_interval = log_step_interval * gradient_accumulation_steps


train_data_config = [ ("train", 1.0) ]
val_data_config = [ ("validation", 1.0) ]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

wandb_logger = WandbLogger(project="pretrain-llm", name=name)

def setup(
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    load_dir: Optional[Path] = None,
    resume: Union[bool, Literal["auto"], Path] = False,
    train_model: str = None,
    train_name: str = "scaling_mup_tie_rbase_prolong_varlen",
    #"scaling_mup_tie_rbase_prolong_varlen", "scaling_mup", 
    depth: int = 16,
    max_tokens: float = None,
    ctx_len: int = None,
    swa_len: int = None,
    data_mixture: str = None,
    fsdp_save_mem: bool = False,
    base_hps: BaseHyperparameters = None,
) -> None:
    global model_name, train_config, name, out_dir, ckpt_dir, devices, learning_rate, nodes, train_tokens, \
        global_batch_size, micro_batch_size, total_evals, warmup_tokens, log_step_interval, \
        eval_iters, min_lr, batch_size, gradient_accumulation_steps, log_iter_interval, hparams, \
        depth_global, weight_decay, eps, beta1, beta2, muon_beta, weight_lr_scale, label_smoothing, \
        mup, wandb_logger, seq_len, local_window, num_extrapol, use_cu_seqlen, train_data_config, original_mup
    
    # Update global base_hps if provided
    if base_hps is not None:
        globals()['base_hps'] = base_hps
    else:
        base_hps = globals()['base_hps']
         
    seq_len = ctx_len if ctx_len is not None else seq_len
    model_name = train_model if train_model is not None else model_name
    train_config = train_name if train_name is not None else train_config
    local_window = swa_len if swa_len is not None else local_window
    ckpt_dir = load_dir if load_dir is not None else ckpt_dir
    if data_mixture is not None:
        # Load dataset weights from JSON file
        with open(data_mixture, 'r') as f:
            dataset_weights = json.load(f)

        # Convert dataset weights to train_data_config format
        train_data_config = [(dir_path, weight) for dir_path, weight in dataset_weights.items()]

    assert depth is not None
    mup= "mup" in train_config # use mup++
    original_mup = "_ori_mup" in train_config # use original mup
    rope_base = 640_000 if "_rbase_" in train_config else 10_000 # rope base for 32k ctx len
    use_cu_seqlen = "_varlen" in train_config # use cu_seqlen for variable length training
    label_smoothing = 0.1 if "_ls_" in train_config else 0.0
    model_name = model_name+"_d"+str(depth)
    
    if seq_len == 4096: # hardcoded for now
        if use_cu_seqlen:
            micro_batch_size = 1
            seq_len = 8192
        else:
            micro_batch_size = 2
    else:
        # long context
        micro_batch_size = 1
        num_extrapol = 2
        
    eos_token_id = 2 # llama2 token id
        
    ar = Config.from_name(model_name).ar # model aspect ratio
    mult = 14.5 * (ar ** 2) # 237568 # transformer
    if "samba" in model_name:
        mult = 15 * (ar ** 2) + 160 * ar
    if "sambay" in model_name or "sambay2" in model_name or "phi4miniflash" in model_name:
        mult = 14.5 * (ar ** 2) + 144 * ar
    if "sambayoco" in model_name or "sambayattn" in model_name:
        mult = 13.5 * (ar ** 2) + 208 * ar
    if "mambay" in model_name or "gdny" in model_name or "mambay2" in model_name:
        mult = 16 * (ar ** 2) + 64 * ar 
    if "sambaymlp" in model_name:
        mult = 15.5 * (ar ** 2) + 144 * ar      
    
    depth_global = depth
    
    # Calculate target parameters
    n_target = mult * (depth**3)
    if max_tokens is not None:
        train_tokens = max_tokens
    else:
        # Scale tokens based on parameter count (Chinchilla scaling)
        train_tokens = int(base_hps.t0 * n_target / base_hps.n0)
    
    # Scale learning rate and batch size using base hyperparameters
    raw_b = base_hps.b0
    multiple = nodes * devices * micro_batch_size * seq_len
    b = (raw_b // multiple) * multiple
    
    learning_rate = base_hps.eta0 * math.sqrt(b/base_hps.b0)
    if mup:
        if not original_mup:
            learning_rate = learning_rate * math.sqrt(base_hps.d0/depth)
            #learning_rate = learning_rate * (base_hps.t0/train_tokens)**(1/3)
    
    # Use base hyperparameters for other parameters
    weight_decay = base_hps.weight_decay0
    eps = base_hps.eps0
    min_lr = base_hps.min_lr0
    warmup_tokens = base_hps.warmup_tokens0
    beta1 = base_hps.beta1_0
    beta2 = base_hps.beta2_0
    muon_beta = base_hps.muon_beta_0
    weight_lr_scale = base_hps.weight_lr_scale_0
    
    # Apply independent weight decay if needed
    #weight_decay = weight_decay * base_hps.eta0 / learning_rate
    
    global_batch_size =  b // (seq_len * nodes)

        
    name = train_config +"_" + model_name+ "_ctx" + str(seq_len)
    if local_window is not None:
        name = name+ "_swa" + str(local_window)
    wandb_logger = WandbLogger(project="pretrain-llm", name=name)
    out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name
    

    if micro_batch_size == 1: # because eval batch size is ceil devided by 2
        eval_iters = total_evals // micro_batch_size // 2 # 50 # 25
    else:
        eval_iters = total_evals // micro_batch_size # 50 # 25

    batch_size = global_batch_size // devices
    gradient_accumulation_steps = batch_size // micro_batch_size
    assert gradient_accumulation_steps > 0

    log_iter_interval = log_step_interval * gradient_accumulation_steps
    # Include both local and global variables in hparams
    hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    # Add relevant global variables
    global_vars = {k: v for k, v in globals().items() 
                  if isinstance(v, (int, float, str)) and not k.startswith("_") 
                  and k not in hparams}
    hparams.update(global_vars)
    
    # Add base hyperparameters to logged hparams
    base_hps_dict = {f"base_hps.{k}": v for k, v in base_hps.__dict__.items()}
    hparams.update(base_hps_dict)

    if "muon" in train_config:     
        # todo: support tensor parallel
        # todo: currently muon_fsdp2 only works for transformer, need to support other archs
        strategy = ModelParallelStrategy(data_parallel_size=devices*nodes, tensor_parallel_size=1, parallelize_fn=configure_model)
    else:
        if fsdp_save_mem:
            strategy = FSDPStrategy(auto_wrap_policy={Block}, activation_checkpointing_policy={Block}, sharding_strategy = "HYBRID_SHARD", cpu_offload=True, state_dict_type="full")
            #strategy = FSDPStrategy(auto_wrap_policy={Block}, activation_checkpointing_policy={Block}, sharding_strategy = "HYBRID_SHARD", cpu_offload=False, state_dict_type="full")
        else:
            strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full",sharding_strategy = "HYBRID_SHARD")
    fabric = L.Fabric(devices=devices, num_nodes=nodes, strategy=strategy, precision="bf16-mixed", loggers=[wandb_logger])
    fabric.launch()
    fabric.print(hparams)

    overides = {"mup": mup, "mup_d0": base_hps.d0, 
            "mup_hd0": base_hps.hd0, "w_init0": base_hps.w_init0, "block_size": seq_len,
            "use_cu_seqlen": use_cu_seqlen,
            "original_mup": original_mup,
            "rope_base": rope_base,
            "eos_token_id": eos_token_id,
            }
    if swa_len is not None:
        overides["local_window"] = swa_len
    if "_tie" in train_config: # use tied embedding
        overides["tied_embed"] = True
    main(fabric, train_data_dir, val_data_dir, resume, fsdp_save_mem, **overides)


    
def convert_to_float8(model):
    # Check if we're running on H100
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:  # H100 has compute capability 9.0
        # float8_config = Float8LinearConfig(
        #     # pip install -U --index-url <https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/> triton-nightly  # noqa
        #     pad_inner_dim=True,
        # )

        def module_filter_fn(mod: torch.nn.Module, fqn: str):
            # don't convert linear modules with weight dimensions not divisible by 16
            if isinstance(mod, torch.nn.Linear):
                if "lm_head" in fqn:
                    return False
                if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    return False
                if mod.in_features >= 2048 and mod.out_features >= 4096: # only these situations fp8 is faster
                    return True
                
            return False
            # return True

        convert_to_float8_training(model, module_filter_fn=module_filter_fn) # config=float8_config,


def configure_model(model: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
    #convert_to_float8(model) # fp8 still buggy
    if "muon" in train_config:
        device_mesh = device_mesh["data_parallel"]
        
    for module in model.modules():
        if isinstance(module, Block):
            fully_shard(module, mesh=device_mesh)

    fully_shard(model, mesh=device_mesh)

    return torch.compile(model)


vector_names = ["conv1d", "wte", "lm_head", "norm", "ln", "layernorm"]

def get_param_groups(model):
    """Group parameters by their types to apply different learning rate multipliers"""
        
    no_decay= vector_names
    for n, p in model.named_parameters():
        print(n, p.shape)

    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not "weight" in n.lower() or any( nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": 0.0 if not original_mup else weight_decay,
            "lr_mult": 1.0,  # Base multiplier for no-decay parameters (vectors)
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "weight" in n.lower() and not any( nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr_mult": base_hps.d0 / depth_global,  # Base multiplier for weights
        }
    ]

    return param_groups

def main(fabric, train_data_dir, val_data_dir, resume, fsdp_save_mem, **overides):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name, **overides)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))
 

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")
    fabric.print(model)

    #convert_to_float8(model) # fp8 still buggy
    if not "muon" in train_config and not fsdp_save_mem:
        model = torch.compile(model) # comment this out for TP 
    model = fabric.setup(model)
    if mup:
        param_groups = get_param_groups(model)
        fabric.print(param_groups)
        
        if "hadam" in train_config:
            optimizer1 = torch.optim.AdamW(
                [param_groups[0]], lr=learning_rate, betas=(0.8, 0.95), eps=1e-10, fused=True
            )
            optimizer2 = torch.optim.NAdam(
                [param_groups[1]], lr=learning_rate * weight_lr_scale, betas=(0.9, 0.95), eps=1e-10, decoupled_weight_decay=True
            )  
            #optimizer2 = torch.optim.SGD([param_groups[1]], momentum=0.95, lr=learning_rate, nesterov=True, fused=True)
            optimizer = [fabric.setup_optimizers(optimizer1), fabric.setup_optimizers(optimizer2)]
        elif "muon" in train_config:
            optimizer1 = torch.optim.AdamW(
                [param_groups[0]], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True
            )
            optimizer2 = Muon_fsdp2([param_groups[1]], lr=learning_rate * weight_lr_scale, beta=muon_beta, ns_steps=5, muon_mode="old_large")
            optimizer = [fabric.setup_optimizers(optimizer1), fabric.setup_optimizers(optimizer2)]
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(beta1, beta2), eps=eps, fused=True)
            optimizer = fabric.setup_optimizers(optimizer)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
        )
        optimizer = fabric.setup_optimizers(optimizer)

    if ckpt_dir is not None:
        state = {"model": model}
        if os.path.exists(ckpt_dir):
            fabric.print(f"Loading ckpt from {ckpt_dir}")
            fabric.load(ckpt_dir, state)
            state.update({"optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}) # not load optimizer
            resume = False # not resume dataloader
        else:
            raise ValueError(f"No ckpt found in {ckpt_dir}")
    else:   
        state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}
        resume = find_resume_path(resume, out_dir)
        if resume :
            fabric.print(f"Resuming training from {resume}")
            fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume) # resume for dataloader
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def get_grad_norm(model):
    """Compute gradient norm for logging"""
    total_norm = 0.0
    param_norms = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            # Store individual parameter gradient norms
            param_norms[n] = param_norm
    total_norm = total_norm ** 0.5
    return total_norm, param_norms

def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        # sanity check
        validate(fabric, model, val_dataloader, state, monitor, sanity_check= False if "validation" in train_config else True)
        
    if "validation" in train_config:
        return
    
    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    train_tokens_per_device = train_tokens // fabric.world_size
    tokens_per_iter = micro_batch_size * model.config.block_size
    max_iters = train_tokens_per_device // tokens_per_iter
    max_steps = max_iters // gradient_accumulation_steps
    warmup_iters = warmup_tokens // fabric.world_size // tokens_per_iter
    initial_iter = state["iter_num"]
    curr_iter = 0
    if model.config.vocab_size > 100_000:
        loss_func = FusedLinearCrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        loss_func = FusedCrossEntropyLoss(label_smoothing=label_smoothing)
    
    for train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. 
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break
        
        # determine and set the learning rate for this iteration
        if type(optimizer) == list or type(optimizer) == tuple:
            lr = get_lr(state["iter_num"], warmup_iters, max_iters, learning_rate)
            for i, opt in enumerate(optimizer):
                if i == 0: #vectors
                    lr = get_lr(state["iter_num"], warmup_iters, max_iters, learning_rate) 
                else: #weights
                    lr = get_lr(state["iter_num"], warmup_iters, max_iters, learning_rate * weight_lr_scale) 
                for param_group in opt.param_groups:
                    param_group["lr"] = lr * param_group["lr_mult"] if "lr_mult" in param_group else lr
        else:
            lr = get_lr(state["iter_num"], warmup_iters, max_iters, learning_rate) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * param_group["lr_mult"] if "lr_mult" in param_group else lr


        iter_t0 = time.perf_counter()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            if model.config.vocab_size > 100_000:
                output = model(input_ids)
                loss = loss_func(output.logits, output.weight, targets)
            else:
                logits = model(input_ids).logits
                loss = loss_func(logits, targets)
            fabric.backward(loss / gradient_accumulation_steps)
        
        state["iter_num"] += 1
        if not is_accumulating:
            state["step_count"] += 1
        if state["iter_num"] % log_iter_interval == 0:
            # Log gradient norms before clipping
            grad_norm, param_norms = get_grad_norm(model)
            fabric.log_dict({
                "metric/grad_norm": grad_norm,
                "metric/grad_norm_clip_ratio": grad_norm / grad_clip
            }, state["step_count"])
            
            # Log individual parameter group gradient norms
            weight_grad_norm = 0.0
            vector_grad_norm = 0.0
            for n, norm in param_norms.items():
                if "weight" in n.lower() and not any(nd in n.lower() for nd in vector_names):
                    weight_grad_norm += norm ** 2
                else:
                    vector_grad_norm += norm ** 2
                    
            if type(optimizer) == list or type(optimizer) == tuple:
                weight_lr = optimizer[1].param_groups[0]["lr"]
            else:
                weight_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]
            fabric.log_dict({
                "metric/weight_grad_norm": weight_grad_norm ** 0.5,
                "metric/vector_grad_norm": vector_grad_norm ** 0.5,
                "metric/weight_learning_rate": weight_lr,
                "metric/vector_learning_rate": optimizer[0].param_groups[0]["lr"] if type(optimizer) == list or type(optimizer) == tuple \
                    else optimizer.param_groups[0]["lr"],
            }, state["step_count"])
        if not is_accumulating:
            if type(optimizer) == list or type(optimizer) == tuple:
                for opt in optimizer:
                    if grad_clip > 0:
                        fabric.clip_gradients(model, opt, max_norm=grad_clip)
                    opt.step()
                    opt.zero_grad()
            else:
                if grad_clip > 0:
                    fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        elif fabric.device.type == "xla":
            xm.mark_step()

        # input_id: B L 
        total_lengths += input_ids.size(1) * input_ids.size(0) // micro_batch_size
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" input_length: {input_ids.size(1)} total_length: {total_lengths} "
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item(),
            model = model,
        )

        if val_dataloader is not None and not is_accumulating and (state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_steps - 1):
            validate(fabric, model, val_dataloader, state, monitor)
        if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_steps - 1):
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, state: dict, monitor: Monitor, sanity_check=False) -> torch.Tensor:
    t0 = time.perf_counter()
    fabric.print("Validating ...")
    model.eval()
    global num_extrapol
    losses = torch.zeros(eval_iters, num_extrapol, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        
        extrapol_list = [(i + 1) * seq_len  for i in range(num_extrapol)]   
        for i, length in enumerate(extrapol_list): 
            input_ids = val_data[:, 0 : length].contiguous()
            targets = val_data[:, 1 : length + 1].contiguous()
            logits = model(input_ids).logits
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            losses[k,i] = loss.item()
        
        
    out = losses.mean(0)
    model.train()
    t1 = time.perf_counter() - t0
    monitor.eval_end(t1)
    if sanity_check:
        return out
    for i in range(num_extrapol):
        fabric.print(f"step {state['iter_num']}: val loss@{str(i+1)}x {out[i]:.4f}, val time: {t1 * 1000:.2f}ms")
        fabric.log_dict({"metric/val_loss@"+str(i+1)+"x": out[i].item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
        fabric.log_dict({"metric/val_ppl@"+str(i+1)+"x": math.exp(out[i].item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
    fabric.barrier()
    return out


def list_folders_with_bin_files(root_dir):
    folders_with_bin = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fname.endswith('.bin') for fname in filenames):
            folders_with_bin.append(dirpath)

    return folders_with_bin


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    folders = list_folders_with_bin_files(data_dir)
    print(folders)
    
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*.bin")))
        random.seed(seed)
        random.shuffle(filenames)
        use_large_chunk = "phi4miniflash" in model_name
        if split != "train":
            n_chunks = - (8 // -nodes) # ceil division
        else:
            n_chunks = 128 if use_large_chunk else 8
        dataset = PackedDataset(
            filenames,
            # n_chunks control the prefetch buffer size. increase n_chunks for better sample-level randomness
            n_chunks=n_chunks,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
            drop_last= False if use_large_chunk else True,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size= - (batch_size // -2), # ceil division
            block_size= num_extrapol * block_size + 1, # val 4* extrapolation
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader

def rampup_func(k, step_width = 16, max_len = 4096, warmup_step = 10000 ):
    # k: global step
    assert step_width * warmup_step >= max_len
    rampup_step = k // (warmup_step// (max_len // step_width) ) # (2*2k)/(64k/2k)
    if rampup_step < max_len//step_width:
        x = step_width  * (rampup_step + 1)
    else:
        x = max_len
    return x


# learning rate scheduler with warmup, stable period, and decay
def get_lr(it: int, warmup_iters: int, max_iters: int, lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return lr * it / warmup_iters
    
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    
    if "wsd" in train_config:
        # 3) stable period for 5/7 of training after warmup (deepseekv3 schedule) 
        # empirically, 5/7 is better than 9/10
        stable_iters = int(5/7 * (max_iters - warmup_iters))
        if it < warmup_iters + stable_iters:
            return lr
            
        # 4) decay period for remaining iterations
        decay_iters = max_iters - warmup_iters - stable_iters
        decay_ratio = (it - warmup_iters - stable_iters) / decay_iters
        assert 0 <= decay_ratio <= 1
    else:
        # 3) in between, use linear or cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    if "cosine" in train_config:
        # Cosine decay
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (lr - min_lr)
    else:
        # Linear decay
        return lr + decay_ratio * (min_lr - lr)



def find_resume_path(resume: Union[bool, Literal["auto"], Path], out_dir: Path) -> Optional[Path]:
    if not resume or isinstance(resume, Path):
        return resume

    resume_path = max(out_dir.rglob("iter-*"), key=(lambda p: int(p.name.split("-")[1])), default=None)
    if resume == "auto":
        return resume_path
    if resume is True and resume_path is None:
        raise FileNotFoundError(
            f"You passed `--resume=True`, but no checkpoint file was found in `--out_dir={out_dir}`."
        )
    return resume_path


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)

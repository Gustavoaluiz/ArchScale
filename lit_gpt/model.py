# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
from typing import Any, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from typing_extensions import Self
from lit_gpt.config import Config
from .fused_rotary_embedding import apply_rotary_emb_func
from torch import Tensor
from .mamba_simple import Mamba
from functools import partial
from einops import rearrange
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn
except:
    causal_conv1d_fn = None
    
from .diff_attn import FlashDiffAttention
from flash_attn.modules.mha import FlashCrossAttention
from flash_attn.bert_padding import pad_input, unpad_input
torch._dynamo.config.capture_scalar_outputs = True
unpad_input =torch.compiler.disable(unpad_input)
pad_input = torch.compiler.disable(pad_input)

from .mamba2 import Mamba2
from .gated_memory_unit import swiglu, GMUWrapper
from .gated_deltanet import GatedDeltaNet
import copy
from collections import namedtuple

CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "weight"], defaults=[None, None])
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]


def truncated_normal_(tensor, mean=0.0, std=0.02):
   
    tensor=torch.nn.init.trunc_normal_(tensor, mean, std, -2*std, 2*std)
   
    return tensor


def get_rnn(config: Config, layer_idx: int, gmu_save: bool = False, **factory_kwargs):
    # Create the appropriate RNN module based on rnn_type
    if config.rnn_type == "mamba":
        return Mamba(config.n_embd, layer_idx=layer_idx, gmu_save=gmu_save, config=config, **factory_kwargs)
    elif config.rnn_type == "mamba2":
        mamba2_expand = 8 * math.ceil(config.n_embd * 2 / 64 / 8) * 64 / config.n_embd 
        return Mamba2(config.n_embd, expand=mamba2_expand, layer_idx=layer_idx, gmu_save=gmu_save, config=config, **factory_kwargs)
    elif config.rnn_type == "gdn":
        return GatedDeltaNet(hidden_size=config.n_embd, num_heads=math.ceil(int(config.n_embd*0.75)/256), head_dim=256, mode='chunk', gmu_save=gmu_save, use_short_conv=True, allow_neg_eigval=True)
    else:
        raise ValueError(f"Unknown RNN type: {config.rnn_type}. Supported types: mamba, mamba2, gdn")


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.mup = config.mup
        
        if config.mup:
            self.logit_scale = config.mup_d0 / config.n_layer 
        else:
            self.logit_scale = 1.0
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        num_layer = config.n_layer 

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(num_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.max_len = self.config.block_size
        self.scale_embed = config.scale_embed
        self.tied_embed = config.tied_embed
        if self.tied_embed:
            self.tie_weights()

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Embedding):
            std = 1e-4 # RWKV init
            if self.tied_embed:
                std = 0.02
            if self.scale_embed:
                std = std /math.sqrt(self.config.n_embd)
            torch.nn.init.normal_(module.weight, std=std)
        elif isinstance(module, nn.Linear):
            if not self.mup:
                torch.nn.init.normal_(module.weight, std=0.02)
            #truncated_normal_(module.weight, mean=0.0, std=0.02)
            #nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            with torch.no_grad():
                module.weight *= self.config.w_init0 
                # if self.mup: # only needed for non kaiming init
                #     module.weight *=  math.sqrt(self.config.mup_d0/self.config.n_layer)
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias) 
        if self.mup and not self.tied_embed:
            for name, p in module.named_parameters():
                if "lm_head" in name:
                    torch.nn.init.zeros_(p)  # mup zero_readout trick
        else:
            # GPT-2 per-layer output projection intialization multiplier
            for name, p in module.named_parameters():
                if (name == "out_proj.weight") \
                    or (name == "o_proj.weight") \
                        or (name == "proj.weight" and isinstance(module, LLaMAMLP)) \
                        or name == "w3.weight" \
                        or (name=="proj.weight" and isinstance(module, CausalSelfAttention)):       
                        #if use xformer swiglu, fc2 layer will be renamed to w3       
                    if not self.config.mlp:
                        n_residuals_per_layer = 1  
                    else:
                        n_residuals_per_layer = 2
                    #nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)

    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight
        
    
    def reset_cache(self) -> None:
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        B, T = idx.size()
        if self.config.use_cu_seqlen:
            assert idx.size(0) == 1, "only support batch size 1 for variable length training"
            attn_mask = (idx.flatten()==self.config.eos_token_id)
            
        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size

        if not self.config.nope:
            if self.rope_cache is None:
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            elif T> self.max_len:
                self.max_len = T
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            cos, sin = self.rope_cache   

        if not self.config.nope:
            cos = cos[:T]
            sin = sin[:T]
        if self.config.nope:
            rope = None
        else:
            rope = (cos, sin)
        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.scale_embed:
            x = x * math.sqrt(self.config.n_embd)

        kv_cache = None
        gmu_mems = None
        for block in self.transformer.h:
            x, kv_cache, gmu_mems = block(x, rope, max_seq_length, attn_mask, kv_cache = kv_cache, gmu_mems = gmu_mems)

        x = self.transformer.ln_f(x.to(dtype=self.transformer.ln_f.weight.dtype))
        x = x * self.logit_scale
        if self.config.vocab_size > 100_000 and self.training:
            return CausalLMOutput(logits=x, weight=self.lm_head.weight) # (b, t, vocab_size)
        else:
            lm_logits = self.lm_head(x)
            return CausalLMOutput(logits=lm_logits) # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        return build_rope_cache(
            seq_len=seq_len,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            base = self.config.rope_base,
            condense_ratio=self.config.condense_ratio,
        )


    
class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        factory_kwargs = {"jamba_norm": config.jamba_norm, "device": "cuda", "dtype": torch.float32}

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        
        # Initialize flags
        self.use_rnn = False # use rnn for this layer
        self.rnn_type = config.rnn_type  # Store the actual RNN type being used
        self.use_gmu = False # use gmu for this layer
        self.yoco_kv = False # save kv for yoco
        self.gmu_save = False # save memory for gmu
        self.yoco_cross = False # use cross attention for yoco
        self.last_layer = layer_idx == config.n_layer - 1

        if config.yoco:
            config = copy.deepcopy(config)
            assert config.n_layer % 4 == 0, 'n_layer should be divisible by 4 for samba + yoco'
            if layer_idx < config.n_layer//2:
                self.use_rnn = config.rnn_per_layer > 0 and layer_idx % config.rnn_per_layer == 0
                self.use_full = False
            else:
                if config.gmu_yoco and not config.gmu_attn:
                    self.gmu_save = (layer_idx >= (config.n_layer//2))
                else:
                    self.gmu_save = False
                self.yoco_kv = (layer_idx >= (config.n_layer//2 +1))
                self.yoco_cross = (layer_idx >= (config.n_layer//2 +2))
                self.use_full = (layer_idx >= (config.n_layer//2 +1))
                if layer_idx == (config.n_layer//2):
                    self.use_rnn = config.rnn_per_layer > 0 
                if config.gmu_yoco and layer_idx >= (config.n_layer//2+2):
                    self.use_gmu = layer_idx % config.gmu_per_layer == 0
 
            if self.use_full:
                config.local_window = -1

        else: 
            if config.attn_layer_pos is not None:
                # For attn_layer_pos, RNN is used when NOT in the attention layer positions
                self.use_rnn = layer_idx not in eval(config.attn_layer_pos)
            else:
                self.use_rnn = config.rnn_per_layer > 0 and layer_idx % config.rnn_per_layer == 0
                
        ### token mixer
        mamba2_expand = 8 * math.ceil(config.n_embd * 2 / 64 / 8) * 64 / config.n_embd
        if self.use_gmu:
            if config.gmu_attn:
                gmu_inner = config.head_size * config.n_head
            elif config.gmu_mlp:
                gmu_inner = config.intermediate_size
            elif config.rnn_per_layer > 0 and config.rnn_type == "mamba2":
                gmu_inner = int(config.n_embd * mamba2_expand)
            elif config.rnn_per_layer > 0 and config.rnn_type == "gdn":
                gmu_inner = math.ceil(int(config.n_embd*0.75)/256)* 256 * 2
            else:
                gmu_inner = config.n_embd * 2
            use_norm = config.rnn_per_layer > 0 and (config.rnn_type == "mamba2" or config.rnn_type == "gdn")
            self.attn = GMUWrapper(config.n_embd, gmu_inner, bias=config.bias, use_norm=use_norm)
        elif self.use_rnn:
            self.attn = get_rnn(config, layer_idx, gmu_save=self.gmu_save, **factory_kwargs)
        else:
            self.attn = CausalSelfAttention(config, n_embd= config.n_embd, layer_idx= layer_idx, yoco_cross=self.yoco_cross)
            
        # mlp
        if config.mlp:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if config.mlp:
            config._mlp_class = "LLaMAMLP"
            self.mlp = config.mlp_class(config,)
                
        self.config = config
        
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        gmu_mems = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        ox = x
        ox = ox.to(torch.float32) # reduce error accumulation across layers during inference
        n_1 = self.norm_1(x.to(dtype=self.norm_1.weight.dtype))
        seq_idx = None
        if self.config.use_cu_seqlen:
            if self.use_rnn and self.rnn_type == "mamba2":
                new_seq_pos = F.pad(mask[:-1].to(torch.int32), (1, 0)).unsqueeze(0)
                seq_idx = torch.cumsum(new_seq_pos, dim=-1).to(torch.int32)
            else:
                def get_cu_seqlen(a,):
                    return torch.cat([torch.zeros(1).to(a.device).long(), 
                              (a).nonzero().flatten()+1, 
                              torch.tensor([a.shape[0]]).to(a.device).long() ],dim=-1)
                mask = get_cu_seqlen(mask)
                if self.use_rnn and self.rnn_type == "mamba":
                    seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, device=mask.device) 
                                for i, s in enumerate(mask[1:]-mask[:-1])], dim=0).unsqueeze(0)
        else:
            if self.use_rnn and self.rnn_type == "mamba2" and mask is not None:
                seq_idx  = mask.to(torch.int32)
 
        if self.use_rnn:
            if self.rnn_type in ["mamba", "mamba2"]:
                h, gmu_mems = self.attn(n_1, seq_idx=seq_idx, mask=mask, gmu_mems=gmu_mems)
                new_kv_cache = kv_cache
            elif self.rnn_type in ["gdn"]:
                cu_seqlens = mask if self.config.use_cu_seqlen else None
                attn_mask = None if self.config.use_cu_seqlen else mask
                h, gmu_mems = self.attn(n_1, attention_mask=attn_mask, cu_seqlens=cu_seqlens, gmu_mems=gmu_mems)
                new_kv_cache = kv_cache
            else: # self.rnn_type in ["retnet", "gla", "delta"]:
                h, _, new_kv_cache = self.attn(n_1)
        elif self.use_gmu:
            h, gmu_mems = self.attn(n_1, gmu_mems)
            new_kv_cache = kv_cache 
        else:
            # attention
            h, new_kv_cache, gmu_mems = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache, gmu_mems)
        if self.config.mup and not self.config.original_mup:
            h = h / math.sqrt(2 * self.config.n_layer)
        x = ox + h
        
        if self.config.mlp:
            ox = x
            n_2 = self.norm_2(x.to(dtype=self.norm_2.weight.dtype))
            h, gmu_mems = self.mlp(n_2, self.layer_idx,gmu_mems)

            if self.config.mup and not self.config.original_mup:
                h = h / math.sqrt(2 * self.config.n_layer)
            x = ox + h
        return x, new_kv_cache, gmu_mems


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int , n_embd: int, yoco_cross = False,) -> None:
        super().__init__()
        self.yoco_cross = yoco_cross
        self.local = layer_idx % config.full_per_layer < config.full_per_layer-1
            
        self.head_size = config.head_size
        self.n_head = config.n_head
        self.n_query_groups = config.n_query_groups
        if yoco_cross:
            shape = self.head_size * self.n_head
        else:
            shape = (self.n_head + 2 * self.n_query_groups) * self.head_size

        if config.add_sink:
            self.k_sink = nn.Parameter(torch.zeros(self.n_query_groups * self.head_size))
            self.v_sink = torch.zeros(self.n_query_groups * self.head_size)
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=config.attn_bias)
        # output projection
        self.config = config
        self.scale = config.attn_scale if config.attn_scale is not None else 1.0 / math.sqrt(self.head_size) 
        if self.config.mup:
            self.scale = self.scale * math.sqrt(self.config.mup_hd0)/ math.sqrt(self.head_size) 
        if self.local and self.config.local_window > -1:
            self.win_tuple = (self.config.local_window-1, 0)
        else:
            self.win_tuple = (-1,-1)
        self.use_cu_seqlen = config.use_cu_seqlen
        self.use_da = config.use_da
        if self.use_da:       
            depth = 10000 if config.da_const_lamb else layer_idx
            self.da = FlashDiffAttention(self.head_size, depth, causal=True, softmax_scale= self.scale, window_size = self.win_tuple)
        else:
            self.attn_func = FlashCrossAttention(causal=True, softmax_scale= self.scale, window_size = self.win_tuple)
        self.proj = nn.Linear(self.head_size * self.n_head, n_embd, bias=config.attn_out_bias)
        self.sc = config.sc_attn
        if self.sc:
            self.q_dim = self.n_head * self.head_size
            self.kv_dim = self.n_query_groups * self.head_size
            d_conv = 4
            self.q_conv1d = nn.Conv1d(
                in_channels=self.q_dim,
                out_channels=self.q_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.q_dim,
                padding=d_conv - 1,
            )
            self.k_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )
            self.v_conv1d = nn.Conv1d(
                in_channels= self.kv_dim,
                out_channels= self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups= self.kv_dim,
                padding=d_conv - 1,
            ) 

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        gmu_mems = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        if self.yoco_cross:
            q = self.attn(x)
            q = q.reshape(B,  T, -1, self.head_size) 
            if not self.config.nope and not self.config.yoco_nope:         
                cos, sin = rope
                # apply rope in fp32 significanly stabalize training
                # fused rope expect (batch_size, seqlen, nheads, headdim)
                q = apply_rotary_emb_func(q, cos, sin, False, True)       
             
            k, v = kv_cache
            y = self.scaled_dot_product_attention(q, k, v, attention_mask=mask)
            y = y.reshape(B, T, -1)  # re-assemble all head outputs side by side

            # output projection
            y = self.proj(y)
            return y, kv_cache, gmu_mems
        
        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B,  T, -1 )  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1 )  
        v = v.reshape(B,  T, -1 )  
        if self.sc:
            q = causal_conv1d_fn(
                        x = q.transpose(-1,-2),
                        weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                        bias=self.q_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            k = causal_conv1d_fn(
                        x = k.transpose(-1,-2),
                        weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                        bias=self.k_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            v = causal_conv1d_fn(
                        x = v.transpose(-1,-2),
                        weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                        bias=self.v_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2) 

        q = q.reshape(B,  T, -1, self.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.head_size)  
        v = v.reshape(B,  T, -1, self.head_size)

        if not self.config.nope and not (self.config.yoco_nope and self.win_tuple == (-1,-1)):         
            cos, sin = rope
            # apply rope in fp32 significanly stabalize training
            # fused rope expect (batch_size, seqlen, nheads, headdim)
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)

        if self.config.add_sink:
            k = torch.cat([self.k_sink.to(k).reshape(1,  1, -1, self.head_size).repeat(B,1,1,1), k], dim=1)
            v = torch.cat([self.v_sink.to(v).reshape(1,  1, -1, self.head_size).repeat(B,1,1,1), v], dim=1)
        kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, attention_mask=mask)

        y = y.reshape(B, T, -1)  # re-assemble all head outputs side by side
        if self.config.gmu_attn:
            gmu_mems = y
        # output projection
        y = self.proj(y)
        return y, kv_cache, gmu_mems

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = None, None, None, None
        if not self.use_da:
            kv = torch.stack([k, v], dim=2)
        if attention_mask is not None:
            if self.use_cu_seqlen:
                if self.use_da:
                    k = k.reshape(-1, k.shape[-2], k.shape[-1])
                    v = v.reshape(-1, v.shape[-2], v.shape[-1])
                else:
                    ckv = kv.reshape(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])
                cu_seqlens_k = cu_seqlens_q = attention_mask.int()
                max_seqlen_q = max_seqlen_k = (attention_mask - attention_mask.roll(1)).max().cpu().item()
                q = q.reshape(-1, q.shape[-2], q.shape[-1])
            else:
                if self.use_da:
                    k, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, attention_mask.to(k.device))
                    v, _, _, _, _ = unpad_input(v, attention_mask.to(v.device))
                else:
                    ckv, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(kv, attention_mask.to(kv.device))

                if seqlen_q == 1:
                    attention_mask = torch.ones(batch_size, 1, device=q.device)
                elif seqlen_q != seqlen_k:
                    attention_mask = attention_mask[:, -seqlen_q:]

                q, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, attention_mask.to(q.device))
        else:
            if not self.use_da:
                ckv = kv

        if self.config.full_swa_extend and self.win_tuple == (-1,-1) and not self.training:
            wintuple = (self.config.block_size -1, 0)
            if self.use_da:
                self.da.window_size = wintuple
            else:   
                self.attn_func.window_size = wintuple
        if self.use_da:
            attn_output =self.da(q, k, v, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q, 
                                    cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k,)
        else:
            attn_output =self.attn_func(q, ckv, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q, 
                                    cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k,)
        if self.use_cu_seqlen:
            attn_output = attn_output.reshape(batch_size, seqlen_q, q.shape[-2], q.shape[-1])
        else:
            attn_output = (
                pad_input(attn_output, indices_q, batch_size, max_seqlen_q)
                if attention_mask is not None
                else attn_output
            )
        return attn_output

    
class LLaMAMLP(nn.Module):
    def __init__(self, config: Config,) -> None:
        super().__init__()
        self.relu2 = config.mlp_relu2
        self.config = config
        self.legacy_swiglu = False
        if self.relu2:
            self.w1 = nn.Linear(config.n_embd, int(config.intermediate_size*1.5), bias=config.bias)
            self.w3 = nn.Linear(int(config.intermediate_size*1.5), config.n_embd, bias=config.bias)
        else:
            if self.legacy_swiglu:
                self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False)
            else:
                self.w1 = nn.Linear(config.n_embd, int(config.intermediate_size*2), bias=config.bias)
                self.w3 = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)


    def forward(self, x: torch.Tensor, layer_idx: int, gmu_mems: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.relu2:
            x = self.w1(x)
            x = F.relu(x).square()
            x = self.w3(x)
        else:
            # SwiGLU implementation: split input into two parts, apply SwiGLU activation
            if self.legacy_swiglu:
                x = self.swiglu(x)
            else:
                x_gate, x_inp = self.w1(x).chunk(2, dim=-1)
                if self.config.gmu_mlp and layer_idx == self.config.n_layer//2+1:
                    gmu_mems = x_inp
                x = swiglu(x_gate, x_inp)
                x = self.w3(x)
        return x, gmu_mems

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, bias=False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)
        # from xformers.ops import SwiGLU
        # if config.no_mlp_bias:
        #     self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False) 
        # else:
        #     self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=config.bias, _pack_weights=False)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x1) * x2
        x = self.w3(x)
        return x

    
def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


    
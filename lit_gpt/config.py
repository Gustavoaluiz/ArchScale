# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import lit_gpt.model
from lit_gpt.utils import find_multiple


@dataclass
class Config:
    org: str = "Microsoft"
    name: str = "transformer_d8"
    block_size: int = 4096 # sequence length for training 
    vocab_size: int = 50254 # vocab size for training
    padding_multiple: int = 512 # vocab size should be at least muliple of 8 to be efficient on hardware. compute the closest value
    padded_vocab_size: Optional[int] = None # vocab size after padding. will overide padding_multiple
    n_layer: int = 16 # number of layers
    n_head: int = 32 # number of attention heads
    n_embd: int = 4096 # embedding dimension
    ar: int = None # Aspect ratio: n_embed = ar * n_layer
    mlp_expand: int = 4 # MLP expand ratio: intermediate_size = mlp_expand * n_embd
    rotary_percentage: float = 1.0 # percentage of rotary embedding
    bias: bool = False # use bias for linear layers
    attn_bias: bool = False # use bias for attention qkv linear layers
    attn_out_bias: bool = False # use bias for attention output linear layers
    local_window: int = -1 # window size for sliding window attention
    mlp: bool = True # use MLP
    full_per_layer: int = 1000000 # use full attention at the end of every x layers
    rnn_per_layer: int = -1  # use rnn at the beginning of every x layers
    rnn_type: str = "mamba"  # Options: "mamba", "mamba2", "retnet", "gla", "delta", "gdn"
    nope: bool = False # not use position embedding
    sc_attn: bool = False # use short convolution with attention
    rms_norm: bool= True # use RMSNorm
    attn_layer_pos: str = None # For attn_layer_pos, RNN is used when NOT in the attention layer positions
    scale_embed: bool = False # scale embedding with 1/sqrt(n_embd)
    use_cu_seqlen: bool = False # use cu_seqlen for variable length training
    eos_token_id: int = 2 # llama2 token id for eos
    mlp_relu2: bool = False # use relu^2 activation in MLP
    head_dim: int = None # head dimension. will overide head_size
    jamba_norm: bool = False # use Jamba-style normalization in mamba layer
    add_sink: bool = False # add one attention sink kv every layer
    rope_base: int = 10000  # base frequency for rope
    use_da: bool = False # use differential attention
    full_swa_extend: bool = False # extrapolate the full attention with swa
    yoco: bool = False # use YOCO: you only cache once decoder-decoder architecture
    gmu_yoco: bool = False # use deocder-hybrid-decoder architecture with GMU
    gmu_per_layer: int = 2 # use GMU every x layers
    gmu_attn: bool = False # use GMU in attention
    gmu_mlp: bool = False # use GMU in MLP
    attn_scale: float = None # attention logits multiplier
    w_init0: float = 1.0 # weight initialization multiplier
    da_const_lamb: bool = False # use constant lambda for differential attention
    no_mlp_bias: bool = False # not use bias in MLP
    use_sigmoid: bool = False # use sigmoid attention
    yoco_nope: bool = False # use YOCO with nope full attention and rope swa
    mup_d0: int = 16 # base depth for muP++
    mup: bool = False # use muP++
    mup_hd0: int = 128 # base head dimension for muP++
    sp_init: bool = False # use Standard Parametrization init
    mup_tie: bool = False # mup++ with tied embedding
    original_mup: bool = False  # original muP with only width scaling
    n_query_groups: Optional[int] = None # equal to n_kv_heads
    _norm_class: Literal["LayerNorm", "RMSNorm","FusedRMSNorm"] = "FusedRMSNorm"
    norm_eps: float = 1e-5 # epsilon for normalization
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "LLaMAMLP"
    intermediate_size: Optional[int] = None # intermediate size for MLP
    condense_ratio: int = 1 # condense ratio for rope

    def __post_init__(self):
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # default intermediate size for MLP if not set
        if self.intermediate_size is None:
            self.intermediate_size = self.mlp_expand * self.n_embd
        
        if self.ar is not None:
            self.n_embd = self.ar * self.n_layer
            self.intermediate_size = self.mlp_expand * self.n_embd
            
        # error checking
        assert self.n_embd % self.n_head == 0

    @property
    def head_size(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        else:
            return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(lit_gpt.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm
            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)


configs=[]

phi4_mini_flash_configs = [
    dict(
        org="Microsoft",
        name="phi4miniflash_d32", 
        block_size=8192,
        vocab_size=200064,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        attn_bias = True,
        attn_out_bias = True,
        n_layer=32,
        n_head=40,
        use_da = True,
        head_dim=64,
        ar = 80,
        sp_init = True,
        _norm_class = "LayerNorm",
        n_query_groups= 20, 
        mlp_expand= 4, 
        local_window = 512, 
    )
]
configs.extend(phi4_mini_flash_configs)


scaling_xformer_configs = [
        dict(
        org="Microsoft",
        name="transformer_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= d//4, 
        mlp_expand= 4, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformer_configs)

scaling_xformerls_configs = [
        dict(
        org="Microsoft",
        name="transformerls_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        full_per_layer = 4,
        n_layer= d,
        n_head= d,
        ar = 128,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_xformerls_configs)

scaling_samba_configs = [
    dict(
        org="Microsoft",
        name="samba_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 122,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 2048, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_samba_configs)

abaltion_configs = [
    dict(
        org="Microsoft",
        name="mambay_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=1,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 120,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="mambay2_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=1,
        rnn_type="mamba2",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 120,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="gdny_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=1,
        rnn_type="gdn",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 120,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sgdny_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="gdn",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 126,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sambay2_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba2",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 124,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sambayattn_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        gmu_attn = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 126,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sambayattnall_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        gmu_attn = True,
        gmu_per_layer = 1,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 126,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
    dict(
        org="Microsoft",
        name="sambaymlp_d16", 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        gmu_mlp = True,
        nope = True,
        n_layer=16,
        n_head=16,
        head_dim=128,
        ar = 120,
        n_query_groups= 16//4, 
        mlp_expand= 4, 
        local_window = 128, 
    ),
]
configs.extend(abaltion_configs)

scaling_sambay_configs = [
    dict(
        org="Microsoft",
        name="sambay_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 124,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambay_configs)

scaling_sambay_da_configs = [
    dict(
        org="Microsoft",
        name="sambayda_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = True,
        nope = True,
        n_layer=d,
        n_head=d,
        use_da = True,
        head_dim=128,
        ar = 124,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambay_da_configs)

scaling_sambayoco_configs = [
    dict(
        org="Microsoft",
        name="sambayoco_d"+str(d), 
        block_size=4096,
        vocab_size=32000,
        padding_multiple=64,
        rnn_per_layer=2,
        rnn_type="mamba",
        yoco = True,
        gmu_yoco = False,
        nope = True,
        n_layer=d,
        n_head=d,
        head_dim=128,
        ar = 126,
        n_query_groups= d//4, 
        mlp_expand= 4, 
        local_window = 128, 
    )
    for d in [8,12,16,20,24]
]
configs.extend(scaling_sambayoco_configs)

name_to_config = {config["name"]: config for config in configs}

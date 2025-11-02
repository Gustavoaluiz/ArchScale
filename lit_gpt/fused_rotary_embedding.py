# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2023, Tri Dao.

from flash_attn.ops.triton.rotary import apply_rotary
import torch

class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        # x: (B, T, H, D); cos/sin: (T_ro, D_ro/2)
        _, seqlen, _, _ = x.shape
        cos2 = cos[:seqlen]
        sin2 = sin[:seqlen]
        if cos2.ndim == 3 and cos2.shape[1] == 1:
            cos2 = cos2.squeeze(1)
        if sin2.ndim == 3 and sin2.shape[1] == 1:
            sin2 = sin2.squeeze(1)
        out = apply_rotary(
            x,
            cos2.contiguous(),
            sin2.contiguous(),
            interleaved=interleaved,
            inplace=inplace,
            conjugate=False,
        )
        ctx.save_for_backward(cos2, sin2)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out

    @staticmethod
    def backward(ctx, do):
        cos2, sin2 = ctx.saved_tensors
        _, seqlen, _, _ = do.shape
        cos2 = cos2[:seqlen].contiguous()
        sin2 = sin2[:seqlen].contiguous()
        dx = apply_rotary(
            do,
            cos2,
            sin2,
            interleaved=ctx.interleaved,
            inplace=False,
            conjugate=True,
        )
        return dx, None, None, None, None


apply_rotary_emb_func = ApplyRotaryEmb.apply


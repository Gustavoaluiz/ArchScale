"""Fallback implementation of rotary embedding utilities.

This module provides a compatible interface for the legacy
``rotary_emb`` extension that used to ship with flash-attention.
The current flash-attention wheels no longer expose the compiled
``apply_rotary`` function, but :mod:`lit_gpt.fused_rotary_embedding`
still imports it.  To keep the data preparation and training scripts
working out of the box, we re-implement the required routine using
PyTorch tensor operations.
"""

from __future__ import annotations

import torch


def _broadcast_trig(trig: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast ``trig`` to match ``target``'s shape.

    The legacy kernel expected ``trig`` to have shape ``(seqlen, 1, dim)``;
    by unsqueezing the leading dimension we obtain ``(1, seqlen, 1, dim)``
    which broadcasts against ``(batch, seqlen, nheads, dim)``.
    """

    # Ensure the tensor lives on the same device/dtype as the activations.
    trig = trig.to(device=target.device, dtype=target.dtype)
    if trig.ndim == target.ndim - 1:
        trig = trig.unsqueeze(0)
    return trig


def apply_rotary(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conjugate: bool = False,
) -> None:
    """Apply rotary position embeddings.

    Parameters
    ----------
    x1, x2:
        Tensors containing the split rotary dimensions. They are expected to
        have shape ``(batch, seqlen, nheads, rotary_dim // 2)``.
    cos, sin:
        Trigonometric caches with shape ``(seqlen, 1, rotary_dim // 2)``.
    out1, out2:
        Destination tensors with the same shape as ``x1``/``x2``.
    conjugate:
        When ``True`` the gradients are accumulated using the conjugate
        rotation, matching the behaviour of the original CUDA kernel.
    """

    cos = _broadcast_trig(cos, x1)
    sin = _broadcast_trig(sin, x1)

    if conjugate:
        rot1 = x1 * cos + x2 * sin
        rot2 = x2 * cos - x1 * sin
    else:
        rot1 = x1 * cos - x2 * sin
        rot2 = x2 * cos + x1 * sin

    out1.copy_(rot1)
    out2.copy_(rot2)

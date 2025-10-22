#!/usr/bin/env python3
"""Utility to match Transformer++ width to a target parameter count.

The script is tailored for comparing the pure Transformer family against
SambaY or other hybrid models.  It can either read the target parameter
count from an existing configuration (by instantiating the model) or use a
value provided on the command line.  Afterwards it scans a range of aspect
ratios (``ar``) and reports the value that minimises the absolute difference
in parameter counts.

Example (match Transformer++ depth 8 against SambaY depth 8 with tied
embeddings):

```
python scripts/match_transformer_params.py \
    --target-model sambay --target-depth 8 \
    --base-model transformer --base-depth 8 \
    --tie-embed
```

The result prints the recommended ``ar`` value and a ready-to-use snippet for
``pretrain.py`` via the new ``--transformer_ar`` CLI flag.
"""

from __future__ import annotations

import argparse
from typing import Optional

from lit_gpt.config import Config
from lit_gpt.model import GPT
from lit_gpt.utils import num_parameters


def transformer_param_count(config: Config) -> int:
    """Analytically count parameters for the pure Transformer++ configs."""
    if config.rnn_per_layer > 0 or config.yoco or config.use_da:
        raise ValueError(
            "Parameter estimation only supports the Transformer++ blocks; "
            "received a config that enables recurrent or hybrid layers."
        )

    n_layer = config.n_layer
    n_embd = config.n_embd
    n_head = config.n_head
    head_dim = config.head_size
    n_query_groups = config.n_query_groups
    intermediate = config.intermediate_size

    # Attention projections (qkv + output)
    attn_projs = n_embd * head_dim * (n_head + 2 * n_query_groups)
    attn_out = n_embd * n_embd

    # SwiGLU MLP has two input projections (gate + value) and one output
    mlp_params = 3 * n_embd * intermediate

    # Layer norms: one always-on norm and an additional norm when MLP is used
    norms_per_layer = n_embd * (2 if config.mlp else 1)

    block_params = attn_projs + attn_out + mlp_params + norms_per_layer

    # Token embedding + optional tied output head + final norm
    embed_params = config.padded_vocab_size * n_embd
    head_params = 0 if config.tied_embed else embed_params
    final_norm = n_embd

    return n_layer * block_params + embed_params + head_params + final_norm


def count_params_from_name(name: str) -> int:
    """Instantiate a model configuration and return its parameter count."""
    config = Config.from_name(name)
    model = GPT(config)
    if hasattr(model, "to_empty"):
        model.to_empty(device="meta")  # release storage pressure
    return num_parameters(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="sambay", help="Architecture name to match (without _d depth suffix).")
    parser.add_argument("--target-depth", type=int, default=8, help="Depth of the target architecture.")
    parser.add_argument(
        "--target-params",
        type=int,
        default=None,
        help="Manually provide the target parameter count instead of instantiating the model.",
    )
    parser.add_argument("--base-model", default="transformer", help="Base model family to adjust (without _d suffix).")
    parser.add_argument("--base-depth", type=int, default=8, help="Depth of the base architecture.")
    parser.add_argument("--min-ar", type=int, default=80, help="Minimum aspect ratio (inclusive) to explore.")
    parser.add_argument("--max-ar", type=int, default=160, help="Maximum aspect ratio (inclusive) to explore.")
    parser.add_argument(
        "--tie-embed",
        action="store_true",
        help="Assume tied input/output embeddings for the base Transformer.",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print the full scan table instead of only the best candidate.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    target_name = f"{args.target_model}_d{args.target_depth}"
    if args.target_params is None:
        target_params = count_params_from_name(target_name)
    else:
        target_params = args.target_params

    base_kwargs: dict[str, object] = {"tied_embed": args.tie_embed}
    base_name = f"{args.base_model}_d{args.base_depth}"

    best: Optional[tuple[int, int, int]] = None  # (ar, n_embd, params)
    rows: list[tuple[int, int, int, int]] = []  # (ar, n_embd, params, diff)

    for ar in range(args.min_ar, args.max_ar + 1):
        config = Config.from_name(base_name, ar=ar, **base_kwargs)
        params = transformer_param_count(config)
        diff = abs(params - target_params)
        rows.append((ar, config.n_embd, params, diff))
        if best is None or diff < best[2]:
            best = (ar, config.n_embd, params)

    assert best is not None

    if args.print_table:
        header = "ar  | n_embd | params (M) | diff (M)"
        print(header)
        print("-" * len(header))
        for ar, n_embd, params, diff in rows:
            print(
                f"{ar:3d} | {n_embd:6d} | {params/1e6:9.3f} | {diff/1e6:8.3f}"
            )
        print()

    ar, n_embd, params = best
    diff = params - target_params

    print(f"Target {target_name}: {target_params/1e6:.3f}M parameters")
    print(
        "Best Transformer++ width: ar={ar} (n_embd={n_embd}) -> {params/1e6:.3f}M parameters"
        .format(ar=ar, n_embd=n_embd, params=params)
    )
    print(f"Difference: {diff/1e6:+.3f}M parameters")

    tied_suffix = "_tie" if args.tie_embed else ""
    suggested_train_name = f"scaling_mup{tied_suffix}" if args.tie_embed else "scaling_mup"
    print()
    print("Use the following CLI overrides with pretrain.py:")
    print(
        "  --train_model {base} --depth {depth} --train_name {train_name} --transformer_ar {ar}".format(
            base=args.base_model,
            depth=args.base_depth,
            train_name=suggested_train_name,
            ar=ar,
        )
    )


if __name__ == "__main__":
    main(parse_args())

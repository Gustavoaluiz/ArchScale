# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

from importlib import import_module
from lightning_utilities.core.imports import RequirementCache

if not bool(RequirementCache("torch>=2.1.0dev")):
    raise ImportError(
        "Lit-GPT requires torch nightly (future torch 2.1). Please follow the installation instructions in the"
        " repository README.md"
    )
_LIGHTNING_AVAILABLE = RequirementCache("lightning>=2.1.0.dev0")
if not bool(_LIGHTNING_AVAILABLE):
    raise ImportError(
        "Lit-GPT requires Lightning nightly (future lightning 2.1). Please run:\n"
        f" pip uninstall -y lightning; pip install -r requirements.txt\n{str(_LIGHTNING_AVAILABLE)}"
    )


__all__ = ["GPT", "Config", "Tokenizer", "FusedCrossEntropyLoss"]


def __getattr__(name):
    if name == "GPT":
        return import_module("lit_gpt.model").GPT
    if name == "Config":
        return import_module("lit_gpt.config").Config
    if name == "Tokenizer":
        return import_module("lit_gpt.tokenizer").Tokenizer
    if name == "FusedCrossEntropyLoss":
        return import_module("lit_gpt.fused_cross_entropy").FusedCrossEntropyLoss
    raise AttributeError(f"module 'lit_gpt' has no attribute {name!r}")

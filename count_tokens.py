from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List


HDR_MAGIC = b"LITPKDS"

"""
Example
python count_tokens.py /home/jovyan/persist/datasets/slimpajama_packed \
  --glob_pattern "train_slimpajama_*_*.bin"

python count_tokens.py /home/jovyan/persist/datasets/slimpajama_packed \
  --glob_pattern "validation_slimpajama_*_*.bin"
"""

def read_chunk_size(path: Path) -> int:
    with open(path, "rb") as f:
        assert f.read(len(HDR_MAGIC)) == HDR_MAGIC
        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1
        f.read(1)  # dtype code
        chunk_size = struct.unpack("<Q", f.read(8))[0]
    return chunk_size


def count_tokens(
    destination_path: Path,
    glob_pattern: str = "*.bin",
    verbose: bool = True,
) -> Dict[str, int]:
    bins: List[Path] = sorted(destination_path.rglob(glob_pattern))
    if not bins:
        if verbose:
            print("arquivos: 0")
            print("tokens: 0")
        return {"arquivos": 0, "tokens": 0}

    chunk_sizes = [read_chunk_size(p) for p in bins]
    total_tokens = sum(chunk_sizes)

    if verbose:
        unique_sizes = sorted(set(chunk_sizes))
        print(f"arquivos: {len(bins)}")
        if len(unique_sizes) == 1:
            print(f"chunk_size: {unique_sizes[0]:,}")
        else:
            print(f"chunk_size (múltiplos): {', '.join(map(str, unique_sizes))}")
        print(f"tokens: {total_tokens:,}")

    return {"arquivos": len(bins), "tokens": total_tokens}


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(count_tokens)
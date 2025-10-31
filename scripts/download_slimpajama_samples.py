from __future__ import annotations
from pathlib import Path
import random
from huggingface_hub import list_repo_files, snapshot_download

REPO = "cerebras/SlimPajama-627B"
OUT = Path(r"/home/jovyan/persist/datasets/slimpajama/raw")  
BUDGET = {
    "train/chunk1/": 200,  # 2.2 B tokens
    # você pode distribuir para reduzir viés:
    # "train/chunk1/": 70, "train/chunk2/": 70, "train/chunk3/": 60,
    "validation/chunk1/": "all",  # 27M tokens
}
RANDOM_SEED = 42
MAX_WORKERS = 6  # paralelismo do download

def main():
    files = list_repo_files(REPO, repo_type="dataset")
    random.seed(RANDOM_SEED)

    allow = []
    for prefix, k in BUDGET.items():
        if k in ("all", -1, None):
            allow.append(f"{prefix}*.jsonl.zst")  # pega tudo do chunk
        else:
            cand = sorted(f for f in files if f.startswith(prefix) and f.endswith(".jsonl.zst"))
            if not cand:
                print(f"[warn] sem arquivos em {prefix}")
                continue
            take = random.sample(cand, k=min(k, len(cand)))
            allow.extend(take)

    if not allow:
        raise SystemExit("Nada a baixar (verifique prefixes em BUDGET).")

    print(f"N Arquivos a baixar: {len(allow)}\n Exemplos: {allow[:3]} ... {allow[-3:]}")
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=allow,           
        local_dir=str(OUT),
        local_dir_use_symlinks=False,   
        max_workers=MAX_WORKERS,               
    )

if __name__ == "__main__":
    main()

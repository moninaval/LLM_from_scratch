import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# your project imports
from src.data.text_dataset import TextDataset
from src.tokenizer.tokenizer_manager import TokenizerManager


def infer_dtype(vocab_size: int):
    # pick compact dtype for ids (uint16 if <= 65535 else uint32)
    if vocab_size <= 0xFFFF:
        return np.uint16, "uint16"
    elif vocab_size <= 0xFFFFFFFF:
        return np.uint32, "uint32"
    else:
        raise ValueError(f"vocab_size {vocab_size} too large for uint32; use uint64 if needed.")


def write_memmap(out_path: str, seq_len: int, dtype_np, dataset, indices, meta_base: dict):
    """Write sequences into a contiguous memmap in the order of `indices`."""
    N = len(indices)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mm = np.memmap(out_path, dtype=dtype_np, mode="w+", shape=(N, seq_len))

    # write in chunks to avoid holding too much in RAM
    # (we still index the dataset one by one to keep it simple)
    for row, idx in enumerate(indices):
        x = dataset[idx]  # Tensor of shape [seq_len], dtype long
        arr = x.cpu().numpy().astype(dtype_np, copy=False)
        if arr.shape[0] != seq_len:
            raise ValueError(f"Sequence at idx {idx} has length {arr.shape[0]} != seq_len={seq_len}")
        mm[row, :] = arr
        if (row + 1) % 10000 == 0:
            print(f"  wrote {row + 1}/{N} sequences...")
    mm.flush()

    meta = dict(meta_base)
    meta.update({
        "n_sequences": int(N),
        "dtype": str(dtype_np.__name__),
        "output_file": out_path,
        "created_at": datetime.now().isoformat(timespec="seconds")
    })
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Wrote {N} sequences to {out_path}")
    print(f"📄 Meta: {meta_path}")


def maybe_write_jsonl(out_path_jsonl: str, dataset, indices, key: str = "input_ids"):
    """Optional JSONL dump in the same permuted order (slower, for inspection)."""
    os.makedirs(os.path.dirname(out_path_jsonl) or ".", exist_ok=True)
    with open(out_path_jsonl, "w", encoding="utf-8") as f:
        for i, idx in enumerate(indices):
            ids = dataset[idx].tolist()
            f.write(json.dumps({key: ids}) + "\n")
            if (i + 1) % 10000 == 0:
                print(f"  jsonl wrote {i + 1}/{len(indices)}")
    print(f"📝 JSONL written: {out_path_jsonl}")


def main():
    ap = argparse.ArgumentParser(description="Tokenize raw text and pack into contiguous .bin (+ optional train/val split)")
    ap.add_argument("--model", required=True, help="Model name to load config from (configs/<model>.json)")
    ap.add_argument("--val_ratio", type=float, default=0.0, help="Fraction for validation split (e.g., 0.01 for 1%)")
    ap.add_argument("--shuffle", action="store_true", help="Pre-shuffle sequences once before writing")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.add_argument("--emit_jsonl", action="store_true", help="Also emit JSONL in the same permuted order")
    ap.add_argument("--out_prefix", type=str, default=None, help="Override output prefix (e.g., data/corpus_seq2048)")
    args = ap.parse_args()

    # Load config
    cfg_path = os.path.join("configs", f"{args.model}.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Resolve paths and knobs
    raw_path = config.get("raw_data_path", None) 
    if not raw_path:
        raise ValueError("Config must include 'data_path' (raw text) or 'raw_data_path'.")

    tokenizer_path = config.get("tokenizer_path")
    tokenizer_type = config.get("tokenizer_type", "huggingface")
    seq_len = int(config.get("max_position_embeddings", 2048))
    vocab_size = int(config.get("vocab_size", 65536))  # used to select dtype

    if not tokenizer_path:
        raise ValueError("Config must include 'tokenizer_path' (existing tokenizer).")

    # Build tokenizer
    tokenizer_config = {
        "model_type": config.get("model_type", "generic"),
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": tokenizer_type,
        "max_position_embeddings": seq_len
    }
    tokenizer = TokenizerManager(tokenizer_config).tokenizer

    # Build dataset from raw text → fixed-length sequences
    # This uses your TextDataset packing; it should:
    #  - insert EOS between docs
    #  - pack to fixed seq_len
    #  - yield tensors of shape [seq_len]
    print(f"📚 Building dataset from: {raw_path}")
    ds = TextDataset(
        file_path=raw_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        cache_path=None,          # no side JSONL unless emit_jsonl
        pretokenized=False,
        force_rebuild=True
    )

    N = len(ds)
    if N == 0:
        raise ValueError("No sequences produced. Check your raw data and tokenizer settings.")
    print(f"✅ Total sequences: {N} (seq_len={seq_len})")

    # Pre-shuffle once (recommended) so file is randomized; otherwise write in natural order
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        perm = np.arange(N)
        rng.shuffle(perm)
        print(f"🔀 Pre-shuffled with seed={args.seed}")
    else:
        perm = np.arange(N)

    # Optional train/val split
    val_ratio = max(0.0, min(0.5, float(args.val_ratio)))
    split = int(math.floor(N * (1.0 - val_ratio)))
    train_idx = perm[:split]
    val_idx = perm[split:] if val_ratio > 0.0 else np.array([], dtype=int)

    # Decide output prefix
    base = args.out_prefix
    if base is None:
        base_name = os.path.splitext(os.path.basename(raw_path))[0]
        base = f"data/{base_name}_seq{seq_len}"

    # Choose dtype from vocab size
    dtype_np, dtype_name = infer_dtype(vocab_size)

    # Common meta
    meta_base = {
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": tokenizer_type,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "pre_shuffled": bool(args.shuffle),
        "shuffle_seed": int(args.seed) if args.shuffle else None,
        "val_ratio": val_ratio
    }

    # Write train bin
    train_out = f"{base}.train.bin" if val_ratio > 0 else f"{base}.bin"
    print(f"💾 Writing train memmap: {train_out}  ({len(train_idx)} sequences, {dtype_name})")
    write_memmap(train_out, seq_len, dtype_np, ds, train_idx, meta_base)

    # Write optional val bin
    if val_ratio > 0.0 and len(val_idx) > 0:
        val_out = f"{base}.val.bin"
        print(f"💾 Writing val memmap:   {val_out}  ({len(val_idx)} sequences, {dtype_name})")
        write_memmap(val_out, seq_len, dtype_np, ds, val_idx, meta_base)

    # Optional JSONL (slower; for inspection/debug)
    if args.emit_jsonl:
        jsonl_out = f"{base}.train.jsonl" if val_ratio > 0 else f"{base}.jsonl"
        print(f"📝 Also emitting JSONL: {jsonl_out}")
        maybe_write_jsonl(jsonl_out, ds, train_idx)
        if val_ratio > 0 and len(val_idx) > 0:
            maybe_write_jsonl(f"{base}.val.jsonl", ds, val_idx)

    print("🎯 Done. Point your training config's data_path to the new .bin file(s).")


if __name__ == "__main__":
    main()

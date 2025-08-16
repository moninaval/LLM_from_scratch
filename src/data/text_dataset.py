# src/data/text_dataset.py
import os
import json
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Two main uses:

    1) Preprocessing (raw -> sequences):
       TextDataset(file_path=RAW_TXT, tokenizer=tok, seq_len=L,
                   pretokenized=False, add_eos=True, cache_path=OPTIONAL_JSONL,
                   stride=None or int, force_rebuild=True/False)
       - Reads raw text line-by-line
       - Tokenizes each line (no special tokens)
       - Inserts EOS between lines/docs (if add_eos and EOS available)
       - Packs ACROSS lines into fixed windows of length seq_len
       - If cache_path is given, writes a JSONL cache and reads lazily via offsets;
         else stores sequences in RAM (OK for tiny corpora)

    2) Training (pretokenized -> memmap/jsonl):
       TextDataset(file_path=BIN_OR_JSONL, tokenizer=None, seq_len=L,
                   pretokenized=True)
       - If .bin: uses numpy memmap with shape (N, L) + side .meta.json
       - If .jsonl: lazy read via byte offsets

    __getitem__ always returns torch.long of shape [seq_len].
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int,
        cache_path: Optional[str] = None,
        pretokenized: bool = False,
        jsonl_key: str = "input_ids",
        add_eos: bool = True,
        eos_id: Optional[int] = None,
        stride: Optional[int] = None,         # None => non-overlap; else e.g. 64 for overlap
        drop_remainder: bool = True,
        force_rebuild: bool = False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.jsonl_key = jsonl_key
        self.drop_remainder = bool(drop_remainder)
        self.stride = int(stride) if stride is not None else None

        # Resolve EOS id (if available)
        self.eos_id = eos_id
        if self.eos_id is None and add_eos and hasattr(tokenizer, "eos_token_id"):
            self.eos_id = tokenizer.eos_token_id

        ext = os.path.splitext(file_path)[1].lower()

        if pretokenized:
            # ---- Pretokenized paths: .bin (memmap) or .jsonl (offset-indexed) ----
            if ext == ".bin":
                self._mode = "memmap"
                self._init_memmap(file_path)
            elif ext == ".jsonl":
                self._mode = "jsonl"
                self._jsonl_path = file_path
                self._offsets = self._index_jsonl(self._jsonl_path)
            else:
                raise ValueError(f"Pretokenized=True expects .bin or .jsonl, got {file_path}")
        else:
            # ---- RAW mode: build sequences from text; JSONL cache optional ----
            if cache_path:
                if force_rebuild or (not os.path.exists(cache_path)):
                    self._build_jsonl_cache_from_raw(
                        raw_path=file_path,
                        tokenizer=tokenizer,
                        cache_path=cache_path,
                        add_eos=add_eos,
                    )
                self._mode = "jsonl"
                self._jsonl_path = cache_path
                self._offsets = self._index_jsonl(cache_path)
            else:
                self._mode = "ram"
                self._samples: List[torch.Tensor] = []
                self._build_in_ram_from_raw(
                    raw_path=file_path,
                    tokenizer=tokenizer,
                    add_eos=add_eos,
                )

    # -------------------- RAW -> RAM builder --------------------
    def _build_in_ram_from_raw(self, raw_path: str, tokenizer, add_eos: bool):
        buf: List[int] = []
        eos = int(self.eos_id) if (add_eos and self.eos_id is not None) else None
        step = self.seq_len if self.stride is None else max(1, self.stride)

        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if text:
                    buf.extend(self._tokenize_line(text, tokenizer))
                    if eos is not None:
                        buf.append(eos)
                else:
                    # blank line → treat as doc boundary
                    if eos is not None:
                        buf.append(eos)

                while len(buf) >= self.seq_len:
                    chunk = buf[:self.seq_len]
                    self._samples.append(torch.tensor(chunk, dtype=torch.long))
                    del buf[:step]

        if (not self.drop_remainder) and buf:
            pad = buf + [0] * (self.seq_len - len(buf))
            self._samples.append(torch.tensor(pad[:self.seq_len], dtype=torch.long))

    # -------------------- RAW -> JSONL cache builder --------------------
    def _build_jsonl_cache_from_raw(self, raw_path: str, tokenizer, cache_path: str, add_eos: bool):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        buf: List[int] = []
        eos = int(self.eos_id) if (add_eos and self.eos_id is not None) else None
        step = self.seq_len if self.stride is None else max(1, self.stride)

        print(f"🛠 Tokenizing and caching to {cache_path}")
        with open(raw_path, "r", encoding="utf-8") as fin, open(cache_path, "w", encoding="utf-8") as fout:
            for line in fin:
                text = line.rstrip("\n")
                if text:
                    buf.extend(self._tokenize_line(text, tokenizer))
                    if eos is not None:
                        buf.append(eos)
                else:
                    if eos is not None:
                        buf.append(eos)

                while len(buf) >= self.seq_len:
                    chunk = buf[:self.seq_len]
                    fout.write(json.dumps({self.jsonl_key: chunk}) + "\n")
                    del buf[:step]

            if (not self.drop_remainder) and buf:
                pad = buf + [0] * (self.seq_len - len(buf))
                fout.write(json.dumps({self.jsonl_key: pad[:self.seq_len]}) + "\n")

    # -------------------- JSONL lazy index --------------------
    def _index_jsonl(self, jsonl_path: str) -> List[int]:
        offsets: List[int] = []
        with open(jsonl_path, "rb") as f:
            pos = f.tell()
            line = f.readline()
            while line:
                offsets.append(pos)
                pos = f.tell()
                line = f.readline()
        print(f"📄 JSONL indexed: {jsonl_path} ({len(offsets)} samples)")
        return offsets

    # -------------------- Memmap .bin + .meta.json --------------------
    def _init_memmap(self, bin_path: str):
        meta_path = bin_path + ".meta.json"
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing meta file: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        nseq = int(meta["n_sequences"])
        seq_len_meta = int(meta["seq_len"])
        if seq_len_meta != self.seq_len:
            raise ValueError(f"seq_len mismatch: meta={seq_len_meta}, dataset={self.seq_len}")

        dtype_name = meta.get("dtype", "uint16")
        dtype_map = {
            "uint16": np.uint16,
            "uint32": np.uint32,
            "int64": np.int64,
            "int32": np.int32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported dtype in meta: {dtype_name}")

        dtype_np = dtype_map[dtype_name]
        self._mm = np.memmap(bin_path, dtype=dtype_np, mode="r", shape=(nseq, self.seq_len))
        self._length = nseq
        print(f"📦 Memmap loaded: {bin_path} [{nseq}×{self.seq_len}] dtype={dtype_name}")

    # -------------------- helpers --------------------
    def _tokenize_line(self, text: str, tokenizer):
    # Prefer a direct encode() if available (works for HF + SentencePiece)
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            ids = tokenizer.encode(text, add_special_tokens=False)
        else:
        # Fallback to __call__ returning a BatchEncoding/dict
            out = tokenizer(text, add_special_tokens=False)
            if isinstance(out, dict):
                ids = out.get("input_ids")
            elif hasattr(out, "input_ids"):
                ids = out.input_ids
            else:
                raise TypeError("Tokenizer output has no 'input_ids' field")

    # Flatten if tokenizer returned [[...]] for a single example
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]

        if ids is None:
            raise ValueError("Tokenizer returned None for input_ids")

        return [int(t) for t in ids]

    # -------------------- Dataset API --------------------
    def __len__(self) -> int:
        if self._mode == "ram":
            return len(self._samples)
        elif self._mode == "jsonl":
            return len(self._offsets)
        elif self._mode == "memmap":
            return self._length
        else:
            raise RuntimeError("Dataset not initialized properly.")

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._mode == "ram":
            return self._samples[idx]

        if self._mode == "jsonl":
            with open(self._jsonl_path, "rb") as f:
                f.seek(self._offsets[idx])
                line = f.readline().decode("utf-8")
            obj = json.loads(line)
            ids = obj.get(self.jsonl_key, None)
            if ids is None or len(ids) != self.seq_len:
                raise ValueError(f"Bad JSONL record at idx={idx}")
            return torch.as_tensor(ids, dtype=torch.long)

        if self._mode == "memmap":
            arr = self._mm[idx]  # view [L]
            return torch.as_tensor(arr, dtype=torch.long)

        raise RuntimeError("Unknown dataset mode.")


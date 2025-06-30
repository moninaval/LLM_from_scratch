# src/data/text_dataset.py

import os
import json
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len, cache_path=None, pretokenized=False):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []
        self.pretokenized = pretokenized

        if pretokenized:
            self.cache_path = file_path  # direct use of pretokenized file (.jsonl or .bin)
            self._load_from_cache()
        else:
            if cache_path is None:
                cache_path = file_path + f".cached.{seq_len}.jsonl"
            self.cache_path = cache_path

            if os.path.exists(self.cache_path):
                self._load_from_cache()
            else:
                self._build_and_cache(file_path)

    def _build_and_cache(self, file_path):
        print(f"🛠 Tokenizing and caching dataset to {self.cache_path}")
        with open(file_path, 'r', encoding='utf-8') as f_in, open(self.cache_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                token_ids = self.tokenizer.encode(line, truncation=False, add_special_tokens=False)
                for i in range(0, len(token_ids) - self.seq_len + 1, self.seq_len):
                    chunk = token_ids[i:i + self.seq_len]
                    json.dump({"input_ids": chunk}, f_out)
                    f_out.write("\n")
                    self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def _load_from_cache(self):
        print(f"✅ Loading dataset from cache: {self.cache_path}")
        ext = os.path.splitext(self.cache_path)[1]
        if ext == ".jsonl":
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if "input_ids" in data:
                        chunk = data["input_ids"]
                        if len(chunk) == self.seq_len:
                            self.samples.append(torch.tensor(chunk, dtype=torch.long))
        elif ext == ".bin":
            print("📦 Loading binary tokenized dataset")
            with open(self.cache_path, 'rb') as f:
                try:
                    while True:
                        chunk = torch.load(f)
                        if chunk.shape[0] == self.seq_len:
                            self.samples.append(chunk)
                except EOFError:
                    pass
        else:
            raise ValueError(f"Unsupported cache format: {ext}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

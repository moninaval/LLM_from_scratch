# src/tokenizer/tokenizer_manager.py

import os
from typing import List

class TokenizerManager:
    def __init__(self, config):
        self.model_type = config["model_type"]
        self.tokenizer_path = config["tokenizer_path"]
        self.tokenizer_type = config.get("tokenizer_type", "huggingface")  # default to huggingface
        self.max_length = config.get("max_position_embeddings", 2048)

        if self.tokenizer_type == "tiktoken":
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.backend = "tiktoken"

        elif self.tokenizer_type == "sentencepiece":
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(os.path.join(self.tokenizer_path, "tokenizer.model"))
            self.backend = "sentencepiece"

        elif self.tokenizer_type == "custom":
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(os.path.join(self.tokenizer_path, "tokenizer.json"))
            self.backend = "custom"

        else:  # huggingface
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.backend = "huggingface"

    def encode(self, text: str) -> List[int]:
        if self.backend == "tiktoken":
            return self.tokenizer.encode(text)
        elif self.backend == "sentencepiece":
            return self.sp.encode(text, out_type=int)
        elif self.backend == "custom":
            return self.tokenizer.encode(text).ids
        else:
            return self.tokenizer.encode(text, truncation=True, max_length=self.max_length)

    def decode(self, token_ids: List[int]) -> str:
        if self.backend == "tiktoken":
            return self.tokenizer.decode(token_ids)
        elif self.backend == "sentencepiece":
            return self.sp.decode(token_ids)
        elif self.backend == "custom":
            return self.tokenizer.decode(token_ids)
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def batch_encode(self, texts: List[str]):
        if self.backend == "huggingface":
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
        else:
            return [self.encode(text) for text in texts]

    def get_vocab_size(self):
        if self.backend == "sentencepiece":
            return self.sp.get_piece_size()
        elif self.backend == "custom":
            return self.tokenizer.get_vocab_size()
        return self.tokenizer.vocab_size

    def save_tokenizer(self, path):
        if self.backend == "huggingface":
            self.tokenizer.save_pretrained(path)
        elif self.backend == "custom":
            self.tokenizer.save(path)

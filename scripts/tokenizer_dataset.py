# scripts/tokenize_dataset.py

import argparse
import torch
import json
import os
from datetime import datetime
from src.data.text_dataset import TextDataset
from src.tokenizer.tokenizer_manager import TokenizerManager

def main():
    parser = argparse.ArgumentParser(description="Tokenize raw text into .jsonl or .bin chunks")
    parser.add_argument("--model", type=str, required=True, help="Model name to load config from (e.g., phi-2)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join("configs", f"{args.model}.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_path = config.get("data_path")
    tokenizer_path = config.get("tokenizer_path")
    tokenizer_type = config.get("tokenizer_type", "huggingface")
    seq_len = config.get("max_position_embeddings", 2048)
    save_bin = config.get("save_bin", False)

    if not input_path or not tokenizer_path:
        raise ValueError("Config must include 'data_path' and 'tokenizer_path'")

    # Auto-generate unique output path
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_jsonl = f"data/{base_name}.cached.{seq_len}.{timestamp}.jsonl"

    # Load tokenizer
    tokenizer_config = {
        "model_type": config.get("model_type", "generic"),
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": tokenizer_type,
        "max_position_embeddings": seq_len
    }
    tokenizer = TokenizerManager(tokenizer_config).tokenizer

    # Tokenize and build dataset
    dataset = TextDataset(
        file_path=input_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        cache_path=output_jsonl,
        pretokenized=False,
        force_rebuild=True
    )

    if save_bin:
        bin_path = output_jsonl.replace(".jsonl", ".bin")
        print(f"💾 Saving tokenized dataset to binary file: {bin_path}")
        with open(bin_path, "wb") as f:
            for sample in dataset:
                torch.save(sample, f)

    # Save metadata
    meta_path = output_jsonl.replace(".jsonl", ".meta.json")
    meta = {
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": tokenizer_type,
        "seq_len": seq_len,
        "input_file": input_path,
        "output_file": output_jsonl,
        "save_bin": save_bin,
        "num_samples": len(dataset)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Tokenization complete. Total samples: {len(dataset)}")
    print(f"📄 JSONL: {output_jsonl}")
    print(f"📄 Metadata: {meta_path}")

if __name__ == "__main__":
    main()

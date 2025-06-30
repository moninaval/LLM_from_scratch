# scripts/tokenize_dataset.py

import argparse
import torch
from src.data.text_dataset import TextDataset
from src.tokenizer.tokenizer_manager import TokenizerManager


def main():
    parser = argparse.ArgumentParser(description="Tokenize raw text into .jsonl or .bin chunks")
    parser.add_argument("--input", type=str, required=True, help="Path to raw .txt file")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer name or path")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--output", type=str, help="Optional custom cache output path")
    parser.add_argument("--save_bin", action="store_true", help="Also save tokenized samples as .bin")

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = TokenizerManager(args.tokenizer_path).tokenizer

    # Tokenize and build dataset (will save to jsonl cache)
    dataset = TextDataset(
        file_path=args.input,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        cache_path=args.output,
        pretokenized=False
    )

    if args.save_bin:
        bin_path = (args.output or args.input + f".cached.{args.seq_len}.bin")
        print(f"💾 Saving tokenized dataset to binary file: {bin_path}")
        with open(bin_path, 'wb') as f:
            for sample in dataset:
                torch.save(sample, f)

        # Save metadata
    meta_path = (args.output or args.input + f".cached.{args.seq_len}.meta.json")
    meta = {
        "tokenizer_path": args.tokenizer_path,
        "seq_len": args.seq_len,
        "input_file": args.input,
        "output_file": args.output or args.input + f".cached.{args.seq_len}.jsonl",
        "save_bin": args.save_bin,
        "num_samples": len(dataset)
    }
    with open(meta_path, 'w', encoding='utf-8') as meta_file:
        import json
        json.dump(meta, meta_file, indent=2)

    print(f"✅ Tokenization complete. Total samples: {len(dataset)}")
    print(f"📄 Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()

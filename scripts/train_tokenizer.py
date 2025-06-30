import argparse
import json
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

def train_bpe_from_config(config):
    corpus_path = config["data_path"]
    vocab_size = config.get("vocab_size", 30000)
    save_dir = config["tokenizer_path"]

    # Setup tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )

    # Read corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    tokenizer.train_from_iterator(lines, trainer=trainer)

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"✅ Tokenizer saved to {save_dir}/tokenizer.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to load config from (e.g., my-llm)")
    args = parser.parse_args()

    config_path = os.path.join("configs", f"{args.model}.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    train_bpe_from_config(config)

if __name__ == "__main__":
    main()

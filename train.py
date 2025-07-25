import argparse
import json
import os
import glob
import torch
from torch.utils.data import DataLoader, ConcatDataset
from src.tokenizer.tokenizer_manager import TokenizerManager
from src.data.text_dataset import TextDataset
from src.model.llm_model import LLMModel

def load_seen_files(seen_path):
    if os.path.exists(seen_path):
        with open(seen_path, 'r') as f:
            return set(json.load(f).get("trained_files", []))
    return set()

def save_seen_files(seen_path, seen_files):
    with open(seen_path, 'w') as f:
        json.dump({"trained_files": sorted(list(seen_files))}, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Train LLM on .jsonl datasets")
    parser.add_argument("--model", type=str, required=True, help="Model name to load config from (e.g., phi-2)")
    parser.add_argument("--only_new", action="store_true", help="Only train on unseen .jsonl files")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join("configs", f"{args.model}.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    tokenizer = TokenizerManager(config).tokenizer
    seq_len = config["max_position_embeddings"]
    batch_size = config.get("batch_size", 2)
    epochs = config.get("epochs", 1)

    # Seen files tracking
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    seen_path = os.path.join(checkpoint_dir, "seen_datasets.json")
    seen_files = load_seen_files(seen_path)

    all_jsonl = sorted(glob.glob("data/*.jsonl"))
    jsonl_files = [f for f in all_jsonl if f not in seen_files] if args.only_new else all_jsonl

    if not jsonl_files:
        print("✅ No new .jsonl files to train on.")
        return

    print(f"📂 Training on: {len(jsonl_files)} file(s)\n" + "\n".join(jsonl_files))

    # Build and combine datasets
    datasets = [
        TextDataset(file_path=f, tokenizer=tokenizer, seq_len=seq_len, pretokenized=True)
        for f in jsonl_files
    ]
    full_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Init model and optimizer
    model = LLMModel(**config)
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 3e-4))

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch.to(device)
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Step {step} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f"✅ Epoch {epoch+1} Summary → Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity.item():.2f}")
        # Save model checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")

    # Update seen files
    seen_files.update(jsonl_files)
    save_seen_files(seen_path, seen_files)
    print(f"📜 Seen file list updated: {seen_path}")

if __name__ == "__main__":
    main()

from src.tokenizer.tokenizer_manager import TokenizerManager
import json

# Load config file
with open("configs/phi-2.json") as f:
    config = json.load(f)

tokenizer = TokenizerManager(config)

test_text = "Transformers are amazing!"
token_ids = tokenizer.encode(test_text)
decoded_text = tokenizer.decode(token_ids)

print("🧾 Original Text:", test_text)
print("🔢 Token IDs:", token_ids)
print("📜 Decoded Text:", decoded_text)
print("🔠 Vocab Size:", tokenizer.get_vocab_size())

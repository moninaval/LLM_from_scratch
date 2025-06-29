# test/test_llm_model.py

import json
import torch
from src.model.llm_model import LLMModel
from src.tokenizer.tokenizer_manager import TokenizerManager

# Load config from phi-2.json
with open("configs/phi-2.json") as f:
    config = json.load(f)

# Create model from config
model = LLMModel(
    vocab_size=config["vocab_size"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    intermediate_size=config["intermediate_size"],
    max_position_embeddings=config["max_position_embeddings"],
    dropout=config.get("dropout", 0.0),
    use_rotary=config.get("use_rotary", False),
    rotary_dim=config.get("rotary_dim", None),
    norm_type=config.get("norm_type", "layernorm"),
    ffn_type=config.get("ffn_type", "gelu"),
    qkv_proj=config.get("qkv_proj", "fused"),
    tie_weights=config.get("tie_weights", False),
    debug=True
)

# Create dummy input from vocab size
batch_size = 2
seq_len = 16
input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

# Forward pass
model.eval()
with torch.no_grad():
    output = model(input_ids)

print("✅ LLMModel test passed. Output shape:", output.shape)

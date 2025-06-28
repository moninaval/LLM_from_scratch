# src/model/llm_model.py

import torch
import torch.nn as nn
from src.model.embedding import TokenEmbedding
from src.model.transformer_stack import TransformerStack
from src.model.norm import Norm
from src.util.debug_util import DebugLogger

class LLMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings=2048, dropout=0.0, use_rotary=False, norm_type="layernorm", debug=False):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            use_rotary=use_rotary
        )

        self.transformer = TransformerStack(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            use_rotary=use_rotary,
            debug=debug,
            norm_type=norm_type
        )

        self.final_norm = Norm(hidden_size, norm_type)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.logger = DebugLogger(enabled=debug, prefix="[LLMModel]")

    def forward(self, input_ids, mask=None):
        self.logger.log(f"Input IDs shape: {input_ids.shape}")

        x = self.token_embedding(input_ids)
        self.logger.log(f"After TokenEmbedding: {x.shape}")

        x = self.transformer(x, mask)
        self.logger.log(f"After TransformerStack: {x.shape}")

        x = self.final_norm(x)
        self.logger.log(f"After Final Norm: {x.shape}")

        logits = self.lm_head(x)
        self.logger.log(f"Logits shape: {logits.shape}")

        return logits

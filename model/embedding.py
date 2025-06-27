# src/model/embedding.py
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=2048, use_rotary=False):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.use_rotary = use_rotary

        if not self.use_rotary:
            self.pos_embed = nn.Embedding(max_position_embeddings, hidden_size)
        else:
            self.pos_embed = None  # rotary will be applied in attention later

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        token_embeddings = self.token_embed(input_ids)

        if self.use_rotary:
            return token_embeddings  # pos info will be added during attention
        else:
            # Generate positions for each token in the batch
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            pos_embeddings = self.pos_embed(positions)
            return token_embeddings + pos_embeddings
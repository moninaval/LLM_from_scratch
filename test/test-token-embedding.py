import torch
from src.model.embedding import TokenEmbedding

def test_embedding(use_rotary):
    print(f"\n🔁 Testing TokenEmbedding (use_rotary={use_rotary})")
    
    vocab_size = 100
    hidden_size = 16
    max_position_embeddings = 10
    batch_size = 2
    seq_len = 5

    # Create input IDs (random integers)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Create embedding module
    embedding = TokenEmbedding(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        use_rotary=use_rotary
    )

    # Forward pass
    output = embedding(input_ids)

    # Print results
    print("Input IDs:\n", input_ids)
    print("Output Shape:", output.shape)
    print("Output Sample (batch 0, token 0):", output[0, 0])

if __name__ == "__main__":
    test_embedding(use_rotary=False)
    test_embedding(use_rotary=True)
    print("\n✅ All tests passed!")

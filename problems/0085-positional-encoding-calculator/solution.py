import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    # Your code here

    # pe = torch.zeros((position, d_model), dtype=torch.float32)
    # pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)
    # base_term = 1 / (
    #     10000 ** (
    #         torch.arange(0, d_model, 2, dtype=torch.float32)
    #         / d_model
    #     )
    # )
    # freqs = torch.einsum('i,j->ij', pos, base_term)
    # emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
    # cos = emb.cos()[:, :]
    # sin = emb.sin()[:, :]
    # pe[:, 0::2] = sin
    # pe[:, 1::2] = cos
    
    # return pe.to(torch.float16)

    # 1. Handle invalid inputs
    if position <= 0 or d_model <= 0:
        return -1

    # 2. Initialize positional encoding matrix
    pe = torch.zeros(position, d_model, dtype=torch.float32)

    # 3. Generate positions: (position, 1)
    pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)

    # 4. Calculate frequency scaling factors
    div_term = 1 / (
        10000 ** (
            torch.arange(0, d_model, 2, dtype=torch.float32)
            / d_model
        )
    )

    # 5. Apply sine to even dimensions
    pe[:, 0::2] = torch.sin(pos * div_term)

    # 6. Apply cosine to odd dimensions
    pe[:, 1::2] = torch.cos(
        pos * div_term[:d_model // 2]
    )

    # 7. Convert to float16
    return pe.to(torch.float16)

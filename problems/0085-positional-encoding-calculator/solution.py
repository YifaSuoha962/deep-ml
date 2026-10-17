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
    if position == 0 or d_model <= 0:
        return -1

    pe = torch.zeros((position, d_model), dtype=torch.float32)

    # (d_model // 2,)
    div_term = 1 / (
        10000 ** (
            torch.arange(0, d_model, 2, dtype=torch.float32)
            / d_model
        )
    )

    # frequencies: (position, d_model // 2)
    # explicit out product
    pos = torch.arange(position, dtype=torch.float32)   
    freqs = torch.einsum('i,j->ij', pos, div_term)
    """
    # broadcast version.
    pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)
    freqs = pos * div_term  
    """
    
    # 偶数维度用 sin，奇数维度用 cos
    pe[:, 0::2] = torch.sin(freqs)    # 位置是对上的，特征维度穿插
    pe[:, 1::2] = torch.cos(freqs)

    return pe.to(torch.float16)
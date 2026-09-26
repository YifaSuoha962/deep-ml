import numpy as np

def unet_time_embedding(timesteps, embed_dim, W1, b1, W2, b2, max_period=10000):
    """
    Compute time embeddings for a diffusion model U-Net.
    
    Args:
        timesteps: list or 1D array of shape (B,) with timestep values
        embed_dim: dimension of sinusoidal embedding (must be even)
        W1: weight matrix of first linear layer, shape (embed_dim, hidden_dim)
        b1: bias of first linear layer, shape (hidden_dim,)
        W2: weight matrix of second linear layer, shape (hidden_dim, output_dim)
        b2: bias of second linear layer, shape (output_dim,)
        max_period: controls the frequency range for sinusoidal embedding
    
    Returns:
        numpy array of shape (B, output_dim) with time embeddings
    """
    # 转换为 numpy 数组 (B,)
    t = np.asarray(timesteps, dtype=np.float32)
    if t.ndim == 0:
        t = t[None]  # 如果是标量，转为 (1,)
    
    half_dim = embed_dim // 2
    # 频率: exp(-ln(max_period) * i / half_dim)
    i = np.arange(half_dim, dtype=np.float32)
    freqs = np.exp(-np.log(max_period) * i / half_dim)  # (half_dim,)
    
    # args = t * freqs_i  -> (B, half_dim)
    args = t[:, None] * freqs[None, :]
    
    # 正弦嵌入: [sin(args), cos(args)]
    sin_emb = np.sin(args)
    cos_emb = np.cos(args)
    emb = np.concatenate([sin_emb, cos_emb], axis=1)  # (B, embed_dim)
    
    # MLP: Linear -> SiLU -> Linear
    h = emb @ W1 + b1  # (B, hidden_dim)
    # SiLU (Swish): x * sigmoid(x)
    h = h * (1.0 / (1.0 + np.exp(-h)))
    out = h @ W2 + b2  # (B, output_dim)
    
    return out
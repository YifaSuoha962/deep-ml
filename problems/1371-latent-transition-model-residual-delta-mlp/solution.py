import numpy as np

def gelu(x):
    # tanh approximation of GELU
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def layer_norm(z, eps):
    # Layer normalization over the last axis, biased variance, no learnable scale/shift
    mu = np.mean(z, axis=-1, keepdims=True)
    var = np.var(z, axis=-1, keepdims=True)  # biased variance (ddof=0)
    return (z - mu) / np.sqrt(var + eps)

def latent_transition(h, x_emb, W1, b1, W2, b2, W3, b3, eps=1e-5):
    """
    Predict the next hidden state as a residual delta.

    Args:
        h:     (D,) or (T, D) current hidden state(s)
        x_emb: (D,) or (T, D) embedding(s) of the next token
        W1, b1: (2D, H) and (H,)
        W2, b2: (H, H) and (H,)
        W3, b3: (H, D) and (D,)
        eps: layer-norm epsilon

    Returns:
        array shaped like h: the predicted next hidden state
    """
    # 1. Concatenate along the last axis → width 2D
    z = np.concatenate([h, x_emb], axis=-1)
    # 2. Layer-normalize over the last axis
    z_norm = layer_norm(z, eps)
    # 3. Three-layer MLP with GELU
    a1 = gelu(z_norm @ W1 + b1)
    a2 = gelu(a1 @ W2 + b2)
    delta = a2 @ W3 + b3
    # 4. Residual connection
    return delta + h
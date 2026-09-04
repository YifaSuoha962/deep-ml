import torch
import math

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    d_k = torch.tensor(Q.shape[-1])
    # (seq_len, seq_len)
    scores = (Q @ K.T) / math.sqrt(d_k)

    # 每个 query 对所有 key 的概率
    attn_weights = torch.softmax(scores, dim=-1)

    # 对所有 value 加权求和，不是哈达玛积
    output = attn_weights @ V
    return output

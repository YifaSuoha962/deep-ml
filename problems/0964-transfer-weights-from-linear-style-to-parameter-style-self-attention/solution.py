import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Transfer linear-style weights (shape (d_out, d_in)) into parameter-style
    weights (shape (d_in, d_out)) and compute single-head causal self-attention.

    Args:
        X: input array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: linear-style weight matrices of shape (d_out, d_in)

    Returns:
        Nested list of shape (seq_len, d_out): the self-attention output.
    """
    # Your code here
    # 1. Convert inputs to NumPy arrays
    X = np.asarray(X, dtype=float)
    W_q_lin = np.asarray(W_q_lin, dtype=float)
    W_k_lin = np.asarray(W_k_lin, dtype=float)
    W_v_lin = np.asarray(W_v_lin, dtype=float)

    # 2. Transfer weights: (d_out, d_in) -> (d_in, d_out) and Compute Q, K, V
    Q = X @ W_q_lin.T
    K = X @ W_k_lin.T
    V = X @ W_v_lin.T

    # 3. Scaled dot-product attention
    d_out = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_out)

    # 5. Causal mask: prevent attending to future tokens
    """
    triu - 上三角矩阵, tril - 下三角矩阵
    k表示有效区域相对对角线的偏移位置： 0-从对角线本身算起, 1-从对象线上面的1行1列区域开始, -1 -从对象线下面的1行1列区域开始
    """
    # 上三角写法
    # mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    # scores[mask] = -1e9
    # 下三角写法
    mask = np.tril(np.ones(scores.shape, dtype=bool), k=0)
    scores[~mask] = -1e9

    """
    延伸练习：
    def banded_diagonal_mask(n: int, k: int) -> np.ndarray:
        
        构建带状对角掩码矩阵。
        
        参数:
            n: 矩阵大小 (n x n)
            k: 对角线两侧的斜线条数（带宽）
        
        返回:
            (n, n) 的整数矩阵，满足 |i - j| <= k 的位置为 1，其余为 0
        
        # 创建行索引和列索引
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        # 判断绝对值差是否在 k 以内
        mask = (np.abs(i - j) <= k).astype(np.int8)
        return mask
    """

    # 6. Numerically stable softmax over keys
    """ keepdims=True 才能保证逐位计算 """
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 7. Weighted sum of values
    output = attn @ V

    return output.tolist()



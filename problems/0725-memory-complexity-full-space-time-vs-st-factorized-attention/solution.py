def attention_memory(T: int, H: int, W: int, num_heads: int, bytes_per_element: int) -> list:
    """
    Compute memory (in bytes) of attention score tensors for full space-time
    attention and for space-time factorized attention.

    Args:
        T: number of temporal frames
        H: spatial height in tokens
        W: spatial width in tokens
        num_heads: number of attention heads
        bytes_per_element: bytes per stored score value (e.g. 4 for fp32, 2 for fp16)

    Returns:
        [full_bytes, factorized_bytes]
    """
    S = H * W
    N = T * S

    # 全时空注意力：num_heads 个 N × N 矩阵。
    full_bytes = num_heads * N * N * bytes_per_element

    # 空间注意力：每帧一个 S × S 矩阵。
    spatial_scores = T * S * S

    # 时间注意力：每个空间位置一个 T × T 矩阵。
    temporal_scores = S * T * T

    factorized_bytes = (
        num_heads * (spatial_scores + temporal_scores) * bytes_per_element
    )

    return [full_bytes, factorized_bytes]
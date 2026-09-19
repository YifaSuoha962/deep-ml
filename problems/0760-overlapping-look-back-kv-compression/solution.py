import numpy as np

def lookback_compressed_kv(
    H,
    W_aKV,
    W_bKV,
    W_aZ,
    W_bZ,
    B_a,
    B_b,
    m
):
    """
    H: (n, d) hidden states; n must be divisible by m

    W_aKV, W_bKV: (d, c) KV projection weights
    W_aZ, W_bZ: (d, c) compression-weight projection matrices

    B_a, B_b: (m, c) positional biases

    m: block size

    Returns:
        list of lists of shape (n // m, c)
    """

    # Convert inputs to NumPy arrays
    H = np.asarray(H, dtype=float)
    W_aKV = np.asarray(W_aKV, dtype=float)
    W_bKV = np.asarray(W_bKV, dtype=float)
    W_aZ = np.asarray(W_aZ, dtype=float)
    W_bZ = np.asarray(W_bZ, dtype=float)
    B_a = np.asarray(B_a, dtype=float)
    B_b = np.asarray(B_b, dtype=float)

    n, d = H.shape

    if n % m != 0:
        raise ValueError("n must be divisible by m")

    num_blocks = n // m

    # ------------------------------------------------------------
    # 1. Project every token into two KV streams
    #    and two compression-logit streams
    # ------------------------------------------------------------
    """
    Note the different roles of a (current blk) and b (previous blk)
    """
    a_kv = H @ W_aKV          # (n, c)
    b_kv = H @ W_bKV          # (n, c)

    a_logits = H @ W_aZ       # (n, c)
    b_logits = H @ W_bZ       # (n, c)

    outputs = []

    # ------------------------------------------------------------
    # 2. Compress each non-overlapping block
    # ------------------------------------------------------------
    for i in range(num_blocks):

        cur_start = i * m
        cur_end = (i + 1) * m

        # Current block uses stream a
        cur_kv = a_kv[cur_start:cur_end]               # (m, c)
        cur_logits = a_logits[cur_start:cur_end] + B_a # (m, c)

        if i == 0:
            # ----------------------------------------------------
            # First block has no previous block.
            # Stream-b logits are -inf, therefore softmax weight=0.
            # KV values themselves can be arbitrary because they
            # will receive zero weight.
            # ----------------------------------------------------
            prev_kv = np.zeros_like(cur_kv)             # (m, c)
            prev_logits = np.full_like(
                cur_logits,
                -np.inf
            )                                           # (m, c)

        else:
            prev_start = (i - 1) * m
            prev_end = i * m

            # Previous block uses stream b
            prev_kv = b_kv[prev_start:prev_end]         # (m, c)
            prev_logits = (
                b_logits[prev_start:prev_end] + B_b
            )                                           # (m, c)

        # --------------------------------------------------------
        # 3. Combine current + look-back streams
        #    Shape: (2m, c)
        # --------------------------------------------------------
        kv = np.concatenate(
            [cur_kv, prev_kv],
            axis=0
        )                                               # (2m, c)

        logits = np.concatenate(
            [cur_logits, prev_logits],
            axis=0
        )                                               # (2m, c)

        # --------------------------------------------------------
        # 4. Softmax over the 2m token axis independently
        #    for each output feature dimension
        #
        #    axis=0 because rows correspond to the 2m tokens
        # --------------------------------------------------------
        max_logits = np.max(logits, axis=0, keepdims=True)  # (1, c)

        exp_logits = np.exp(logits - max_logits)            # (2m, c)
        weights = exp_logits / np.sum(
            exp_logits,
            axis=0,
            keepdims=True
        )                                                   # (2m, c)

        # --------------------------------------------------------
        # 5. Weighted sum of each feature over the 2m tokens into a compacted feature
        # --------------------------------------------------------
        compressed = np.sum(
            weights * kv,
            axis=0
        )                                                   # (c,)

        outputs.append(compressed)

    return np.asarray(outputs).tolist()



import numpy as np

def compute_log_prob_ratios(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    mask: np.ndarray
) -> dict:
    """
    Compute log-probability ratios between policy and reference models.

    Args:
        policy_log_probs: (B, T) log-probs under the policy model
        reference_log_probs: (B, T) log-probs under the reference model
        mask: (B, T) binary mask, 1 for valid tokens, 0 for padding

    Returns:
        Dictionary with:
          - 'per_token_log_ratios': (B, T) array
          - 'sequence_log_ratios': (B,) array
          - 'mean_sequence_log_ratio': float scalar
    """
    per_token_log_ratios = policy_log_probs - reference_log_probs
    
    """
    在布尔索引中，任何非零值都被视为 True，所以 ~mask 的所有元素（-2 或 -1）都为真，导致 per_token_log_ratios[~mask] = 0 把所有 token 的 log ratio 全部置为 0，最终序列 log ratio 和均值也都变成 0。
    """
    # 将 mask 转为布尔型，避免整数按位取反
    mask_bool = mask.astype(bool)
    per_token_log_ratios[~mask_bool] = 0

    sequence_log_ratios = np.sum(per_token_log_ratios, axis=1)

    n_batch = policy_log_probs.shape[0]
    mean_sequence_log_ratio = np.sum(sequence_log_ratios) / n_batch

    return {'per_token_log_ratios': per_token_log_ratios, 'sequence_log_ratios': sequence_log_ratios, 'mean_sequence_log_ratio': mean_sequence_log_ratio}
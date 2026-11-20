import numpy as np

def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    """
    Identity Preference Optimization (IPO) loss.

    Args:
        policy_chosen_logps: log πθ(y_w) per example
        policy_rejected_logps: log πθ(y_l) per example
        ref_chosen_logps: log πref(y_w) per example
        ref_rejected_logps: log πref(y_l) per example
        beta: KL temperature (target gap is 1/(2β))

    Returns:
        Mean IPO loss, rounded to 6 decimals.
    """
    # Your code here
    if beta <= 0:
        raise ValueError("beta must be positive")

    # 转为浮点数组，支持列表或 NumPy 数组输入
    policy_chosen = np.asarray(policy_chosen_logps, dtype=float)
    policy_rejected = np.asarray(policy_rejected_logps, dtype=float)
    ref_chosen = np.asarray(ref_chosen_logps, dtype=float)
    ref_rejected = np.asarray(ref_rejected_logps, dtype=float)

    # 分别计算 chosen 和 rejected 相对于参考模型的 log-ratio
    chosen_log_ratio = policy_chosen - ref_chosen
    rejected_log_ratio = policy_rejected - ref_rejected

    # 偏好 margin：chosen 的 log-ratio 减去 rejected 的 log-ratio
    h = chosen_log_ratio - rejected_log_ratio

    # IPO 将 margin 拉向有限目标 1/(2β)，再对批次取均方误差
    target_gap = 0.5 / beta
    loss = np.mean((h - target_gap) ** 2)

    return round(float(loss), 6)

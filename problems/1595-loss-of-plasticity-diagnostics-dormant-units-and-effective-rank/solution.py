import math
import numpy as np


def dormant_ratio(H, tau):
    """返回休眠单元比例，以及按升序排列的休眠单元索引。"""
    H = np.asarray(H, dtype=float)

    if H.ndim != 2:
        raise ValueError("H must be a 2D array with shape (N, U)")

    num_inputs, num_units = H.shape
    if num_units == 0:
        raise ValueError("H must contain at least one unit")
    if num_inputs == 0:
        raise ValueError("H must contain at least one input")

    # 每个单元在 N 个输入上的平均绝对激活，形状为 (U,)。
    unit_means = np.abs(H).mean(axis=0)

    # 分母是所有单元平均激活的均值。
    layer_mean = unit_means.mean()  # mean along u axis 

    # 若整层的平均绝对激活为 0，题目规定所有分数为 0。
    if layer_mean == 0:
        scores = np.zeros(num_units, dtype=float)
    else:
        scores = unit_means / layer_mean

    # 分数小于等于阈值的单元为休眠单元。
    """ 直接用比较运算符得到 """
    dormant_mask = scores <= tau
    dormant_indices = np.flatnonzero(dormant_mask).tolist()

    # 转成 Python float。
    ratio = float(dormant_mask.sum() / num_units)

    return ratio, sorted(dormant_indices)


def effective_rank(H, eps=1e-12):
    """返回列中心化激活矩阵的有效秩。"""
    H = np.asarray(H, dtype=float)

    if H.ndim != 2:
        raise ValueError("H must be a 2D array with shape (N, U)")

    # 对每一列减去该列均值，消除平均激活的影响。
    Hc = H - H.mean(axis=0, keepdims=True)

    # 直接计算奇异值，不需要 U、V 矩阵。
    singular_values = np.linalg.svd(Hc, compute_uv=False)

    # 忽略小于或等于 eps 的奇异值。
    singular_values = singular_values[singular_values > eps]

    # 所有奇异值都低于阈值时，有效秩为 0。
    if singular_values.size == 0:
        return 0.0

    # 将保留的奇异值归一化为概率分布。
    probabilities = singular_values / singular_values.sum()

    # 计算 exp(熵)，并返回 Python float。
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(math.exp(entropy))


def plasticity_report(H, W, tau, decimals=4):
    """返回包含休眠比例、有效秩和权重 Frobenius 范数的报告。"""
    H = np.asarray(H, dtype=float)
    W = np.asarray(W, dtype=float)

    ratio, _ = dormant_ratio(H, tau)
    rank = effective_rank(H)
    weight_norm = float(np.linalg.norm(W, ord="fro"))

    return {
        "dormant_ratio": round(ratio, decimals),
        "effective_rank": round(rank, decimals),
        "weight_norm": round(weight_norm, decimals),
    }
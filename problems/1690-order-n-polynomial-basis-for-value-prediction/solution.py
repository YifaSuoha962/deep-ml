import numpy as np
from itertools import product

def polynomial_basis(s, n):
    # 转为浮点状态向量，k 为状态维度
    s = np.asarray(s, dtype=float)

    # 按字典序枚举指数：最后一个维度变化最快
    # 每个特征为 prod_j s[j] ** c[j]；NumPy 将 0**0 视为 1
    features = [
        np.prod(s ** np.asarray(c, dtype=int))
        for c in product(range(n + 1), repeat=len(s))
    ]

    return np.asarray(features, dtype=float)


def gradient_mc_poly(samples, n, alpha, n_passes):
    # 需要至少一个样本，才能确定状态维度和权重数量
    if len(samples) == 0:
        raise ValueError("samples must contain at least one (state, return) pair")

    k = len(samples[0][0])
    w = np.zeros((n + 1) ** k, dtype=float)

    # 特征只依赖状态，可预先计算，重复训练时复用
    feature_samples = [
        (polynomial_basis(s, n), G)
        for s, G in samples
    ]

    # 每一轮严格按样本原始顺序，逐个更新权重
    for _ in range(n_passes):
        for x, G in feature_samples:
            prediction = w @ x
            error = G - prediction

            # 线性价值函数对 w 的梯度为 x
            w += alpha * error * x

    return w
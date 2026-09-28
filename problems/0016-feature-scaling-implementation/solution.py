import numpy as np
from typing import Tuple

def feature_scaling(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对数据集进行标准化和最小-最大归一化。
    输入 data: 形状 (n_samples, n_features)
    返回: (standardized_data, normalized_data)
    """
    # 1. 标准化（对每个特征列）
    mu_feats = np.mean(data, axis=0)          # 形状 (n_features,)
    std_feats = np.std(data, axis=0)          # 形状 (n_features,)
    # 防止除以零
    std_feats = np.where(std_feats == 0, 1e-8, std_feats)
    standardized_data = (data - mu_feats) / std_feats

    # 2. 最小-最大归一化（对每个特征列）
    min_feats = np.min(data, axis=0)          # 形状 (n_features,)
    max_feats = np.max(data, axis=0)          # 形状 (n_features,)
    range_feats = max_feats - min_feats
    # 防止除以零
    range_feats = np.where(range_feats == 0, 1e-8, range_feats)
    normalized_data = (data - min_feats) / range_feats

    # 3. 四舍五入到 4 位小数
    standardized_data = np.round(standardized_data, 4)
    normalized_data = np.round(normalized_data, 4)

    return standardized_data, normalized_data
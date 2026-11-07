import torch

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    # 确保输入为 float32 的 Tensor
    data = torch.as_tensor(data, dtype=torch.float32)
    
    # 1. 标准化数据集 (均值为 0，标准差为 1)
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0, unbiased=False)
    # 避免除以 0 (对于常数特征)
    std = torch.where(std == 0, torch.ones_like(std), std)
    data_std = (data - mean) / std
    
    # 2. 计算协方差矩阵
    n_samples = data.shape[0]
    # 样本协方差矩阵 (这里使用 n-1，实际上特征向量与缩放因子无关)
    cov = (data_std.T @ data_std) / max(n_samples - 1, 1)
    
    # 3. 使用 eigh 进行特征分解 (eigh 返回的特征值按升序排列)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # 4. 按特征值降序排序，选出前 k 个
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_indices = sorted_indices[:k]
    top_eigenvectors = eigenvectors[:, top_indices].clone()
    
    # 5. 应用符号约定
    for i in range(top_eigenvectors.shape[1]):
        vec = top_eigenvectors[:, i]
        # 找到第一个绝对值大于 1e-10 的元素
        idx = torch.where(torch.abs(vec) > 1e-10)[0]
        if len(idx) > 0:
            first_idx = idx[0]
            # 如果该元素为负数，则将整个特征向量翻转
            if vec[first_idx] < 0:
                top_eigenvectors[:, i] = -vec
                
    # 6. 将结果四舍五入到 4 位小数
    # 使用 * 10000 再 round 再 / 10000 以避免浮点精度问题
    top_eigenvectors = torch.round(top_eigenvectors * 10000) / 10000
    
    return top_eigenvectors
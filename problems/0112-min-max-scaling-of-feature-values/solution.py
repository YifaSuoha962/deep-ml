import torch

def min_max(x: torch.Tensor) -> torch.Tensor:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A tensor of numerical values
    
    Returns:
        A new tensor with values normalized to [0, 1]
    """
    # Your code here
    
    # 计算全局最小值和最大值（适用于任意形状，按整体归一化）
    min_val = x.min()
    max_val = x.max()
    
    # 处理所有值相同的情况，避免除以零
    if max_val == min_val:
        return torch.zeros_like(x)
    
    return (x - min_val) / (max_val - min_val)
import torch

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
    """
    Detect and remove outliers using the IQR method.
    
    Args:
        data: List of numerical values
        k: IQR multiplier for determining outlier bounds (default 1.5)
    
    Returns:
        Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
    """
    # Your code here — use torch.quantile() for percentile computation
    data_tensor = torch.tensor(data, dtype=torch.float32)
    # 计算 Q1 和 Q3（默认线性插值）
    q1 = torch.quantile(data_tensor, 0.25)
    q3 = torch.quantile(data_tensor, 0.75)
    iqr = q3 - q1
    # bounds
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    # tag outliers
    outlier_mask = (data_tensor > upper_bound) | (data_tensor < lower_bound)
    # get outliers's indices
    outlier_indices = torch.where(outlier_mask)[0].tolist()
    # filter clean data
    cleaned = data_tensor[~outlier_mask].tolist()
    cleaned = [round(x, 4) for x in cleaned]
    
    return {
        'cleaned_data': cleaned,
        'outlier_indices': outlier_indices,
        'lower_bound': round(lower_bound.item(), 4),
        'upper_bound': round(upper_bound.item(), 4)
    }
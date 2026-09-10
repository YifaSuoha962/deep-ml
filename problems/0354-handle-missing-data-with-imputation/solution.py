import torch

def impute_missing_data(data: torch.Tensor, strategy: str = 'mean') -> torch.Tensor:
    """
    Impute missing values in a 2D tensor using the specified strategy.

    Args:
        data: 2D torch.Tensor with missing values represented as float('nan')
        strategy: Imputation strategy - 'mean', 'median', or 'mode'

    Returns:
        2D torch.Tensor with missing values imputed
    """
    # Your code here
    
    # Work on a copy to avoid modifying the input in-place
    result = data.clone()
    n_rows, n_cols = result.shape
    
    for j in range(n_cols):
        col = result[:, j]
        mask = ~torch.isnan(col)
        
        # If the entire column is NaN, leave it unchanged
        if mask.sum() == 0:
            continue
        
        valid = col[mask]
        
        if strategy == 'mean':
            stat = valid.mean()
        elif strategy == 'median':
            stat = valid.median()
        else:  # mode
            values, counts = torch.unique(valid, return_counts=True)
            max_count = counts.max()
            # values is sorted ascending; pick the first (smallest) value with max count
            idx = torch.where(counts == max_count)[0][0]
            stat = values[idx]
        
        # Replace NaN values in this column with the computed statistic
        col[torch.isnan(col)] = stat
    
    return result

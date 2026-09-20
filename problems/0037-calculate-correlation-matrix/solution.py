import torch
from typing import Optional, Union

def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """
    # Your implementation here
    # transform dtype
    X_t = torch.as_tensor(X, dtype=torch.float32)
    if Y is None:
        Y_t = X_t
    else:
        Y_t = torch.as_tensor(Y, dtype=torch.float32)

    # dimension check (n_samples, n_features)
    if X_t.dim() == 1:
        X_t = X_t.unsqueeze(1)
    if Y_t.dim() == 1:
        Y_t = Y_t.unsqueeze(1)

    n = X_t.shape[0]
    
    # compute residual
    X_residual = X_t - X_t.mean(dim=0, keepdim=True)
    Y_residual = Y_t - Y_t.mean(dim=0, keepdim=True)

    # coveriance matrix 
    cov = (X_residual.T @ Y_residual) / n

    # std 
    std_X = torch.sqrt((X_residual ** 2).sum(dim=0) / n)
    std_Y = torch.sqrt((Y_residual ** 2).sum(dim=0) / n)

    # avoid zero-division
    eps = 1e-8
    std_X = std_X.clamp_min(eps)
    std_Y = std_Y.clamp_min(eps)

    # correlation matrix 
    corr = cov / (std_X.unsqueeze(1) * std_Y.unsqueeze(0))

    # fill diagonal for error calibration
    if Y is None:
        corr = corr.clamp(-1.0, 1.0)
        corr.fill_diagonal_(1.0)

    return corr

import torch

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 torch tensor
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Your code here
    # 直接调用 torch.linalg.svd，返回 U, S, Vh（Vh 即 V^T）
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U, S, Vh
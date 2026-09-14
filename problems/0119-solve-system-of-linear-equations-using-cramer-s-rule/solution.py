import torch

def cramers_rule(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve system of linear equations Ax = b using Cramer's Rule.
    Returns solution vector x or -1 if no unique solution exists.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = A.shape[0]

    if b.numel() != n:
        raise ValueError("b must have the same number of elements as rows of A.")
    b = b.reshape(-1)

    det_A = torch.linalg.det(A)

    if torch.isclose(det_A, torch.tensor(0.0, dtype=A.dtype)):
        return -1
    
    x = torch.zeros(n, dtype=A.dtype)

    # replace each col by b
    for i in range(n):
        A_i = A.clone()
        A_i[:, i] = b
        det_A_i = torch.linalg.det(A_i)
        x[i] = det_A_i / det_A
    
    return x
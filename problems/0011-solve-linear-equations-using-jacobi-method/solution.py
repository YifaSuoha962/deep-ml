import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Your implementation here
    
    m = A_t.shape[0]
    # Initial solution x^(0) = 0
    x = torch.zeros(m, dtype=torch.float)

    diag = torch.diag(A_t)

    if torch.any(diag == 0):
        raise ValueError("Diagonal elements of A must be non-zero.")

    # Off-diagonal part: R = A - D
    R = A_t - torch.diag(diag)

    # Perform exactly n Jacobi iterations
    for _ in range(n):
        x = (b_t - R @ x) / diag
    
    # Round only the final result
    return torch.round(x * 10000) / 10000


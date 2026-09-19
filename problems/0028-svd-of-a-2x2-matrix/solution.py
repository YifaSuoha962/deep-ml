import torch


def svd_2x2(
    A: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix without using torch.linalg.svd.

    Returns U, s, V such that:

        A ≈ U @ diag(s) @ V

    Here V is V^T in the conventional SVD notation.
    """

    A = torch.as_tensor(A)

    if A.shape != (2, 2):
        raise ValueError("A must be a 2x2 matrix.")

    # Save output dtype.
    # For normal floating inputs, preserve their dtype.
    # Integer input is converted to float32.
    if torch.is_floating_point(A):
        output_dtype = A.dtype
    else:
        output_dtype = torch.float32

    # --------------------------------------------------
    # Use float64 internally for numerical stability
    # --------------------------------------------------
    A64 = A.to(torch.float64)

    # --------------------------------------------------
    # 1. Compute A^T A
    # --------------------------------------------------
    ATA = A64.T @ A64

    # A^T A is symmetric, so eigh is appropriate
    eigenvalues, eigenvectors = torch.linalg.eigh(ATA)

    # Sort descending
    order = torch.argsort(eigenvalues, descending=True)

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    # --------------------------------------------------
    # 2. Singular values
    # --------------------------------------------------
    s = torch.sqrt(eigenvalues)

    # --------------------------------------------------
    # 3. Construct right singular vectors
    # --------------------------------------------------
    v0 = eigenvectors[:, 0]
    v0 = v0 / torch.linalg.vector_norm(v0)

    # In 2D, directly construct an orthogonal vector
    v1 = torch.stack([
        -v0[1],
        v0[0]
    ])

    # Match sign of second eigenvector
    if torch.dot(v1, eigenvectors[:, 1]) < 0:
        v1 = -v1

    V_cols = torch.stack(
        [v0, v1],
        dim=1
    )

    # --------------------------------------------------
    # 4. Construct left singular vectors
    # --------------------------------------------------
    eps = 1e-12

    if s[0] > eps:
        u0 = A64 @ V_cols[:, 0]
        u0 = u0 / torch.linalg.vector_norm(u0)
    else:
        # Zero matrix
        u0 = torch.tensor(
            [1.0, 0.0],
            dtype=torch.float64
        )

    # Construct second vector directly perpendicular to u0
    u1 = torch.stack([
        -u0[1],
        u0[0]
    ])

    # Choose sign so reconstruction agrees with A
    if s[1] > eps:
        Av1 = A64 @ V_cols[:, 1]

        if torch.dot(u1, Av1) < 0:
            u1 = -u1

    U = torch.stack(
        [u0, u1],
        dim=1
    )

    # Conventional SVD uses V^T
    V = V_cols.T

    # --------------------------------------------------
    # 5. Convert back to input dtype
    # --------------------------------------------------
    U = U.to(output_dtype)
    s = s.to(output_dtype)
    V = V.to(output_dtype)

    return U, s, V
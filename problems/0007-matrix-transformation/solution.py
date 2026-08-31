import numpy as np


def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]]
):
    """
    Compute the matrix transformation T^(-1) A S.

    T and S must both be invertible.
    Return -1 if the transformation cannot be performed.
    """

    # Convert to NumPy arrays
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)

    # Check that all matrices are 2-dimensional
    if A.ndim != 2 or T.ndim != 2 or S.ndim != 2:
        return -1

    # T and S must be square
    if T.shape[0] != T.shape[1]:
        return -1

    if S.shape[0] != S.shape[1]:
        return -1

    # Dimensions must allow T^(-1) @ A @ S
    if T.shape[1] != A.shape[0]:
        return -1

    if A.shape[1] != S.shape[0]:
        return -1

    try:
        # Check that T and S are invertible
        inv_T = np.linalg.inv(T)
        np.linalg.inv(S)

    except np.linalg.LinAlgError:
        return -1

    # T^(-1) A S
    result = inv_T @ A @ S

    # Convert NumPy array back to ordinary Python lists
    return result.tolist()
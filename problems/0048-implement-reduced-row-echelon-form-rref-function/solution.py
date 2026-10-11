import torch

def rref(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts a matrix to its Reduced Row Echelon Form (RREF).

    Args:
        matrix: Input matrix as a 2D tensor

    Returns:
        Matrix in RREF form
    """

    # Convert to float and avoid modifying original tensor
    A = matrix.clone().to(torch.float64)

    rows, cols = A.shape

    # Current pivot row
    pivot_row = 0
    eps = 1e-10

    # Iterate over columns
    for col in range(cols):

        # Stop if all rows have pivots
        if pivot_row >= rows:
            break

        # Find row with largest absolute pivot candidate
        max_row = pivot_row + torch.argmax(
            torch.abs(A[pivot_row:, col])
        ).item()

        # Skip column if no valid pivot
        if torch.abs(A[max_row, col]).item() < eps:
            continue

        # Swap rows if necessary
        if max_row != pivot_row:
            temp = A[pivot_row].clone()
            A[pivot_row] = A[max_row]
            A[max_row] = temp

        # Normalize pivot row
        pivot = A[pivot_row, col].clone()
        A[pivot_row] = A[pivot_row] / pivot

        # Eliminate all other entries in pivot column
        for row in range(rows):
            if row != pivot_row:
                factor = A[row, col].clone()
                A[row] = A[row] - factor * A[pivot_row]

        # Move to next pivot row
        pivot_row += 1

    # Clean up numerical errors
    A[torch.abs(A) < eps] = 0.0

    return A

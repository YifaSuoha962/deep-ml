# import torch

# def determinant_4x4(matrix) -> float:
#     """
#     Compute the determinant of a 4×4 matrix using PyTorch.
#     Input can be a Python list, NumPy array, or torch Tensor of shape (4,4).
#     Returns a Python float.
#     """
#     # Convert to tensor
#     m = torch.as_tensor(matrix, dtype=torch.float)
#     # Your implementation here
    
#     if m.shape != (4, 4):
#         raise ValueError("Input matrix must have shape (4, 4)")
    
#     def determinant(mat):
#         n = mat.shape[0]
#         # boundary case: 1x1 
#         if n == 1:
#             return mat[0, 0]
#         elif n == 2:
#             return (mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
#         else:
#             # (recursive) Laplace expansion along the first row (i=0)
#             det = torch.tensor(0.0, dtype=mat.dtype)
#             for j in range(n):
#                 minor = torch.cat((mat[1:, :j], mat[1:, j + 1:]), dim=1)
#                 cofactor = ((-1) ** j) * mat[0, j] * determinant(minor)

#                 det += cofactor
#         return cofactor
    
#     return determinant(m).item()



import torch

def determinant_4x4(matrix) -> float:
    """
    Compute the determinant of a 4×4 matrix using Laplace Expansion.

    Input can be a Python list, NumPy array, or torch Tensor of shape (4,4).

    Returns a Python float.
    """

    # Convert to tensor
    m = torch.as_tensor(matrix, dtype=torch.float)

    if m.shape != (4, 4):
        raise ValueError("Input matrix must have shape (4, 4)")

    def determinant(mat):
        n = mat.shape[0]

        # Base case: 1x1
        if n == 1:
            return mat[0, 0]

        # Base case: 2x2
        if n == 2:
            return (
                mat[0, 0] * mat[1, 1]
                - mat[0, 1] * mat[1, 0]
            )

        # Laplace expansion along the first row
        det = torch.tensor(0.0, dtype=mat.dtype)

        for j in range(n):
            # Delete row 0 and column j to construct the minor
            minor = torch.cat(
                (mat[1:, :j], mat[1:, j + 1:]),
                dim=1
            )

            cofactor = ((-1) ** j) * mat[0, j] * determinant(minor)

            det += cofactor

        return det

    return determinant(m).item()
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    trans_a = list()
    m = len(a)
    n = len(a[0])
    for j in range(n):
        row = [a[i][j] for i in range(m)]
        trans_a.append(row)
    return trans_a
    # pass
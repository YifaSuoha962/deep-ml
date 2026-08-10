from typing import List, Union

def matrix_dot_vector(a: List[List[Union[int, float]]], b: List[Union[int, float]]) -> Union[List[Union[int, float]], int]:
    # If matrix is empty, cannot determine columns -> incompatible
    if not a:
        return -1

    num_cols = len(a[0])
    # Check that every row has the same number of columns and that
    # the number of columns equals the length of vector b.
    if any(len(row) != num_cols for row in a):
        return -1
    if num_cols != len(b):
        return -1

    # Compute dot product for each row
    result = []
    for row in a:
        dot = sum(row[i] * b[i] for i in range(num_cols))
        result.append(dot)
    return result
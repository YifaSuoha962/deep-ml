import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    mat_a = np.asarray(a)
    mat_b = np.asarray(b)
    if mat_a.shape[1] != mat_b.shape[0]:
        return -1
    c = mat_a @ mat_b
    return c.tolist()
import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    # 修正 axis 映射：'row' -> 行均值 (axis=1)，其他（如 'column'）-> 列均值 (axis=0)
    axis = 1 if mode == 'row' else 0
    np_mat = np.asarray(matrix)          # 修正拼写：np.as_array -> np.asarray
    means = np_mat.mean(axis=axis)       # 使用 NumPy 数组而非原 matrix 列表
    # 如果结果是标量（例如 1D 矩阵），转为列表
    if np.isscalar(means):
        means = np.array([means])
    return means.tolist()                # 返回 Python 列表
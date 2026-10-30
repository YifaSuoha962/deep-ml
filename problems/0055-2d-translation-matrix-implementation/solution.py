import numpy as np

def translate_object(points, tx, ty):
    np_points = np.asarray(points, dtype=float)

    if np_points.size == 0:
        return np_points.reshape(0, 2)

    if np_points.ndim != 2 or np_points.shape[1] != 2:
        raise ValueError("points 必须是形状为 (N, 2) 的点集")

    # 每一行从 [x, y] 扩展为 [x, y, 1]
    homogeneous_points = np.column_stack(
        (np_points, np.ones(np_points.shape[0]))
    )

    # 列向量约定下的 2D 平移矩阵
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ], dtype=float)

    # 点目前按行存储，因此转置后作为列向量矩阵相乘
    translated_homogeneous = (T @ homogeneous_points.T).T

    # 去掉齐次坐标，只返回 x、y
    return translated_homogeneous[:, :2]
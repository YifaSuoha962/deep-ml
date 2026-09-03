import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round

	# 转换为NumPy数组
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1, 1)

    # 正规方程求解
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    # 四舍五入到四位小数并转换为列表
    theta = np.round(theta, 4).flatten().tolist()
    return theta
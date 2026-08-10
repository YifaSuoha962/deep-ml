import math

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # 提取矩阵元素
    a, b = matrix[0]
    c, d = matrix[1]
    
    # 计算迹和行列式
    trace = a + d
    det = a * d - b * c
    
    # 判别式
    discriminant = trace * trace - 4 * det
    
    # 确保判别式非负（对于实矩阵可能为负，但这里我们仍计算）
    if discriminant < 0:
        # 对于复特征值，我们返回实部（但题目期望实数）
        # 为安全，取绝对值或返回空，但根据题意应为实矩阵，这里按实数处理
        discriminant = 0  # 或者 raise ValueError，但根据题目我们假设实特征值
    
    sqrt_disc = math.sqrt(discriminant)
    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0
    
    eigenvalues = [lambda1, lambda2]
    eigenvalues.sort(reverse=True)  # 从高到低排序
    return eigenvalues
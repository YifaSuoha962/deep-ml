import numpy as np

def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L, rounded to 3 decimals
    """
    v = np.asarray(v, dtype=float)
    L = np.asarray(L, dtype=float)
    
    # 计算点积
    dot_vL = np.dot(v, L)      # v · L
    dot_LL = np.dot(L, L)      # L · L
    
    # 投影公式：proj_L(v) = (v·L / L·L) * L
    proj = (dot_vL / dot_LL) * L
    
    # 转为列表并四舍五入到 3 位小数
    return [round(float(x), 3) for x in proj]
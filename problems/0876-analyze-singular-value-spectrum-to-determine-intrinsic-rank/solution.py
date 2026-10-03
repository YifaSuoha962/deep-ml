import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
    """
    Return the smallest rank k such that the top-k singular values of delta_W
    capture at least `energy_threshold` of the total squared-singular-value energy.
    
	-error-log:
	2026/10/3 -- 不熟悉svd调用函数的参数即返回形式,不熟悉对矩阵求范数的方式(求平方和)
	"""
    # 计算奇异值（full_matrices=False 更高效，S 为一维数组）
    _, S, _ = np.linalg.svd(delta_W, full_matrices=False)
    
    # 能量 = 奇异值的平方
    energy = S ** 2
    total_energy = np.sum(energy)
    
    # 全零矩阵返回 0
    if total_energy == 0:
        return 0
    
    # 累积能量，找到最小的 k 使累积比例达到阈值
    cumulative_energy = 0.0
    for k, e in enumerate(energy, start=1):
        cumulative_energy += e
        if cumulative_energy / total_energy >= energy_threshold:
            return k
    
    # 理论上不会执行到这里（阈值 >= 1），但为了鲁棒性返回非零奇异值个数
    return len(energy)
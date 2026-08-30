import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # 确保输入为numpy数组
    x = np.asarray(x)
    analytical_grad = np.asarray(analytical_grad)
    shape = x.shape
    numerical_grad = np.zeros_like(x)
    
    # 展平以方便迭代
    flat_x = x.ravel()
    flat_num_grad = np.zeros_like(flat_x, dtype=float)
    
    for i in range(flat_x.size):
        # 保存原始值
        original = flat_x[i]
        
        # 前向扰动
        flat_x[i] = original + epsilon
        f_plus = f(flat_x.reshape(shape))
        
        # 后向扰动
        flat_x[i] = original - epsilon
        f_minus = f(flat_x.reshape(shape))
        
        # 恢复原始值
        flat_x[i] = original
        
        # 中心差分近似
        flat_num_grad[i] = (f_plus - f_minus) / (2 * epsilon)
    
    numerical_grad = flat_num_grad.reshape(shape)
    
    # 计算相对误差
    diff = numerical_grad - analytical_grad
    norm_diff = np.linalg.norm(diff)
    norm_num = np.linalg.norm(numerical_grad)
    norm_ana = np.linalg.norm(analytical_grad)
    
    # 防止除以零：分母加小量
    denominator = norm_num + norm_ana + 1e-12
    relative_error = norm_diff / denominator
    
    return numerical_grad, relative_error
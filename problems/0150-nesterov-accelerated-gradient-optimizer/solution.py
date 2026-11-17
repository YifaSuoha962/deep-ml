import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """NAG look-ahead update; returns (updated_parameter, updated_velocity)."""
    # 转为浮点数组，同时支持标量和数组输入
    parameter = np.asarray(parameter, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    # 先沿动量方向预测位置：θ_lookahead = θ - γv
    lookahead = parameter - momentum * velocity

    # 在预测位置计算梯度，而不是在当前参数位置
    gradient = np.asarray(grad_fn(lookahead), dtype=float)

    # 更新速度：v_new = γv + ηg
    velocity = momentum * velocity + learning_rate * gradient

    # 更新参数：θ_new = θ - v_new
    parameter = parameter - velocity

    return np.round(parameter, 5), np.round(velocity, 5)
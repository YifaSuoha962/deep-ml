import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    # 计算通道均值 (N, H, W) 上的均值，保留维度便于广播
    mean = x.mean(dim=(0, 2, 3), keepdim=True)          # (1, C, 1, 1)
    # 计算总体方差（有偏）
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)  # (1, C, 1, 1)
    
    # 归一化
    x_norm = (x - mean) / torch.sqrt(var + eps)         # (N, C, H, W)
    
    # 重塑 gamma 和 beta 以广播 （这俩都是向量，x 的 特征维度在第2维）
    gamma = gamma.view(1, -1, 1, 1)                     # (1, C, 1, 1)
    beta = beta.view(1, -1, 1, 1)                       # (1, C, 1, 1)
    
    return gamma * x_norm + beta

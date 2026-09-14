import torch

def compute_arithmetic_intensity(
    flops: torch.Tensor,
    bytes_accessed: torch.Tensor,
    peak_performance: torch.Tensor,
    peak_bandwidth: torch.Tensor
) -> dict:
    # 转换为双精度，避免 float32 精度误差
    flops = flops.double()
    bytes_accessed = bytes_accessed.double()
    peak_performance = peak_performance.double()
    peak_bandwidth = peak_bandwidth.double()
    
    arithmetic_intensity = flops / bytes_accessed
    ridge_point = peak_performance / peak_bandwidth
    
    if arithmetic_intensity < ridge_point:
        bottleneck = 'memory-bound'
        achieved_performance = arithmetic_intensity * peak_bandwidth
    else:
        bottleneck = 'compute-bound'
        achieved_performance = peak_performance
    
    utilization_percent = (achieved_performance / peak_performance) * 100.0
    
    # arithmetic_intensity = round((flops / bytes_accessed).item(), 6)
    # ridge_point = round((peak_performance / peak_bandwidth).item(), 6)

    return {
        'arithmetic_intensity': arithmetic_intensity.item(),
        'ridge_point': ridge_point.item(),
        'bottleneck': bottleneck,
        'achieved_performance': achieved_performance.item(),
        'utilization_percent': utilization_percent.item()
    }
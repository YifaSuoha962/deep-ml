def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    # Return [latency_ms, tokens_per_sec]
    
    # 计算每个token的解码延迟（秒）
    latency_sec = (num_params * bytes_per_param) / bandwidth_bytes_per_s
    # 转换为毫秒
    latency_ms = latency_sec * 1000.0
    # 吞吐量：每秒生成的token数
    tokens_per_sec = 1.0 / latency_sec
    return [latency_ms, tokens_per_sec]
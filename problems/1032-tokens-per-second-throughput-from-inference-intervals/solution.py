def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    
    step_tps = list()
    cumulative_tps = list()
    overall_tps = 0.0

    if intervals:
        cum_n_tokens = 0.0
        cum_t_elapsed = 0.0

        for tokens_processed, elapsed_seconds in intervals:
            # 处理零耗时：无法计算吞吐量，抛出 ValueError
            if elapsed_seconds == 0:
                raise ValueError("elapsed_seconds must be non-zero")
            tps = tokens_processed / elapsed_seconds
            cum_n_tokens += tokens_processed
            cum_t_elapsed += elapsed_seconds
            cum_tps = cum_n_tokens / cum_t_elapsed

            step_tps.append(round(tps, 2))
            cumulative_tps.append(round(cum_tps, 2))

        overall_tps = cumulative_tps[-1]

    return {'step_tps': step_tps, 'cumulative_tps': cumulative_tps, 'overall_tps': overall_tps}
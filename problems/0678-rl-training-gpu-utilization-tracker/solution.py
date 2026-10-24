from collections import defaultdict


def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    """
    Analyze GPU utilization during RL training and compute per-phase statistics.
    """
    n_intervals = len(timestamps_ms) - 1

    total_time_ms = timestamps_ms[-1] - timestamps_ms[0]

    # 按阶段累计：时长、加权利用率、浪费的 GPU 时间
    """
    针对每个阶段，分别维护 time span 和 gpu utility 的 dict，这样避免构建嵌套 dict 加大难度
    """
    phase_duration = defaultdict(float)
    phase_weighted_util = defaultdict(float)
    phase_wasted = defaultdict(float)

    # 记录每个阶段第一次出现的位置，用于平局判定
    phase_first_index = {}

    weighted_util_sum = 0.0

    for i in range(n_intervals):
        duration = timestamps_ms[i + 1] - timestamps_ms[i]
        util = gpu_util_pct[i]
        phase = phase_labels[i]

        if phase not in phase_first_index:
            phase_first_index[phase] = i

        phase_duration[phase] += duration
        phase_weighted_util[phase] += util * duration
        phase_wasted[phase] += duration * (100.0 - util)
        weighted_util_sum += util * duration

    avg_gpu_util_pct = weighted_util_sum / total_time_ms if total_time_ms > 0 else 0.0

    """
    分阶段处理，统计完后再整理成想要的dict
    """
    # 构造 phase_stats，按阶段名排序
    phase_stats = {}
    for phase in sorted(phase_duration):
        duration = phase_duration[phase]
        avg_util = phase_weighted_util[phase] / duration if duration > 0 else 0.0
        time_fraction = (duration / total_time_ms * 100.0) if total_time_ms > 0 else 0.0
        phase_stats[phase] = {
            "total_time_ms": round(duration, 2),
            "avg_util_pct": round(avg_util, 2),
            "time_fraction_pct": round(time_fraction, 2),
        }

    # 瓶颈阶段：浪费 GPU 时间最多的阶段；平局取首次出现最早的
    bottleneck_phase = min(
        phase_wasted,
        key=lambda p: (-phase_wasted[p], phase_first_index[p]),
    ) if phase_wasted else None

    gpu_idle_fraction_pct = 100.0 - avg_gpu_util_pct

    return {
        "total_time_ms": round(total_time_ms, 2),
        "avg_gpu_util_pct": round(avg_gpu_util_pct, 2),
        "phase_stats": phase_stats,
        "bottleneck_phase": bottleneck_phase,
        "gpu_idle_fraction_pct": round(gpu_idle_fraction_pct, 2),
    }
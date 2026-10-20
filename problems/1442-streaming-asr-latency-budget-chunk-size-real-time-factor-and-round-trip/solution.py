import math


def chunk_latency(chunk_s, rtf, rtt_s):
    # 第一个 chunk_s：等待音频块填满才送入模型
    # rtf * chunk_s：模型推理时间
    # rtt_s：网络往返时间
    return chunk_s + rtf * chunk_s + rtt_s


def largest_chunk_under(budget_s, rtf, rtt_s, step=0.05):
    # 在总延迟不超过 budget_s 的前提下，找到最大的、为 step 整数倍的音频块时长 (multiple of step)。
    # 比较时加入微小容差
    eps = 1e-9
    multiple = 1
    lat_c = chunk_latency(multiple * step, rtf, rtt_s)
    while lat_c <= budget_s + eps:
        multiple += 1
        lat_c = chunk_latency(multiple * step, rtf, rtt_s)
    return round((multiple - 1) * step, 2) if multiple > 1 else None


def concurrent_streams(rtf, gpu_share=1.0):
    # rtf 是每路流每秒音频占用的 GPU 秒数
    return math.floor(gpu_share / rtf)


def end_of_speech_latency(hangover_s, chunk_s, rtf, rtt_s):
    return hangover_s + chunk_latency(chunk_s, rtf, rtt_s)


def budget_breakdown(chunk_s, rtf, rtt_s):
    # TODO: {'wait', 'model', 'network'} fractions rounded to 4 decimals
    total = chunk_latency(chunk_s, rtf, rtt_s)
    if total == 0:
        return {"wait": 0.0, "model": 0.0, "network": 0.0}

    wait = chunk_s / total
    model = (rtf * chunk_s) / total
    network = rtt_s / total

    return {
        "wait": round(wait, 4),
        "model": round(model, 4),
        "network": round(network, 4),
    }

from collections import defaultdict
import numpy as np

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # 手搓
    # if not samples:
    #     return []
    # n = len(samples)
    # freq = defaultdict(int)
    # for x in samples:
    #     freq[x] += 1
    # # 按值排序并计算概率
    # pmf = [(val, count / n) for val, count in sorted(freq.items())]
    # return pmf

    # numpy版本
    if not samples:
        return []
    values, counts = np.unique(samples, return_counts=True)
    probs = counts / len(samples)
    return list(zip(values.tolist(), probs.tolist()))
import numpy as np


def logit_margin(z, t):
    """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably; returns a Python float."""
    z = np.asarray(z, dtype=np.float64)
    others = np.delete(z, t)  # 去掉目标 logit

    # 对 others 做稳定的 log-sum-exp ！！！
    max_other = np.max(others)
    logsumexp_others = max_other + np.log(np.sum(np.exp(others - max_other)))

    return float(z[t] - logsumexp_others)


def precision_cliff(dtype):
    """Smallest m >= 0 at which 1/(1 + exp(-m)) == 1 when evaluated in dtype; bisection on a float64 m.

    Returns the margin rounded to 4 decimals.
    """
    dt = np.dtype(dtype)
    one = np.array(1, dtype=dt)

    def saturated(m):
        # exp(-m) 在 float64 中计算，再舍入到目标 dtype
        e = np.array(np.exp(-m), dtype=dt)
        # 加法和除法都在目标 dtype 中完成
        denom = one + e
        q = one / denom
        return bool(q == one)

    lo, hi = 0.0, 100.0
    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        if saturated(mid):
            hi = mid
        else:
            lo = mid

    return round(float(hi), 4)


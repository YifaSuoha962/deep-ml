import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    # Your code here
    n = a.shape[0]
    # np.arange(n) 返回一个从 0 开始、到 n-1 结束、步长为 1 的一维整数数组
    return a[n - 1 - np.arange(n)]

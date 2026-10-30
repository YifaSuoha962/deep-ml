import numpy as np

def deterministic_hash(s):
    '''Converts a string to a deterministic integer.'''
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h


def create_hv(dim, seed):
    '''Creates a bipolar hypervector of given dimension using the seed.'''
    np.random.seed(seed % (2**32 - 1))
    return np.random.choice([-1, 1], dim)


def create_row_hv(row, dim, random_seeds):
    '''Create composite hypervector for a dataset row.
    
    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.
    '''
    bundled = np.zeros(dim, dtype=int)

    for name, value in row.items():
        base = random_seeds[name]

        # 字段名的超向量
        name_hv = create_hv(dim, base)

        # 值的种子由字段基础种子和值的哈希共同决定
        value_seed = (base + deterministic_hash(value)) % (2**32 - 1)
        value_hv = create_hv(dim, value_seed)

        # 绑定字段名与字段值，再累加到捆绑结果中
        bundled += name_hv * value_hv

    # 将累加结果映射回双极性向量；恰好为 0 时映射为 1
    return np.where(bundled >= 0, 1, -1)
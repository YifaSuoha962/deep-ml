def term_bounds(terms, ranges):
    """返回线性表达式可能取得的最小值和最大值，均为闭区间端点。"""
    vmin = vmax = terms.get("", 0)

    for var, coefficient in terms.items():
        # 常数已经计入；零系数不需要查询变量范围。
        if var == "" or coefficient == 0:
            continue

        lo, hi = ranges[var]
        last = hi - 1

        if coefficient > 0:
            vmin += coefficient * lo
            vmax += coefficient * last
        else:
            # 乘以负数后，大小关系反转。
            vmin += coefficient * last
            vmax += coefficient * lo

    return vmin, vmax


def fold_div(terms, c, ranges):
    """将 expr // c 化为 quotient + remainder // c。"""
    # 两个字典都要求常数键排在最前面。
    quotient = {"": 0}
    remainder = {"": 0}

    for var, coefficient in terms.items():
        if coefficient == 0:
            continue

        if coefficient % c == 0:
            # 可以整除的项直接进入商，包括可以整除的常数。
            quotient[var] = coefficient // c
        else:
            # 不能整除的项保留原系数，进入余项。
            remainder[var] = coefficient

    vmin, vmax = term_bounds(remainder, ranges)

    # 若 0 <= remainder < c，则 remainder // c 恒为 0。
    if 0 <= vmin and vmax < c:
        return quotient, None

    return quotient, remainder


def fold_mod(terms, c, ranges):
    """删除 c 的整数倍项，并判断剩余表达式是否无需继续取模。"""
    # 字典推导式保持输入顺序，同时删除零系数和可整除项。
    remainder = {
        var: coefficient
        for var, coefficient in terms.items()
        if coefficient != 0 and coefficient % c != 0
    }

    # 若常数被删除或输入未提供常数键，将其补在末尾。
    # 若非整除常数仍在字典中，则保留它原来的位置。
    if "" not in remainder:
        remainder[""] = 0

    vmin, vmax = term_bounds(remainder, ranges)
    exact = 0 <= vmin and vmax < c

    return remainder, exact


def unflatten(shape, flat_terms, ranges):
    """逐轴尝试将 (flat // stride) % axis_size 折叠为线性表达式。"""
    # 行优先布局：每一轴的 stride 等于其右侧所有维度的乘积。
    strides = [1] * len(shape)
    stride = 1

    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        stride *= shape[axis]

    result = []

    for axis_size, stride in zip(shape, strides):
        quotient, remainder = fold_div(flat_terms, stride, ranges)

        # 除法仍有未消除的余项时，不能仅用 quotient 表示 flat // stride。
        if remainder is not None:
            result.append("div/mod")
            continue

        axis_terms, exact = fold_mod(quotient, axis_size, ranges)

        # 只有除法和取模都成功消除，才返回该轴的线性表达式。
        result.append(axis_terms if exact else "div/mod")

    return result
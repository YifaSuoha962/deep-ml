from numbers import Number


def _is_expr(expr):
    return (
        isinstance(expr, tuple)
        and (
            (len(expr) == 2 and expr[0] == "neg")
            or (len(expr) == 3 and expr[0] in ("add", "mul"))
        )
    )


def size(expr):
    if not _is_expr(expr):
        return 1

    if expr[0] == "neg":
        return 1 + size(expr[1])

    return 1 + size(expr[1]) + size(expr[2])


def simplify(expr):
    if not _is_expr(expr):
        return expr

    op = expr[0]

    if op == "neg":
        node = ("neg", simplify(expr[1]))
    else:
        node = (op, simplify(expr[1]), simplify(expr[2]))

    while _is_expr(node):
        op = node[0]

        if op == "neg":
            child = node[1]

            if isinstance(child, Number):
                node = -child
                continue

            if _is_expr(child) and child[0] == "neg":
                node = simplify(child[1])
                continue

            return node

        left, right = node[1], node[2]

        # 常量折叠
        if isinstance(left, Number) and isinstance(right, Number):
            node = left + right if op == "add" else left * right
            continue

        if op == "add":
            if isinstance(left, Number) and left == 0:
                node = right
                continue
            if isinstance(right, Number) and right == 0:
                node = left
                continue

            # 常量移到右侧
            if isinstance(left, Number) and not isinstance(right, Number):
                node = ("add", right, left)
                continue

            # (x + c1) + c2 -> x + (c1 + c2)
            if (
                _is_expr(left)
                and left[0] == "add"
                and isinstance(left[2], Number)
                and isinstance(right, Number)
            ):
                node = simplify(("add", left[1], left[2] + right))
                continue

            # x + (y + c) -> (x + y) + c
            if (
                _is_expr(right)
                and right[0] == "add"
                and isinstance(right[2], Number)
            ):
                node = simplify(("add", ("add", left, right[1]), right[2]))
                continue

            # (x + c) + y -> (x + y) + c
            if (
                _is_expr(left)
                and left[0] == "add"
                and isinstance(left[2], Number)
                and not isinstance(right, Number)
            ):
                node = simplify(("add", ("add", left[1], right), left[2]))
                continue

            return node

        # 乘法恒等式与零
        if isinstance(left, Number) and left == 0:
            return 0
        if isinstance(right, Number) and right == 0:
            return 0
        if isinstance(left, Number) and left == 1:
            node = right
            continue
        if isinstance(right, Number) and right == 1:
            node = left
            continue

        # 常量移到右侧
        if isinstance(left, Number) and not isinstance(right, Number):
            node = ("mul", right, left)
            continue

        # (x * c1) * c2 -> x * (c1 * c2)
        if (
            _is_expr(left)
            and left[0] == "mul"
            and isinstance(left[2], Number)
            and isinstance(right, Number)
        ):
            node = simplify(("mul", left[1], left[2] * right))
            continue

        # (a + b) * c -> a * c + b * c
        if (
            isinstance(right, Number)
            and _is_expr(left)
            and left[0] == "add"
        ):
            node = simplify(
                (
                    "add",
                    ("mul", left[1], right),
                    ("mul", left[2], right),
                )
            )
            continue

        return node

    return node
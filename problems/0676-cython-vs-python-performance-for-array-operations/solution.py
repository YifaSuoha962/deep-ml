import math


def crossover_size(op: dict) -> int:
    """Return ceil((numpy_oh - cython_oh) / (cython_pe - numpy_pe)), 0 if <=0, -1 if NumPy never wins."""
    numpy_oh = op["numpy_overhead_ns"]
    cython_oh = op["cython_overhead_ns"]
    numpy_pe = op["numpy_per_elem_ns"]
    cython_pe = op["cython_per_elem_ns"]

    if cython_pe <= numpy_pe:
        return -1

    ratio = max(0, (numpy_oh - cython_oh) / (cython_pe - numpy_pe))

    return int(math.ceil(ratio))


def analyze_array_performance(n: int, operations: list) -> dict:
    """
    Analyze and compare array operation performance across Python, Cython, and NumPy.
    """
    ops_res_list = []
    crossovers = {}

    pipeline_python_us = 0.0
    pipeline_cython_us = 0.0
    pipeline_numpy_us = 0.0

    for op_dict in operations:
        name = op_dict["name"]

        # 未四舍五入的原始时间
        python_us = (op_dict["python_per_elem_ns"] * n + op_dict["python_overhead_ns"]) / 1000
        cython_us = (op_dict["cython_per_elem_ns"] * n + op_dict["cython_overhead_ns"]) / 1000
        numpy_us = (op_dict["numpy_per_elem_ns"] * n + op_dict["numpy_overhead_ns"]) / 1000

        # pipeline 用未四舍五入的时间累加
        pipeline_python_us += python_us
        pipeline_cython_us += cython_us
        pipeline_numpy_us += numpy_us

        # speedup 用未四舍五入的时间计算，最后再 round
        cython_speedup = python_us / cython_us if cython_us > 0 else 0.0
        numpy_speedup = python_us / numpy_us if numpy_us > 0 else 0.0

        # best 基于未四舍五入时间，平局按字母序
        best = min(
            [("python", python_us), ("cython", cython_us), ("numpy", numpy_us)],
            key=lambda item: (item[1], item[0]),
        )[0]

        ops_res = {
            "name": name,
            "python_us": round(python_us, 2),
            "cython_us": round(cython_us, 2),
            "numpy_us": round(numpy_us, 2),
            "cython_speedup": round(cython_speedup, 2),
            "numpy_speedup": round(numpy_speedup, 2),
            "best": best,
        }
        ops_res_list.append(ops_res)

        # crossover 存整数
        crossovers[name] = crossover_size(op_dict)

    # recommended 基于未四舍五入的 pipeline 总时间，平局按字母序
    recommended = min(
        [
            ("python", pipeline_python_us),
            ("cython", pipeline_cython_us),
            ("numpy", pipeline_numpy_us),
        ],
        key=lambda item: (item[1], item[0]),
    )[0]

    return {
        "operations": ops_res_list,
        "pipeline_python_us": round(pipeline_python_us, 2),
        "pipeline_cython_us": round(pipeline_cython_us, 2),
        "pipeline_numpy_us": round(pipeline_numpy_us, 2),
        "recommended": recommended,
        "crossover_sizes": crossovers,
    }
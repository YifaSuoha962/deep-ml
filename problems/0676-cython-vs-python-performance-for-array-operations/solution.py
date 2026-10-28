import math
from collections import defaultdict

def crossover_size(op: dict) -> int:
    """Return ceil((numpy_oh - cython_oh) / (cython_pe - numpy_pe)), 0 if that is <= 0, -1 if NumPy never wins."""
    numpy_oh = op["numpy_overhead_ns"]
    cython_oh = op["cython_overhead_ns"]
    numpy_pe = op["numpy_per_elem_ns"]
    cython_pe = op["cython_per_elem_ns"]
    if cython_pe <= numpy_pe:
        return {op["name"]: -1}
    return {op["name"]: int(max(0, math.ceil((numpy_oh - cython_oh) / (cython_pe - numpy_pe))))}

def analyze_array_performance(n: int, operations: list) -> dict:
    """
    Analyze and compare array operation performance across Python, Cython, and NumPy.
    
    Args:
        n: Array size (number of elements)
        operations: List of dicts with performance parameters for each operation
    
    Returns:
        dict with per-operation metrics, pipeline totals, recommendation,
        and crossover sizes.
    """
    ops_res_list = []
    crossovers = defaultdict()

    # times over all operations
    pipeline_python_us = 0
    pipeline_cython_us = 0
    pipeline_numpy_us = 0

    for op_dict in operations:
        # per method
        python_us = round((op_dict["python_per_elem_ns"] * n + op_dict["python_overhead_ns"]) / 1000, 2)
        cython_us = round((op_dict["cython_per_elem_ns"] * n + op_dict["cython_overhead_ns"]) / 1000, 2)
        numpy_us = round((op_dict["numpy_per_elem_ns"] * n + op_dict["numpy_overhead_ns"]) / 1000, 2)
        
        times = {"cython": cython_us, "python": python_us, "numpy": numpy_us}
        pipeline_python_us += python_us
        pipeline_cython_us += cython_us
        pipeline_numpy_us += numpy_us

        cython_speedup = round(python_us / cython_us, 2)
        numpy_us_speedup = round(python_us / numpy_us, 2)

        bst_ver = sorted(times, key=lambda name: (times[name], name))[0]

        ops_res = {"name": op_dict["name"], "python_us": python_us, "cython_us": cython_us, "numpy_us": numpy_us, "cython_speedup": cython_speedup, "numpy_speedup": numpy_us_speedup, "best": bst_ver}
        ops_res_list.append(ops_res)

        cr = crossover_size(op_dict)
        crossovers.update(cr)
    
    recommended = min([
            ("python", pipeline_python_us),
            ("cython", pipeline_cython_us),
            ("numpy", pipeline_numpy_us),
        ],key=lambda item: (item[1], item[0]),)[0]

    return {"operations": ops_res_list, "pipeline_python_us": pipeline_python_us, "pipeline_cython_us": pipeline_cython_us, "pipeline_numpy_us": pipeline_numpy_us, "recommended": recommended, "crossover_sizes": crossovers}
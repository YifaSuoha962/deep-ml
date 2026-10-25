import math

def _space_dim(space: dict) -> int:
    """Return the flattened dimension of a gym-style space description."""
    if space["type"] == "Box":
        return math.prod(space["shape"])
    # Discrete / MultiBinary
    return space["n"]


def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Automatically determine MLP architecture from environment space definitions.

    Args:
        obs_space: dict with 'type' and type-specific keys defining the observation space
        act_space: dict with 'type' and type-specific keys defining the action space
        hidden_layers: list of ints specifying hidden layer widths

    Returns:
        dict with keys:
          'input_dim': int
          'output_dim': int
          'layer_shapes': list of ((weight_rows, weight_cols), (bias_dim,)) tuples
          'total_params': int
    """
    input_dim = _space_dim(obs_space)
    output_dim = _space_dim(act_space)

    # 完整维度序列：输入 -> 隐藏层... -> 输出
    dims = [input_dim] + hidden_layers + [output_dim]

    layer_shapes = []
    total_params = 0

    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layer_shapes.append(((in_dim, out_dim), (out_dim,)))
        total_params += in_dim * out_dim + out_dim

    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "layer_shapes": layer_shapes,
        "total_params": total_params,
    }
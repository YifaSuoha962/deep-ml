def cal_dim(ins_space):
    dim = None
    type_sp = ins_space["type"]
    if type_sp == "Box":
        dim = 1
        for i in ins_space["shape"]:
            dim *= i
    else:
        dim = ins_space["n"]
    return dim


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
    dim_in = cal_dim(obs_space)
    dim_out = cal_dim(act_space)
    elems = hidden_layers + [dim_out]
    mlp_list = list()
    tot_params = 0
    for i, dim in enumerate(elems):
        pre_dim = dim_in if i == 0 else elems[i-1]
        mlp_list.append(((pre_dim, dim), (dim,)))
        tot_params += pre_dim * dim + dim
            
    return {'input_dim': dim_in, 'output_dim': dim_out, 'layer_shapes': mlp_list, 'total_params': tot_params}
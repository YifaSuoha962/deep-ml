import torch
import torch.nn as nn

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    # TODO: return a Sequential of Linear -> ReLU -> Linear
    linear_1 = nn.Linear(in_dim, hidden_dim)
    linear_2 = nn.Linear(hidden_dim, out_dim)
    relu_func = nn.ReLU()
    mlp = nn.Sequential(linear_1, relu_func, linear_2)
    return mlp

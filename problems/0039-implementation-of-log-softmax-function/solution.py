import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here
    scores = torch.tensor(scores)
    log_exp_sum = torch.log(torch.sum(torch.exp(scores)))
    log_softmax = scores - log_exp_sum
    return log_softmax
    # return [x.item() for x in log_softmax]

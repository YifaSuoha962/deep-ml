import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    scores = torch.tensor(scores, dtype=torch.float32)
    exp_scores = torch.exp(scores - scores.max())  # 数值稳定性技巧
    return (exp_scores / exp_scores.sum()).tolist()

import torch

def mutual_information(joint_prob: torch.Tensor) -> float:
    """
    Compute the mutual information between two random variables.
    
    Args:
        joint_prob: 2D joint probability distribution P(X,Y) as a torch.Tensor
    
    Returns:
        Mutual information I(X;Y)
    """
    # Your code here
    # Use float64 for better numerical precision
    p_xy = torch.as_tensor(joint_prob, dtype=torch.float64)

    # marginal distribution
    p_x = p_xy.sum(dim=1, keepdim=True)
    p_y = p_xy.sum(dim=0, keepdim=True)
    
    # independent prob multiplication
    px_py = p_x * p_y

    # Only compute terms where P(x,y) > 0
    mask = p_xy > 0

    mi = torch.sum(p_xy[mask] * torch.log(p_xy[mask] / px_py[mask]))

    return mi.item()
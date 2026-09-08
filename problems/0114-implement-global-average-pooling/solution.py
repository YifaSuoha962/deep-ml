import torch

def global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Performs Global Average Pooling on a 3D tensor representing feature maps.
    
    Args:
        x: Input tensor of shape (height, width, channels)
    
    Returns:
        1D tensor of shape (channels,) with average values per channel
    """
    return x.flatten(0, 1).mean(dim=0)
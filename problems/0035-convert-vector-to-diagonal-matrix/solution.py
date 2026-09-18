import torch
from typing import Union

def make_diagonal(x: Union[torch.Tensor, list, "np.ndarray"]) -> torch.Tensor:
    """Return a diagonal matrix whose diagonal elements are the 1-D values in `x`.
    If `x` is not a torch tensor it will be converted automatically.
    
    Hint: `torch.diag_embed` makes this very short!
    """
    # ✏️  Your code here
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    return torch.diag_embed(x)
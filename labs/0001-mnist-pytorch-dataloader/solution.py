import torch

class MyTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        Must be non-identity and deterministic.
        """
        # TODO: implement your custom transformation logic here
        # 1. normalization 
        eps = 1e-6
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / (std + eps)
        noise = torch.randn_like(x) * 0.05 
        x_out = x_norm + noise
        x_out = x_out.clamp(-3.0, 3.0)
        return x_out
        # pass

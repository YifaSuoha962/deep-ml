import torch

def forward_diffusion(x_0: torch.Tensor, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: torch.Tensor) -> torch.Tensor:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (torch.Tensor)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise tensor (same shape as x_0)
    
    Returns:
        Noisy sample x_t as torch.Tensor
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    alpha_bar_t = alphas_cum[t-1]   # left shift time to align with index
    return x_0 * torch.sqrt(alpha_bar_t) + noise * torch.sqrt((1 - alpha_bar_t))

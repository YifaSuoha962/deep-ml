import torch
from torch.distributions import Normal, kl_divergence as kl_div_fn
import torch.nn.functional as F

def vae_loss(x: torch.Tensor, x_reconstructed: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: torch.Tensor of shape (batch_size, features), original input
        x_reconstructed: torch.Tensor of shape (batch_size, features), reconstructed input
        mu: torch.Tensor of shape (batch_size, latent_dim), latent mean
        log_var: torch.Tensor of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    # Your code here
    l_recontruction = torch.mean(torch.sum((x - x_reconstructed) ** 2, dim=-1))
    kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=-1))
    l_tot = l_recontruction + kl_div
    return (l_tot.item(), l_recontruction.item(), kl_div.item())
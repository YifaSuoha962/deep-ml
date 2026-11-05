import torch

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) -> torch.Tensor:
    """
    Compute the KL divergence between two normal distributions P and Q.
    KL(P || Q) = log(sigma_q / sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2 * sigma_q^2) - 0.5
    """
    # 将输入统一转换为 float32 的 Tensor
    mu_p = torch.as_tensor(mu_p, dtype=torch.float32)
    sigma_p = torch.as_tensor(sigma_p, dtype=torch.float32)
    mu_q = torch.as_tensor(mu_q, dtype=torch.float32)
    sigma_q = torch.as_tensor(sigma_q, dtype=torch.float32)

    # 按照公式计算：注意分子是 sigma_p^2
    term1 = torch.log(sigma_q / sigma_p)  # 等价于 log(sigma_q) - log(sigma_p)
    term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
    kl = term1 + term2 - 0.5
    
    return kl
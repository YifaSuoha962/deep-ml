import torch

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    # pass
    probs = torch.tensor(probabilities, dtype=torch.float32)
    log_ps = torch.log(probs)
    mean_log_ps = log_ps.mean()
    ppl = torch.exp(-mean_log_ps)
    return ppl.item()

import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    # compact stability
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    target_logps = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # l_ce = -1 * torch.mean(target_logps)
    l_ce = - target_logps.mean()
    return l_ce


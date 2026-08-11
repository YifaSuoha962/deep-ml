import numpy as np
import torch

def train(train_chosen, train_rejected, n_items, beta=0.5):
    """
    Train a preference policy with DPO on unpaired item indices.

    A frozen uniform reference over `n_items` catalog items is assumed.
    You only observe pairwise preferences (chosen_idx, rejected_idx).

    Args:
        train_chosen: np.ndarray[int] shape (N,) — preferred item indices
        train_rejected: np.ndarray[int] shape (N,) — rejected item indices
        n_items: size of the discrete item catalog
        beta: DPO temperature

    Returns:
        score: callable(indices: np.ndarray[int]) -> np.ndarray[float]
               Higher score = more preferred. Used to evaluate whether
               score(chosen) > score(rejected) on held-out pairs.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chosen = torch.tensor(train_chosen, dtype=torch.long, device=device)
    rejected = torch.tensor(train_rejected, dtype=torch.long, device=device)

    # Learnable logits (un-normalized scores) for each item
    logits = torch.zeros(n_items, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([logits], lr=0.01)

    n_epochs = 500
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        log_pi = torch.log_softmax(logits, dim=0)          # (n_items,)

        log_pi_chosen = log_pi[chosen]                     # (N,)
        log_pi_rejected = log_pi[rejected]                 # (N,)
        diff = log_pi_chosen - log_pi_rejected             # (N,)

        # DPO loss: -log σ(β * (logπ(chosen) - logπ(rejected)))
        loss = -torch.log(torch.sigmoid(beta * diff)).mean()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}")

    # Return a score function that returns the raw logits for the queried indices.
    def score(indices):
        indices = np.asarray(indices, dtype=int)
        with torch.no_grad():
            logits_np = logits.detach().cpu().numpy()
        return logits_np[indices]

    return score
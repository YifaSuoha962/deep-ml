import numpy as np

def mc_q_estimation(episodes: list, n_states: int, n_actions: int, gamma: float, method: str = "first_visit") -> list:
    """
    Estimate action-value function Q(s, a) from episodes using Monte Carlo methods.
    
    Args:
        episodes: List of episodes, each episode is a list of (state, action, reward) tuples.
        n_states: Number of states.
        n_actions: Number of actions.
        gamma: Discount factor.
        method: "first_visit" or "every_visit".
    
    Returns:
        Nested list of shape (n_states, n_actions) with Q-values rounded to 4 decimal places.
    """
    # 1. Initialize return sums and visit counts
    returns_sum = np.zeros((n_states, n_actions))
    counts = np.zeros((n_states, n_actions))

    # 2. traverse episodes
    for ep in episodes:
        # Compute discounted returns backward along each trajectory
        G = 0.0
        times = len(ep)
        returns = [0.0] * len(ep)

        for t in range(times - 1, -1, -1):
            state, action, reward = ep[t]
            G = reward + gamma * G
            returns[t] = G

        # track the first visited state 
        visited = set()

        # 3. accumulate returns for each pairs
        for t, (state, action, reward) in enumerate(ep):
            pair = (state, action)
            if method == "first_visit":
                if pair in visited:
                    continue
                visited.add(pair)  # update by the latest reward

            returns_sum[state, action] += returns[t]
            counts[state, action] += 1
    
    # 4. Compute average returns
    Q = np.zeros((n_states, n_actions))

    """
    trick: out=Q -- 计算结果会直接写入 Q，而不是返回一个新数组。这样避免了额外内存分配。
    note: out 必须是与输入兼容的数组
    """
    np.divide(
        returns_sum,
        counts,
        out=Q,
        where=counts > 0
    )

    # 5. Round to 4 decimal places and return nested list
    return np.round(Q, 4).tolist()            

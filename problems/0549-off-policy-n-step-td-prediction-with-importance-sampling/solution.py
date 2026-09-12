import numpy as np


def off_policy_nstep_td(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    n: int,
    alpha: float,
    gamma: float
) -> np.ndarray:
    """
    Off-policy n-step TD prediction for state values using importance sampling.

    Args:
        episodes: List of episodes, each a list of
                  (state, action, reward) tuples.
        behavior_policy: b(a|s), shape (num_states, num_actions).
        target_policy: pi(a|s), shape (num_states, num_actions).
        num_states: Number of states.
        num_actions: Number of actions.
        n: Number of steps for the n-step return.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        V: numpy array of shape (num_states,)
    """

    # Initialize state values to zero
    V = np.zeros(num_states, dtype=float)

    behavior_policy = np.asarray(behavior_policy, dtype=float)
    target_policy = np.asarray(target_policy, dtype=float)

    if n <= 0:
        raise ValueError("n must be at least 1.")

    # Process episodes in order
    for episode in episodes:

        T = len(episode)

        # Process time steps from beginning to end
        for t in range(T):

            state_t = episode[t][0]

            # Number of rewards included in this n-step window
            h = min(n, T - t)

            # ---------------------------------
            # 1. Compute n-step return G
            # ---------------------------------
            G = 0.0

            for k in range(h):
                reward = episode[t + k][2]
                G += (gamma ** k) * reward

            # Bootstrap only if n-step lookahead
            # does not reach the terminal state
            if t + n < T:
                next_state = episode[t + n][0]
                G += (gamma ** n) * V[next_state]

            # ---------------------------------
            # 2. Importance sampling ratio
            # ---------------------------------
            rho = 1.0

            for k in range(h):
                state = episode[t + k][0]
                action = episode[t + k][1]

                b_prob = behavior_policy[state, action]
                pi_prob = target_policy[state, action]

                if b_prob == 0:
                    raise ValueError(
                        "Behavior policy probability cannot be zero "
                        "for an observed action."
                    )

                rho *= pi_prob / b_prob

            # ---------------------------------
            # 3. TD update
            # ---------------------------------
            V[state_t] += alpha * rho * (G - V[state_t])

    return V
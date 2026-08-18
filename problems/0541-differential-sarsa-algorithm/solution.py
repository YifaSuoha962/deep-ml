def differential_sarsa(
    transitions: dict,
    initial_state: str,
    alpha: float,
    beta: float,
    num_steps: int
) -> tuple:
    """
    Differential Sarsa for the average-reward continuing setting.
    
    Args:
        transitions: dict mapping (state, action) -> (reward, next_state)
        initial_state: starting state
        alpha: step size for Q-value updates
        beta: step size for average reward estimate
        num_steps: number of steps to simulate
    
    Returns:
        Tuple of (Q, R_bar) where Q is a dict {(state, action): float}
        and R_bar is a float.
    """

    # Initialize Q for every state-action pair in transitions
    Q = {state_action: 0.0 for state_action in transitions}

    # Initialize average reward estimate
    R_bar = 0.0

    # Collect available actions for each state
    actions_by_state = {}
    for state, action in transitions:
        actions_by_state.setdefault(state, []).append(action)

    # Sort actions so ties are resolved lexicographically
    for state in actions_by_state:
        actions_by_state[state].sort()

    def greedy_action(state):
        """Choose the greedy action, breaking ties lexicographically."""
        actions = actions_by_state[state]

        return min(
            actions,
            key=lambda action: (-Q[(state, action)], action)
        )

    # Initial state and action
    state = initial_state
    action = greedy_action(state)

    for _ in range(num_steps):
        # Take current action
        reward, next_state = transitions[(state, action)]

        # Choose next action greedily from the current Q-table
        next_action = greedy_action(next_state)

        # Differential TD error
        delta = (
            reward
            - R_bar
            + Q[(next_state, next_action)]
            - Q[(state, action)]
        )

        # Update average reward estimate
        R_bar += beta * delta

        # Update action value
        Q[(state, action)] += alpha * delta

        # Move to the next state-action pair
        state = next_state
        action = next_action

    return Q, float(R_bar)
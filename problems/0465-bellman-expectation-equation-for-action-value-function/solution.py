import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    """
    Compute Q^pi(s, a) for all state-action pairs using iterative policy evaluation
    based on the Bellman expectation equation for action-values.

    Args:
        states: List of state identifiers
        actions: List of action identifiers
        transition_probs: Dict mapping (s, a, s') -> P(s'|s,a)
        rewards: Dict mapping (s, a, s') -> R(s,a,s')
        policy: Dict mapping (s, a) -> pi(a|s)
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        Dict mapping (s, a) -> Q-value rounded to 4 decimal places
    """
    # 初始化所有 Q(s, a) 为 0
    q = {(s, a): 0.0 for s in states for a in actions}

    # 没有状态动作对时，无需迭代
    if not q:
        return {}

    while True:
        new_q = {}
        max_change = 0.0

        for s in states:
            for a in actions:
                value = 0.0

                # 对所有可能的下一状态求期望
                for next_state in states:
                    key = (s, a, next_state)
                    prob = transition_probs.get(key, 0.0)
                    reward = rewards.get(key, 0.0)

                    # 下一状态下，按照策略对后续动作价值求期望
                    next_value = sum(
                        policy.get((next_state, next_action), 0.0)
                        * q[(next_state, next_action)]
                        for next_action in actions
                    )

                    value += prob * (reward + gamma * next_value)

                new_q[(s, a)] = value
                max_change = max(max_change, abs(value - q[(s, a)]))

        # 整轮计算完成后再替换，确保本轮都使用上一轮的 Q 值
        q = new_q

        if max_change < theta:
            break

    return {key: round(value, 4) for key, value in q.items()}
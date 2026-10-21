def random_walk_td_mc(
    episodes: list,
    n_states: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
) -> tuple:
    td_values = [0.0] * n_states

    for episode in episodes:
        for state, reward, next_state, done in episode:
            successor_value = 0.0 if done else td_values[next_state]
            target = reward + gamma * successor_value
            td_values[state] += alpha * (target - td_values[state])

    return_sums = [0.0] * n_states
    visit_counts = [0] * n_states

    for episode in episodes:
        returns = [0.0] * len(episode)
        running_return = 0.0

        for i in range(len(episode) - 1, -1, -1):
            _, reward, _, _ = episode[i]
            running_return = reward + gamma * running_return
            returns[i] = running_return

        visited = set()
        for i, (state, _, _, _) in enumerate(episode):
            if state not in visited:
                visited.add(state)
                return_sums[state] += returns[i]
                visit_counts[state] += 1

    mc_values = [
        total / count if count else 0.0
        for total, count in zip(return_sums, visit_counts)
    ]

    return td_values, mc_values
from typing import List, Tuple


def random_walk_td_mc(
    episodes: list,
    n_states: int,
    gamma: float = 1.0,
    alpha: float = 0.1,
) -> tuple:
    """
    Run TD(0) and first-visit Monte Carlo prediction on random-walk episodes.

    Each transition is ``(state, reward, next_state, done)``.  ``next_state``
    is ignored for terminal transitions because terminal states are implicit.
    """
    # 两种方法都从全 0 的价值函数开始。
    # td_values 会在每条转移之后立即更新。
    td_values = [0.0] * n_states

    # TD(0)：按 episode 和转移顺序在线更新。
    # 当前状态的目标值由即时奖励和后继状态的当前估计组成。
    for episode in episodes:
        """
        注意计算顺序，是从起点到终点！
        """
        for state, reward, next_state, done in episode:
            # 到达终止状态后没有后继状态，因此后继价值定义为 0。
            successor_value = 0.0 if done else td_values[next_state]

            # TD 目标：r + gamma * V(s')。
            # 用当前估计和目标之间的误差进行增量更新。
            td_target = reward + gamma * successor_value
            td_values[state] += alpha * (td_target - td_values[state])

    # MC 方法保存每个状态在各个 episode 中的回报总和与出现次数。
    # 最后用总和除以次数，得到每个状态的平均回报。
    return_sums = [0.0] * n_states
    visit_counts = [0] * n_states

    for episode in episodes:
        # returns[index] 表示从第 index 条转移对应状态开始的完整折扣回报。
        returns = [0.0] * len(episode)
        running_return = 0.0

        # 从 episode 末尾反向计算：G_t = R_{t+1} + gamma * G_{t+1}。
        for index in range(len(episode) - 1, -1, -1):
            _, reward, _, _ = episode[index]
            running_return = reward + gamma * running_return
            returns[index] = running_return

        # 首访 MC：同一 episode 内一个状态只统计第一次出现的位置。
        seen_states = set()
        for index, (state, _, _, _) in enumerate(episode):
            if state not in seen_states:
                seen_states.add(state)
                return_sums[state] += returns[index]
                visit_counts[state] += 1

    # 没有被任何 episode 访问过的状态保持初始估计 0.0。
    mc_values = [
        total / count if count else 0.0
        for total, count in zip(return_sums, visit_counts)
    ]

    """
    TD performs online update and estimation, 
    while mc performs offline statistics 
    """
    return td_values, mc_values


def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    # Write code here
    T = len(rewards)
    advantages = [0.0] * T
    last_adv = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lam * last_adv
        last_adv = advantages[t]
    return advantages
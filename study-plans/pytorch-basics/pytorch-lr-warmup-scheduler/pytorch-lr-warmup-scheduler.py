import math

def warmup_cosine_schedule(base_lr: float, warmup_steps: int, total_steps: int) -> list[float]:
    """
    Returns one Python float learning rate per step.
    """
    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        lrs.append(lr)
    return lrs

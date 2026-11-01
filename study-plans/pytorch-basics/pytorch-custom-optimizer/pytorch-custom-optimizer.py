import torch

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Returns the closure loss when provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if momentum != 0:
                        state = self.state[p]
                        if "velocity" not in state:
                            state["velocity"] = torch.zeros_like(p)
                        old_velocity = state["velocity"]
                        # v_t = momentun * v_{t-1} + grad
                        state["velocity"] = momentum * old_velocity + grad
                        # p_{t+1} = p_t - lr * v_t
                        new_p = p - lr * state["velocity"]
                    else:
                        new_p = p - lr * grad
                    p.copy_(new_p)
        return loss
                    
                

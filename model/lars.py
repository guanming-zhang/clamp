import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """
    LARS (Layer-wise Adaptive Rate Scaling) optimizer for large-batch training.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, eta=0.001, eps=1e-8):
        """
        params: Model parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay (L2 penalty) factor (default: 0)
        eta: Trust coefficient for layer-wise learning rate adaptation
        eps: A small value to prevent division by zero
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0 and p_norm != 0 and g_norm != 0:
                    lars_lr = p_norm / (g_norm + p_norm * weight_decay + self.eps)
                    lars_lr *= self.eta

                    d_p = d_p.add(p, alpha=weight_decay)
                    d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
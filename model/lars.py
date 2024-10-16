import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """
    LARS (Layer-wise Adaptive Rate Scaling) optimizer for large-batch training.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, eta=0.001, eps=1e-9):
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
    def step(self):
        """
        Perform a single optimization step.
        """
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            eta = group['eta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                param_norm = torch.norm(p)
                grad_norm = torch.norm(grad)

                # Apply weight decay if applicable
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # LARS scaling
                if param_norm > eps and grad_norm > eps:
                    local_lr = eta * param_norm / (grad_norm + weight_decay * param_norm + eps)
                else:
                    local_lr = 1.0

                # Momentum update
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=local_lr * lr)

                p.add_(buf, alpha=-1.0)
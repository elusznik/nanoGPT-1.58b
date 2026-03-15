import os
import torch
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = [p for p in params if p.numel() > 0]
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    raise RuntimeError("Muon strictly requires all parameters to have >= 2 dimensions.")
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                
                buf.mul_(group['momentum']).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=group['momentum'])
                else:
                    g = buf
                
                # Orthogonalize the update
                update = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                update *= max(1, update.size(0)/update.size(1))**0.5
                
                p.data.add_(update, alpha=-group['lr'])

        return loss
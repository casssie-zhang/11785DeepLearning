import numpy as np

from mytorch.optim.optimizer import Optimizer
from mytorch.tensor import Tensor


class Adam(Optimizer):
    """The Adam optimizer (Kingma and Ba 2015)
    https://arxiv.org/abs/1412.6980

    >>> optimizer = Adam(model.parameters(), lr=1e-3)

    Args:
        params (list): <some module>.parameters()
        lr (float): learning rate (eta)
        betas (tuple(float, float)): coefficients for computing running avgs
                                     of gradient and its square
        eps (float): term added to denominator to improve numerical stability

    Inherits from:
        Optimizer (optim.optimizer.Optimizer)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params)
        self.betas = betas
        self.eps = eps
        self.lr = lr
        
        # Initialize moment estimates for each parameter tensor as zero arrays
        self.state = [{'m_t': np.zeros(p.shape), 'v_t': np.zeros(p.shape)} for p in self.params]
            
        # To keep track of the number of steps so far
        self.t = 0
    
    def step(self):
        b1, b2 = self.betas # for convenience
        self.t += 1 # increment step num

        b1_correction = 1 - pow(b1, self.t)
        b2_correction = 1 - pow(b2, self.t)
        for param, state in zip(self.params, self.state):
            grad = param.grad.data

            state['m_t'] = b1 * state['m_t'] + (1 - b1)  * grad
            state['v_t'] = b2 * state['v_t'] + (1 - b2) * grad * grad

            step_size = self.lr / b1_correction
            d = np.sqrt(state['v_t']) / np.sqrt(b2_correction) + self.eps
            param.data = param.data - state['m_t'] * step_size / d


        # TODO: Implement ADAM

from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """

        batch_size = Tensor(x.shape[0])



        mean = x.sum(axis=0) / batch_size
        variance = pow((x - mean), 2).sum(axis=0) / batch_size

        if self.is_train:
            x_norm = (x - mean) / pow(self.eps + variance, 0.5)
            y_norm = self.gamma * x_norm + self.beta

            unbiased_variance = pow((x - mean), 2).sum(axis=0) / (batch_size - 1)
            self.running_mean = (Tensor(1) - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (Tensor(1) - self.momentum) * self.running_var + \
                               self.momentum * unbiased_variance
        else:
            x_norm = (x - self.running_mean) / pow(self.eps + self.running_var, 0.5)
            y_norm = self.gamma * x_norm + self.beta

        return y_norm





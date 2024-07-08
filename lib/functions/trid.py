import numpy as np

from lib.functions.function import Function


class Trid(Function):
    def __init__(self, interval=None, n_dims=2):
        self.n_dims = n_dims
        self.name = 'Trid'
        bounds = np.full((n_dims, 2), [-n_dims, n_dims])
        super().__init__(bounds)

    def eval(self, X):
        return np.sum(
                [ (X[:,i] - 1) ** 2 for i in range(self.n_dims)], axis=0
            ) - np.sum(
                [ X[:,i] * X[:,i+1] for i in range(self.n_dims - 1)], axis=0
            ) + (self.n_dims * (self.n_dims + 4) * (self.n_dims - 1)) / 6

import numpy as np

from lib.functions.function import Function


class Zakharov(Function):
    def __init__(self, interval=[-5, 10], n_dims=2):
        self.n_dims = n_dims
        self.name = 'Zakharov'
        bounds = np.full((n_dims, 2), interval)
        super().__init__(bounds)

    def eval(self, X):
        return np.sum(
                [ X[:,i] ** 2 for i in range(self.n_dims)], axis=0
            ) + np.sum(
                [ 0.5 * i * X[:,i] for i in range(self.n_dims)], axis=0
            ) ** 2 + np.sum(
                [ 0.5 * i * X[:,i] for i in range(self.n_dims)], axis=0
            ) ** 4

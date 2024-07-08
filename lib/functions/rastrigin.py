import numpy as np

from lib.functions.function import Function


class Rastrigin(Function):
    def __init__(self, interval=[-5.12, 5.12], n_dims=2):
        self.n_dims = n_dims
        self.name = 'Rastrigin'
        bounds = np.full((n_dims, 2), interval)
        super().__init__(bounds)

    def eval(self, X):
        return 10 * self.n_dims + np.sum([
            X[:,i] ** 2 - 
            10 * np.cos(2 * np.pi * X[:,i])
            for i in range(self.n_dims)], axis=0)

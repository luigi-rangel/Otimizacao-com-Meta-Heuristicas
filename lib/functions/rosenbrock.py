import numpy as np

from lib.functions.function import Function


class Rosenbrock(Function):
    def __init__(self, interval=[-2.048, 2.048], n_dims=2):
        self.n_dims = n_dims
        self.name = 'Rosenbrock'
        bounds = np.full((n_dims, 2), interval)
        super().__init__(bounds)

    def eval(self, X):
        return np.sum([
            100 * (X[:, i + 1] - X[:, i] ** 2) ** 2
            + (X[:, i] - 1) ** 2
            for i in range(self.n_dims - 1)], axis=0)

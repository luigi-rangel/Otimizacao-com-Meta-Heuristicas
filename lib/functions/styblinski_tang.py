import numpy as np

from lib.functions.function import Function


class StyblinskiTang(Function):
    def __init__(self, interval=[-5, 10], n_dims=2):
        self.n_dims = n_dims
        self.name = 'Styblinski'
        bounds = np.full((n_dims, 2), interval)
        super().__init__(bounds)

    def eval(self, X):
        return 0.5 * np.sum(
            [X[:, i] ** 4 - 16 * X[:, i] ** 2 + 5 * X[:, i] for i in range(self.n_dims)], axis=0
        ) + 39.16599 * self.n_dims

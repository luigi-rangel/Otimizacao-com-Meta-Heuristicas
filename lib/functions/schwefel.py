import numpy as np

from lib.functions.function import Function


class Schwefel(Function):
    def __init__(self, interval=[-500, 500], n_dims=2):
        self.n_dims = n_dims
        self.name = 'Schwefel'
        bounds = np.full((n_dims, 2), interval)
        super().__init__(bounds)

    def eval(self, X):
        return 418.9829 * self.n_dims - np.sum([
            X[:,i] * np.sin(np.sqrt(abs(X[:,i])))
            for i in range(self.n_dims)], axis=0)

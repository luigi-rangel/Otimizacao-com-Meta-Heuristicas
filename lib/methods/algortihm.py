import numpy as np


class Algorithm:
    def __init__(self, F, n_inds, n_gens, keep_x=False,
                 keep_best=False):
        self.rng = np.random.default_rng()
        self.stop = False

        self.i = 1
        self.n_inds = n_inds
        self.n_gens = n_gens
        self.F = F
        self.keep_x = keep_x
        self.keep_best = keep_best

        self.X = self.rng.uniform(
            *self.F.bounds.transpose(), [self.n_inds, self.F.bounds.shape[0]])
        self.apt_X = self.F.eval(self.X)
        idx_g = np.argmin(self.apt_X)
        self.best = self.X[idx_g]
        self.apt_best = self.apt_X[idx_g]

        self.X_history = []
        self.apt_X_history = []
        self.best_history = []
        self.apt_best_history = []
        self.update_stop()

    def run(self):
        while not self.stop:
            self.step()
        return self

    def step(self):
        pass

    def update_stop(self):
        self.stop = self.i == self.n_gens

    def update_history(self):
        if self.keep_x:
            self.X_history.append(self.X)
            self.apt_X_history.append(self.apt_X)
        if self.keep_best:
            self.best_history.append(self.best)
            self.apt_best_history.append(self.apt_best)

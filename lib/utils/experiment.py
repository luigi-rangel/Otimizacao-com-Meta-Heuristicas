import numpy as np


class Experiment:
    def __init__(self, tag, algorithm, params):
        self.tag = tag
        self.algorithm = algorithm
        self.params = params
        self.x_histories = []
        self.apt_x_histories = []
        self.best_histories = []
        self.apt_best_histories = []

    def run(self, n_gens, keep_x, keep_best):
        alg = self.algorithm(**self.params, n_gens=n_gens, keep_x=keep_x,
                                keep_best=keep_best).run()
        if keep_x:
            self.x_histories.append(alg.X_history)
            self.apt_x_histories.append(alg.apt_X_history)
        if keep_best:
            self.best_histories.append(alg.best_history)
            self.apt_best_histories.append(alg.apt_best_history)
        return self

import numpy as np
from lib.methods.algortihm import Algorithm


class DE(Algorithm):
    def __init__(self, F, n_inds, n_gens, mut=0.8, crossp=0.7,
                 keep_x=False, keep_best=False, variant=['rand', 1]):
        super().__init__(F, n_inds, n_gens, keep_x, keep_best)

        self.mut = mut
        self.crossp = crossp
        self.variant = variant
        self.best_idx = self.apt_X.argmin()
        self.mutant = np.zeros((self.n_inds, self.F.bounds.shape[0]))
        self.trial = np.zeros((self.n_inds, self.F.bounds.shape[0]))
        self.apt_trial = np.zeros(self.n_inds)
        super().update_history()

    def mutate(self):
        if self.variant[0] == 'rand':
            n_terms = 2 * self.variant[1] + 1
            r_idxs = np.zeros(n_terms)
            r = np.zeros((self.n_inds, n_terms, self.F.bounds.shape[0]))
            for i in range(self.n_inds):
                pos = []
                for j in range(self.n_inds):
                    if i != j:
                        pos.append(j)
                r_idxs = self.rng.choice(pos, n_terms, replace=False)
                r[i] = self.X[np.array(r_idxs, int)]
        if self.variant[0] == 'best':
            n_terms = 2 * self.variant[1] + 1
            r_idxs = np.zeros(n_terms)
            r = np.zeros((self.n_inds, n_terms, self.F.bounds.shape[0]))
            for i in range(self.n_inds):
                pos = []
                for j in range(self.n_inds):
                    if i != j:
                        pos.append(j)
                r_idxs[0] = self.best_idx
                r_idxs[1:] = self.rng.choice(pos, n_terms - 1, replace=False)
                r[i] = self.X[np.array(r_idxs, int)]
        if self.variant[0] == 'current-to-best':
            n_terms = 2 * self.variant[1] + 3
            r_idxs = np.zeros(n_terms)
            r = np.zeros((self.n_inds, n_terms, self.F.bounds.shape[0]))
            for i in range(self.n_inds):
                pos = []
                for j in range(self.n_inds):
                    if i != j:
                        pos.append(j)
                r_idxs[0] = i
                r_idxs[1] = self.best_idx
                r_idxs[2] = i
                r_idxs[3:] = self.rng.choice(pos, n_terms - 3, replace=False)
                r[i] = self.X[np.array(r_idxs, int)]

        self.mutant = r[:, 0] + self.mut * \
            (r[:, 1::2].sum(1) - r[:, 2::2].sum(1))

    def crossover(self):
        cross_points = self.rng.random(self.X.shape) < self.crossp
        empty_lines = np.where(np.any(cross_points, 1))
        cross_points[empty_lines] = self.rng.random(
            cross_points[empty_lines].shape) < self.crossp
        self.trial = np.where(cross_points, self.mutant, self.X)

    def select(self):
        self.apt_trial = self.F.eval(self.trial)
        selection_condition = self.apt_trial < self.apt_X
        self.X[selection_condition] = self.trial[selection_condition]
        self.apt_X[selection_condition] = self.apt_trial[selection_condition]
        self.best_idx = self.apt_X.argmin()
        self.best = self.X[self.best_idx]
        self.apt_best = self.apt_X[self.best_idx]

    def step(self):
        self.mutate()
        self.crossover()
        self.select()
        self.i += 1
        super().update_stop()
        super().update_history()

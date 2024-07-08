import numpy as np

from lib.methods.algortihm import Algorithm


class CDEEPSO(Algorithm):
    def __init__(self, F, t = 0.1, p = 0.8, mut = 0.6, mc = 0.3, memory=0.1, n_inds=30,
                 n_gens=100, variant='rand', perturb_best=False,
                 keep_x=False, keep_best=False, search=False):
        super().__init__(F, n_inds, n_gens, keep_x, keep_best)

        self.perturb_best = perturb_best

        self.t = t
        self.p = p
        self.mut = mut
        self.mc = mc
        self.pbest = np.array(self.X)
        self.apt_pbest = np.array(self.apt_X)
        self.improved = np.full(self.n_inds, True)

        self.memsize = int(np.round(memory * self.n_inds))
        sm = np.argpartition(self.apt_X, self.memsize)[:self.memsize]
        self.mem = self.X[sm]
        self.apt_mem = self.apt_X[sm]

        self.variant = variant

        self.w = self.rng.uniform(0, 1, [3, self.n_inds])
        self.wg = self.rng.uniform(0, 1, self.best.shape)
        self.P = np.diag(self.rng.choice(
            [0, 1], self.X.shape[0], p=[1 - self.p, self.p]))
        self.v = np.zeros(self.X.shape)

        super().update_history()

        self.search = search

        self.set_pos()

    def step(self):
        self.mutate()
        self.update_velocity()
        self.update_x()
        self.update_best()
        self.update_memory()
        super().update_history()
        self.i += 1
        super().update_stop()

    def get_Xst(self):
        xs = np.arange(self.n_inds)

        if self.variant == "rand":
            r = np.zeros((self.n_inds, 3, self.F.bounds.shape[0]))
            idxs = self.rng.choice(self.n_inds - 1, self.n_inds)
            r[:, 0] = self.X[self.pos[xs, idxs]]
            idxs = self.rng.choice(self.n_inds - 1, self.n_inds)
            r[:, 1] = self.X[self.pos[xs, idxs]]
            r[:, 2] = self.X

        if self.variant == "best":
            r = np.zeros((self.n_inds, 3, self.F.bounds.shape[0]))
            r[:, 0] = np.full(self.X.shape, self.best)
            idxs = self.rng.choice(self.n_inds - 1, self.n_inds)
            r[:, 1] = self.X[self.pos[xs, idxs]]
            r[:, 2] = self.X

        return r[:, 0] + self.mut * (r[:, 1] - r[:, 2])

    def mutate(self):
        w0 = np.clip(self.rng.normal(
            self.w[0], self.t * self.w[0], self.w[0].shape), 0, 1)
        w1 = np.clip(self.rng.normal(
            self.w[1], self.t * self.w[1], self.w[1].shape), 0, 2)
        w2 = np.clip(self.rng.normal(
            self.w[2], self.t * self.w[2], self.w[2].shape), 0, 2)
        self.w[0] = np.where(
            np.logical_or(self.rng.uniform(size=self.n_inds) <
                          self.mc, np.logical_not(self.improved)),
            w0, self.w[0]
        )
        self.w[1] = np.where(
            np.logical_or(self.rng.uniform(size=self.n_inds) <
                          self.mc, np.logical_not(self.improved)),
            w1, self.w[1]
        )
        self.w[2] = np.where(
            np.logical_or(self.rng.uniform(size=self.n_inds) <
                          self.mc, np.logical_not(self.improved)),
            w2, self.w[2]
        )
        self.P = np.diag(self.rng.choice(
            [0, 1], self.n_inds, p=[1 - self.p, self.p]))

        if self.perturb_best:
            self.wg = np.clip(
                self.wg * (1 + self.t * self.rng.standard_normal(self.wg.shape)), 0, 2)
            cand = np.clip(self.best * (1 + self.wg * self.rng.standard_normal(self.best.shape)),
                           *self.F.bounds.transpose())
            apt_cand = self.F.eval(np.array([cand]))[0]
            if apt_cand < self.apt_best:
                self.best = cand
                self.apt_best = apt_cand

    def update_velocity(self):
        Xst = self.get_Xst()
        t1 = np.diag(self.w[0]) @ self.v
        t2 = np.diag(self.w[1]) @ (Xst - self.X)
        t3 = np.diag(
            self.w[2]) @ self.P @ (np.full(self.X.shape, self.best) - self.X)
        self.v = np.clip(t1 + t2 + t3, -5, 5)

    def update_x(self):
        self.X = np.clip(self.X + self.v, *self.F.bounds.transpose())
        self.apt_X = self.F.eval(self.X)

    def update_best(self):
        g_and_x = np.array([self.best, *self.X])
        apt_g_and_x = np.array([self.apt_best, *self.apt_X])
        best_idx = np.argmin(apt_g_and_x)
        self.best = g_and_x[best_idx]
        self.apt_best = apt_g_and_x[best_idx]
        improved = self.apt_X < self.apt_pbest
        self.pbest[improved] = self.X[improved]
        self.apt_pbest[improved] = self.apt_X[improved]
        self.improved = np.full(self.n_inds, False)
        self.improved[improved] = True

    def update_memory(self):
        x_and_mem = np.array([*self.X, *self.mem])
        apt_X_and_mem = np.array([*self.apt_X, *self.apt_mem])
        sm = np.argpartition(apt_X_and_mem, self.memsize)[:self.memsize]
        self.mem = x_and_mem[sm]
        self.apt_mem = apt_X_and_mem[sm]

    def update_history(self):
        if self.keep_x:
            self.X_history.append([self.X, self.v])
            self.apt_X_history.append(self.apt_X)
        if self.keep_best:
            self.best_history.append(self.best)
            self.apt_best_history.append(self.apt_best)

    def set_pos(self):
        self.pos = np.full((self.n_inds, self.n_inds - 1),
                           np.arange(self.n_inds - 1))
        self.pos += np.triu(np.ones_like(self.pos))

import numpy as np

from lib.methods.algortihm import Algorithm


class DEEPSO(Algorithm):
    def __init__(self, F, t = 0.1, p = 0.8, memory=0.1, n_inds=30,
                 n_gens=100, variant='Pb', perturb_best=False,
                 keep_x=False, keep_best=False, search=False):
        super().__init__(F, n_inds, n_gens, keep_x, keep_best)

        self.perturb_best = perturb_best

        self.t = t
        self.p = p

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

    def step(self):
        self.mutate()
        self.update_velocity()
        self.update_x()
        self.update_best()
        self.hill_climbing_search()
        self.update_memory()
        super().update_history()
        self.i += 1
        super().update_stop()

    def get_xr(self):
        if self.variant == "Sg":
            idxs_xr = self.rng.choice(self.X.shape[0], self.X.shape[0])
            xr = self.X[idxs_xr]
            order = np.where(self.apt_X[idxs_xr] < self.apt_X, 1, -1)
            return xr, order

        if self.variant == "Pb":
            idxs_xr = self.rng.choice(self.mem.shape[0], self.X.shape[0])
            xr = self.mem[idxs_xr]
            order = np.where(self.apt_mem[idxs_xr] < self.apt_X, 1, -1)
            return xr, order

        if self.variant == "SgPb":
            idxs_xr_sg = self.rng.choice(
                self.X.shape[0], self.X.shape[0] // 2, False)
            idxs_xr_pb = self.rng.choice(
                self.mem.shape[0], self.X.shape[0] - self.X.shape[0] // 2)
            xr_sg = self.X[idxs_xr_sg]
            apt_xr_sg = self.apt_X[idxs_xr_sg]
            xr_pb = self.mem[idxs_xr_pb]
            apt_xr_pb = self.apt_mem[idxs_xr_pb]
            idxs = self.rng.choice(self.n_inds, self.n_inds, False)
            xr = np.append(xr_sg, xr_pb, 0)[idxs]
            apt_xr = np.append(apt_xr_sg, apt_xr_pb, 0)[idxs]
            order = np.where(apt_xr < self.apt_X, 1, -1)
            return xr, order

    def mutate(self):
        self.w[0] = np.clip(self.rng.normal(
            self.w[0], self.t * self.w[0], self.w[0].shape), 0, 1)
        self.w[1] = np.clip(self.rng.normal(
            self.w[1], self.t * self.w[1], self.w[1].shape), 0, 2)
        self.w[2] = np.clip(self.rng.normal(
            self.w[2], self.t * self.w[2], self.w[2].shape), 0, 2)
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
        xr, order = self.get_xr()
        t1 = np.diag(self.w[0]) @ self.v
        t2 = np.diag(self.w[1]) @ (np.diag(order) @ (xr - self.X))
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

    def update_memory(self):
        x_and_mem = np.array([*self.X, *self.mem])
        apt_X_and_mem = np.array([*self.apt_X, *self.apt_mem])
        sm = np.argpartition(apt_X_and_mem, self.memsize)[:self.memsize]
        self.mem = x_and_mem[sm]
        self.apt_mem = apt_X_and_mem[sm]

    def hill_climbing_search(self):
        if self.search and self.i == 20:
            for _ in range(10):
                cand = self.rng.normal([self.best], np.exp(- self.i / 5))
                apt_cand = self.F.eval(cand)
                if apt_cand < self.apt_best:
                    self.best = cand[0]
                    self.apt_best = apt_cand[0]

    def update_history(self):
        if self.keep_x:
            self.X_history.append([self.X, self.v])
            self.apt_X_history.append(self.apt_X)
        if self.keep_best:
            self.best_history.append(self.best)
            self.apt_best_history.append(self.apt_best)

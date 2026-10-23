"""Corrected re-implementation of Miguel, Vespignani, Zaiser & Zapperi (2002), cond-mat/0205217.

Units as in the paper: lengths in y_e, stresses in Db/y_e, time in y_e^2/(chi D b^3).
Adaptive Dormand-Prince RK45; annihilation is checked between accepted steps (never inside
the right-hand side), and annihilated pairs are removed from the system outright.

An L-stable Rosenbrock (ROS2) integrator was also tried, since stacked same-sign
dislocations make the system stiff; it came out slower than this explicit loop, so it's gone.
"""
import numpy as np
from numba import njit

PI = np.pi


@njit(cache=True, fastmath=True)
def velocity(x, b, C, YSh, L, sigma):
    """v_i = b_i (sum_j b_j sigma_s(r_i - r_j) + sigma), sigma_s summed over all x-images."""
    n = x.size
    k = 2 * PI / L
    v = b * sigma
    for i in range(n):
        for j in range(i + 1, n):
            xa = k * (x[i] - x[j])
            s, c = np.sin(xa), np.cos(xa)
            dc = c - C[i, j]
            f = -PI * s * (L * c + YSh[i, j] - L * C[i, j]) / (L * L * dc * dc)
            w = b[i] * b[j] * f
            v[i] += w  # sigma_s is odd under r -> -r, so the pair is computed once
            v[j] -= w
    return v



@njit(cache=True)
def close_pairs(x, y, b, L, ye):
    """Opposite-sign pairs closer than ye (periodic in x), as (d2, i, j) rows."""
    n = x.size
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if b[i] * b[j] < 0:
                dx = x[i] - x[j]
                dx -= L * np.round(dx / L)
                d2 = dx * dx + (y[i] - y[j]) ** 2
                if d2 < ye * ye:
                    out.append((d2, i, j))
    return out


class Cell:
    def __init__(self, xs, ys, bs, L, ye=1.0):
        self.xs, self.ys, self.bs = xs.astype(float) % L, ys.astype(float), bs.astype(float)
        self.L, self.ye = float(L), ye
        self.ids = np.arange(self.xs.size)
        self.wraps = np.zeros(self.xs.size)  # whole periods travelled, for unwrapped trajectories
        self._prep()

    def _prep(self):
        # glide only: y never changes, so the hyperbolic half of the kernel is fixed
        Y = self.ys[:, None] - self.ys[None, :]
        ya = 2 * PI * Y / self.L
        self.C = np.ascontiguousarray(np.cosh(ya))
        self.YSh = np.ascontiguousarray(2 * PI * Y * np.sinh(ya))

    @property
    def n(self):
        return self.xs.size

    def v(self, x, sigma):
        return velocity(x, self.bs, self.C, self.YSh, self.L, sigma)

    def annihilate(self, x):
        pairs = close_pairs(x % self.L, self.ys, self.bs, self.L, self.ye)
        self.wraps = self.wraps + np.floor(x / self.L)
        self.xs = x % self.L
        if not pairs:
            return 0
        alive = np.ones(self.n, bool)
        for _, i, j in sorted(pairs):
            if alive[i] and alive[j]:
                alive[i] = alive[j] = False
        self.xs, self.ys, self.bs = self.xs[alive], self.ys[alive], self.bs[alive]
        self.ids, self.wraps = self.ids[alive], self.wraps[alive]
        self._prep()
        return int((~alive).sum())


def random_cell(n0, L, rng):
    bs = rng.permutation(np.r_[np.ones(n0 // 2), -np.ones(n0 // 2)])
    return Cell(rng.random(n0) * L, rng.random(n0) * L, bs, L)


# Dormand-Prince 5(4) tableau
_A = [[],
      [1 / 5],
      [3 / 40, 9 / 40],
      [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


def run(cell, sigma, t_end, t_record=(), rtol=1e-5, atol=1e-6, max_move=0.2, record_x=False):
    """Integrate cell for t_end time units under applied stress sigma.

    Returns sum(b*v) and N at each t_record (plus position snapshots if record_x).
    max_move caps how far any dislocation glides per step (in y_e), so an opposite-sign
    pair can't step past each other between annihilation checks.
    """
    t_record = np.asarray(t_record, float)
    rate = np.full(t_record.size, np.nan)
    nalive = np.zeros(t_record.size, int)
    snaps = []
    cell.annihilate(cell.xs)
    t, x, dt, ri = 0.0, cell.xs.copy(), 1e-2, 0
    k1 = cell.v(x, sigma)
    while t < t_end and cell.n > 0:
        h = min(dt, max_move / (np.abs(k1).max() + 1e-12), t_end - t)
        if ri < t_record.size:
            h = min(h, max(t_record[ri] - t, 1e-12))
        ks = [k1]
        for s in range(1, 7):
            ks.append(cell.v(x + h * sum(a * kk for a, kk in zip(_A[s], ks)), sigma))
        K = np.array(ks)
        x5 = x + h * (_B5 @ K)
        err = h * ((_B5 - _B4) @ K)
        en = np.sqrt(np.mean((err / (atol + rtol * np.maximum(np.abs(x), np.abs(x5)))) ** 2))
        if en > 1.0:
            dt = h * max(0.2, 0.9 * en ** -0.2)
            continue
        t, x = t + h, x5
        dt = h * min(5.0, 0.9 * (en + 1e-12) ** -0.2)
        k1 = K[6]  # first-same-as-last
        while ri < t_record.size and t_record[ri] <= t * (1 + 1e-12):
            rate[ri] = np.sum(cell.bs * k1)
            nalive[ri] = cell.n
            if record_x:
                snaps.append((t_record[ri], x + cell.wraps * cell.L, cell.ys.copy(), cell.bs.copy(),
                              cell.ids.copy()))
            ri += 1
        if cell.annihilate(x):
            k1 = cell.v(cell.xs, sigma)
        x = cell.xs.copy()
    return rate, nalive, snaps


# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numba",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from numba import njit

    PI = np.pi


@app.cell
def _():
    mo.md(r"""
    # Dislocation jamming and Andrade creep: corrected replication

    Re-implementation of Miguel, Vespignani, Zaiser & Zapperi (2002),
    [cond-mat/0205217](https://arxiv.org/abs/cond-mat/0205217). Units as in the paper:
    lengths in $y_e$, stresses in $Db/y_e$, time in $y_e^2/(\chi D b^3)$, so Eq. 2 becomes
    $v_n = b_n\left(\sum_{m \ne n} b_m \tilde\sigma_s(r_{nm}) + \tilde\sigma\right)$.

    What changed from `dimensionless.py` (the stress kernel and the nondimensionalization
    there were already right; the analytic image sum matches a brute-force sum to ~7 digits):

    1. **Annihilated dislocations no longer exert stress.** The `elim_mult != 0` mask in
       `sigma_s_analytic_sum` broadcast across columns (targets), so dead dislocations
       stopped *feeling* stress but kept *exerting* it from where they froze, leaving a fixed
       dipole behind every annihilation. Here annihilated pairs are removed from the arrays.
    2. **The strain rate uses the run's own stress.** `strain_rate` hardcoded
       `sigma_ext=0.03`, and since $\sum b_i v_i = \sum(\text{internal}) + N\sigma$, that added
       a false floor of $0.03N$ to every curve, which flattens them into fake plateaus.
    3. **Annihilation happens between accepted steps, not inside the right-hand side.**
       `solve_ivp` calls the RHS at trial stages, including rejected steps, so annihilations
       were irreversible side effects of steps that never happened. Here a hand-written
       Dormand–Prince RK45 (the paper's "adaptive fifth-order Runge-Kutta") checks for
       annihilation after each accepted step, and caps how far any dislocation can glide per
       step so an opposite-sign pair can't hop past each other between checks.
    4. **Annihilation distance is periodic in $x$**, time is recorded at log-spaced real
       times rather than array indices, and runs go to $t = 10^4$.

    The inner loop is compiled with numba, which is what makes 192-seed ensembles feasible.
    """)
    return


@app.function
@njit(fastmath=True)
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
            # sigma_s is odd under r -> -r, so each pair is computed once
            v[i] += w
            v[j] -= w
    return v


@app.function
@njit
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


@app.class_definition
# a periodic-in-x cell of glide-only edge dislocations
class Cell:
    def __init__(self, xs, ys, bs, L, ye=1.0):
        self.xs, self.ys, self.bs = xs.astype(float) % L, ys.astype(float), bs.astype(float)
        self.L, self.ye = float(L), ye
        # identity and whole periods travelled, so trajectories can be followed and unwrapped
        self.ids = np.arange(self.xs.size)
        self.wraps = np.zeros(self.xs.size)
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
        """Remove opposite-sign pairs closer than ye, closest first; returns how many died."""
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


@app.function
def random_cell(n0, L, rng, height=None):
    """n0 dislocations (half b=+1, half b=-1) placed uniformly in [0, L) x [0, height)."""
    height = L if height is None else height
    bs = rng.permutation(np.r_[np.ones(n0 // 2), -np.ones(n0 // 2)])
    return Cell(rng.random(n0) * L, rng.random(n0) * height, bs, L)


@app.cell
def _():
    # Dormand-Prince 5(4) tableau
    DP_A = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    DP_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    DP_B4 = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    )
    return DP_A, DP_B4, DP_B5


@app.cell
def _(DP_A, DP_B4, DP_B5):
    def run(cell, sigma, t_end, t_record=(), rtol=1e-5, atol=1e-6, max_move=0.2,
            record_x=False):
        """Integrate cell for t_end time units under applied stress sigma (mutates cell).

        Returns sum(b*v) and N at each t_record, plus (t, unwrapped x, y, b, ids)
        snapshots if record_x.
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
                ks.append(cell.v(x + h * sum(a * kk for a, kk in zip(DP_A[s], ks)), sigma))
            K = np.array(ks)
            x5 = x + h * (DP_B5 @ K)
            err = h * ((DP_B5 - DP_B4) @ K)
            scale = atol + rtol * np.maximum(np.abs(x), np.abs(x5))
            en = np.sqrt(np.mean((err / scale) ** 2))
            if en > 1.0:
                dt = h * max(0.2, 0.9 * en**-0.2)
                continue
            t, x = t + h, x5
            dt = h * min(5.0, 0.9 * (en + 1e-12) ** -0.2)
            k1 = K[6]  # first-same-as-last
            while ri < t_record.size and t_record[ri] <= t * (1 + 1e-12):
                rate[ri] = np.sum(cell.bs * k1)
                nalive[ri] = cell.n
                if record_x:
                    snaps.append((t_record[ri], x + cell.wraps * cell.L, cell.ys.copy(),
                                  cell.bs.copy(), cell.ids.copy()))
                ri += 1
            if cell.annihilate(x):
                k1 = cell.v(cell.xs, sigma)
            x = cell.xs.copy()
        return rate, nalive, snaps
    return (run,)


@app.cell
def _():
    mo.md(r"""
    ## One realization

    Relax $N_0 = 100$ random dislocations in an $L = 50$ cell with no applied stress, then
    switch on $\sigma$. Same density as the paper's $N_0 = 400$, $L = 100$, at a sixteenth
    of the cost per step.
    """)
    return


@app.cell
def _(run):
    demo = random_cell(100, 50, np.random.default_rng(0))
    run(demo, 0.0, 1000.0)
    demo_relaxed_n = demo.n
    demo_t = np.logspace(0, 4, 49)
    demo_rate, demo_n, _ = run(demo, 0.0125, 1e4, demo_t)

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ok = demo_rate > 0
    _ax.plot(np.log10(demo_t[_ok]), np.log10(demo_rate[_ok]), color="#2a78d6", lw=2)
    _ax.set_xlabel("$\\log_{10} t$")
    _ax.set_ylabel("$\\log_{10} \\sum b v$")
    _ax.set_title(f"σ = 0.0125, {demo_relaxed_n} dislocations after relaxation", loc="left")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax
    return


@app.cell
def _():
    mo.md(r"""
    ## Ensemble (Fig. 1)

    The figure in `replication-vs-bug.png` used 192 seeds per stress (about 40 minutes on 12
    cores via `multiprocessing`). The default here is small enough to run in a few minutes;
    raise `n_seeds` for smooth curves.
    """)
    return


@app.cell
def _():
    n_seeds = mo.ui.slider(4, 192, value=16, step=4, label="seeds per stress")
    go = mo.ui.run_button(label="run ensemble")
    mo.hstack([n_seeds, go], justify="start")
    return go, n_seeds


@app.cell
def _(go, n_seeds, run):
    mo.stop(not go.value, mo.md("Press **run ensemble** to start."))

    STRESSES = [0.0025, 0.0075, 0.01, 0.0175, 0.0325]
    T_REC = np.logspace(0, 4, 49)
    ens_rate = {s: [] for s in STRESSES}
    ens_n = {s: [] for s in STRESSES}
    for _seed in mo.status.progress_bar(range(n_seeds.value), title="seeds"):
        _cell = random_cell(100, 50, np.random.default_rng(_seed))
        run(_cell, 0.0, 1000.0)
        for _s in STRESSES:
            _r, _n, _ = run(copy.deepcopy(_cell), _s, 1e4, T_REC)
            ens_rate[_s].append(_r)
            ens_n[_s].append(_n)
    return STRESSES, T_REC, ens_n, ens_rate


@app.cell
def _(STRESSES, T_REC, ens_n, ens_rate):
    # one-hue ramp, light -> dark with stress
    _cols = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for _ax, _buggy in zip(_axes, [False, True]):
        _labels = []
        for _s, _c in zip(STRESSES, _cols):
            _r, _n = np.array(ens_rate[_s]), np.array(ens_n[_s])
            if _buggy:  # what dimensionless.py's strain_rate reported
                _r = np.abs(_r - _s * _n + 0.03 * _n)
            _m = np.nanmean(_r, axis=0)
            _ok = _m > 0
            _ax.plot(np.log10(T_REC[_ok]), np.log10(_m[_ok]), color=_c, lw=2)
            _labels.append([np.log10(_m[_ok][-1]), f"σ = {_s}"])
        _labels.sort()
        for _i in range(1, len(_labels)):  # keep direct labels from overlapping
            _labels[_i][0] = max(_labels[_i][0], _labels[_i - 1][0] + 0.2)
        for _y, _txt in _labels:
            _ax.text(4.06, _y, _txt, va="center", fontsize=9.5, color="#52514e")
        _ax.set_xlim(0, 4.75)
        _ax.set_xlabel("$\\log_{10} t$")
        _ax.spines[["top", "right"]].set_visible(False)
    _x = np.array([0.5, 2.5])
    _axes[0].plot(_x, -2 / 3 * (_x - 0.5) - 0.2, color="k", lw=1, ls=(0, (4, 3)))
    _axes[0].text(1.55, -0.75, "$t^{-2/3}$ (Andrade)", fontsize=10, rotation=-21)
    _axes[0].set_ylabel("$\\log_{10} \\langle \\sum b v \\rangle$")
    _axes[0].set_title("Corrected", loc="left")
    _axes[1].set_title("Same runs through the hardcoded σ = 0.03", loc="left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## Banner art

    The total shear stress $\sigma_s$ of a relaxed (jammed) configuration, drawn as contour
    lines. Each dislocation's field is a four-lobed flower; the cell is periodic in $x$, so a
    strip 100 wide and 25 tall tiles seamlessly at LinkedIn's 4:1 banner ratio. The field is
    passed through a signed square root so the far field shows instead of just the cores.
    """)
    return


@app.function
def stress_field(xs, ys, bs, LX, H, nx=1584, ny=396):
    """Total glide shear stress on a grid, periodic in x (the paper's image sum)."""
    GX, GY = np.meshgrid(np.linspace(0, LX, nx, endpoint=False), np.linspace(0, H, ny))
    total = np.zeros_like(GX)
    for x0, y0, b in zip(xs, ys, bs):
        X, Y = GX - x0, GY - y0
        xa, ya = 2 * PI * X / LX, 2 * PI * Y / LX
        num = PI * np.sin(xa) * (LX * np.cos(xa) + 2 * PI * Y * np.sinh(ya) - LX * np.cosh(ya))
        den = LX**2 * (np.cos(xa) - np.cosh(ya)) ** 2
        with np.errstate(all="ignore"):
            total += b * np.where(den > 1e-12, -num / den, 0)
    return GX[0], GY[:, 0], total


@app.cell
def _(run):
    _LX, _H = 100.0, 25.0
    _cell = random_cell(100, _LX, np.random.default_rng(1000), height=_H)
    run(_cell, 0.0, 1000.0)
    run(_cell, 0.0125, 5000.0)
    _gx, _gy, _F = stress_field(_cell.xs, _cell.ys, _cell.bs, _LX, _H)
    _Fs = np.sign(_F) * np.sqrt(np.abs(_F))
    _lim = np.percentile(np.abs(_Fs), 97)

    _bg = "#f4f1ea"
    _cmap = LinearSegmentedColormap.from_list("cream", ["#2f5d8a", _bg, "#b5523b"])
    _fig = plt.figure(figsize=(15.84, 3.96), dpi=100)
    _ax = _fig.add_axes([0, 0, 1, 1])
    _ax.set_axis_off()
    _fig.patch.set_facecolor(_bg)
    _ax.contour(_gx, _gy, _Fs, levels=np.linspace(-_lim, _lim, 24), cmap=_cmap, linewidths=0.8,
                norm=TwoSlopeNorm(0, -_lim, _lim))
    _ax.set_xlim(0, _LX)
    _ax.set_ylim(0, _H)
    _fig
    return


if __name__ == "__main__":
    app.run()

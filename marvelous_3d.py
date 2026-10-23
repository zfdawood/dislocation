# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    # import numpy
    # try:
    #     import cupy as np
    #     if not np.is_available():
    #         raise ImportError("Cannot use cupy")
    # except ImportError:
    #     # potential double numpy import to keep the codebase consistent between cuda and non-cuda vers
    #     import numpy as np

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy.integrate import solve_ivp
    from types import SimpleNamespace
    import matplotlib
    import marimo as mo
    import pickle
    import os
    import time

    pi = np.pi


@app.cell
def _():
    mo.md(r"""
    # marvelous 3d

    The 2D model (`corrected.py`) treats every dislocation as a point: really it's a perfectly straight
    edge line running along z, so all the action happens in the x-y plane. Here the lines get to bend.

    - Every dislocation is a line on its own glide plane y = const, periodic in z with period `Lz`, made of
      `M` nodes joined by straight segments. Burgers vector is still ±x. Each node glides perpendicular to
      the line *within its glide plane* (so in x and z), and also slides along the line to keep the nodes
      evenly spaced, which doesn't change the shape.
    - The stress on a node is (1) the periodic 2D sum from the paper, treating every *other* line as if it
      were straight at its average x, plus (2) a correction for the wiggles: the Peach-Koehler line integral
      over the real line minus the same integral over its straightened version. Both use the nonsingular
      stress of Cai, Arsenlis, Weinberger & Bulatov (2006), core radius `a`, and the integral along each
      segment is done exactly. Straight lines give the 2D model (exactly the paper's when a = 0).
    - A line doesn't feel its own stress field; that's replaced with line tension, which pulls it straight.
    - Two opposite-sign lines annihilate (both whole lines go) as soon as any part of them gets within
      `ye`. In real life they touch at one spot, reconnect, and the two halves zip up, so whole-line
      removal is the cheap version of that. `solve_ivp` stops at each annihilation (an event) and restarts,
      so nothing gets annihilated inside the right-hand side anymore.

    Units are the paper's: lengths in ye, stress in Db/ye, time in ye²/(χDb³).
    """)
    return


@app.function
# infinite sum from eqn 1 in the paper, but from straight lines at (src_xs, src_ys) onto any field points,
# and with the core spread out over a radius a so it matches the peach-koehler part below.
# a straight edge line with a spread core has sigma_s = x (x^2 - y^2 + a^2) / (x^2 + y^2 + a^2)^2
# = x / (x^2 + c^2) - 2 y^2 x / (x^2 + c^2)^2 with c^2 = y^2 + a^2, and both of those pieces have
# closed-form sums over the periodic copies in x. a = 0 gives back exactly the paper's formula
# returns a (field, source) matrix: [i, j] is sigma_s at field point i from a b = +1 line at source j
def sigma_s_analytic_sum(field_xs, field_ys, src_xs, src_ys, L, a=0):
    # X[i, j] is field point i's distance from source j (Y too WLOG)
    X = field_xs.reshape(-1,1) - src_xs
    Y = field_ys.reshape(-1,1) - src_ys
    c = np.sqrt(Y**2 + a**2)

    k = 2 * pi / L
    den = np.cosh(k * c) - np.cos(k * X)
    # sum over copies of x / (x^2 + c^2)...
    sum_1 = np.divide(pi / L * np.sin(k * X), den, out=np.zeros(X.shape), where=(den != 0))
    # ... and of x / (x^2 + c^2)^2, which is -1/(2c) d/dc of the first one
    sum_2 = np.divide(pi * k * np.sin(k * X) * np.sinh(k * c), 2 * c * L * den**2,
                      out=np.zeros(X.shape), where=(den != 0))

    # den is only 0 when a field point sits right on a line (with a = 0), which we don't want anyway
    return sum_1 - 2 * Y**2 * sum_2


@app.function
# nonsingular version of the peach-koehler line integral (cai, arsenlis, weinberger & bulatov 2006):
# swap R = |x - x'| for R_a = sqrt(|x - x'|^2 + a^2) so nothing blows up near the core.
# we only ever need sigma_xy since b always points along x and every glide plane is y = const.
# field_pts is (n, 3), src_pts is (q, 3) quadrature points along a line, dls is (q, 3), the little
# dx' vector each quadrature point stands for. returns sigma_xy at each field point from a b = +1 line
# signs are picked so a straight line along +z gives the 2D model's sigma_s (checked in a cell below)
# this is the formula written out plainly; the simulation uses sigma_xy_segments, which is the same
# integral done exactly along straight segments
def sigma_xy_pk(field_pts, src_pts, dls, a, nu):
    # r[i, j] is field point i minus source point j, one matrix per component
    rx = field_pts[:,0].reshape(-1,1) - src_pts[:,0]
    ry = field_pts[:,1].reshape(-1,1) - src_pts[:,1]
    rz = field_pts[:,2].reshape(-1,1) - src_pts[:,2]

    Ra2 = rx**2 + ry**2 + rz**2 + a**2
    Ra = np.sqrt(Ra2)
    Ra3 = Ra2 * Ra
    Ra5 = Ra3 * Ra2

    # d/dz of laplacian(R_a): picks up the bits of the line pointing along x (screw character)
    dz_lap = -(2 / Ra3 + 3 * a**2 / Ra5) * rz
    # d^3 R_a / dx dy dy: picks up the bits of the line pointing along z (edge character)
    dxyy = -rx / Ra3 + 3 * rx * ry**2 / Ra5

    # D = mu / (2 pi (1 - nu)) = 1 in our units, so mu / 8pi = (1 - nu) / 4 and mu / (4pi (1 - nu)) = 1/2
    return ((1 - nu) / 4 * dz_lap @ dls[:,0]) - (0.5 * dxyy @ dls[:,2])


@app.function
# same integral as sigma_xy_pk, but done exactly along straight segments (seg_starts + t * seg_vecs,
# t from 0 to 1) so it doesn't matter how close a field point gets to a segment.
# segments have to lie in a glide plane (no y component). pass Lz if things are periodic in z
def sigma_xy_segments(field_pts, seg_starts, seg_vecs, a, nu, Lz=None):
    # A[i, j] is field point i minus the start of segment j, one matrix per component
    Ax = field_pts[:,0].reshape(-1,1) - seg_starts[:,0]
    Ay = field_pts[:,1].reshape(-1,1) - seg_starts[:,1]
    Az = field_pts[:,2].reshape(-1,1) - seg_starts[:,2]
    # periodic in z: each segment counts from whichever copy of it is closest (so one period of line,
    # centered on the field point). only ok for the wiggle correction, which dies off fast
    if Lz is not None:
        Az = Az - Lz * np.round(Az / Lz)

    seg_len = np.sqrt(seg_vecs[:,0]**2 + seg_vecs[:,2]**2)
    ux = seg_vecs[:,0] / seg_len
    uz = seg_vecs[:,2] / seg_len

    # along the segment, r = A - t u. write t = p + s where p is how far along the segment the field
    # point's foot is, so R_a^2 = s^2 + q2 with q2 = (distance off the segment's line)^2 + a^2
    p = Ax * ux + Az * uz
    q2 = Ax**2 + Ay**2 + Az**2 - p**2 + a**2
    # the parts of A perpendicular to the segment: r_x = rpx - s ux, r_z = rpz - s uz, r_y = Ay
    rpx = Ax - p * ux
    rpz = Az - p * uz

    # antiderivatives of 1/R^3, s/R^3, 1/R^5, s/R^5 in s
    def antiderivs(s):
        R = np.sqrt(s**2 + q2)
        R3 = R**3
        return (s / (q2 * R), -1 / R, s * (2 * s**2 + 3 * q2) / (3 * q2**2 * R3), -1 / (3 * R3))

    lo = antiderivs(-p)
    hi = antiderivs(seg_len - p)
    d3_0, d3_1, d5_0, d5_1 = [h - l for h, l in zip(hi, lo)]

    # integral of r_z (2/R^3 + 3a^2/R^5), the dz_lap part (up to sign)...
    I1 = rpz * (2 * d3_0 + 3 * a**2 * d5_0) - uz * (2 * d3_1 + 3 * a**2 * d5_1)
    # ... and of r_x (-1/R^3 + 3 r_y^2/R^5), the dxyy part
    I2 = rpx * (-d3_0 + 3 * Ay**2 * d5_0) - ux * (-d3_1 + 3 * Ay**2 * d5_1)

    return np.sum(-(1 - nu) / 4 * I1 * ux - 0.5 * I2 * uz, axis=1)


@app.function
# right hand side for solve_ivp. annihilation is NOT in here anymore - solve_ivp tries out
# positions it later throws away, so killing dislocations in here killed them at fake times
def dxsdt_func(t, state, sigma_ext, elim, lc):
    return lc.vs(state, sigma_ext, elim.mults)


@app.class_definition
# class to help remove annihilated dislocations from the system
class Eliminator:
    def __init__(self, size):
        # turn off annihilated lines' contributions to various computations by
        # multiplying them by 0 elementwise (if it does contribute, multiply by one)
        self.mults = np.ones(size)

        # time at which the ith line was annihilated, nan if it's still around
        self.elim_times = np.full(size, np.nan)

        self.size = size

    def copy(self):
        new_elim = Eliminator(self.size)
        new_elim.mults = self.mults.copy()
        new_elim.elim_times = self.elim_times.copy()
        return new_elim

    # which lines were still around at time t
    def alive_at(self, t):
        return np.isnan(self.elim_times) | (self.elim_times > t)


@app.class_definition
# this class represents a 3d periodic cell with wiggly dislocation lines in it
# the state solve_ivp sees is every node's x, then every node's z, all flattened
class LatticeCell3D:

    def __init__(self, N, sigma_ext, ye, L, Lz, M, wiggle, a=0.2, nu=0.3, cutoff=None, rng_seed=None):
        self.rng = np.random.default_rng(rng_seed)

        self.N = N
        self.M = M
        self.L = L
        self.Lz = Lz
        self.nu = nu
        self.sigma_ext = sigma_ext
        self.ye = ye
        # core spreading radius for the nonsingular stress. both halves of the stress use it so they
        # cancel properly for straight lines; a = 0 is the paper's singular 2D model.
        # roughly b, and ye is about 6 b for copper, hence the 0.2
        self.a = a
        # only lines whose glide planes are within cutoff of a node bother computing the wiggle correction
        # (the correction falls off fast since it's the difference of two lines with the same b)
        self.cutoff = np.inf if cutoff is None else cutoff

        # each line gets a glide plane, a sign, and an average x position...
        self.ys = self.rng.random(N) * L
        self.bs = self.rng.permuted(
            np.concatenate((
                np.ones(int(N/2)),
                 -(np.ones(int(N/2)))
            ))
        )
        xbars = self.rng.random(N) * L

        # ... starts out with its nodes evenly spaced in z...
        self.z0s = np.tile(np.arange(M) * Lz / M, (N, 1))
        # ... plus some wiggles: a few random fourier modes along z, smaller for the shorter wavelengths
        # (wiggle = 0 gives perfectly straight lines, which is exactly the 2D model)
        self.x0s = np.repeat(xbars.reshape(-1,1), M, axis=1)
        for n in range(1, 4):
            amps = wiggle * self.rng.normal(size=N) / n
            phases = self.rng.random(N) * 2 * pi
            self.x0s = self.x0s + amps.reshape(-1,1) * np.cos(2 * pi * n * self.z0s / Lz + phases.reshape(-1,1))

        # line tension of a dislocation, the usual mu b^2 / (4 pi) ln(R / r0) with R ~ Lz, which in units
        # of D b^2 is (1 - nu) / 2 * ln(Lz / r0). r0 HAS to be the same core radius a the stress uses:
        # with r0 = ye, two antiparallel bits of line closer than ye had negative energy, so wherever two
        # lines crossed they zipped up into a screw dipole that grew forever.
        # (a = 0 only makes sense for straight lines, where line tension never comes up)
        self.line_tension = (1 - nu) / 2 * np.log(Lz / a) if a > 0 else 0
        # how fast nodes slide along the line to even out their spacing
        self.spacing_rate = 1.0

        self.elim_relax = None
        self.elim_stress = None


    def pack(self, xs, zs):
        return np.concatenate((xs.flatten(), zs.flatten()))

    def unpack(self, state):
        return state[:self.N * self.M].reshape(self.N, self.M), state[self.N * self.M:].reshape(self.N, self.M)


    # segment j of each line runs from node j to node j + 1; the last one goes back to node 0,
    # one period up in z
    def segments(self, xs, zs):
        x_next = np.roll(xs, -1, axis=1)
        z_next = np.roll(zs, -1, axis=1)
        z_next[:,-1] = z_next[:,-1] + self.Lz
        return x_next - xs, z_next - zs


    # tangents, normals, curvature etc for every node. everything here is in the (x, z) glide plane
    def geometry(self, xs, zs):
        seg_dx, seg_dz = self.segments(xs, zs)
        seg_len = np.sqrt(seg_dx**2 + seg_dz**2)
        # unit tangent of each segment...
        seg_tx = seg_dx / seg_len
        seg_tz = seg_dz / seg_len
        # ... and of each node, splitting the difference between the segments either side of it
        # (np.roll(..., 1) is the segment coming into the node, the plain one is the segment leaving it)
        tx = seg_tx + np.roll(seg_tx, 1, axis=1)
        tz = seg_tz + np.roll(seg_tz, 1, axis=1)
        t_len = np.sqrt(tx**2 + tz**2)
        tx = tx / t_len
        tz = tz / t_len
        # normal in the glide plane: +x for a line going up in z, which is the direction b * sigma pushes
        # a b = +1 line, same as the 2D model
        nx = tz
        nz = -tx
        # how much line each node is responsible for
        node_len = (seg_len + np.roll(seg_len, 1, axis=1)) / 2
        # curvature: how fast the tangent turns, per length of line
        curvature = ((seg_tx - np.roll(seg_tx, 1, axis=1)) * nx
                     + (seg_tz - np.roll(seg_tz, 1, axis=1)) * nz) / node_len
        return SimpleNamespace(seg_dx=seg_dx, seg_dz=seg_dz, seg_len=seg_len, tx=tx, tz=tz, nx=nx, nz=nz,
                               node_len=node_len, curvature=curvature)


    # the x a line would sit at if you straightened it out without sweeping any net area
    def xbars(self, xs, zs):
        seg_dx, seg_dz = self.segments(xs, zs)
        return np.sum((xs + seg_dx / 2) * seg_dz, axis=1) / self.Lz


    # sigma_xy from every line onto every field point (by default, onto every node of every line)
    # field_line[i] says which line field point i belongs to, so lines skip themselves (-1 = no line)
    def stress_xy(self, xs, zs, mults, field_pts=None, field_line=None):
        if field_pts is None:
            field_pts = np.column_stack((
                xs.flatten(),
                np.repeat(self.ys, self.M),
                zs.flatten(),
            ))
            field_line = np.repeat(np.arange(self.N), self.M)
        if field_line is None:
            field_line = np.full(field_pts.shape[0], -1)

        xbars = self.xbars(xs, zs)
        seg_dx, seg_dz = self.segments(xs, zs)

        # part 1: every line as if it were straight, sitting at its average x (the 2D sum)
        S = sigma_s_analytic_sum(field_pts[:,0], field_pts[:,1], xbars, self.ys, self.L, self.a)
        # lines don't push on themselves - their own wiggles are what line tension is for
        S[field_line.reshape(-1,1) == np.arange(self.N)] = 0
        sigmas = S @ (self.bs * mults)

        # part 2: the wiggle correction, one source line at a time
        # dead field points don't need anything computed
        field_alive = np.ones(field_pts.shape[0], dtype=bool)
        field_alive[field_line >= 0] = mults[field_line[field_line >= 0]] != 0
        for m in np.nonzero(mults)[0]:
            # perfectly straight lines have no correction (this is what makes wiggle = 0 the 2D model)
            if np.ptp(xs[m]) < 1e-12:
                continue
            # field points that are...
            near = (
                # ... not on line m itself...
                (field_line != m)
                # ... on a line that's still around...
                & field_alive
                # ... and close enough in y to care
                & (np.abs(field_pts[:,1] - self.ys[m]) < self.cutoff)
            )
            if not np.any(near):
                continue
            fp = field_pts[near].copy()
            # periodic in x: move each field point to the copy of the box closest to line m
            fp[:,0] = fp[:,0] - self.L * np.round((fp[:,0] - xbars[m]) / self.L)

            # the real line's segments, and the straightened line cut at the same heights
            real_starts = np.column_stack((xs[m], np.full(self.M, self.ys[m]), zs[m]))
            real_vecs = np.column_stack((seg_dx[m], np.zeros(self.M), seg_dz[m]))
            straight_starts = np.column_stack((np.full(self.M, xbars[m]), np.full(self.M, self.ys[m]), zs[m]))
            straight_vecs = np.column_stack((np.zeros(self.M), np.zeros(self.M), seg_dz[m]))
            sigmas[near] += self.bs[m] * (
                sigma_xy_segments(fp, real_starts, real_vecs, self.a, self.nu, self.Lz)
                - sigma_xy_segments(fp, straight_starts, straight_vecs, self.a, self.nu, self.Lz)
            )
        return sigmas


    # how fast each node moves perpendicular to its line (in the glide plane), plus the geometry
    def normal_velocity(self, xs, zs, sigma_ext, mults):
        sigmas = self.stress_xy(xs, zs, mults).reshape(self.N, self.M)
        geo = self.geometry(xs, zs)
        # glide: b * (resolved shear stress), plus line tension pulling the line straight
        v_n = self.bs.reshape(-1,1) * (sigmas + sigma_ext) + self.line_tension * geo.curvature
        return v_n, geo


    # d(state)/dt, flattened the way solve_ivp likes it
    def vs(self, state, sigma_ext=0, mults=None):
        if mults is None:
            mults = np.ones(self.N)
        xs, zs = self.unpack(state)
        v_n, geo = self.normal_velocity(xs, zs, sigma_ext, mults)

        # slide nodes along the line toward whichever neighbor is further away, so the spacing stays even
        # (moving along the line doesn't change its shape, so this isn't physics, just bookkeeping)
        v_t = self.spacing_rate * (geo.seg_len - np.roll(geo.seg_len, 1, axis=1))

        vx = v_n * geo.nx + v_t * geo.tx
        vz = v_n * geo.nz + v_t * geo.tz
        return self.pack(mults.reshape(-1,1) * vx, mults.reshape(-1,1) * vz)


    # squared in-plane distance from every node of line k to the closest point on line m
    def node_to_line_dist2(self, xs, zs, seg_dx, seg_dz, k, m):
        # rel[i, j] is node i of line k minus the start of segment j of line m (closest periodic copies)
        rel_x = xs[k].reshape(-1,1) - xs[m]
        rel_z = zs[k].reshape(-1,1) - zs[m]
        rel_x = rel_x - self.L * np.round(rel_x / self.L)
        rel_z = rel_z - self.Lz * np.round(rel_z / self.Lz)
        # how far along each segment the closest point is, clamped to the segment
        t = np.clip((rel_x * seg_dx[m] + rel_z * seg_dz[m]) / (seg_dx[m]**2 + seg_dz[m]**2), 0, 1)
        return np.min((rel_x - t * seg_dx[m])**2 + (rel_z - t * seg_dz[m])**2)


    # gap[k, m] is how close lines k and m get, for opposite-sign living pairs on glide planes closer than
    # ye (nobody else can annihilate); everything else is inf
    def closest_approach(self, state, mults):
        xs, zs = self.unpack(state)
        seg_dx, seg_dz = self.segments(xs, zs)
        dy = self.ys.reshape(-1,1) - self.ys

        candidates = (
            # ... opposite burgers vectors...
            (self.bs.reshape(-1,1) != self.bs)
            # ... both still around...
            & np.outer(mults, mults).astype(bool)
            # ... and glide planes close enough that they could ever get within ye
            & (np.abs(dy) < self.ye)
        )
        gap = np.full((self.N, self.N), np.inf)
        for k, m in zip(*np.nonzero(np.triu(candidates))):
            # two lines that aren't crossing are closest at a node of one or the other
            d2 = min(self.node_to_line_dist2(xs, zs, seg_dx, seg_dz, k, m),
                     self.node_to_line_dist2(xs, zs, seg_dx, seg_dz, m, k))
            gap[k, m] = gap[m, k] = np.sqrt(d2 + dy[k, m]**2)
        return gap


    def annihilate(self, state, t, elim):
        gap = self.closest_approach(state, elim.mults)
        # the 1e-9 slack is so a pair that solve_ivp stopped at *exactly* ye gets caught
        ks, ms = np.nonzero(gap < self.ye * (1 + 1e-9))
        # closest pairs go first, and a line can only be annihilated once
        for idx in np.argsort(gap[ks, ms]):
            k, m = ks[idx], ms[idx]
            if (elim.mults[k] != 0) and (elim.mults[m] != 0):
                # print(f"annihilation event with {k} and {m} at t = {t}")
                elim.mults[k] = 0
                elim.mults[m] = 0
                elim.elim_times[k] = t
                elim.elim_times[m] = t


    # solve_ivp from t_start to t_end, stopping and restarting at every annihilation
    # returns something shaped like a solve_ivp result: .t and .y at the t_record times
    def run_ivp(self, state_start, t_start, t_end, sigma_ext, elim, t_record):
        assert np.isclose(t_record[-1], t_end), "t_record has to end at t_end"
        t_record = t_record.copy()
        t_record[-1] = t_end

        t = t_start
        state = state_start.copy()
        all_t = []
        all_y = []
        while t < t_end and np.any(elim.mults):
            self.annihilate(state, t, elim)

            # solve_ivp stops when this hits 0, ie when the closest opposite pair is ye apart
            # (solve_ivp hands events the same args as dxsdt_func, which we don't need)
            def collision(t_, state_, *args):
                return min(np.min(self.closest_approach(state_, elim.mults)) - self.ye, 1e3)
            collision.terminal = True
            collision.direction = -1

            t_eval = t_record[(t_record > t) & (t_record <= t_end)]
            ivp_result = solve_ivp(dxsdt_func,
                                   (t, t_end),
                                   state,
                                   t_eval=t_eval,
                                   events=collision,
                                   method="RK45",
                                   # the default rtol of 1e-3 is loose enough that runs visibly drift
                                   rtol=1e-5,
                                   atol=1e-6,
                                   # don't let anything glide far enough in one step to skip past a partner
                                   max_step=1.0,
                                   args=(sigma_ext, elim, self)
                                  )
            # (if two annihilations land between record times, this stretch recorded nothing)
            if len(ivp_result.t) > 0:
                all_t.append(ivp_result.t)
                all_y.append(ivp_result.y)
            if ivp_result.status == 1:
                # stopped for an annihilation, pick up from there
                t = ivp_result.t_events[0][0]
                state = ivp_result.y_events[0][0]
            else:
                t = t_end
                state = ivp_result.y[:,-1]

        return SimpleNamespace(t=np.concatenate(all_t), y=np.concatenate(all_y, axis=1))


    def relax_ivp(self, t_end=100, num_tpoints=40):
        if self.elim_relax is not None:
            raise TypeError("cannot relax_ivp more than once")

        self.elim_relax = Eliminator(self.N)
        self.relax_ivp_result = self.run_ivp(self.pack(self.x0s, self.z0s), 0, t_end, 0, self.elim_relax,
                                             np.logspace(-1, np.log10(t_end), num_tpoints))
        self.relax_end = t_end
        return self.relax_ivp_result

    # time keeps going from where relax_ivp stopped (so no more "+ 1000"s), and strain_rate
    # hands back the time since the stress was switched on
    def stress_ivp(self, t_end=100, num_tpoints=40):
        if self.elim_relax is None:
            raise TypeError("run relax_ivp first")
        if self.elim_stress is not None:
            raise TypeError("cannot run stress_ivp more than once")

        self.elim_stress = self.elim_relax.copy()
        relaxed = self.relax_ivp_result.y[:,-1]
        self.stress_ivp_result = self.run_ivp(relaxed, self.relax_end, self.relax_end + t_end,
                                              self.sigma_ext, self.elim_stress,
                                              self.relax_end + np.logspace(-1, np.log10(t_end), num_tpoints))
        return self.stress_ivp_result


    # plastic strain rate: sum over lines of b * (area swept per unit time) / Lz, which for
    # straight lines is the 2D sum of b * v. uses the sigma_ext it's given this time (not 0.03)
    def strain_rate(self, elim, ivp_result, sigma_ext, t_offset=0):
        strains = np.zeros(ivp_result.t.size)
        for i, t in enumerate(ivp_result.t):
            alive = elim.alive_at(t).astype(float)
            xs, zs = self.unpack(ivp_result.y[:,i])
            v_n, geo = self.normal_velocity(xs, zs, sigma_ext, alive)
            strains[i] = np.sum(alive * self.bs * np.sum(v_n * geo.node_len, axis=1)) / self.Lz
        return ivp_result.t - t_offset, strains

    def strain_rate_graph(self, elim, ivp_result, sigma_ext, t_offset=0):
        ts, strains = self.strain_rate(elim, ivp_result, sigma_ext, t_offset)

        fig, ax = plt.subplots()
        ax.scatter(np.log10(ts), np.log10(np.abs(strains)))
        ax.set_xlabel("$log_{10}(t)$")
        ax.set_ylabel("$log_{10}$(strain rate)")
        ax.set_title(rf"strain rate vs time when $\sigma={sigma_ext}$")
        return fig


    # 3d picture of the lines, b = +1 and b = -1 in two colors (or two grays, for anaglyphs)
    def plot_lines(self, ax, state, alive, pos_color='y', neg_color='m', lw=1.5):
        xs, zs = self.unpack(state)
        for k in np.nonzero(alive)[0]:
            # close the loop (back to node 0, one period up), fold into the box, then break the line
            # wherever it wraps around in x or z
            xk = np.append(xs[k], xs[k][0]) % self.L
            zk = np.append(zs[k], zs[k][0] + self.Lz) % self.Lz
            jumps = np.nonzero((np.abs(np.diff(xk)) > self.L / 2) | (np.abs(np.diff(zk)) > self.Lz / 2))[0] + 1
            for x_piece, z_piece in zip(np.split(xk, jumps), np.split(zk, jumps)):
                ax.plot(x_piece, np.full(x_piece.size, self.ys[k]), z_piece,
                        c=(pos_color if self.bs[k] > 0 else neg_color), lw=lw)
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_zlim(0, self.Lz)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax


@app.function
# render a 3d matplotlib drawing to a grayscale image from one viewing angle
# draw_fn(ax) does the drawing; it gets called fresh for every view
def render_view(draw_fn, elev, azim, figsize=(8, 6), dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='w')
    ax = fig.add_subplot(projection='3d')
    draw_fn(ax)
    ax.view_init(elev=elev, azim=azim)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255
    plt.close(fig)
    # luminance, so both eyes see the same brightness
    return 0.299 * rgba[...,0] + 0.587 * rgba[...,1] + 0.114 * rgba[...,2]


@app.function
# red/cyan anaglyph, 2011 style: left eye in the red channel, right eye in green + blue
# (red lens over the left eye). eye_sep is how many degrees apart the two views are
def anaglyph(draw_fn, elev=20, azim=-60, eye_sep=4, **kwargs):
    left = render_view(draw_fn, elev, azim - eye_sep / 2, **kwargs)
    right = render_view(draw_fn, elev, azim + eye_sep / 2, **kwargs)
    return np.dstack((left, right, right))


@app.function
# wigglegram: flip back and forth between the two eyes' views. no glasses needed, very 2011 tumblr
def wigglegram(draw_fn, filename, elev=20, azim=-60, eye_sep=4, frames=6, **kwargs):
    angles = azim + eye_sep / 2 * np.sin(np.linspace(0, 2 * pi, frames, endpoint=False))
    images = [render_view(draw_fn, elev, ang, **kwargs) for ang in angles]
    fig = plt.figure(figsize=(images[0].shape[1] / 100, images[0].shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    shown = ax.imshow(images[0], cmap='gray', vmin=0, vmax=1)

    def animate(i):
        shown.set_data(images[i])
        return (shown,)

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True)
    ani.save(filename, writer=animation.PillowWriter(fps=12))
    plt.close(fig)


@app.function
def write_to_disk(filename, obj):
    if os.path.exists(filename):
        raise ValueError("would overwrite an existing file")
    else:
        with open(filename, "wb") as file:
            pickle.dump(obj, file)


@app.cell
def _():
    mo.md(r"""
    ## checks

    First the stress formula against textbook results for straight lines a long way out, where the core
    spreading doesn't matter: an edge line along z (which is the 2D model's sigma_s), a screw line along x,
    and a mixed line tilted 0.6 rad, all with b along x. Then the exact segment integrals against brute
    force quadrature of the same formula, for random segments close to the field points.
    """)
    return


@app.cell
def _():
    # one long straight line through the origin along `direction`, sampled finely near the middle
    def long_line(direction):
        u = np.sinh(np.linspace(-12, 12, 100001))
        u = u / u.max() * 5e4
        mid = (u[1:] + u[:-1]) / 2
        d = np.array(direction, float) / np.linalg.norm(direction)
        return mid.reshape(-1,1) * d, np.diff(u).reshape(-1,1) * d

    check_rng = np.random.default_rng(1)
    check_nu = 0.3
    check_pts = check_rng.uniform(-8, 8, (20, 3))

    # edge line along z: should be x (x^2 - y^2) / (x^2 + y^2)^2
    edge_pts = check_pts * [1, 1, 0]
    line_pts, line_dls = long_line([0, 0, 1])
    edge_num = sigma_xy_pk(edge_pts, line_pts, line_dls, 1e-3, check_nu)
    ex, ey = edge_pts[:,0], edge_pts[:,1]
    edge_exact = ex * (ex**2 - ey**2) / (ex**2 + ey**2)**2

    # screw line along x: sigma_xy = -(mu / 2pi) z / (y^2 + z^2), and mu / 2pi = 1 - nu here
    screw_pts = check_pts * [0, 1, 1]
    line_pts, line_dls = long_line([1, 0, 0])
    screw_num = sigma_xy_pk(screw_pts, line_pts, line_dls, 1e-3, check_nu)
    sy, sz = screw_pts[:,1], screw_pts[:,2]
    screw_exact = -(1 - check_nu) * sz / (sy**2 + sz**2)

    # mixed line tilted by th in the x-z plane: split b into its edge and screw parts in the line's
    # own frame (e1, e2 = y, e3 = along the line), use the textbook fields there, rotate back
    th = 0.6
    e3 = np.array([np.sin(th), 0, np.cos(th)])
    e2 = np.array([0., 1, 0])
    e1 = np.cross(e2, e3)
    line_pts, line_dls = long_line(e3)
    mixed_num = sigma_xy_pk(check_pts, line_pts, line_dls, 1e-3, check_nu)
    b_edge, b_screw = e1[0], e3[0]
    X1, X2 = check_pts @ e1, check_pts @ e2
    r2 = X1**2 + X2**2
    local = np.zeros((20, 3, 3))
    local[:,0,0] = -b_edge * X2 * (3 * X1**2 + X2**2) / r2**2
    local[:,1,1] = b_edge * X2 * (X1**2 - X2**2) / r2**2
    local[:,0,1] = local[:,1,0] = b_edge * X1 * (X1**2 - X2**2) / r2**2
    local[:,2,2] = check_nu * (local[:,0,0] + local[:,1,1])
    local[:,0,2] = local[:,2,0] = -(1 - check_nu) * b_screw * X2 / r2
    local[:,1,2] = local[:,2,1] = (1 - check_nu) * b_screw * X1 / r2
    rot = np.array([e1, e2, e3]).T
    mixed_exact = np.einsum('ai,nij,bj->nab', rot, local, rot)[:,0,1]

    for check_name, num_, exact_ in [("edge", edge_num, edge_exact), ("screw", screw_num, screw_exact),
                                     ("mixed", mixed_num, mixed_exact)]:
        print(f"{check_name}: max relative error {np.max(np.abs(num_ - exact_)) / np.max(np.abs(exact_)):.1e}")

    # exact segment integrals vs chopping each segment into 20000 pieces
    seg_starts = np.column_stack((check_rng.uniform(-2, 2, 30), np.full(30, 0.3), check_rng.uniform(-2, 2, 30)))
    seg_vecs = np.column_stack((check_rng.uniform(-2, 2, 30), np.zeros(30), check_rng.uniform(-2, 2, 30)))
    near_pts = check_rng.uniform(-3, 3, (40, 3))
    t_fine = (np.arange(20000) + 0.5) / 20000
    fine_pts = (seg_starts.reshape(30, 1, 3) + t_fine.reshape(1, -1, 1) * seg_vecs.reshape(30, 1, 3)).reshape(-1, 3)
    fine_dls = np.repeat(seg_vecs / 20000, 20000, axis=0)
    seg_brute = sigma_xy_pk(near_pts, fine_pts, fine_dls, 0.2, check_nu)
    seg_exact = sigma_xy_segments(near_pts, seg_starts, seg_vecs, 0.2, check_nu)
    print(f"segments: max relative error {np.max(np.abs(seg_exact - seg_brute)) / np.max(np.abs(seg_brute)):.1e}")
    return


@app.cell
def _():
    mo.md(r"""
    Next: straight lines (`wiggle=0`) should move exactly like the 2D model. Then one wiggly line on its
    own should straighten out under line tension, with a small cosine wiggle decaying like exp(-T k² t).
    """)
    return


@app.cell
def _():
    straight = LatticeCell3D(20, 0.01, 1, 30, 30, 16, wiggle=0, rng_seed=3)
    # every node of a straight line should move at the 2D model's v = b (sum b sigma_s + sigma)
    v_3d_x, v_3d_z = straight.unpack(straight.vs(straight.pack(straight.x0s, straight.z0s), 0.01))
    S_2d = sigma_s_analytic_sum(straight.x0s[:,0], straight.ys, straight.x0s[:,0], straight.ys, straight.L, straight.a)
    v_2d = straight.bs * (straight.bs @ S_2d.T + 0.01)
    print("straight lines vs 2D model, max |difference|:", np.max(np.abs(v_3d_x - v_2d.reshape(-1,1))),
          "and nothing moves in z:", np.max(np.abs(v_3d_z)))

    # lone line with a single small cosine wiggle
    lone = LatticeCell3D(2, 0, 1, 30, 30, 32, wiggle=0, rng_seed=0)
    # second line switched off with mults so only the first one is around
    lone_mults = np.array([1., 0.])
    amp0 = 0.2
    lone_x = lone.x0s.copy()
    lone_x[0] = lone_x[0] + amp0 * np.cos(2 * pi * lone.z0s[0] / lone.Lz)
    lone_res = solve_ivp(lambda t, s: lone.vs(s, 0, lone_mults), (0, 50), lone.pack(lone_x, lone.z0s),
                         t_eval=[50], rtol=1e-8, atol=1e-10)
    lone_final, _ = lone.unpack(lone_res.y[:,-1])
    amp_final = (lone_final[0].max() - lone_final[0].min()) / 2
    # on a grid of nodes, k^2 turns into (2 - 2 cos(k dz)) / dz^2
    k = 2 * pi / lone.Lz
    dz = lone.Lz / lone.M
    k2_grid = (2 - 2 * np.cos(k * dz)) / dz**2
    print(f"wiggle amplitude after t = 50: {amp_final:.5f}, line tension says {amp0 * np.exp(-lone.line_tension * k2_grid * 50):.5f}")
    return


@app.cell
def _():
    mo.md(r"""
    ## a run

    36 lines in a 30 × 30 × 30 box (same density as the paper's 400 in 100 × 100), relaxed at zero
    stress and then loaded, once with wiggly lines and once with the same lines straightened. At t = 100
    for each phase this takes about a minute (it waits for the button); t = 1000 each takes more like
    six and shows the decay a lot better.
    """)
    return


@app.cell
def _():
    run_button = mo.ui.run_button(label="run wiggly + straight")
    run_button
    return (run_button,)


@app.cell
def _(run_button):
    N_3d = 36
    sigma_3d = 0.0125
    # 1000 and 1000 is about six minutes
    t_relax = 100
    t_stress = 100

    if run_button.value:
        cells = {}
        for name, wig in [("wiggly", 1.5), ("straight", 0)]:
            start = time.time()
            lc = LatticeCell3D(N_3d, sigma_3d, 1, 30, 30, 24, wiggle=wig, rng_seed=0)
            lc.relax_ivp(t_relax)
            lc.stress_ivp(t_stress)
            print(f"{name}: {time.time() - start:.0f}s, {int(np.sum(lc.elim_stress.mults))} of {N_3d} lines left")
            cells[name] = lc
    else:
        cells = None
    return cells, sigma_3d


@app.cell
def _(cells, sigma_3d):
    mo.stop(cells is None, mo.md("hit the button above to run"))

    fig_rate, ax_rate = plt.subplots(figsize=(7, 5))
    for name_, lc_ in cells.items():
        ts_, strains_ = lc_.strain_rate(lc_.elim_stress, lc_.stress_ivp_result, sigma_3d, t_offset=lc_.relax_end)
        ok = strains_ > 0
        ax_rate.plot(np.log10(ts_[ok]), np.log10(strains_[ok]), 'o-', ms=3, label=name_)
    ax_rate.set_xlabel("$log_{10}(t)$ since loading")
    ax_rate.set_ylabel("$log_{10}$(strain rate)")
    ax_rate.set_title(rf"strain rate vs time when $\sigma={sigma_3d}$")
    ax_rate.legend()
    mo.mpl.interactive(fig_rate)
    return


@app.cell
def _():
    mo.md(r"""
    ## marvelous 3d

    Rotatable version first, then the real thing: grab some red/cyan glasses (red over the left eye).
    Without a run these show the starting configuration.
    """)
    return


@app.cell
def _(cells):
    # whatever the latest configuration is: the end of the wiggly run if there was one, else the start
    if cells is not None:
        show_lc = cells["wiggly"]
        show_state = show_lc.stress_ivp_result.y[:,-1]
        show_alive = show_lc.elim_stress.mults != 0
    else:
        show_lc = LatticeCell3D(36, 0.0125, 1, 30, 30, 24, wiggle=1.5, rng_seed=0)
        show_state = show_lc.pack(show_lc.x0s, show_lc.z0s)
        show_alive = np.ones(show_lc.N, dtype=bool)
    return show_alive, show_lc, show_state


@app.cell
def _(show_alive, show_lc, show_state):
    fig_3d = plt.figure(figsize=(8, 7))
    ax_3d = fig_3d.add_subplot(projection='3d')
    show_lc.plot_lines(ax_3d, show_state, show_alive)
    mo.mpl.interactive(fig_3d)
    return


@app.cell
def _(show_alive, show_lc, show_state):
    # for anaglyphs the colors have to go: b = +1 black, b = -1 gray
    def draw_lines(ax):
        show_lc.plot_lines(ax, show_state, show_alive, pos_color='k', neg_color='0.55', lw=2)

    lines_anaglyph = anaglyph(draw_lines, elev=18, azim=-55, eye_sep=4)
    fig_ana, ax_ana = plt.subplots(figsize=(10, 7.5))
    ax_ana.imshow(lines_anaglyph)
    ax_ana.set_axis_off()
    fig_ana
    return (draw_lines,)


@app.cell
def _(show_alive, show_lc, show_state):
    # sigma_xy over one glide plane, as a surface. the plane sits just above the bendiest line,
    # since straight lines just make boring stripes
    show_xs, show_zs = show_lc.unpack(show_state)
    bendiest = np.argmax(np.ptp(show_xs, axis=1) * show_alive)
    plane_y = show_lc.ys[bendiest] + 0.5
    grid_x, grid_z = np.meshgrid(np.linspace(0, show_lc.L, 90), np.linspace(0, show_lc.Lz, 90))
    plane_pts = np.column_stack((grid_x.flatten(), np.full(grid_x.size, plane_y), grid_z.flatten()))
    plane_sigma = show_lc.stress_xy(show_xs, show_zs, show_alive.astype(float),
                                    field_pts=plane_pts).reshape(grid_x.shape)
    # squash the spikes near the cores so the rest of the landscape shows up
    plane_height = np.sign(plane_sigma) * np.abs(plane_sigma)**0.5

    def draw_surface(ax):
        ax.plot_surface(grid_x, grid_z, plane_height, cmap='gray', rstride=1, cstride=1, linewidth=0,
                        antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel(r"sign $\cdot \sqrt{|\sigma_{xy}|}$")
        ax.set_title(f"shear stress on the glide plane y = {plane_y:.1f}")

    surface_anaglyph = anaglyph(draw_surface, elev=35, azim=-60, eye_sep=4)
    fig_surf, ax_surf = plt.subplots(figsize=(10, 7.5))
    ax_surf.imshow(surface_anaglyph)
    ax_surf.set_axis_off()
    fig_surf
    return


@app.cell
def _(draw_lines):
    # no glasses? wigglegram. writes a gif next to the notebook
    wigglegram(draw_lines, "wigglegram.gif", elev=18, azim=-55, eye_sep=4)
    mo.image("wigglegram.gif")
    return


if __name__ == "__main__":
    app.run()

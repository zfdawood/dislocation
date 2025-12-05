import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Attempting to duplicate the results of [Dislocation Jamming and Andrade Creep](https://emory-my.sharepoint.com/:b:/g/personal/jburto7_emory_edu/Eb4he1EhqCtDk79MQ6LvvdQBhMWK9qSw14aaEDOr7Y_ygQ?e=MvFXOW)
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import HTML
    import matplotlib.animation as animation
    from functools import partial
    from scipy.integrate import solve_ivp
    import matplotlib
    matplotlib.rcParams['animation.embed_limit'] = float('inf')
    return HTML, animation, np, partial, plt, solve_ivp


@app.cell
def _(np):
    N = 300 # number of edge dislocations 
    sigma_ext = 0.01 # external shear stress 
    b = 2.56e-10 #3e-8 # burgers vector magnitude for copper use 2.56 A
    chi_d = 1 # effective mobility 
    mu = 44e9 # shear modulus (same units as sheer stress), for ice, use 3 GPa, for copper use 44 GPa
    nu = 0.34 # poisson ratio, for ice use 0.3, for copper use 0.34
    ye = 1.6e-9 # burgers vector annihilation distance. For copper use 1.6 nm 
    D = mu / (2 * np.pi * (1 - nu))
    L = 100 * ye # size of cell
    return D, L, N, b, chi_d, sigma_ext


@app.cell
def _(L, b, np):
    # create "perfect" grid points with interatomic spacing equal to b 
    grid = np.arange(0,L,b) # spacing sholud be b 
    grid_xs = np.repeat(grid, grid.size)
    grid_ys = np.tile(grid, grid.size)
    return grid_xs, grid_ys


@app.cell
def _(L, N, b, grid_xs, grid_ys, np):
    mod_grid_xs = grid_xs.copy()
    mod_grid_ys = grid_ys.copy()
    extras_xs = np.zeros(0)
    extras_ys = np.zeros(0)
    dislocation_sign = np.array([1 if _x < 0.5 else -1 for _x in np.random.rand(N)]).reshape(-1, 1)
    dislocation_pos = np.hstack((L * np.random.rand(N, 1), b * np.random.randint(0, high=L / b, size=(N, 1)), dislocation_sign))
    for [_x, _y, _] in dislocation_pos:
        mod_grid_xs = mod_grid_xs + (b - (mod_grid_xs - _x) % b) * (mod_grid_xs > _x) * (mod_grid_ys <= _y)
        mod_grid_xs = mod_grid_xs + ((_x - mod_grid_xs) % b - b) * (mod_grid_xs < _x) * (mod_grid_ys <= _y)
        extras_xs = np.append(extras_xs, np.repeat(_x, _y / b))
        extras_ys = np.append(extras_ys, np.linspace(0, _y, int(_y / b)))
    mod_grid_xs = mod_grid_xs % L
    return dislocation_pos, extras_xs, extras_ys, mod_grid_xs, mod_grid_ys


@app.cell
def _(extras_xs, extras_ys):
    print(extras_xs.shape)
    print(extras_ys.shape)
    return


@app.cell
def _(
    b,
    dislocation_pos,
    extras_xs,
    extras_ys,
    grid_xs,
    grid_ys,
    mod_grid_xs,
    mod_grid_ys,
    plt,
):
    _fig, (_ax, ax2) = plt.subplots(1, 2)
    _ax.scatter(mod_grid_xs, mod_grid_ys, c='g', marker='1', label='modified')
    _ax.scatter(grid_xs, grid_ys, c='b', label='perfect', marker='2')
    _ax.scatter(extras_xs, extras_ys, c='m', label='extra rows', marker=2)
    _ax.scatter(dislocation_pos[:, 0], dislocation_pos[:, 1], c='r', marker=1, label='dislocation')
    _ax.legend()
    ax2.scatter(mod_grid_xs, mod_grid_ys, c='g', marker='1', label='modified')
    ax2.scatter(grid_xs, grid_ys, c='b', label='perfect', marker='2')
    ax2.scatter(extras_xs, extras_ys, c='m', label='extra rows', marker=2)
    ax2.scatter(dislocation_pos[:, 0], dislocation_pos[:, 1], c='r', marker=1, label='dislocation')
    ax2.set_xlim(dislocation_pos[0, 0] - 3 * b, dislocation_pos[0, 0] + 3 * b)
    ax2.set_ylim(dislocation_pos[0, 1] - 3 * b, dislocation_pos[0, 1] + 3 * b)
    plt.show()
    return


@app.cell
def _(D, b, np):
    def sigma_s(dislocation_xs, dislocation_ys, actee_x, actee_y):
        _x = actee_x - dislocation_xs
        _y = actee_y - dislocation_ys
        dem = (_x ** 2 + _y ** 2) ** 2
    # try to do a "dumb" sinusoidal sigma_s
        return D * b * np.divide(_x * (_x ** 2 - _y ** 2), dem, where=dem != 0)
    return (sigma_s,)


@app.cell
def _(b, chi_d, sigma_ext):
    def v_i(sigma_fn, xs, curr_x, dislocation_poss, dislocation_i):
        current = dislocation_poss[dislocation_i,:]
        return chi_d * b**2 * current[2] * (dislocation_poss[:,2] @ sigma_fn(xs, dislocation_poss[:,1], curr_x, current[1]) + sigma_ext)
    return (v_i,)


@app.cell
def _(L, N, dislocation_pos, np, sigma_s, v_i):
    def dxsdt_func(t, xs_p):
        xs = xs_p % L
        dxsdt = np.zeros(N)
        for _i, _x in enumerate(xs):
            dxsdt[_i] = v_i(sigma_s, xs, _x, dislocation_pos, _i)
        return dxsdt
    return (dxsdt_func,)


@app.cell
def _():
    # get vs by just running the individual calculation again with the intermediate points
    return


@app.cell
def _(dislocation_pos, dxsdt_func, np, solve_ivp):
    ivp_result = solve_ivp(dxsdt_func, [0, 1000], dislocation_pos[:,0], t_eval=np.linspace(0,1000,10000))
    return (ivp_result,)


@app.cell
def _(
    HTML,
    animation,
    dislocation_pos,
    grid_xs,
    grid_ys,
    ivp_result,
    np,
    partial,
    plt,
):
    _anifig, _aniax = plt.subplots()
    _aniax.scatter(grid_xs, grid_ys, c='b', label='perfect', marker='2')
    _scat = _aniax.scatter([], [], c='r')

    def _animate(i, dislocations):
        thing = np.hstack((dislocations[:, _i].reshape(-1, 1), dislocation_pos[:, 1].reshape(-1, 1)))
        _scat.set_offsets(thing)
        return (_scat, _scat)
    ani = animation.FuncAnimation(_anifig, partial(_animate, dislocations=ivp_result.y), frames=1000, blit=True)
    HTML(ani.to_jshtml())
    return (ani,)


@app.cell
def _(ani):
    print(ani)
    return


@app.cell
def _(
    HTML,
    animation,
    dislocation_pos,
    grid_xs,
    grid_ys,
    ivp_result,
    np,
    partial,
    plt,
):
    _anifig, _aniax = plt.subplots()
    _aniax.scatter(grid_xs, grid_ys, c='b', label='perfect', marker='2')
    _scat = _aniax.scatter([], [], c='r')

    def _animate(i, dislocations):
        thing = np.hstack((dislocations[:, _i].reshape(-1, 1), dislocation_pos[:, 1].reshape(-1, 1)))
        _scat.set_offsets(thing)
        return (_scat, _scat)
    ani_1 = animation.FuncAnimation(_anifig, partial(_animate, dislocations=ivp_result.y), frames=9000, blit=True)
    HTML(ani_1.to_jshtml())
    return


@app.cell
def _(dislocation_pos, dxsdt_func, ivp_result, np, plt):
    v_ts = np.zeros(ivp_result.t.size)
    for _i, _t in enumerate(ivp_result.t):
        v_ts[_i] = dislocation_pos[:, 2] @ dxsdt_func(_t, ivp_result.y[:, _i])
    _fig, _ax = plt.subplots()
    _ax.plot(ivp_result.t, v_ts)
    plt.show()
    return


@app.cell
def _(dxsdt_func, ivp_result, np):
    v_ts_1 = np.zeros(ivp_result.t.size)
    for _i, _t in enumerate(ivp_result.t):
        v_ts_1[_i] = np.sum(dxsdt_func(_t, ivp_result.y[:, _i]))
    return (v_ts_1,)


@app.cell
def _(ivp_result, plt, v_ts_1):
    _fig, _ax = plt.subplots()
    _ax.plot(ivp_result.t, v_ts_1)
    plt.show()
    return


@app.cell
def _(np, plt):
    _x = np.linspace(0, 10, 100)
    _y = np.sin(_x)
    _fig, _ax = plt.subplots()
    _ax.plot(_x, _y)
    _ax.set_ylim(-0.5, 0.5)
    plt.show()  # Set y-axis limits from -0.5 to 0.5 for this specific Axes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    talk to alex about the way that standard crystal rows move when the extra rows move through it - show some demos below. Shouldn't necessarily affect the simulation results. for now, we move the intermediate points (rows between start and end points) back by b

    +\vec b is like a salagtite -\vec b is like a salagmite
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    tues morn - compute \sum bivi and get a good accounting of possibly-fualty assumptions
    """)
    return


@app.cell
def _():
    # mod_grid_xs_2 = mod_grid_xs.copy()
    # mod_grid_ys_2 = mod_grid_ys.copy()
    # extras_xs_2 = extras_xs.copy()
    # extras_ys_2 = extras_ys.copy()
    # dislocation_pos_2 = dislocation_pos.copy()

    # for [x, y, s] in dislocation_pos_2:
    #     mod_grid_xs_2 = (mod_grid_xs_2 - s * b * (mod_grid_xs_2 >= x) * (mod_grid_xs_2 <= x + 2 * b) * (mod_grid_ys < y)) % L
    #     x =  (x + 2 * b) % L
    #     extras_xs_2 = (extras_xs_2 + 2 * b) % L

    # fig, (ax, ax2) = plt.subplots(1,2)
    # ax.scatter(mod_grid_xs, mod_grid_ys, c='g', marker="1", label="modified")
    # ax.scatter(grid_xs, grid_ys, c = 'b', label="perfect", marker="2")
    # ax.scatter(extras_xs, extras_ys, c = 'm', label = "extra rows", marker = 2)
    # ax.scatter(dislocation_pos[:,0], dislocation_pos[:,1], c='r', marker = 1, label="dislocation")
    # ax.legend()

    # ax2.scatter(mod_grid_xs_2, mod_grid_ys, c='g', marker="1", label="modified")
    # ax2.scatter(grid_xs, grid_ys, c = 'b', label="perfect", marker="2")
    # ax2.scatter(extras_xs_2, extras_ys, c = 'm', label = "extra rows", marker = 2)
    # ax2.scatter(dislocation_pos_2[:,0], dislocation_pos_2[:,1], c='r', marker = 1, label="dislocation")
    # plt.show()
    return


@app.cell
def _():
    # anifig, aniax = plt.subplots()
    # aniax.scatter(grid_xs, grid_ys, c = 'b', label="perfect", marker="2")
    # scat = aniax.scatter([],[], c='r')
    # dislocation_pos_2 = dislocation_pos.copy()
    # def animate(i, dislocation_pos):
    #     dislocation_pos += np.vstack(posadjust(dislocation_pos[0,:]) % b,
    #                         np.zeros(N)))
    #     for [x1, y1] in dislocation_pos.T:
    #         for [x2, y2] in dislocation_pos.T:
    #             if np.sqrt((x1 - x2)**2 + (y1-y2)**2) < ye:
    #                 x1 = x2 = y1 = y2 = 0 # get rid of suffficiently close dislocations
    #     scat.set_offsets(dislocation_pos.T)
    #     return (scat, scat)
    # ani = animation.FuncAnimation(anifig, partial(animate, dislocation_pos = dislocation_pos_2), frames=1000, blit=True)
    # HTML(ani.to_jshtml())
    return


@app.cell
def _(np):
    test = np.array([[1,2,3],[4,5,6]])
    print(test)
    print(np.sum(test, axis=0))
    print(np.sum(test, axis=1))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()


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
    import marimo as mo 
    rng = np.random.default_rng()
    matplotlib.rcParams['animation.embed_limit'] = float('inf')
    return HTML, animation, mo, np, partial, plt, rng, solve_ivp


@app.cell
def _(np):
    N = 30 # number of edge dislocations, must be even 
    sigma_ext = 0 # external shear stress 
    b = 4e-8 # burgers vector magnitude for copper use 2.56 A
    chi_d = 1e-12 # effective mobility, try something like 1 \mu m /sMPa
    mu = 44e9 # shear modulus (same units as sheer stress), for ice, use 3 GPa, for copper use 44 GPa
    nu = 0.34 # poisson ratio, for ice use 0.3, for copper use 0.34
    ye = 1.6e-9 # burgers vector annihilation distance. For copper use 1.6 nm 
    D = mu / (2 * np.pi * (1 - nu))
    L = 100 * ye # size of cell
    return D, L, N, b, chi_d, sigma_ext


@app.cell
def _(mo):
    mo.md(r"""
    $$\sigma_{xy}^{\text{per}}(x,y)
    = \sum_{m,n=-\infty}^{\infty}
    \sigma_{xy}^{\text{disl}}\big(x - x_0 - mL,; y - y_0 - nL\big)$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $$ \boxed{
    \sigma_{xy}^{\text{per}}(x,y)
    -,\frac{\mu b,\pi}{4(1-\nu)L^{2}},
    Y,
    \Big[
    \csc^{2}!\Big(\frac{\pi}{L}(X + iY)\Big)
    +
    \csc^{2}!\Big(\frac{\pi}{L}(X - iY)\Big)
    \Big]
    }$$
    """)
    return


app._unparsable_cell(
    r"""
    sum = 0
    for n in range(-10000, bb10000):
        for m in range(-1000a, 10000):
           sum += sigma_s() 
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    mo.md(\"$$\boxed{
    \sigma_{xy}^{\text{per}}(x,y)
    -,\frac{\mu b,\pi}{4(1-\nu)L^{2}},
    Y,
    \Big[
    \csc^{2}!\Big(\frac{\pi}{L}(X + iY)\Big)
    +
    \csc^{2}!\Big(\frac{\pi}{L}(X - iY)\Big)
    \Big]
    }$$\")
    """,
    name="_"
)


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


@app.cell
def _(N, np, plt, rng):

    def calculate_displacements(X, Y, center_x, center_y, b_mag, poisson_ratio):
        """
        Calculates the displacement field (u_x, u_y) generated by a single edge 
        dislocation centered at (center_x, center_y).
        """

        # Coordinates relative to the dislocation core
        X_rel = X - center_x
        Y_rel = Y - center_y

        # Polar Coordinates
        # Add a tiny epsilon (1e-9) to r to avoid division by zero near the core
        r = np.sqrt(X_rel**2 + Y_rel**2)
        theta = np.arctan2(Y_rel, X_rel)

        # Constants for the equations
        pi = np.pi
        nu = poisson_ratio
        const = b_mag / (2 * pi)

        # Displacement in X (u_x) - Volterra solution
        u_x = const * (theta + (np.sin(2*theta) / (4*(1-nu))))

        # Displacement in Y (u_y) - Volterra solution
        u_y = -const * ( ((1-2*nu)/(2*(1-nu))) * np.log(r + 1e-9) + (np.cos(2*theta) / (4*(1-nu))) )

        return u_x, u_y

    def generate_multiple_dislocations(
        grid_size=30, 
        b_mag=1.0, 
        poisson_ratio=0.3,
        dislocations=None
    ):
        """
        Simulates a lattice with multiple edge dislocations using superposition.

        Args:
            grid_size (int): The width/height of the lattice (number of atoms).
            dislocations (list): List of dictionaries defining each dislocation.
                Example: [{'center': (x1, y1), 'b_mag': b1}, ...]

        Returns:
            X_new, Y_new: Arrays of the displaced atomic coordinates.
        """

        if dislocations is None:
            raise ValueError("Dislocations list cannot be None.")

        # 1. Create a perfect 2D Lattice
        range_val = grid_size // 2
        x_coords = np.linspace(-range_val, range_val, grid_size) * b_mag
        y_coords = np.linspace(-range_val, range_val, grid_size) * b_mag
        X_perfect, Y_perfect = np.meshgrid(x_coords, y_coords)

        # Flatten coordinates for vector calculation
        X_flat = X_perfect.flatten()
        Y_flat = Y_perfect.flatten()

        # Initialize total displacement arrays
        U_x_total = np.zeros_like(X_flat)
        U_y_total = np.zeros_like(Y_flat)

        # 2. Superposition Principle: Sum the displacement fields
        for d in dislocations:
            center_x, center_y = d[0], d[1]
            b = d[2]

            u_x, u_y = calculate_displacements(
                X_flat, Y_flat, center_x, center_y, b, poisson_ratio
            )

            # Add the displacement of this dislocation to the total
            U_x_total += u_x
            U_y_total += u_y

        # 3. Apply total displacements to original coordinates
        X_new = X_flat + U_x_total
        Y_new = Y_flat + U_y_total

        return X_new, Y_new, [(d[0], d[1]) for d in dislocations]

    def plot_lattice(x, y, centers, title="Lattice"):
        plt.figure(figsize=(9, 9))

        # Plot atoms
        plt.scatter(x, y, c='cornflowerblue', edgecolors='k', s=70, zorder=10)

        # Highlight the cores
        for cx, cy in centers:
            plt.plot(cx, cy, 'rx', markersize=12, markeredgewidth=3, label='Dislocation Core' if cx == centers[0][0] else "")

        # Aesthetic settings
        plt.title(title, fontsize=15)
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.legend()
        plt.show()

    # --- Run the Simulation for a Dislocation Dipole ---
    if __name__ == "__main__":

        # Define the Dipole: Two dislocations with opposite Burgers vectors
        # This creates a stable configuration where they attract each other vertically.
        DISLOCATIONS_DIPOLE = dislocations = np.column_stack((
        (rng.random(N) * 30) - 15, # xs
        (rng.random(N) * 30) - 15, # ys
        # b vector directions, making sure the number of positive bs == number of negative bs
        rng.permuted(
            np.concatenate(
                (np.ones(int(N/2)),
                 -(np.ones(int(N/2))))))
    ))

    #     DISLOCATIONS_DIPOLE = dislocations = np.column_stack((
    #     (rng.random(N) * L), # xs
    #     (rng.random(N) * L), # ys
    #     # b vector directions, making sure the number of positive bs == number of negative bs
    #     rng.permuted(
    #         np.concatenate(
    #             (np.ones(int(N/2)),
    #              -(np.ones(int(N/2))))))
    # ))


        print("Generating Lattice with a Dislocation Dipole (Opposite Burgers Vectors)...")
        x_displaced, y_displaced, cores = generate_multiple_dislocations(
            grid_size=35, 
            b_mag=1.0, 
            dislocations=DISLOCATIONS_DIPOLE
        )

        # The resulting structure shows the atomic arrangement being less distorted 
        # than a single dislocation, especially in the far field, because the 
        # stress fields tend to cancel out.
        plot_lattice(
            x_displaced, y_displaced, cores, 
            title="Lattice with Dislocation Dipole (Superposition)"
        )
    return


if __name__ == "__main__":
    app.run()

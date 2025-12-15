import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium", auto_download=["html"])


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
    return mo, np, plt, rng


@app.cell
def _(np):
    # constants
    N = 400 # number of edge dislocations, must be even 
    sigma_ext = 0 # external shear stress 
    b = 2.56e-10 #4e-8 # burgers vector magnitude for copper use 2.56 A
    chi_d = 1e-12 # effective mobility, try something like 1 \mu m /sMPa
    mu = 3e9 # shear modulus (same units as sheer stress), for ice, use 3 GPa, for copper use 44 GPa
    nu = 0.34 # poisson ratio, for ice use 0.3, for copper use 0.34
    ye = 1.6e-9 # burgers vector annihilation distance. For copper use 1.6 nm 
    D = mu / (2 * np.pi * (1 - nu))
    L = 100 * ye # size of cell
    LATTICE_EDGE_CT = int(L / b) # 30
    return L, LATTICE_EDGE_CT, N, b, nu


@app.cell
def _(np, nu):
    def calculate_displacements(X, Y, disloc_x, disloc_y, b):
        # not a constant because it depends on whether b is pos or negative
        b2pi = b / (2 * np.pi)

        # existing lattice points
        X_rel = X - disloc_x
        Y_rel = Y - disloc_y
        # in polar 
        r = np.sqrt(X_rel**2 + Y_rel**2)
        theta = np.arctan2(Y_rel, X_rel)

        # horizontal and vertical displacement 
        u_x = b2pi * (theta + (np.sin(2*theta) / (4*(1-nu))))
        # the additive constant np.log(r + 1e-19) is to prevent log(0) errors 
        u_y = -b2pi * ( ((1-2*nu)/(2*(1-nu))) * np.log(r + 1e-19) + (np.cos(2*theta) / (4*(1-nu))) ) 
        return u_x, u_y
    return (calculate_displacements,)


@app.cell
def _(L, LATTICE_EDGE_CT, calculate_displacements, np):
    def apply_dislocations(dislocations, b):
        # non-dislocated lattice 
        # xs = np.linspace(0, LATTICE_EDGE_CT, LATTICE_EDGE_CT) 
        # ys = np.linspace(0, LATTICE_EDGE_CT, LATTICE_EDGE_CT) 
        xs = np.linspace(0, L, LATTICE_EDGE_CT)
        ys = np.linspace(0, L, LATTICE_EDGE_CT)
        X_perfect, Y_perfect = np.meshgrid(xs, ys)
        X_flat = X_perfect.flatten()
        Y_flat = Y_perfect.flatten()

        # accumulate displacements from dislocations
        disp_x = np.zeros_like(X_flat)
        disp_y = np.zeros_like(Y_flat)

        for c_x, c_y, b in dislocations:
            u_x, u_y = calculate_displacements(X_flat, Y_flat, c_x, c_y, b)
            disp_x += u_x
            disp_y += u_y

        X_new = (X_flat + disp_x) % L
        Y_new = (Y_flat + disp_y) % L
        return X_new, Y_new
    return (apply_dislocations,)


@app.cell
def _(plt):
    def plot_lattice(x, y, dislocations):
        fig, ax = plt.subplots(figsize=(9, 9))

        ax.scatter(x, y)

        ax.scatter(dislocations[:,0], dislocations[:,1], c=dislocations[:,2], label="dislocation lines")

        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        return fig
    return (plot_lattice,)


@app.cell
def _(L, N, apply_dislocations, b, np, plot_lattice, rng):
    dislocations = np.column_stack((
        rng.random(N) * L, # xs
        rng.random(N) * L, # ys
        # b vector directions, making sure the number of positive bs == number of negative bs
        rng.permuted(
            np.concatenate(
                (np.ones(int(N/2)),
                 -(np.ones(int(N/2))))))
    ))

    # dislocations = np.column_stack((
    #     ((rng.random(N) * LATTICE_EDGE_CT) * b), # xs
    #     ((rng.random(N) * LATTICE_EDGE_CT) * b), # ys
    #     # b vector directions, making sure the number of positive bs == number of negative bs
    #     rng.permuted(
    #         np.concatenate(
    #             (np.ones(int(N/2)),
    #              -(np.ones(int(N/2))))))
    # ))

    # dislocations = np.array([
    #     [4*b, 0, -1],
    #     [2*b, 0, 1]
    # ])

    X, Y = apply_dislocations(dislocations, b)
    plot_lattice(X,Y, dislocations)
    return X, Y, dislocations


@app.cell
def _(X, Y, dislocations, mo, plot_lattice):
    mo.mpl.interactive(plot_lattice(X, Y, dislocations))
    return


if __name__ == "__main__":
    app.run()

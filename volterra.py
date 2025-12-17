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
    pi = np.pi # adding the "np" every time honestly adds up 
    sigma_ext = 0 # external shear stress 
    b = 2.56e-10 #4e-8 # burgers vector magnitude for copper use 2.56 A
    chi_d = 1e-12 # effective mobility, try something like 1 \mu m /sMPa
    mu = 44e9 # shear modulus (same units as sheer stress), for ice, use 3 GPa, for copper use 44 GPa
    nu = 0.34 # poisson ratio, for ice use 0.3, for copper use 0.34
    ye = 1.6e-9 # burgers vector annihilation distance. For copper use 1.6 nm 
    D = mu / (2 * np.pi * (1 - nu))
    L = 100 * ye # size of cell
    LATTICE_EDGE_CT = int(L / b) # 30
    return D, L, LATTICE_EDGE_CT, N, b, chi_d, nu, pi, sigma_ext


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
        u_y = -b2pi * ( ((1-2*nu)/(2*(1-nu))) * np.log(r) + 
                        (np.cos(2*theta) / (4*(1-nu))) ) 
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

        # # accumulate displacements from dislocations
        # disp_x = np.zeros_like(X_flat)
        # disp_y = np.zeros_like(Y_flat)

        # for c_x, c_y, b in dislocations:
        #     u_x, u_y = calculate_displacements(X_flat, Y_flat, c_x, c_y, b)
        #     disp_x += u_x
        #     disp_y += u_y

        # X_new = (X_flat + disp_x)  % L
        # Y_new = (Y_flat + disp_y)  % L

        for c_x, c_y, b in dislocations:
            u_x, u_y = calculate_displacements(X_flat, Y_flat, c_x, c_y, b)
            X_flat = (X_flat + u_x) % L
            Y_flat = (Y_flat + u_y) % L

        return X_flat, Y_flat
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
        return fig
    return (plot_lattice,)


@app.cell
def _(L, N, apply_dislocations, b, mo, np, plot_lattice, rng):
    dislocations = np.column_stack((
        rng.random(N) * L, # xs
        rng.random(N) * L, # ys
        # b vector directions, making sure the number of positive bs == number of negative bs
        b * rng.permuted(
            np.concatenate((
                np.ones(int(N/2)),
                 -(np.ones(int(N/2)))
            ))
        )
    ))

    # dislocations = np.array([
    #     [4*b, 0, -1],
    #     [2*b, 0, 1]
    # ])

    X, Y = apply_dislocations(dislocations, b)
    mo.mpl.interactive(plot_lattice(X, Y, dislocations))
    return (dislocations,)


@app.cell
def _(D, L, N, b, np, pi):
    # infinite sum from eqn 1 in the paper
    def sigma_s_analytic_sum(target_x, target_y, dislocations):
        X = target_x - dislocations[:,0]
        Y = target_y - dislocations[:,1]

        x_arg = 2 * pi * X / L
        y_arg = 2 * pi * Y / L 

        num = pi * D * b * np.sin(x_arg) * (
            L * np.cos(x_arg) + 2 * pi * Y * np.sinh(y_arg)
            - L * np.cosh(y_arg)
        )
        den = L**2 * (np.cos(x_arg) - np.cosh(y_arg))**2

        #sets stress to zero when X[i] or Y[i] = 0 (ie at the point we dont want to compute)
        return np.divide(num, den, out=np.zeros(N), where=((X!=0) & (Y!=0))) # or den!=0) - add if needed (if getting divide by zero errors but should be taken care of with the other conditions)
    return (sigma_s_analytic_sum,)


@app.cell
def _(dislocations, sigma_s_analytic_sum):
    # probably worth pointing out just how big these stresses are (order of 1e6)
    sigma_s_analytic_sum(dislocations[0,0], dislocations[0,1], dislocations)
    return


@app.cell
def _(b, chi_d, sigma_ext):
    # eqn 2 in the paper
    def v_i(sigma_fn, xs, curr_x, dislocations, dislocation_i):
        _, curr_y, curr_b = dislocations[dislocation_i,:]
        bracketed = dislocations[:,2] @ sigma_fn(curr_x, curr_y, dislocations) + sigma_ext 
        print(bracketed)
        print(curr_b)
        print(curr_b * b**2 * chi_d)
        return curr_b * b**2 * chi_d * bracketed
    return (v_i,)


@app.cell
def _(dislocations, sigma_s_analytic_sum, v_i):
    # but now i feel like this has gone too far in the other direction, size-wise (order of 1e-23)
    v_i(sigma_s_analytic_sum, dislocations[:,0], dislocations[0,0], dislocations, 0)
    return


@app.cell
def _(b, chi_d):
    print(b**3 * chi_d * 1e7)
    return


if __name__ == "__main__":
    app.run()

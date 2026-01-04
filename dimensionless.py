import marimo

__generated_with = "0.18.4"
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
    return mo, np, plt, rng, solve_ivp


@app.cell
def _(np):
    # constants
    # here we use dimensionless constants, omitting eg nu, mu, D, which cancel (see the whiteboard pictures in dimensionless.md in the notes directory). they have an implied tilde over them 
    N = 40 # number of dislocations, must be even 
    assert N % 2 == 0, "N must be even" # hiring managers: I know I can do `1 - (N & 1)`, but this should be readable for physicists 
                                        # even if wasn't, I'd value readability and pythonicity over a tiny speed delta like this 
    pi = np.pi
    sigma_ext = 1 
    ye = 1 
    b = 1
    L = 100 * ye # size of cell
    t0 = 1
    LATTICE_EDGE_CT = int(L / b) 
    return L, LATTICE_EDGE_CT, N, b, pi, t0


@app.cell
def _(L, N, b, np, rng):
    dislocations = np.column_stack((
        rng.random(N) * L, # xs
        rng.random(N) * L, # ys 
        b * rng.permuted(
            np.concatenate((
                np.ones(int(N/2)),
                 -(np.ones(int(N/2)))
            ))
        )
    ))

    # dislocations = np.array([
    #     [4*b, 0, -b],
    #     [2*b, 0, b]
    # ])
    return (dislocations,)


@app.cell
def _(apply_dislocations, dislocations, mo, plot_lattice):
    X, Y = apply_dislocations(dislocations)
    mo.mpl.interactive(plot_lattice(X, Y, dislocations))
    return


@app.cell
def _(dislocations, sigma_s_analytic_sum, vs):
    vs(sigma_s_analytic_sum, dislocations[:,0], dislocations)
    return


@app.cell
def _(L, N, dislocations, np, sigma_s_analytic_sum, solve_ivp, t0, vs):
    # we use this vector to make sure that canceled dislocations don't contribute to dxdt 
    elimination_vector = np.column_stack((np.ones(N), np.zeros(N)))

    # number of points to use in the ivp solver
    num_tpoints = 150


    def dxsdt_func(t, xs_p, sigma_ext):
        # the positions used by the ivp solver aren't bound to the box so we apply that binding here
        xs = xs_p % L

    
        # annihilation_list is a list of ordered pairs of annihilated dislocations 
        # we mark a dislocation to be annihilated when... 
        annihilation_list = np.column_stack(np.nonzero(
            # ... two dislocations are within ye of each other... 
            # (this particular implementation draws more of a square bounding box 
            # than a circle one to check closeness but it's fine)
            (np.isclose(xs.reshape(-1,1), xs.reshape(1,-1), rtol=0, atol=10))
            & (np.isclose(dislocations[:,1].reshape(-1,1), dislocations[:,1].reshape(1,-1), rtol=0, atol=10))
            # ... and the dislocations have opposite burgers vectors... 
            & (dislocations[:,2].reshape(-1,1) != dislocations[:,2].reshape(1,-1)) 
            # ... and they aren't the same dislocation (should be taken care of by the previous point)
            # & (np.logical_not(np.diagflat(np.repeat(True, N))))
        ))

        for (i, j) in annihilation_list: 
            # if (elimination_vector[i] != 0) or (elimination_vector[j] != 0):
            #     print(f"annihilation event with {i} and {j} at t = {t}, the {t/t0}th step")
            if elimination_vector[i,0] != 0:
                elimination_vector[i,:] = [0,t] 
            if elimination_vector[j,0] != 0:
                elimination_vector[j,:] = [0,t]
            
        # this whole thing can probably be improved by concatenating after np.nonzero and then
        # just elimination_vector[concatenated id'd close dislocations, duplicates are fine] = 0
        # but that'll make recording the time difficult 

        return elimination_vector[:,0] * vs(sigma_s_analytic_sum, xs, dislocations)

    # keep this function call with the dxsdt func definition and elimination_vector init
    ivp_result = solve_ivp(dxsdt_func, (t0, num_tpoints*t0), dislocations[:,0], 
                           t_eval=np.linspace(t0, num_tpoints * t0, num_tpoints), method="RK23", args=(0,))
    return dxsdt_func, ivp_result, num_tpoints


@app.cell
def _(L, ivp_result, np):
    print(ivp_result.y[np.array([18,28]),:] % L)
    return


@app.cell
def _():
    # sloppy code to double-check closeness 
    # lastxs = ivp_result.y[:,-1]
    # lastys = dislocations[:,1]
    # for i in range(0,N):
    #     for j in range(0,N):
    #         if (np.sqrt((lastxs[i] - lastxs[j])**2 + (lastys[i] - lastys[j])**2) < ye) and dislocations[i,2] != dislocations[j,2] and (elimination_vector[i] != 0 or elimination_vector[j] != 0):
    #             print(f"{i} and {j} and {abs(ivp_result.y[:,-1][i] - ivp_result.y[:,-1][j])} and {abs(lastys[i]-lastys[j])}")
    return


@app.cell
def _(
    L,
    apply_dislocations,
    b,
    dislocations,
    ivp_result,
    mo,
    np,
    plot_lattice,
):
    post_relaxation_dislocations = np.copy(dislocations)
    post_relaxation_dislocations[:,0] = ivp_result.y[:,-1] % L 

    X_post, Y_post = apply_dislocations(post_relaxation_dislocations, b)
    mo.mpl.interactive(plot_lattice(X_post, Y_post, post_relaxation_dislocations))
    return (post_relaxation_dislocations,)


@app.cell
def _(
    dxsdt_func,
    np,
    num_tpoints,
    post_relaxation_dislocations,
    solve_ivp,
    t0,
):
    ivp_result_ext_stress = solve_ivp(dxsdt_func, (t0, num_tpoints*t0), post_relaxation_dislocations[:,0], 
                           t_eval=np.linspace(t0, num_tpoints * t0, num_tpoints), method="RK23", args=(0.035,))
    return (ivp_result_ext_stress,)


@app.cell
def _(N, dislocations, ivp_result_ext_stress, np, sigma_s_analytic_sum, v_i):
    # rigorify this 
    strain_deriv = np.zeros(800)

    old_elimination = np.ones(N)
    old_elimination[[18, 28]] = 0 

    new_elimination = np.ones(N)
    new_elimination[[18, 28, 38, 39]] = 0

    for i in range(800):
        for j in range(N):
            temp_dislocations = np.copy(dislocations)
            temp_dislocations[:,0] = ivp_result_ext_stress.y[:,i]
            if (i < 448) & (j in [18,28]) :
                continue 
            elif (i > 448) & (j in [18, 28, 38, 39]):
                continue
            else:
                # FIX THE vi and sigma fns to take current xs and not dislocations 
                strain_deriv[i] += dislocations[j,2] * v_i(sigma_s_analytic_sum, ivp_result_ext_stress.y[:,i],
                                             ivp_result_ext_stress.y[j,i], temp_dislocations, j, sigma_ext=0.035)
    print(strain_deriv)
    return (strain_deriv,)


@app.cell
def _(ivp_result_ext_stress, np, plt, strain_deriv):
    fig, ax = plt.subplots()
    ax.plot(np.log10(ivp_result_ext_stress.t[:600]), np.log10(strain_deriv[:600]))
    plt.show()
    return


@app.cell
def _(np):
    def calculate_displacements(X, Y, disloc_x, disloc_y, b, nu=0.3):
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
    def apply_dislocations(dislocations):
        # non-dislocated lattice 
        xs = np.linspace(0, L, LATTICE_EDGE_CT)
        ys = np.linspace(0, L, LATTICE_EDGE_CT)
        X_perfect, Y_perfect = np.meshgrid(xs, ys)
        X_flat = X_perfect.flatten()
        Y_flat = Y_perfect.flatten()

        # iteratively apply dislocations to lattice points 
        for c_x, c_y, b in dislocations:
            u_x, u_y = calculate_displacements(X_flat, Y_flat, c_x, c_y, b)
            X_flat = (X_flat + u_x) % L
            Y_flat = (Y_flat + u_y) % L

        return X_flat, Y_flat
    return (apply_dislocations,)


@app.cell
def _(L, LATTICE_EDGE_CT, calculate_displacements, np):
    # will give similar results as apply_dislocations but the edge behavior is cleaner, though i think it's technically incorrect 
    def apply_dislocations2(dislocations):
        # non-dislocated lattice 
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

        X_new = (X_flat + disp_x)  % L
        Y_new = (Y_flat + disp_y)  % L

        return X_new, Y_new 
    return


@app.cell
def _(plt):
    def plot_lattice(lattice_x, lattice_y, dislocations):
        fig, ax = plt.subplots(figsize=(9, 9))

        ax.scatter(lattice_x, lattice_y)

        ax.scatter(dislocations[:,0], dislocations[:,1], c=dislocations[:,2], label="dislocation centers")

        for i, xy in enumerate(zip(dislocations[:,0], dislocations[:,1])):
            ax.annotate(i, xy, c='r')

        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return fig
    return (plot_lattice,)


@app.cell
def _(L, N, np, pi):
    # infinite sum from eqn 1 in the paper
    # xs and ys should be row vectors - MAKE SURE THEY'RE WITHIN L 
    def sigma_s_analytic_sum(xs, ys):
        # X is a matrix whose ith column is xs[i] - xs, x[i]'s distance from the other dislocations (Y too WLOG)
        X = xs - xs.reshape(N,1)
        Y = ys - ys.reshape(N,1)

        x_arg = 2 * pi * X / L
        y_arg = 2 * pi * Y / L 

        # eliminated leading constants before doing the analytic sum 
        num = pi * np.sin(x_arg) * (
            L * np.cos(x_arg) + 2 * pi * Y * np.sinh(y_arg)
            - L * np.cosh(y_arg)
        )
        den = L**2 * (np.cos(x_arg) - np.cosh(y_arg))**2

        # sets stress to zero when X[i] or Y[i] = 0 (ie at the point we dont want to compute)
        return np.divide(num, den, out=np.zeros((N,N)), where=((X!=0) & (Y!=0))) # or den!=0) - add if needed (if getting divide by zero errors but should be taken care of with the other conditions) 
    return (sigma_s_analytic_sum,)


@app.cell
def _(np):
    # eqn 2 in the paper (but vectorized! MWAHAHA)
    # make sure sigma_fn returns a matrix whose main diagonal elements are zero
    def vs(sigma_fn, xs, dislocations, sigma_ext=0):

        sigmas = sigma_fn(xs, dislocations[:,1])
        assert np.all(np.isclose(0, np.diagonal(sigmas))), "make sure your sigma_fn returns a matrix with zeros on the main diagonal"

        return np.sign(dislocations[:,2]) * (np.sign(dislocations[:,2]) @ sigmas + sigma_ext)
    return (vs,)


if __name__ == "__main__":
    app.run()

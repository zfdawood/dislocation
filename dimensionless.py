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
    import pickle 
    import os 

    rng = np.random.default_rng()
    matplotlib.rcParams['animation.embed_limit'] = float('inf')
    return animation, mo, np, os, partial, pickle, plt, rng, solve_ivp


@app.cell
def _(np):
    # constants
    # here we use dimensionless constants, omitting eg nu, mu, D, which cancel (see the whiteboard pictures in dimensionless.md in the notes directory). they have an implied tilde over them 
    N = 400 # number of dislocations, must be even 
    assert N % 2 == 0, "N must be even" # hiring managers: I know I can do `1 - (N & 1)`, but this should be readable for physicists 
                                        # even if wasn't, I'd value readability and pythonicity over a tiny speed delta like this 
    pi = np.pi
    sigma_ext = 1 
    ye = 1 
    b = 1
    L = 100 * ye # size of cell
    t0 = 1
    LATTICE_EDGE_CT = int(L / b) 
    return L, LATTICE_EDGE_CT, N, b, pi, t0, ye


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
    return X, Y


@app.cell
def _(dislocations, sigma_s_analytic_sum, vs):
    vs(sigma_s_analytic_sum, dislocations[:,0], dislocations)
    return


@app.cell
def _(L, N, dislocations, np, sigma_s_analytic_sum, solve_ivp, t0, vs, ye):
    # we use this vector to make sure that canceled dislocations don't contribute to dxdt 
    elimination_vector = np.column_stack((np.ones(N), np.zeros(N)))

    # number of points to use in the ivp solver
    num_tpoints = 1000


    def dxsdt_func(t, xs_p, sigma_ext):
        # the positions used by the ivp solver aren't bound to the box so we apply that binding here
        xs = xs_p % L
        print(t)

        # annihilation_list is a list of ordered pairs of annihilated dislocations 
        # we mark a dislocation to be annihilated when... 
        annihilation_list = np.column_stack(np.nonzero(
            # ... two dislocations are within ye of each other... 
            # (this particular implementation draws more of a square bounding box 
            # than a circle one to check closeness but it's fine)
            (np.isclose(xs.reshape(-1,1), xs.reshape(1,-1), rtol=0, atol=ye))
            & (np.isclose(dislocations[:,1].reshape(-1,1), dislocations[:,1].reshape(1,-1), rtol=0, atol=ye))
            # ... and the dislocations have opposite burgers vectors... 
            & (dislocations[:,2].reshape(-1,1) != dislocations[:,2].reshape(1,-1)) 
            # ... and they haven't been previously selected...
            # (this looks convoluted but the operation would take literally forever without it)
            & np.outer((elimination_vector[:,0].reshape(-1,1)), (elimination_vector[:,0])).astype(bool)
            # ... and they aren't the same dislocation (should be taken care of by the previous point)
            # & (np.logical_not(np.diagflat(np.repeat(True, N))))
        ))

        for (i, j) in annihilation_list: 
            # if (elimination_vector[i,0] != 0) or (elimination_vector[j,0] != 0):
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
                           t_eval=np.linspace(t0, num_tpoints * t0, num_tpoints), method="RK45", args=(0,))
    return elimination_vector, ivp_result, num_tpoints


@app.cell
def _():
    # with open("./pickles/1000iter_rk45_LAPTOP.pkl", "wb") as file:
    #     pickle.dump((dislocations, ivp_result, elimination_vector), file)
    return


@app.cell
def _():
    # with open("./pickles/1000iter_rk45_LAPTOP.pkl", "rb") as file2:
    #     uhhh = pickle.load(file2)
    # uhhh
    return


@app.cell
def _(uhhh):
    uhhh[1].t
    return


@app.cell
def _(L, apply_dislocations, dislocations, ivp_result, mo, np, plot_lattice):
    post_relaxation_dislocations = np.copy(dislocations)
    post_relaxation_dislocations[:,0] = ivp_result.y[:,-1] % L 

    X_post, Y_post = apply_dislocations(post_relaxation_dislocations)
    mo.mpl.interactive(plot_lattice(X_post, Y_post, post_relaxation_dislocations))
    return


@app.cell
def _(
    L,
    X,
    Y,
    animation,
    apply_dislocations,
    dislocations,
    elimination_vector,
    ivp_result,
    np,
    partial,
    plt,
):
    anifig, aniax = plt.subplots(figsize=(9,9))

    aniax.scatter(X,Y, label="lattice points (base)")
    aniax.scatter(dislocations[:,0], dislocations[:,1], label="dislocations (base)")

    scat_lattice = aniax.scatter([], [], c='g', label="lattice points")
    scat_pos_b = aniax.scatter([], [], c='y', label="dislocations with b = 1")
    scat_neg_b = aniax.scatter([], [], c='m', label="dislocations with b = -1")

    def animate(i, xs_p):
        xs = xs_p[:,i] % L

        lattice_positions = np.column_stack(apply_dislocations(np.column_stack((
            xs, dislocations[:,1:]
        ))))
        scat_lattice.set_offsets(lattice_positions)

        xs_pos = xs[(dislocations[:,2] == 1) & ((elimination_vector[:,1] > i) | (elimination_vector[:,1] == 0))]
        ys_pos = dislocations[(dislocations[:,2] == 1) & ((elimination_vector[:,1] > i) 
                                                          | (elimination_vector[:,1] == 0)), 1]
        dislocations_i_pos = np.column_stack((
            xs_pos,
            ys_pos,
        ))
        scat_pos_b.set_offsets(dislocations_i_pos)

        xs_neg = xs[(dislocations[:,2] == -1) & ((elimination_vector[:,1] > i) | (elimination_vector[:,1] == 0))]
        ys_neg = dislocations[(dislocations[:,2] == -1) & ((elimination_vector[:,1] > i) 
                                                           | (elimination_vector[:,1] == 0)), 1]
        dislocations_i_neg = np.column_stack((
            xs_neg,
            ys_neg
        ))
        scat_neg_b.set_offsets(dislocations_i_neg)

        return (scat_lattice, scat_pos_b, scat_neg_b)

    aniax.set_title("Randomly-placed dislocation relaxation\n400 dislocations, annihilation distance within 1 lattice cell ") 
    aniax.set_xlabel("x")
    aniax.set_ylabel("y")
    aniax.legend()

    ani = animation.FuncAnimation(anifig, 
                                  partial(animate, xs_p=ivp_result.y), 
                                  frames=1000, 
                                  blit=True)    

    # HTML(ani.to_jshtml())

    FFwriter = animation.FFMpegWriter(fps=10)
    return FFwriter, ani


@app.cell
def _(FFwriter, ani, os):
    vidname_relax = './1000iter_relaxation_animation.mp4'
    if not os.path.exists(vidname_relax):
        ani.save('1000iter_relaxation_animation.mp4', writer = FFwriter)
    return


@app.cell
def _(
    L,
    dislocations,
    elimination_vector,
    ivp_result,
    np,
    num_tpoints,
    sigma_s_analytic_sum,
    solve_ivp,
    t0,
    vs,
    ye,
):
    # we use this vector to make sure that canceled dislocations don't contribute to dxdt 
    elimination_vector_2 = elimination_vector.copy()


    def dxsdt_func_2(t, xs_p, sigma_ext):
        # the positions used by the ivp solver aren't bound to the box so we apply that binding here
        xs = xs_p % L
        print(t + 1000)

        # annihilation_list is a list of ordered pairs of annihilated dislocations 
        # we mark a dislocation to be annihilated when... 
        annihilation_list = np.column_stack(np.nonzero(
            # ... two dislocations are within ye of each other... 
            # (this particular implementation draws more of a square bounding box 
            # than a circle one to check closeness but it's fine)
            (np.isclose(xs.reshape(-1,1), xs.reshape(1,-1), rtol=0, atol=ye))
            & (np.isclose(dislocations[:,1].reshape(-1,1), dislocations[:,1].reshape(1,-1), rtol=0, atol=ye))
            # ... and the dislocations have opposite burgers vectors... 
            & (dislocations[:,2].reshape(-1,1) != dislocations[:,2].reshape(1,-1)) 
            # ... and they haven't been previously selected...
            # (this looks convoluted but the operation would take literally forever without it)
            & np.outer((elimination_vector_2[:,0].reshape(-1,1)), (elimination_vector_2[:,0])).astype(bool)
            # ... and they aren't the same dislocation (should be taken care of by the previous point)
            # & (np.logical_not(np.diagflat(np.repeat(True, N))))
        ))

        for (i, j) in annihilation_list:
            if elimination_vector_2[i,0] != 0:
                elimination_vector_2[i,:] = [0,t + 1000] 
            if elimination_vector_2[j,0] != 0:
                elimination_vector_2[j,:] = [0,t + 1000]

        # this whole thing can probably be improved by concatenating after np.nonzero and then
        # just elimination_vector[concatenated id'd close dislocations, duplicates are fine] = 0
        # but that'll make recording the time difficult 

        return elimination_vector_2[:,0] * vs(sigma_s_analytic_sum, xs, dislocations, sigma_ext=sigma_ext)

    # keep this function call with the dxsdt func definition and elimination_vector init
    ivp_result_2 = solve_ivp(dxsdt_func_2, (t0, num_tpoints*t0), ivp_result.y[:,-1], 
                           t_eval=np.linspace(t0, num_tpoints * t0, num_tpoints), method="RK45", args=(0.035,))
    return elimination_vector_2, ivp_result_2


@app.cell
def _():
    import os
    return (os,)


@app.cell
def _(dislocations, elimination_vector_2, ivp_result_2, os, pickle):
    filename = "./pickles/1000iter_rk45_LAPTOP_0.035.pkl"
    if not os.path.exists(filename):
        with open(filename, "wb") as file2:
            pickle.dump((dislocations, ivp_result_2, elimination_vector_2), file2)
    return


@app.cell
def _(
    L,
    animation,
    apply_dislocations,
    dislocations,
    elimination_vector_2,
    ivp_result,
    ivp_result_2,
    np,
    partial,
    plt,
):
    anifig_s, aniax_s = plt.subplots(figsize=(9,9))

    X_relax, Y_relax = apply_dislocations(np.column_stack((
            ivp_result.y[:,-1], dislocations[:,1:]
        )))

    aniax_s.scatter(X_relax,Y_relax, label="lattice points (relaxed)")
    aniax_s.scatter(ivp_result.y[:,-1] % L, dislocations[:,1], label="dislocations (relaxed)")

    scat_lattice_s = aniax_s.scatter([], [], c='g', label="lattice points")
    scat_pos_b_s = aniax_s.scatter([], [], c='y', label="dislocations with b = 1")
    scat_neg_b_s = aniax_s.scatter([], [], c='m', label="dislocations with b = -1")

    def animate_2(i, xs_p):
        xs = xs_p[:,i] % L

        lattice_positions = np.column_stack(apply_dislocations(np.column_stack((
            xs, dislocations[:,1:]
        ))))
        scat_lattice_s.set_offsets(lattice_positions)

        xs_pos = xs[(dislocations[:,2] == 1) & ((elimination_vector_2[:,1] > i + 1000) | (elimination_vector_2[:,1] == 0))]
        ys_pos = dislocations[(dislocations[:,2] == 1) & ((elimination_vector_2[:,1] > i + 1000) 
                                                          | (elimination_vector_2[:,1] == 0)), 1]
        dislocations_i_pos = np.column_stack((
            xs_pos,
            ys_pos,
        ))
        scat_pos_b_s.set_offsets(dislocations_i_pos)

        xs_neg = xs[(dislocations[:,2] == -1) & ((elimination_vector_2[:,1] > i + 1000) | (elimination_vector_2[:,1] == 0))]
        ys_neg = dislocations[(dislocations[:,2] == -1) & ((elimination_vector_2[:,1] > i + 1000) 
                                                           | (elimination_vector_2[:,1] == 0)), 1]
        dislocations_i_neg = np.column_stack((
            xs_neg,
            ys_neg
        ))
        scat_neg_b_s.set_offsets(dislocations_i_neg)

        return (scat_lattice_s, scat_pos_b_s, scat_neg_b_s)

    aniax_s.set_title("Randomly-placed dislocation with $sigma = 0.035$\n400 dislocations, annihilation distance within 1 lattice cell ") 
    aniax_s.set_xlabel("x")
    aniax_s.set_ylabel("y")
    aniax_s.legend()

    ani_s = animation.FuncAnimation(anifig_s, 
                                  partial(animate_2, xs_p=ivp_result_2.y), 
                                  frames=1000, 
                                  blit=True)    

    FFwriter_s = animation.FFMpegWriter(fps=10)
    ani_s.save("dude i dont know.mp4", writer = FFwriter_s)
    return


@app.cell
def _(truth_selector):
    truth_selector
    return


@app.cell
def _(
    L,
    dislocations,
    elimination_vector_2,
    ivp_result_2,
    np,
    sigma_s_analytic_sum,
    vs,
):
    strains = np.zeros(1000)
    for i in range(ivp_result_2.t.size):
        truth_selector = ((elimination_vector_2[:,1] > i + 1000) | (elimination_vector_2[:,1] == 0))
        bs = dislocations[truth_selector,2]
        xs = ivp_result_2.y[truth_selector, i] % L 
        strains[i] = bs @ vs(sigma_s_analytic_sum, xs, dislocations[truth_selector,:], sigma_ext=0.035)
    
    return strains, truth_selector


@app.cell
def _(ax, np, plt, strains):
    fig_line, ax_line = plt.subplots()
    ax.scatter(np.log10(range(1000)), np.log10(strains))
    plt.show()
    return


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
    return (ax,)


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
def _():
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
        X = xs - xs.reshape(-1,1)
        Y = ys - ys.reshape(-1,1)

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


@app.cell
def _(os, pickle):
    def write_to_disk(filename, obj):
        if os.path.exists(filename):
            raise ValueError("would overwrite an existing file")
        else:
            with open(filename, "wb") as file:
                pickle.dump(obj, file)
    return


if __name__ == "__main__":
    app.run()

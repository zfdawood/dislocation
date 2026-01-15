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
    import seaborn as sns
    from IPython.display import HTML
    import matplotlib.animation as animation
    from functools import partial
    from scipy.integrate import solve_ivp
    import matplotlib
    import marimo as mo
    import pickle 
    import os 

    pi = np.pi 
    matplotlib.rcParams['animation.embed_limit'] = float('inf')


@app.function
# TODO: BOTTOM HERE
# infinite sum from eqn 1 in the paper
# xs and ys should be row vectors - MAKE SURE THEY'RE WITHIN L
# TODO: make sure eliminated values don't contribute to sigma_s
def sigma_s_analytic_sum(xs, ys, L):
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
        # might need an extra condition like & elim!=0 - may be causing some of the noise
        return -np.divide(num, den, out=np.zeros((xs.size,xs.size)), where=((X!=0) & (Y!=0)))


@app.function
def dxsdt_func(t, xs_p, sigma_ext, elim, lc):
    # solve_ivp results and inputs don't obey periodic conditions 
    xs = xs_p % lc.L
    # print(t)
    # annihilation_list is a list of ordered pairs of annihilated dislocations 
    # we mark a dislocation to be annihilated when... 
    # annihilation_list = np.column_stack(np.nonzero(
    #     # ... two dislocations are within ye of each other... 
    #     # (this particular implementation draws more of a square bounding box 
    #     # than a circle one to check closeness but it's fine)
    #     (np.isclose(xs.reshape(-1,1), xs.reshape(1,-1), rtol=0, atol=lc.ye))
    #     & (np.isclose(lc.ys.reshape(-1,1), lc.ys.reshape(1,-1), rtol=0, atol=lc.ye))
    #     # ... and the dislocations have opposite burgers vectors... 
    #     & (lc.bs.reshape(-1,1) != lc.bs.reshape(1,-1)) 
    #     # ... and they haven't been previously selected...
    #     # (this looks convoluted but the operation would take literally forever without it)
    #     & np.outer((elim.mults.reshape(-1,1)), (elim.mults)).astype(bool)
    #     # ... and they aren't the same dislocation (should be taken care of by the previous point)
    #     # & (np.logical_not(np.diagflat(np.repeat(True, N))))
    # ))

    # annihilation_list is a list of ordered pairs of annihilated dislocations 
    # we mark a dislocation to be annihilated when...
    annihilation_list = np.column_stack(np.nonzero(
        # ...two dislocations are within ye of each other...
        (np.sqrt((xs - xs.reshape(-1,1))**2 + (lc.ys - lc.ys.reshape(-1,1))**2) < lc.ye)
        # ... and the dislocations have opposite burgers vectors... 
        & (lc.bs.reshape(-1,1) != lc.bs.reshape(1,-1)) 
        # ... and they haven't been previously selected...
        # (this looks convoluted but the operation would take literally forever without it)
        & np.outer((elim.mults.reshape(-1,1)), (elim.mults)).astype(bool)
        # ... and they aren't the same dislocation (should be taken care of by the previous point)
        # & (np.logical_not(np.diagflat(np.repeat(True, N))))
    ))

    for (i, j) in annihilation_list: 
        # if (elimination_vector[i,0] != 0) or (elimination_vector[j,0] != 0):
        #     print(f"annihilation event with {i} and {j} at t = {t}, the {t/t0}th step")
        if elim.mults[i] != 0:
            elim.mults[i] = 0 
            elim.elim_times[i] = t 
        if elim.mults[j] != 0:
            elim.mults[j] = 0 
            elim.elim_times[j] = t 

    # this whole thing can probably be improved by concatenating after np.nonzero and then
    # just elimination_vector[concatenated id'd close dislocations, duplicates are fine] = 0
    # but that'll make recording the time difficult 

    return elim.mults * lc.vs(sigma_s_analytic_sum, xs, sigma_ext=sigma_ext)


@app.function
def dxsdt_func_2(t, xs_p, sigma_ext, elim, lc):
    # the positions used by the ivp solver aren't bound to the box so we apply that binding here
    xs = xs_p % lc.L
    # print(t + 1000)

    # # annihilation_list is a list of ordered pairs of annihilated dislocations 
    # # we mark a dislocation to be annihilated when... 
    # annihilation_list = np.column_stack(np.nonzero(
    #     # ... two dislocations are within ye of each other... 
    #     # (this particular implementation draws more of a square bounding box 
    #     # than a circle one to check closeness but it's fine)
    #     (np.isclose(xs.reshape(-1,1), xs.reshape(1,-1), rtol=0, atol=lc.ye))
    #     & (np.isclose(lc.ys.reshape(-1,1), lc.ys.reshape(1,-1), rtol=0, atol=lc.ye))
    #     # ... and the dislocations have opposite burgers vectors... 
    #     & (lc.bs.reshape(-1,1) != lc.bs.reshape(1,-1)) 
    #     # ... and they haven't been previously selected...
    #     # (this looks convoluted but the operation would take literally forever without it)
    #     & np.outer((elim.mults.reshape(-1,1)), (elim.mults)).astype(bool)
    #     # ... and they aren't the same dislocation (should be taken care of by the previous point)
    #     # & (np.logical_not(np.diagflat(np.repeat(True, N))))
    # ))

    # annihilation_list is a list of ordered pairs of annihilated dislocations 
    # we mark a dislocation to be annihilated when...
    annihilation_list = np.column_stack(np.nonzero(
        # ...two dislocations are within ye of each other...
        (np.sqrt((xs - xs.reshape(-1,1))**2 + (lc.ys - lc.ys.reshape(-1,1))**2) < lc.ye)
        # ... and the dislocations have opposite burgers vectors... 
        & (lc.bs.reshape(-1,1) != lc.bs.reshape(1,-1)) 
        # ... and they haven't been previously selected...
        # (this looks convoluted but the operation would take literally forever without it)
        & np.outer((elim.mults.reshape(-1,1)), (elim.mults)).astype(bool)
        # ... and they aren't the same dislocation (should be taken care of by the previous point)
        # & (np.logical_not(np.diagflat(np.repeat(True, N))))
    ))

    for (i, j) in annihilation_list: 
        # if (elimination_vector[i,0] != 0) or (elimination_vector[j,0] != 0):
        #     print(f"annihilation event with {i} and {j} at t = {t}, the {t/t0}th step")
        if elim.mults[i] != 0:
            elim.mults[i] = 0 
            elim.elim_times[i] = t + 1000
        if elim.mults[j] != 0:
            elim.mults[j] = 0 
            elim.elim_times[j] = t + 1000

    # this whole thing can probably be improved by concatenating after np.nonzero and then
    # just elimination_vector[concatenated id'd close dislocations, duplicates are fine] = 0
    # but that'll make recording the time difficult 

    return elim.mults * lc.vs(sigma_s_analytic_sum, xs, sigma_ext=sigma_ext)


@app.class_definition
# class to help remove annihilated dislocations from the system 
class Eliminator:
    def __init__(self, size):
        # turn off annihilated dislocations' contributions to various computations by
        # multiplying them by 0 elementwise (if it does, contribute, multiply this by one)
        self.mults = np.ones(size)

        # time at which the ith dislocation was annihilated. if i == 0 (TODO: nan), not eliminated   
        self.elim_times = np.zeros(size)

        self.size = size 

    def copy(self):
        new_elim = Eliminator(self.size)
        new_elim.mults = self.mults.copy()
        new_elim.elim_times = self.elim_times.copy()
        return new_elim


@app.class_definition
# this class represents a lattice cell with dislocations 
class LatticeCell:

    def __init__(self, N, sigma_ext, ye, L, t0, b, nu=0.3, rng_seed=None):
        self.rng = np.random.default_rng(rng_seed)

        self.x0s = self.rng.random(N) * L 
        self.ys = self.rng.random(N) * L 


        self.b = b 
        self.bs = b * self.rng.permuted(
            np.concatenate((
                np.ones(int(N/2)),
                 -(np.ones(int(N/2)))
            ))
        )

        self.N = N 
        self.nu = nu
        self.sigma_ext = sigma_ext
        self.ye = ye
        self.L = L 
        self.t0 = t0
        self.LATTICE_EDGE_CT = int(L/b)

        self.elim_relax = None
        self.elim_stress = None


    def apply_dislocations(self, dislocations_xs=None, dislocations_ys=None, bs=None):
        if dislocations_xs is None:
            dislocations_xs = self.x0s
        if dislocations_ys is None:
            dislocations_ys = self.ys
        if bs is None:
            bs = self.bs 

        # non-dislocated lattice 
        xs = np.linspace(0, self.L, self.LATTICE_EDGE_CT)
        ys = np.linspace(0, self.L, self.LATTICE_EDGE_CT)
        X_perfect, Y_perfect = np.meshgrid(xs, ys)
        X_flat, Y_flat = X_perfect.flatten(), Y_perfect.flatten()

        # iteratively apply dislocations to lattice points 
        for c_x, c_y, b in np.column_stack((dislocations_xs, dislocations_ys, bs)):
            u_x, u_y = self.calculate_displacements(X_flat, Y_flat, c_x, c_y, b)
            X_flat = (X_flat + u_x) % self.L
            Y_flat = (Y_flat + u_y) % self.L

        self.lattice_points = np.column_stack((X_flat, Y_flat))
        return X_flat, Y_flat


    def calculate_displacements(self, X, Y, disloc_x, disloc_y, b):
        # not a constant because it depends on whether b is pos or negative
        b2pi = b / (2 * pi)

        # existing lattice points
        X_rel = X - disloc_x
        Y_rel = Y - disloc_y
        # in polar 
        r = np.sqrt(X_rel**2 + Y_rel**2)
        theta = np.arctan2(Y_rel, X_rel)

        # horizontal and vertical displacement 
        u_x = b2pi * (theta + (np.sin(2*theta) / (4*(1-self.nu))))
        # the additive constant np.log(r + 1e-19) is to prevent log(0) errors 
        u_y = -b2pi * ( ((1-2*self.nu)/(2*(1-self.nu))) * np.log(r) + 
                        (np.cos(2*theta) / (4*(1-self.nu))) ) 
        return u_x, u_y


    def plot_lattice(self, 
                     lattice_x, 
                     lattice_y, 
                     dislocations_xs, 
                     dislocations_ys=None, 
                     dislocations_bs=None):
        if dislocations_ys is None:
            dislocations_ys = self.ys
        if dislocations_bs is None:
            dislocations_bs = self.bs 

        fig, ax = plt.subplots(figsize=(9, 9))

        ax.scatter(lattice_x, lattice_y)

        ax.scatter(dislocations_xs, dislocations_ys, c=dislocations_bs, label="dislocation centers")

        for i, xy in enumerate(zip(dislocations_xs, dislocations_ys)):
            ax.annotate(i, xy, c='r')

        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return fig


    def vs(self, sigma_fn, xs, ys=None, bs=None, L=None, sigma_ext=0):

        if ys is None:
            ys = self.ys
        if bs is None:
            bs = self.bs 
        if L is None:
            L = self.L

        sigmas = sigma_fn(xs, ys, L)
        assert np.all(np.isclose(0, np.diagonal(sigmas))), ("make sure your sigma_fn returns a "
                                                            "matrix with zeros on the main diagonal")

        return np.sign(bs) * (np.sign(bs) @ sigmas + sigma_ext)


    def relax_ivp(self, num_tpoints=1000):
        if self.elim_relax is not None:
            raise TypeError("cannot relax_ivp more than once")

        self.elim_relax = Eliminator(self.N)
        ivp_result = solve_ivp(dxsdt_func,
                               (self.t0, num_tpoints*self.t0), 
                               self.x0s, 
                               t_eval=np.linspace(self.t0, num_tpoints*self.t0, num_tpoints),
                               method="RK45",
                               args=(0, self.elim_relax, self)
                              )
        self.relax_ivp_result = ivp_result
        return ivp_result

    def stress_ivp(self, num_tpoints=1000):
        if self.elim_relax is None:
            raise TypeError("run relax_ivp first")
        if self.elim_stress is not None:
            raise TypeError("cannot run stress_ivp more than once")

        self.elim_stress = self.elim_relax.copy()
        relax_last_t = self.relax_ivp_result.t[-1]
        ivp_result = solve_ivp(dxsdt_func,
                               (relax_last_t + self.t0, 
                                relax_last_t + num_tpoints*self.t0), 
                               self.relax_ivp_result.y[:,-1], 
                               t_eval=np.linspace(relax_last_t + self.t0, 
                                                  relax_last_t + num_tpoints*self.t0, 
                                                  num_tpoints),
                               method="RK45",
                               args=(self.sigma_ext, self.elim_stress, self)
                              )
        self.stress_ivp_result = ivp_result
        return ivp_result     

    def stress_ivp_2(self, num_tpoints=1000):
        self.elim_stress_2 = self.elim_relax.copy()
        relax_last_t = self.relax_ivp_result.t[-1]
        ivp_result = solve_ivp(dxsdt_func_2, 
                               (self.t0, num_tpoints*self.t0), 
                               self.relax_ivp_result.y[:,-1], 
                               t_eval=np.linspace(self.t0, num_tpoints * self.t0, 
                                                  num_tpoints), 
                               method="RK45", 
                               args=(self.sigma_ext, self.elim_stress_2, self))
        self.stress_ivp_result_2 = ivp_result
        return ivp_result    

    def relax_rate_graph(self, elim, ivp_result):
        time_length, strains = self.relax_rate(elim, ivp_result)        
        
        fig, ax = plt.subplots()
        ax.scatter(np.log10(range(time_length)), np.log10(strains))
        ax.set_xlabel("$log_{10}(t)$")
        ax.set_ylabel("$log_{10}$(strain rate)")
        ax.set_title(rf"strain rate vs time when when relaxing$")
        return fig

    def relax_rate(self, elim, ivp_result):
        time_length = ivp_result.t.size
        strains = np.zeros(time_length)
        for i in range(time_length):
            truth_selector = ((elim.elim_times > i) 
                              | (elim.elim_times == 0))
            bs = self.bs[truth_selector]
            xs = ivp_result.y[truth_selector, i] % self.L

            strains[i] = abs(bs @ self.vs(sigma_s_analytic_sum,
                                     xs,
                                     ys=self.ys[truth_selector], 
                                     bs=bs,
                                     sigma_ext=0))
        return time_length, strains
    
    def strain_rate_graph(self, elim, ivp_result):

        time_length, strains = self.strain_rate(elim, ivp_result)        
        
        fig, ax = plt.subplots()
        ax.scatter(np.log10(range(time_length)), np.log10(strains))
        ax.set_xlabel("$log_{10}(t)$")
        ax.set_ylabel("$log_{10}$(strain rate)")
        ax.set_title(rf"strain rate vs time when $\sigma={self.sigma_ext}$")
        return fig

    def strain_rate(self, elim, ivp_result):
        time_length = ivp_result.t.size
        strains = np.zeros(time_length)
        for i in range(time_length):
            truth_selector = ((elim.elim_times > i + 1000) 
                              | (elim.elim_times == 0))
            bs = self.bs[truth_selector]
            xs = ivp_result.y[truth_selector, i] % self.L

            strains[i] = abs(bs @ self.vs(sigma_s_analytic_sum,
                                     xs,
                                     ys=self.ys[truth_selector], 
                                     bs=bs,
                                     sigma_ext=0.035))
        return time_length, strains


@app.cell
def _():
    thing = LatticeCell(50, 0.035, 1, 100, 1, 1, rng_seed=0)
    X_thing, Y_thing = thing.apply_dislocations()
    mo.mpl.interactive(thing.plot_lattice(X_thing, Y_thing, dislocations_xs=thing.x0s))
    return (thing,)


@app.cell
def _(thing):
    print(thing.vs(sigma_s_analytic_sum, thing.x0s))
    return


@app.cell
def _(thing):
    res = thing.relax_ivp()
    return (res,)


@app.cell
def _(res, thing):
    print(res.y[:,-1])
    print(thing.elim_relax.mults)
    print(thing.elim_relax.elim_times)
    return


@app.cell
def _(res):
    print(res.t[-1])
    return


@app.cell
def _(thing):
    res2 = thing.stress_ivp_2()
    return


@app.cell
def _(thing):
    print(thing.elim_stress_2.elim_times)
    return


@app.cell
def _(thing):
    mo.mpl.interactive(thing.strain_rate_graph(thing.elim_stress_2, thing.stress_ivp_result_2))
    return


@app.cell
def _(animate_relax):
    animate_relax()
    return


@app.cell
def _():
    N = 50
    npoints = 1000
    ensemble_size = 100

    stresses = [0.0025, 0.0075, 0.01, 0.0125, 0.0175, 0.0225, 0.0325, 0.035]
    averaged_strains = np.zeros((len(stresses), npoints))


    for stress_i, stress in enumerate(stresses):
        strain_storage = np.zeros((ensemble_size, npoints))
        for i in range(0,ensemble_size):
            lc = LatticeCell(N, stress, 1, 100, 1, 1, rng_seed=i)
            lc.relax_ivp()
            lc.stress_ivp_2()
            _, strains = lc.strain_rate(elim=lc.elim_stress_2, ivp_result=lc.stress_ivp_result_2)
            strain_storage[i,:] = strains.copy()
        averaged_strains[stress_i, :] = np.mean(strain_storage, axis=0)

    return averaged_strains, npoints, strains, stresses


@app.cell
def _(averaged_strains, npoints, stresses):
    fig, ax = plt.subplots(figsize=(9,9))
    for stress_j, stressj in enumerate(stresses):
        ax.scatter(np.log10(range(npoints)), np.log10(averaged_strains[stress_j, :]), label=rf"$\sigma = {stressj}$")
    ax.legend()
    ax.set_xlabel("$log_{10}(t)$")
    ax.set_ylabel("$log_{10}$(strain rate)")
    ax.set_title(rf"strain rate vs time")
    mo.mpl.interactive(fig)
    return (ax,)


@app.cell
def _():
    tmp_strain = np.zeros((100,1000))
    for k in range(100):
        lck = LatticeCell(50, 0.035, 1, 100, 1, 1, rng_seed=k)
        lck.relax_ivp()
        lck.stress_ivp_2()
        _, strainsk = lck.strain_rate(elim=lck.elim_stress_2, ivp_result=lck.stress_ivp_result_2)
        tmp_strain[k,:] = strainsk.copy()
    print(tmp_strain)
    return


@app.function
def write_to_disk(filename, obj):
    if os.path.exists(filename):
        raise ValueError("would overwrite an existing file")
    else:
        with open(filename, "wb") as file:
            pickle.dump(obj, file)


@app.cell
def _(X, Y, dislocations):
    def animate_relax(lc, ivp_result, filename):
        anifig, aniax = plt.subplots(figsize=(9,9))
    
        aniax.scatter(X,Y, label="lattice points (base)")
        aniax.scatter(dislocations[:,0], dislocations[:,1], label="dislocations (base)")
    
        scat_lattice = aniax.scatter([], [], c='g', label="lattice points")
        scat_pos_b = aniax.scatter([], [], c='y', label="dislocations with b = 1")
        scat_neg_b = aniax.scatter([], [], c='m', label="dislocations with b = -1")
    
        def animate(i, xs_p):
            xs = xs_p[:,i] % lc.L
    
            lattice_positions = np.column_stack(lc.apply_dislocations(lc.xs))
            scat_lattice.set_offsets(lattice_positions)
    
            xs_pos = xs[(lc.bs == 1) & ((lc.elim_relax.elim_times > i) | (lc.elim_relax.elim_times == 0))]
            ys_pos = lc.ys[(lc.bs == 1) & ((lc.elim_relax.elim_times > i) 
                                                              | (lc.elim_relax_elim_times == 0))]
            dislocations_i_pos = np.column_stack((
                xs_pos,
                ys_pos,
            ))
            scat_pos_b.set_offsets(dislocations_i_pos)

            xs_neg = xs[(lc.bs == -1) & ((lc.elim_relax.elim_times > i) | (lc.elim_relax.elim_times == 0))]
            ys_neg = lc.ys[(lc.bs == -1) & ((lc.elim_relax.elim_times > i) 
                                                              | (lc.elim_relax_elim_times == 0))]

            dislocations_i_neg = np.column_stack((
                xs_neg,
                ys_neg
            ))
            scat_neg_b.set_offsets(dislocations_i_neg)
    
            return (scat_lattice, scat_pos_b, scat_neg_b)
    
        aniax.set_title(f"Randomly-placed dislocation relaxation\n{lc.N} dislocations, annihilation distance within {lc.ye} lattice cell(s)") 
        aniax.set_xlabel("x")
        aniax.set_ylabel("y")
        aniax.legend()
    
        ani = animation.FuncAnimation(anifig, 
                                      partial(animate, xs_p=ivp_result.y), 
                                      frames=1000, 
                                      blit=True)    
    
        # HTML(ani.to_jshtml())
    
        FFwriter = animation.FFMpegWriter(fps=10)
        ani.save(filename, writer=FFwriter)
    return (animate_relax,)


@app.cell
def _(X, Y, dislocations):
    def animate_stress(lc, ivp_result, filename):
        anifig, aniax = plt.subplots(figsize=(9,9))
    
        aniax.scatter(X,Y, label="lattice points (base)")
        aniax.scatter(dislocations[:,0], dislocations[:,1], label="dislocations (base)")
    
        scat_lattice = aniax.scatter([], [], c='g', label="lattice points")
        scat_pos_b = aniax.scatter([], [], c='y', label="dislocations with b = 1")
        scat_neg_b = aniax.scatter([], [], c='m', label="dislocations with b = -1")
    
        def animate(i, xs_p):
            xs = xs_p[:,i] % lc.L
    
            lattice_positions = np.column_stack(lc.apply_dislocations(lc.xs))
            scat_lattice.set_offsets(lattice_positions)
    
            xs_pos = xs[(lc.bs == 1) & ((lc.elim_relax.elim_times > i + 1000) | (lc.elim_relax.elim_times == 0))]
            ys_pos = lc.ys[(lc.bs == 1) & ((lc.elim_relax.elim_times > i + 1000) 
                                                              | (lc.elim_relax_elim_times == 0))]
            dislocations_i_pos = np.column_stack((
                xs_pos,
                ys_pos,
            ))
            scat_pos_b.set_offsets(dislocations_i_pos)

            xs_neg = xs[(lc.bs == -1) & ((lc.elim_relax.elim_times > i + 1000) | (lc.elim_relax.elim_times == 0))]
            ys_neg = lc.ys[(lc.bs == -1) & ((lc.elim_relax.elim_times > i + 1000) 
                                                              | (lc.elim_relax_elim_times == 0))]

            dislocations_i_neg = np.column_stack((
                xs_neg,
                ys_neg
            ))
            scat_neg_b.set_offsets(dislocations_i_neg)
    
            return (scat_lattice, scat_pos_b, scat_neg_b)
    
        aniax.set_title(f"Randomly-placed dislocation relaxation\n{lc.N} dislocations, annihilation distance within {lc.ye} lattice cell(s)") 
        aniax.set_xlabel("x")
        aniax.set_ylabel("y")
        aniax.legend()
    
        ani = animation.FuncAnimation(anifig, 
                                      partial(animate, xs_p=ivp_result.y), 
                                      frames=1000, 
                                      blit=True)    
    
        # HTML(ani.to_jshtml())
    
        FFwriter = animation.FFMpegWriter(fps=10)
        ani.save(filename, writer=FFwriter)
    return


@app.cell
def _(ax, strains):
    fig_line, ax_line = plt.subplots()
    ax.scatter(np.log10(range(1000)), np.log10(strains))
    plt.show()
    return


@app.cell
def _(L, LATTICE_EDGE_CT, calculate_displacements):
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


if __name__ == "__main__":
    app.run()

"""Reproduces the two numerical bugs in dimensionless.py by calling its own functions.

Usage: python replication/check_original_bugs.py   (from the repo root or anywhere)
marimo, seaborn and IPython are stubbed out so the notebook's functions import as plain Python.
"""
import contextlib, os, sys, types
import numpy as np
import matplotlib; matplotlib.use('Agg')


class _App:
    def __init__(self, *a, **k): pass
    setup = property(lambda self: contextlib.nullcontext())
    def function(self, f): return f
    def class_definition(self, c): return c
    def cell(self, f): return f
    def run(self): pass


_mo = types.ModuleType('marimo'); _mo.App = _App
_mo.mpl = types.SimpleNamespace(interactive=lambda f: f)
_ipd = types.ModuleType('IPython.display'); _ipd.HTML = None
sys.modules.update({'marimo': _mo, 'seaborn': types.ModuleType('seaborn'),
                    'IPython': types.ModuleType('IPython'), 'IPython.display': _ipd})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dimensionless as d

# Bug 1: dead dislocations still exert stress
xs = np.array([10., 13., 50.]); ys = np.array([5., 5., 40.])
alive = np.array([0., 1., 1.])  # dislocation 0 has been annihilated
S = d.sigma_s_analytic_sum(xs, ys, 100, alive)
print("bug1 stress matrix (row=source, col=target), dislocation 0 dead:\n", S)
print("  stress FROM dead #0 onto #1:", S[0, 1], "(should be 0)")

# Bug 2: strain_rate hardcodes sigma_ext=0.03
lc = d.LatticeCell(20, 0.0025, 1, 100, 1, 1, rng_seed=0)
xs = lc.x0s; alive = np.ones(20)
inside = lc.bs @ lc.vs(d.sigma_s_analytic_sum, xs, sigma_ext=0, elim_mult=alive)
used = lc.bs @ lc.vs(d.sigma_s_analytic_sum, xs, sigma_ext=0.03, elim_mult=alive)
true = lc.bs @ lc.vs(d.sigma_s_analytic_sum, xs, sigma_ext=0.0025, elim_mult=alive)
print(f"bug2 sum b*v at sigma=0.0025: true={true:.4f}  what strain_rate reports={used:.4f}"
      f"  (internal part alone={inside:.4f}; offset = (0.03 - sigma)*N = {(0.03 - 0.0025) * 20:.2f})")

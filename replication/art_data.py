"""Usage: python art_data.py N_SEEDS OUT.pkl

One strip-shaped realization (period 100 in x, height 25) recorded densely, for banner art."""
import os, sys, pickle
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Cell, run

LX, H, N0, SIGMA = 100.0, 25.0, 100, 0.0125
out = {}
for seed in range(int(sys.argv[1])):
    rng = np.random.default_rng(1000 + seed)
    bs = rng.permutation(np.r_[np.ones(N0 // 2), -np.ones(N0 // 2)])
    cell = Cell(rng.random(N0) * LX, rng.random(N0) * H, bs, LX)
    _, _, relax_snaps = run(cell, 0.0, 1000.0, np.logspace(-2, 3, 400), record_x=True)
    _, _, stress_snaps = run(cell, SIGMA, 5000.0, np.logspace(-1, np.log10(5000), 400), record_x=True)
    out[seed] = dict(relax=relax_snaps, stress=stress_snaps, LX=LX, H=H, sigma=SIGMA, n0=N0)
    print('seed', seed, 'left after relax', len(relax_snaps[-1][1]), 'after stress', len(stress_snaps[-1][1]), flush=True)
pickle.dump(out, open(sys.argv[2], 'wb'))

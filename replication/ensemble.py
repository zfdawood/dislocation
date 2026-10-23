"""Usage: python ensemble.py N_SEEDS N_PROCS OUT.pkl   (or: python ensemble.py probe)

Ensemble replication of Fig. 1: strain rate vs time for several applied stresses."""
import os, sys, time, copy, pickle
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import random_cell, run

L, N0, T_RELAX, T_END = 50, 100, 1000.0, 1e4
STRESSES = [0.0025, 0.0075, 0.01, 0.0125, 0.0175, 0.0225, 0.0325]
T_REC = np.logspace(0, 4, 49)


def one_seed(seed):
    t0 = time.time()
    cell = random_cell(N0, L, np.random.default_rng(seed))
    run(cell, 0.0, T_RELAX)
    n_relaxed = cell.n
    out = {}
    for s in STRESSES:
        rate, nal, _ = run(copy.deepcopy(cell), s, T_END, T_REC)
        out[s] = (rate, nal)
    print(f'seed {seed} done in {time.time()-t0:.0f}s, N after relax {n_relaxed}', flush=True)
    return seed, n_relaxed, out


if __name__ == '__main__':
    if sys.argv[1] == 'probe':
        t0 = time.time(); cell = random_cell(N0, L, np.random.default_rng(0)); run(cell, 0.0, T_RELAX)
        print(f'relax {time.time()-t0:.0f}s N={cell.n}', flush=True)
        t0 = time.time(); r, n, _ = run(cell, 0.0125, T_END, T_REC)
        print(f'stress run {time.time()-t0:.0f}s', flush=True)
        print(np.c_[T_REC, r, n][::6])
    else:
        from multiprocessing import Pool
        seeds, procs, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        with Pool(procs) as pool:
            res = pool.map(one_seed, range(seeds), chunksize=1)
        with open(path, 'wb') as f:
            pickle.dump(dict(L=L, N0=N0, T_RELAX=T_RELAX, T_REC=T_REC, STRESSES=STRESSES, res=res), f)
        print('saved', path, flush=True)

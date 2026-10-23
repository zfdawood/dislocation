"""Usage: python fig1.py ens.pkl OUT.png  -- Fig. 1 of the paper from ensemble.py output."""
import sys, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = pickle.load(open(sys.argv[1], 'rb')); T = d['T_REC']; res = d['res']
SHOW = [0.0025, 0.0075, 0.01, 0.0175, 0.0325]
COLS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
SURF, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#c9c7c0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, facecolor=SURF)
for ax, buggy in zip(axes, [False, True]):
    ax.set_facecolor(SURF)
    labels = []
    for s, c in zip(SHOW, COLS):
        r = np.array([x[2][s][0] for x in res]); n = np.array([x[2][s][1] for x in res])
        if buggy:  # what dimensionless.py's strain_rate reports: sigma_ext hardcoded to 0.03
            r = r - s * n + 0.03 * n
        m = np.nanmean(np.abs(r) if buggy else r, axis=0)
        ok = m > 0
        ax.plot(np.log10(T[ok]), np.log10(m[ok]), color=c, lw=2)
        labels.append([np.log10(m[ok][-1]), np.log10(T[ok][-1]) + 0.06, f'σ = {s}'])
    labels.sort()
    for i in range(1, len(labels)):  # push labels apart so none overlap
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 0.2)
    for y, x, txt in labels:
        ax.text(x, y, txt, color=INK2, va='center', fontsize=9.5)
    if not buggy:
        x = np.array([0.5, 2.5]); ax.plot(x, -2 / 3 * (x - 0.5) - 0.2, color=INK, lw=1, ls=(0, (4, 3)))
        ax.text(1.55, -0.75, 't$^{-2/3}$ (Andrade)', color=INK, fontsize=10, rotation=-21)
    ax.set_xlim(0, 4.75); ax.set_xlabel('log$_{10}$ t')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.grid(color='#ecebe7', lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel('log$_{10}$ ⟨Σ b·v⟩')
axes[0].set_title('Corrected: power-law decay, then flow (high σ) or jam (low σ)', loc='left', fontsize=11.5, color=INK)
axes[1].set_title('Same runs through the hardcoded σ = 0.03 in strain_rate', loc='left', fontsize=11.5, color=INK)
fig.suptitle('Andrade creep replication, 192 realizations per stress (L = 50, N₀ = 100)', x=0.01, ha='left', color=INK, fontsize=13)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130, facecolor=SURF)

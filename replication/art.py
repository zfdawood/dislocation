"""Usage: python art.py art.pkl OUT_DIR  -- renders 1584x396 banner images from art_data.py output."""
import os, sys, pickle
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
OUT = sys.argv[2]
art = pickle.load(open(sys.argv[1], 'rb'))
W, H_PX, DPI = 1584, 396, 100
PI = np.pi

def canvas(bg):
    fig = plt.figure(figsize=(W / DPI * 2, H_PX / DPI * 2), dpi=DPI)  # rendered at 2x for crispness
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    return fig, ax

def field(xs, ys, bs, LX, H, nx=1584, ny=396):
    """Total glide shear stress sigma_s on a grid, periodic in x (the paper's image sum)."""
    gx = np.linspace(0, LX, nx, endpoint=False); gy = np.linspace(0, H, ny)
    GX, GY = np.meshgrid(gx, gy)
    tot = np.zeros_like(GX)
    for x0, y0, b in zip(xs, ys, bs):
        X, Y = GX - x0, GY - y0
        xa, ya = 2 * PI * X / LX, 2 * PI * Y / LX
        num = PI * np.sin(xa) * (LX * np.cos(xa) + 2 * PI * Y * np.sinh(ya) - LX * np.cosh(ya))
        den = LX ** 2 * (np.cos(xa) - np.cosh(ya)) ** 2
        with np.errstate(all='ignore'):
            tot += b * np.where(den > 1e-12, -num / den, 0)
    return tot

# --- A: stress field of the relaxed (jammed) arrangement ---------------------------------
palettes = {
    'ink':    ('#0e0f13', ['#2b6cb0', '#0e0f13', '#d0643b']),
    'plum':   ('#17111c', ['#5b8def', '#17111c', '#e86aa6']),
    'paper':  ('#f4f1ea', ['#2f5d8a', '#f4f1ea', '#b5523b']),
    'mono':   ('#0d0d0d', ['#f2efe6', '#0d0d0d', '#f2efe6']),
}
for seed in [0, 2, 5]:
    d = art[seed]; t, x, y, b, ids = d['stress'][-1]
    F = field(x % d['LX'], y, b, d['LX'], d['H'])
    lim = np.percentile(np.abs(F), 97)
    for name, (bg, cols) in palettes.items():
        fig, ax = canvas(bg)
        cmap = LinearSegmentedColormap.from_list(name, cols)
        Fs = np.sign(F) * np.abs(F) ** 0.5  # compress so the far field shows, not just the cores
        l2 = lim ** 0.5
        ax.imshow(Fs, origin='lower', cmap=cmap, norm=TwoSlopeNorm(0, -l2, l2), aspect='auto', interpolation='bilinear')
        fig.savefig(f'{OUT}/A_field_s{seed}_{name}.png', facecolor=bg); plt.close(fig)
        if name in ('ink', 'paper'):
            fig, ax = canvas(bg)
            gx = np.linspace(0, d['LX'], 1584, endpoint=False); gy = np.linspace(0, d['H'], 396)
            ax.contour(gx, gy, Fs, levels=np.linspace(-l2, l2, 24), cmap=cmap, linewidths=1.2)
            ax.set_xlim(0, d['LX']); ax.set_ylim(0, d['H'])
            fig.savefig(f'{OUT}/A_contour_s{seed}_{name}.png', facecolor=bg); plt.close(fig)

# --- B: worldlines, x(t) on a log-time axis, through relaxation then creep ----------------
def worldlines(d, bg, cpos, cneg, fname, phase='both', lw=1.3, alpha=0.75):
    snaps = (d['relax'] if phase in ('both', 'relax') else []) + (
        [(s[0] + 1000, *s[1:]) for s in d['stress']] if phase in ('both', 'stress') else [])
    tracks = {}
    for t, x, y, b, ids in snaps:
        for xi, yi, bi, i in zip(x, y, b, ids):
            tracks.setdefault(i, []).append((t, xi, bi))
    fig, ax = canvas(bg)
    for i, tr in tracks.items():
        tr = np.array(tr)
        col = cpos if tr[0, 2] > 0 else cneg
        xs = tr[:, 1]
        # draw every periodic copy that can enter the window so threads wrap seamlessly
        for k in range(int(np.floor(xs.min() / d['LX'])) - 1, int(np.ceil(xs.max() / d['LX'])) + 1):
            ax.plot(np.log10(tr[:, 0]), xs - k * d['LX'], color=col, lw=lw, alpha=alpha, solid_capstyle='round')
    tmin, tmax = np.log10(snaps[0][0]), np.log10(snaps[-1][0])
    ax.set_xlim(tmin, tmax); ax.set_ylim(0, d['LX'])
    fig.savefig(fname, facecolor=bg); plt.close(fig)

for seed in [0, 1, 5]:
    d = art[seed]
    worldlines(d, '#0e0f13', '#e9c46a', '#5fa8d3', f'{OUT}/B_world_s{seed}_ink.png')
    worldlines(d, '#f4f1ea', '#b5523b', '#2f5d8a', f'{OUT}/B_world_s{seed}_paper.png', alpha=0.8)
    worldlines(d, '#17111c', '#e86aa6', '#7aa2f7', f'{OUT}/B_world_s{seed}_plum.png')
    worldlines(d, '#0e0f13', '#e9c46a', '#5fa8d3', f'{OUT}/B_world_s{seed}_relaxonly_ink.png', phase='relax')
print('ok')

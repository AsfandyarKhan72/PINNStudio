#!/usr/bin/env python3
"""burgers_comparison.png: u and v of 2D Burgers (Mathias) at t = 0.25 and 1, the regenerated reference (Chebyshev,
N = 512), the COMSOL file, and their difference.   python benchmarks/reference_regeneration/make_figure.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(os.path.dirname(os.path.dirname(HERE)), "reference_data", "2D", "burgers_mathias")
new = np.load(os.path.join(HERE, "convergence", "burgers_N512_dt2.5e-06.npz"))
rows = [("u", 0.25, "0.25"), ("v", 0.25, "0.25"), ("u", 1.0, "1.0"), ("v", 1.0, "1.0")]
fig, ax = plt.subplots(len(rows), 3, figsize=(10, 3.0 * len(rows)), constrained_layout=True)
ext = (0, 1, 0, 1)
for i, (c, t, name) in enumerate(rows):
    N = new[f"{c}_{t}"]
    C = np.loadtxt(os.path.join(REF, f"t_{name}_{c}.txt"))[:, 3].reshape(101, 101)
    lim = np.abs(N).max()
    for j, (A, title, cmap, vmax) in enumerate([(N, f"{c}, t = {t}: regenerated (Chebyshev)", "RdBu_r", lim),
                                                (C, f"{c}, t = {t}: COMSOL", "RdBu_r", lim),
                                                (C - N, "COMSOL − regenerated", "PuOr_r", np.abs(C - N).max())]):
        im = ax[i, j].imshow(A, origin="lower", extent=ext, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax[i, j].set_title(title, fontsize=9)
        ax[i, j].set_xlabel("x", fontsize=8)
        ax[i, j].set_ylabel("y", fontsize=8)
        fig.colorbar(im, ax=ax[i, j], shrink=0.85)
fig.savefig(os.path.join(HERE, "burgers_comparison.png"), dpi=110)
print("wrote", os.path.join(HERE, "burgers_comparison.png"))

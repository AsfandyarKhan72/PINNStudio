"""3D view of the Allen-Cahn 3D (Wight & Zhao) result, in the style of their Fig. 3.9.

Top row: the interface phi = 0 of the PINN solution (the drop) inside the cube
at each reference time. Bottom row: the error phi_ref - phi_PINN on a cut-away
cube (one corner removed, so the cuts through the centre are visible).

    python tools/plot_3d_allen_cahn_isosurface.py --model path/to/model_lbfgs-phase2-*.pt \
        --out assets/Images/3D_AllenCahn-Wight/isosurface.png

The model is the one PINNStudio saves in solution_results/ when the template
is run in Forward mode (network [7, 128, 128, 128, 128, 1] on the periodic
features cos/sin(2*pi*x), cos/sin(2*pi*y), cos/sin(2*pi*z), t).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import find_contours, marching_cubes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference_data", "3D", "allen_cahn_wight")
TIMES = ["0", "0.25", "0.5", "0.75", "1"]

# formatting
FIG_WIDTH = 13.0
FIG_HEIGHT = 5.6
VIEW = dict(elev=22, azim=-58)
DROP_COLOR = "#d62728"
EDGE_COLOR = "0.45"
CMAP = "RdBu_r"
FONT = 10


def load_net(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    n = len([k for k in sd if k.endswith(".weight")])
    layers = [(sd[f"linears.{i}.weight"].double(), sd[f"linears.{i}.bias"].double()) for i in range(n)]

    def net(X):
        x, y, z, t = (torch.tensor(X[:, i]) for i in range(4))
        p = 2 * np.pi
        h = torch.stack([torch.cos(p * x), torch.sin(p * x), torch.cos(p * y), torch.sin(p * y),
                         torch.cos(p * z), torch.sin(p * z), t], 1)
        for i, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if i < n - 1:
                h = torch.tanh(h)
        with torch.no_grad():
            return h[:, 0].numpy()
    return net


def load_ref(t):
    """Reference on its grid, as arrays indexed [ix, iy, iz] (x varies fastest in the file)."""
    d = np.loadtxt(os.path.join(REF, f"t_{t}.txt"))
    g = np.unique(d[:, 0])
    n = len(g)
    return g, d[:, :4], d[:, 4].reshape(n, n, n).transpose(2, 1, 0)


def cube_edges(ax, zorder=2):
    for s in (0, 1):
        for r in (0, 1):
            ax.plot([0, 1], [s, s], [r, r], color=EDGE_COLOR, lw=0.6, zorder=zorder)
            ax.plot([s, s], [0, 1], [r, r], color=EDGE_COLOR, lw=0.6, zorder=zorder)
            ax.plot([s, s], [r, r], [0, 1], color=EDGE_COLOR, lw=0.6, zorder=zorder)


def cut_edges(ax):
    """Outline of the removed corner x > 0.5, y < 0.5, z > 0.5."""
    h = 0.5
    for p, q in [((h, 0, h), (1, 0, h)), ((h, 0, h), (h, h, h)), ((h, 0, h), (h, 0, 1)),
                 ((1, 0, h), (1, h, h)), ((h, h, h), (1, h, h)), ((h, h, h), (h, h, 1)),
                 ((h, 0, 1), (h, h, 1)), ((1, h, h), (1, h, 1)), ((h, h, 1), (1, h, 1))]:
        ax.plot(*zip(p, q), color="0.25", lw=0.8, zorder=3)


def ref_interface_on_cuts(ax, g, ref):
    """The reference interface phi = 0 where it crosses the three cut faces (black lines)."""
    n = len(g); m = n // 2; d = g[1] - g[0]
    for c in find_contours(ref[m, :m + 1, m:], 0.0):        # x = .5, y < .5, z > .5
        ax.plot(np.full(len(c), 0.5), c[:, 0] * d, 0.5 + c[:, 1] * d, color="k", lw=1.0, zorder=4)
    for c in find_contours(ref[m:, m, m:], 0.0):            # y = .5, x > .5, z > .5
        ax.plot(0.5 + c[:, 0] * d, np.full(len(c), 0.5), 0.5 + c[:, 1] * d, color="k", lw=1.0, zorder=4)
    for c in find_contours(ref[m:, :m + 1, m], 0.0):        # z = .5, x > .5, y < .5
        ax.plot(0.5 + c[:, 0] * d, c[:, 1] * d, np.full(len(c), 0.5), color="k", lw=1.0, zorder=4)


def style(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(**VIEW)
    ax.set_axis_off()


def cutaway_faces(g, E, norm, cmap):
    """Quads of the cube's visible faces with the corner x > 0.5, y < 0.5, z > 0.5 removed."""
    n = len(g); m = n // 2  # g[m] = 0.5
    polys, cols = [], []

    def add(plane, fixed, ia, ib):
        # plane: 'x', 'y' or 'z' held at index `fixed`; quads over index ranges ia, ib
        for a in range(ia[0], ia[1]):
            for b in range(ib[0], ib[1]):
                if plane == "x":
                    q = [(g[fixed], g[a], g[b]), (g[fixed], g[a + 1], g[b]), (g[fixed], g[a + 1], g[b + 1]), (g[fixed], g[a], g[b + 1])]
                    v = E[fixed, a:a + 2, b:b + 2].mean()
                elif plane == "y":
                    q = [(g[a], g[fixed], g[b]), (g[a + 1], g[fixed], g[b]), (g[a + 1], g[fixed], g[b + 1]), (g[a], g[fixed], g[b + 1])]
                    v = E[a:a + 2, fixed, b:b + 2].mean()
                else:
                    q = [(g[a], g[b], g[fixed]), (g[a + 1], g[b], g[fixed]), (g[a + 1], g[b + 1], g[fixed]), (g[a], g[b + 1], g[fixed])]
                    v = E[a:a + 2, b:b + 2, fixed].mean()
                polys.append(q); cols.append(cmap(norm(v)))

    last = n - 1
    # outer faces the viewer sees (x = 1, y = 0, z = 1), each without the removed quarter
    add("x", last, (0, last), (0, m)); add("x", last, (m, last), (m, last))          # x = 1: keep z < .5, and y > .5
    add("y", 0, (0, m), (0, last)); add("y", 0, (m, last), (0, m))                  # y = 0: keep x < .5, and z < .5
    add("z", last, (0, m), (0, last)); add("z", last, (m, last), (m, last))          # z = 1: keep x < .5, and y > .5
    # the three cuts left by the removed corner
    add("x", m, (0, m), (m, last))     # x = .5, y < .5, z > .5
    add("y", m, (m, last), (m, last))  # y = .5, x > .5, z > .5
    add("z", m, (m, last), (0, m))     # z = .5, x > .5, y < .5
    return polys, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "Images", "3D_AllenCahn-Wight", "isosurface.png"))
    a = ap.parse_args()
    net = load_net(a.model)
    plt.rcParams.update({"font.size": FONT})
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    cmap = plt.get_cmap(CMAP)
    for j, t in enumerate(TIMES):
        g, X, ref = load_ref(t)
        n = len(g)
        pred = net(X).reshape(n, n, n).transpose(2, 1, 0)
        err = ref - pred
        l2 = np.linalg.norm(err) / np.linalg.norm(ref)

        ax = fig.add_subplot(2, len(TIMES), j + 1, projection="3d")
        verts, faces, _, _ = marching_cubes(pred, 0.0, spacing=(g[1] - g[0],) * 3)
        mesh = Poly3DCollection(verts[faces], facecolors=DROP_COLOR, edgecolors=DROP_COLOR, linewidths=0, shade=True,
                                lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45))
        ax.add_collection3d(mesh)
        cube_edges(ax); style(ax)
        ax.set_title(f"t = {t}", pad=0)

        ax = fig.add_subplot(2, len(TIMES), len(TIMES) + j + 1, projection="3d")
        lim = np.abs(err).max()
        norm = matplotlib.colors.Normalize(-lim, lim)
        polys, cols = cutaway_faces(g, err, norm, cmap)
        ax.computed_zorder = False
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.3, zorder=1))
        cut_edges(ax)
        ref_interface_on_cuts(ax, g, ref)
        cube_edges(ax, zorder=0); style(ax)
        ax.set_title(f"$L_2$ = {l2:.1e}", pad=0, fontsize=FONT - 1)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.6, pad=0.0, fraction=0.05, aspect=25)
        cb.set_ticks([-lim, 0, lim]); cb.set_ticklabels([f"{-lim:.1e}", "0", f"{lim:.1e}"])
        cb.ax.tick_params(labelsize=FONT - 2)

    fig.text(0.5, 0.515, r"(a) PINN interface $\phi = 0$ in the domain $[0,1]^3$", ha="center")
    fig.text(0.5, -0.03, r"(b) Error $\phi_{\mathrm{ref}} - \phi_{\mathrm{PINN}}$ on the cube with one corner cut away (black: reference interface)", ha="center")
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, wspace=0.0, hspace=0.3)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

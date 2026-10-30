"""The paper's two 3 x 3 example figures (benchmarks/figures/paper_forward.png, paper_inverse.png).

Forward: Wave 1D (multi-scale Fourier features), Allen-Cahn 2D (Mattey & Ghosh), Navier-Stokes 3D (Beltrami
flow); columns PINN | reference | |error|. Inverse: Burgers 1D (Raissi et al.), linear elasticity 2D (plate),
Allen-Cahn 3D (Wight & Zhao); columns: the field reconstructed by the trained inverse PINN | its absolute error
against the reference | the unknown parameters' relative error (%) during training. The field and the parameters
come from the same inverse run; no separate forward model is involved.
Each row is PINNStudio's best run of that problem (see benchmarks/README.md).

Two steps:

    # 1. from the saved runs (solution_results/model_config.json + model_lbfgs-*.pt, and the inverse runs'
    #    *_convergence.txt): restore each network with PINNStudio's own restore code and save the arrays
    python tools/make_paper_examples.py export RUNS_DIR      # RUNS_DIR/<key>/{model_config.json, model.pt, ...}

    # 2. the figures, from benchmarks/paper_examples/*.npz (committed, so no model is needed)
    python tools/make_paper_examples.py plot
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "benchmarks", "paper_examples")
FIGS = os.path.join(ROOT, "benchmarks", "figures")
REF = os.path.join(ROOT, "reference_data")

FORWARD = ["wave", "ac2d", "beltrami"]
INVERSE = ["burgers_inv", "plate_inv", "ac3d_inv"]


# ---- 1. export -------------------------------------------------------------------------------------------

def _restored_model(run):
    """PINNStudio's own Restore script for this run, up to and including the restore of the weights;
    returns its namespace (with `model`)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from pinnstudio.ui.main_window import MainWindow
    cfg = json.load(open(os.path.join(run, "model_config.json")))
    win = MainWindow()
    script = win._build_restore_script(os.path.join(run, "model.pt"), cfg, "lbfgs", "Surface", 0, 5,
                                       os.path.join(run, "_restore"))
    win.close()
    cut = script.index("Model restored successfully")
    cut = script.index("\nos.makedirs(", cut)          # the plotting that follows is not needed
    ns = {"__name__": "__restore__"}
    exec(compile(script[:cut], "restore", "exec"), ns)
    return ns["model"], cfg


def _grid(a, b, n):
    return np.linspace(a, b, n)


def _cole_hopf(x, t, nu=0.01 / np.pi, n=200):
    """Burgers u_t + u u_x = nu u_xx, u(x, 0) = -sin(pi x), u(+-1, t) = 0: the exact (Cole-Hopf) solution
    by Gauss-Hermite quadrature."""
    z, w = np.polynomial.hermite.hermgauss(n)
    u = np.zeros((len(t), len(x)))
    for i, tv in enumerate(t):
        if tv == 0:
            u[i] = -np.sin(np.pi * x)
            continue
        eta = np.sqrt(4 * nu * tv) * z[None, :]
        y = x[:, None] - eta
        e = -np.cos(np.pi * y) / (2 * np.pi * nu)
        e -= e.max(axis=1, keepdims=True)
        f = np.exp(e) * w[None, :]
        u[i] = -(np.sin(np.pi * y) * f).sum(1) / f.sum(1)
    return u


def _conv(run):
    """{name: (iterations, values)} from the inverse run's *_convergence.txt files."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run, "*_convergence.txt"))):
        d = np.loadtxt(p, delimiter=",", skiprows=1)
        name = open(p).readline().strip().split(",")[1]
        out[name] = (d[:, 0], d[:, 1])
    return out


def export(runs, only=None):
    os.makedirs(DATA, exist_ok=True)
    for key in FORWARD + INVERSE:
        if only and key not in only:
            continue
        run = os.path.join(runs, key)
        if not os.path.isdir(run):
            print("missing", run)
            continue
        model, cfg = _restored_model(run)
        pred = lambda X: np.asarray(model.predict(X))
        meta = {}
        if key == "wave":                       # u(x, t) on the whole space-time domain
            x, t = _grid(0, 1, 201), _grid(0, 1, 201)
            X, T = np.meshgrid(x, t)
            P = pred(np.column_stack([X.ravel(), T.ravel()]))[:, 0].reshape(T.shape)
            R = np.sin(np.pi * X) * np.cos(10 * np.pi * T) + np.sin(2 * np.pi * X) * np.cos(20 * np.pi * T)
            meta = dict(xlabel="$x$", ylabel="$t$", ext=[0, 1, 0, 1], field="u(x, t)")
        elif key == "ac2d":                     # u(x, y) at the last reference time, t = 1
            d = np.loadtxt(os.path.join(REF, "2D", "allen_cahn_mattey", "t_1.0.txt"))
            xs, ys = np.unique(d[:, 0]), np.unique(d[:, 1])
            R = np.full((len(ys), len(xs)), np.nan)
            R[np.searchsorted(ys, d[:, 1]), np.searchsorted(xs, d[:, 0])] = d[:, 3]
            X, Y = np.meshgrid(xs, ys)
            P = pred(np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)]))[:, 0].reshape(X.shape)
            # the error over all reference times, for the label
            num = den = 0.0
            for f in sorted(glob.glob(os.path.join(REF, "2D", "allen_cahn_mattey", "t_*.txt"))):
                q = np.loadtxt(f)
                p = pred(q[:, :3])[:, 0]
                num += np.sum((p - q[:, 3]) ** 2); den += np.sum(q[:, 3] ** 2)
            meta = dict(xlabel="$x$", ylabel="$y$", ext=[xs[0], xs[-1], ys[0], ys[-1]], field="h(x, y, t = 1)",
                        l2_all=float(np.sqrt(num / den)))
        elif key == "beltrami":                 # u(x, y, z = 0) at t = 1
            g = _grid(-1, 1, 161)
            X, Y = np.meshgrid(g, g)
            Z, T = np.zeros_like(X), np.ones_like(X)
            P = pred(np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), T.ravel()]))[:, 0].reshape(X.shape)
            R = -(np.exp(X) * np.sin(Y + Z) + np.exp(Z) * np.cos(X + Y)) * np.exp(-T)
            # every output's L2 over the 3D grid at t = 1, for the caption
            g3 = _grid(-1, 1, 31)
            A, B, C = np.meshgrid(g3, g3, g3, indexing="ij")
            pts = np.column_stack([A.ravel(), B.ravel(), C.ravel(), np.ones(A.size)])
            pr = pred(pts)
            a, b, c = pts[:, 0], pts[:, 1], pts[:, 2]
            ex = [-(np.exp(a) * np.sin(b + c) + np.exp(c) * np.cos(a + b)),
                  -(np.exp(b) * np.sin(c + a) + np.exp(a) * np.cos(b + c)),
                  -(np.exp(c) * np.sin(a + b) + np.exp(b) * np.cos(c + a)),
                  -0.5 * (np.exp(2 * a) + np.exp(2 * b) + np.exp(2 * c) + 2 * np.sin(a + b) * np.cos(c + a) * np.exp(b + c)
                          + 2 * np.sin(b + c) * np.cos(a + b) * np.exp(c + a) + 2 * np.sin(c + a) * np.cos(b + c) * np.exp(a + b))]
            ex = [e_ * np.exp(-1.0) for e_ in ex[:3]] + [ex[3] * np.exp(-2.0)]
            l2 = {}
            for i, n_ in enumerate("uvwp"):
                e_ = pr[:, i] - ex[i]
                if n_ == "p":                   # the pressure is fixed only up to a constant
                    e_ = e_ - e_.mean()
                l2[n_] = float(np.linalg.norm(e_) / np.linalg.norm(ex[i] - (ex[i].mean() if n_ == "p" else 0)))
            meta = dict(xlabel="$x$", ylabel="$y$", ext=[-1, 1, -1, 1], field="u(x, y, z = 0, t = 1)",
                        l2_outputs=l2)
        elif key == "burgers_inv":              # u(x, t)
            x, t = _grid(-1, 1, 256), _grid(0, 1, 201)
            X, T = np.meshgrid(x, t)
            P = pred(np.column_stack([X.ravel(), T.ravel()]))[:, 0].reshape(T.shape)
            R = _cole_hopf(x, t)
            meta = dict(xlabel="$x$", ylabel="$t$", ext=[-1, 1, 0, 1], field="u(x, t)")
        elif key == "plate_inv":                # ux(x, y)
            g = _grid(0, 1, 161)
            X, Y = np.meshgrid(g, g)
            P = pred(np.column_stack([X.ravel(), Y.ravel()]))[:, 0].reshape(X.shape)
            R = np.cos(2 * np.pi * X) * np.sin(np.pi * Y)
            meta = dict(xlabel="$x$", ylabel="$y$", ext=[0, 1, 0, 1], field="uₓ(x, y)")
        elif key == "ac3d_inv":                 # u(x, y, z = 0.5) at t = 0.5 (measured only at t = 1)
            d = np.loadtxt(os.path.join(REF, "3D", "allen_cahn_wight", "t_0.5.txt"))
            d = d[np.isclose(d[:, 2], 0.5)]
            xs, ys = np.unique(d[:, 0]), np.unique(d[:, 1])
            R = np.full((len(ys), len(xs)), np.nan)
            R[np.searchsorted(ys, d[:, 1]), np.searchsorted(xs, d[:, 0])] = d[:, 4]
            X, Y = np.meshgrid(xs, ys)
            P = pred(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 0.5), np.full(X.size, 0.5)]))[:, 0].reshape(X.shape)
            meta = dict(xlabel="$x$", ylabel="$y$", ext=[0, 1, 0, 1], field="φ(x, y, z = 0.5, t = 0.5)")
        meta["l2"] = float(np.linalg.norm(P - R) / np.linalg.norm(R))
        extra = {}
        if key == "beltrami":                   # u in the whole cube at t = 1, for the 3D view
            g = _grid(-1, 1, 41)
            A, B, C = np.meshgrid(g, g, g, indexing="ij")
            extra["grid3"] = g
            extra["pinn3"] = pred(np.column_stack([A.ravel(), B.ravel(), C.ravel(), np.ones(A.size)]))[:, 0].reshape(A.shape)
            extra["ref3"] = -(np.exp(A) * np.sin(B + C) + np.exp(C) * np.cos(A + B)) * np.exp(-1.0)
        if key == "ac3d_inv":                   # phi in the whole cube at t = 0.5, on the reference grid
            d = np.loadtxt(os.path.join(REF, "3D", "allen_cahn_wight", "t_0.5.txt"))
            g = np.unique(d[:, 0]); n = len(g)
            idx = [np.searchsorted(g, d[:, k]) for k in range(3)]
            R3 = np.zeros((n, n, n)); R3[idx[0], idx[1], idx[2]] = d[:, 4]
            A, B, C = np.meshgrid(g, g, g, indexing="ij")
            extra["grid3"] = g
            extra["ref3"] = R3
            extra["pinn3"] = pred(np.column_stack([A.ravel(), B.ravel(), C.ravel(), np.full(A.size, 0.5)]))[:, 0].reshape(A.shape)
        if key in INVERSE:
            true = {v["name"]: v["true"] for v in (eval(cfg["inverse_variables"]) if isinstance(cfg["inverse_variables"], str)
                                                    else cfg["inverse_variables"])}
            for n_, (it, val) in _conv(run).items():
                extra["it_" + n_], extra["val_" + n_] = it, val
            meta["true"] = true
        np.savez_compressed(os.path.join(DATA, key + ".npz"), pinn=P, ref=R, meta=json.dumps(meta), **extra)
        print(f"{key}: L2 = {meta['l2']:.2e}", {k: v for k, v in meta.items() if k.startswith(("l2_", "true"))})


# ---- 2. plot ---------------------------------------------------------------------------------------------

SYM = {"lambda_1": r"$\lambda_1$", "lambda_2": r"$\lambda_2$", "lmbd": r"$\lambda$", "mu": r"$\mu$",
       "lambda_": r"$\lambda$", "diff_coeff": r"$\lambda\varepsilon^2$"}
PAR_COLORS = ["#1f5fa8", "#d1701c", "#2b8a3e"]
ONE_D = ("wave", "burgers_inv")        # drawn with t across and x up, as PINNStudio draws 1D problems
THREE_D = {"beltrami": False, "ac3d_inv": True}   # 3D view; True: one corner cut away (the field is inside)

# layout, in inches (a Word page's text width is 6.5 in)
WIDTH = 6.5
# every column: room for the y-axis labels, the panel, its colorbar and the colorbar's labels
YLAB, CB_PAD, CB_W, CB_TICKS, TOP, ROW_GAP, BOTTOM = 0.37, 0.06, 0.08, 0.27, 0.32, 0.48, 0.42
PANEL = (WIDTH - 3 * (YLAB + CB_PAD + CB_W + CB_TICKS)) / 3     # 1.39 in
COL = YLAB + PANEL + CB_PAD + CB_W + CB_TICKS
FONT = 8
ALT_JET = False    # --alt --jet: the alternative layout with the approved jet colour maps
ALT = False        # the alternative style (python tools/make_paper_examples.py plot --alt): drawn at the CPC text
                   # width (5.42 in) so it prints at 100 %, perceptually uniform / diverging colour maps


def _use_alt():
    """Layout and colour maps of the alternative style. Same data, same plotted quantities, same colour limits
    (a signed field's limits are made symmetric about zero, which only widens them)."""
    global WIDTH, PANEL, COL, ALT, CB_TICKS
    ALT, WIDTH, CB_TICKS = True, 5.42, 0.30
    PANEL = (WIDTH - 3 * (YLAB + CB_PAD + CB_W + CB_TICKS)) / 3
    COL = YLAB + PANEL + CB_PAD + CB_W + CB_TICKS


def _field_style(lo, hi):
    """Colour map and normalisation of a solution field: jet as approved; in the alternative style a diverging
    map centred on zero for a field of both signs, a sequential one otherwise."""
    import matplotlib
    import matplotlib.pyplot as plt
    if not ALT or ALT_JET:
        return plt.get_cmap("jet"), matplotlib.colors.Normalize(lo, hi)
    m = max(abs(lo), abs(hi))
    if lo < 0 < hi and min(abs(lo), abs(hi)) >= 0.3 * m:
        return plt.get_cmap("RdBu_r"), matplotlib.colors.Normalize(-m, m)
    return plt.get_cmap("cividis"), matplotlib.colors.Normalize(lo, hi)


def _error_cmap():
    import matplotlib.pyplot as plt
    return plt.get_cmap("viridis" if (ALT and not ALT_JET) else "jet")


def _faces(g, F, cut):
    """Quads and values of the cube's three visible faces (x = max, y = min, z = max) on grid g, F[ix, iy, iz];
    with cut, the corner x > mid, y < mid, z > mid removed and its three cut faces added."""
    n = len(g); m = n // 2; last = n - 1
    polys, vals = [], []

    def add(plane, k, ra, rb):
        for a in range(*ra):
            for b in range(*rb):
                if plane == "x":
                    q = [(g[k], g[a], g[b]), (g[k], g[a + 1], g[b]), (g[k], g[a + 1], g[b + 1]), (g[k], g[a], g[b + 1])]
                    v = F[k, a:a + 2, b:b + 2].mean()
                elif plane == "y":
                    q = [(g[a], g[k], g[b]), (g[a + 1], g[k], g[b]), (g[a + 1], g[k], g[b + 1]), (g[a], g[k], g[b + 1])]
                    v = F[a:a + 2, k, b:b + 2].mean()
                else:
                    q = [(g[a], g[b], g[k]), (g[a + 1], g[b], g[k]), (g[a + 1], g[b + 1], g[k]), (g[a], g[b + 1], g[k])]
                    v = F[a:a + 2, b:b + 2, k].mean()
                polys.append(q); vals.append(v)
    if not cut:
        add("x", last, (0, last), (0, last)); add("y", 0, (0, last), (0, last)); add("z", last, (0, last), (0, last))
    else:
        add("x", last, (0, last), (0, m)); add("x", last, (m, last), (m, last))
        add("y", 0, (0, m), (0, last)); add("y", 0, (m, last), (0, m))
        add("z", last, (0, m), (0, last)); add("z", last, (m, last), (m, last))
        add("x", m, (0, m), (m, last)); add("y", m, (m, last), (m, last)); add("z", m, (m, last), (0, m))
    return polys, np.array(vals)


def _cube(ax, g, F, cut, cmap, norm):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys, vals = _faces(g, F, cut)
    cols = cmap(norm(vals))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.2))
    import matplotlib
    a, b = g[0], g[-1]
    ax.set_xlim(a, b); ax.set_ylim(a, b); ax.set_zlim(a, b)
    ax.set_box_aspect((1, 1, 1), zoom=0.8)
    ax.view_init(elev=22, azim=-58)
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))
    for axis, lab in ((ax.xaxis, "$x$"), (ax.yaxis, "$y$"), (ax.zaxis, "$z$")):
        axis.set_ticks([a, (a + b) / 2, b] if lab == "$z$" else [a, b])   # x and y meet at a corner: ends only
        axis.set_major_formatter(fmt)
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_linewidth(0.6)
        axis._axinfo["grid"]["linewidth"] = 0
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.25
    ax.set_xlabel("$x$", labelpad=-8); ax.set_ylabel("$y$", labelpad=-8); ax.set_zlabel("$z$", labelpad=-8)
    ax.tick_params(labelsize=FONT - 1.5, pad=-3)
    ax.grid(False)


def _colorbar(fig, ax_rect, mappable, sci=False):
    import matplotlib
    x0, y0, w, h = ax_rect
    cax = fig.add_axes([x0 + w + CB_PAD / WIDTH, y0, CB_W / WIDTH, h])
    fmt = None
    if sci:
        fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
    cb = fig.colorbar(mappable, cax=cax, format=fmt)
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))   # few, short labels
    if not sci:
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
    cb.ax.tick_params(labelsize=FONT - 1.5, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    if sci:
        cb.ax.yaxis.get_offset_text().set_fontsize(FONT - 1.5)
    return cb


def _value(v, true):
    """A learned value with no trailing zeros and at least five significant figures, more where five would make
    a value that is not exact look exact (equal to the true value)."""
    for n in range(5, 10):
        txt = f"{v:.{n}g}"
        if float(txt) != float(true) or v == true:
            return txt
    return f"{v:.10g}"


def plot(keys, path, inverse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": FONT, "axes.linewidth": 0.6, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    # panel widths; the alternative style gives the parameter-error column of the inverse figure more room
    # for its legend, from the two image columns, whose panels stay square
    CBW = CB_PAD + CB_W + CB_TICKS
    # the alternative forward figure: Reference and PINNStudio share one colour scale, drawn once (after PINNStudio)
    shared = ALT and not inverse
    cbs = [not shared, True, True]
    if ALT and inverse:
        # image panels as large as Figure 5's (whose two field panels share one colour bar); the parameter-error
        # plot takes the rest of the width, its legend entries on two lines
        P5 = (WIDTH - 3 * YLAB - 2 * CBW) / 3
        PW = [P5, P5, WIDTH - 3 * YLAB - 3 * CBW - 2 * P5]
    elif shared:
        PW = [(WIDTH - 3 * YLAB - 2 * CBW) / 3] * 3
    else:
        PW = [PANEL] * 3
    RH = PW[0]                                  # row height
    xs = ([YLAB + sum(PW[:c]) + sum(CBW for k in range(c) if cbs[k]) + c * YLAB for c in range(3)] if ALT
          else [YLAB + c * COL for c in range(3)])
    H = TOP + len(keys) * RH + (len(keys) - 1) * ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(WIDTH, H))
    # forward: reference | PINNStudio | error; inverse: the inverse PINN's field | error | the unknowns
    titles = (["PINNStudio", "Absolute error", "Parameter error (%)"] if inverse
              else ["Reference", "PINNStudio", "Absolute error"])
    for r, key in enumerate(keys):
        d = np.load(os.path.join(DATA, key + ".npz"))
        meta = json.loads(str(d["meta"]))
        y0 = (H - TOP - (r + 1) * RH - r * ROW_GAP) / H
        rects = [(xs[c] / WIDTH, y0, PW[c] / WIDTH, RH / H) for c in range(3)]
        three = key in THREE_D
        if three:
            P, R, g = d["pinn3"], d["ref3"], d["grid3"]
        else:
            P, R = d["pinn"], d["ref"]
        lo, hi = float(np.nanmin(R)), float(np.nanmax(R))
        fcmap, norm = _field_style(lo, hi)
        E = np.abs(P - R)
        enorm = matplotlib.colors.Normalize(0, float(np.nanmax(E)))
        panels = [(P, norm), (E, enorm)] if inverse else [(R, norm), (P, norm), (E, enorm)]
        err_col = 1 if inverse else 2
        for c, (A, nm) in enumerate(panels):
            cmap = _error_cmap() if c == err_col else fcmap
            if three:
                # a little wider than the panel, shifted left, so the z axis labels clear the colorbar
                x0_, y0_, w_, h_ = rects[c]
                r3 = [x0_ - 0.3 / WIDTH, y0_ - 0.1 / H, w_ + 0.3 / WIDTH, h_ + 0.2 / H]
                ax = fig.add_axes(r3, projection="3d")
                ax.set_position(r3)
                _cube(ax, g, A, THREE_D[key], cmap, nm)
                sm = matplotlib.cm.ScalarMappable(norm=nm, cmap=cmap)
            else:
                ax = fig.add_axes(rects[c])
                ext = meta["ext"]
                if key in ONE_D:            # t across, x up
                    sm = ax.imshow(A.T, origin="lower", extent=[ext[2], ext[3], ext[0], ext[1]], aspect="auto",
                                   cmap=cmap, norm=nm, interpolation="bilinear")
                    ax.set_xlabel("$t$", labelpad=1)
                    ax.set_ylabel("$x$", labelpad=1)
                else:
                    sm = ax.imshow(A, origin="lower", extent=ext, aspect="auto", cmap=cmap, norm=nm,
                                   interpolation="bilinear")
                    ax.set_xlabel("$x$", labelpad=1)
                    ax.set_ylabel("$y$", labelpad=1)
                ax.tick_params(labelsize=FONT - 1, length=2)
                for axis in (ax.xaxis, ax.yaxis):    # 0, 0.5, 1 (or -1, 0, 1): short labels that never touch
                    axis.set_major_locator(matplotlib.ticker.LinearLocator(3))
                    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
            if cbs[c]:
                _colorbar(fig, rects[c], sm, sci=(c == err_col))
            if r == 0:
                fig.text(rects[c][0] + rects[c][2] / 2, (H - TOP / 2.4) / H, titles[c], ha="center", va="center",
                         fontsize=FONT + 1, fontweight="bold")
        if inverse:
            x0, y0_, w, h = rects[2]
            dx = 0.035                     # room for the y label next to the colour bar on the left
            ax = fig.add_axes([x0 + dx, y0_, w + (CB_PAD + CB_W + CB_TICKS - 0.05) / WIDTH - dx, h])
            names = [k[3:] for k in d.files if k.startswith("it_")]
            for i, n_ in enumerate(names):
                it, val = d["it_" + n_], d["val_" + n_]
                tv = meta["true"][n_]
                ax.semilogy(it / 1000, np.maximum(100 * np.abs(val - tv) / abs(tv), 1e-4), lw=1.1, color=PAR_COLORS[i],
                            label=f"{SYM.get(n_, n_)}: {_value(val[-1], tv)}" + ("\n" if ALT else " ") + f"(true: {tv:g})")
            ax.set_xlabel("Iteration (×10³)", labelpad=1)
            ax.set_ylabel("Relative error (%)", labelpad=1, fontsize=FONT - 1)   # 100 |θ − θ*| / |θ*| (in the caption)
            ax.set_ylim(1e-4, 1e11 if ALT else 2e4)   # the largest early errors are ~10⁴ %; room below the curves for
            # the legend (approved) or, in the alternative layout, above them
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([10.0 ** k for k in range(-4, 5)]) if ALT else
                                       matplotlib.ticker.LogLocator(base=10, numticks=20))   # (ALT: no ticks in the
            # headroom above 10⁴ % that holds the legend)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: (f"$10^{{{int(round(np.log10(v)))}}}$" if int(round(np.log10(v))) % 2 == 1 else "")))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(color="#e4e7eb", lw=0.4)
            leg = ax.legend(fontsize=FONT - (1.5 if ALT else 1), frameon=False, loc="upper right" if ALT else "lower left", handlelength=1.2,
                            borderaxespad=0.2, handletextpad=0.4, labelspacing=0.3)
            if ALT:                         # the text stays readable where a curve passes behind it
                leg.get_frame().set(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="square,pad=0.15")
            ax.tick_params(labelsize=FONT - 1, length=2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                xc = 0.995 if ALT else rects[2][0] + (rects[2][2] + (CB_PAD + CB_W + CB_TICKS) / WIDTH) / 2
                fig.text(xc, (H - TOP / 2.4) / H, titles[2], ha="right" if ALT else "center", va="center",
                         fontsize=FONT + 1, fontweight="bold")
        fig.text(0.08 / WIDTH, (y0 * H + RH + 0.02) / H, f"({'abc'[r]})", ha="left", va="bottom",
                 fontsize=FONT + 1.5, fontweight="bold")
    fig.savefig(path, dpi=300)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print("wrote", path, f"({WIDTH} x {H:.2f} in)")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3:] or None)
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        sfx = ""
        if "--alt" in sys.argv:
            _use_alt()
            sfx = "_alt"
            if "--jet" in sys.argv:
                global ALT_JET
                ALT_JET, sfx = True, "_alt_jet"
        plot(FORWARD, os.path.join(FIGS, f"paper_forward{sfx}.png"), False)
        plot(INVERSE, os.path.join(FIGS, f"paper_inverse{sfx}.png"), True)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

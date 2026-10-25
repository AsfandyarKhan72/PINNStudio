"""The paper's two 3 x 3 example figures (benchmarks/figures/paper_forward.png, paper_inverse.png).

Forward: Wave 1D (multi-scale Fourier features), Allen-Cahn 2D (Mattey & Ghosh), Navier-Stokes 3D (Beltrami
flow); columns PINN | reference | |error|. Inverse: Burgers 1D (Raissi et al.), linear elasticity 2D (plate),
Allen-Cahn 3D (Wight & Zhao); columns PINN | reference | the unknown parameters' relative error during training.
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


def export(runs):
    os.makedirs(DATA, exist_ok=True)
    for key in FORWARD + INVERSE:
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
        if key in INVERSE:
            true = {v["name"]: v["true"] for v in (eval(cfg["inverse_variables"]) if isinstance(cfg["inverse_variables"], str)
                                                    else cfg["inverse_variables"])}
            for n_, (it, val) in _conv(run).items():
                extra["it_" + n_], extra["val_" + n_] = it, val
            meta["true"] = true
        np.savez_compressed(os.path.join(DATA, key + ".npz"), pinn=P, ref=R, meta=json.dumps(meta), **extra)
        print(f"{key}: L2 = {meta['l2']:.2e}", {k: v for k, v in meta.items() if k.startswith(("l2_", "true"))})


# ---- 2. plot ---------------------------------------------------------------------------------------------

ROWS = {
    "wave": ("Wave 1D", "multi-scale Fourier features"),
    "ac2d": ("Allen-Cahn 2D", "Mattey & Ghosh"),
    "beltrami": ("Navier-Stokes 3D", "Beltrami flow"),
    "burgers_inv": ("Burgers 1D", "Raissi et al."),
    "plate_inv": ("Linear elasticity 2D", "plate"),
    "ac3d_inv": ("Allen-Cahn 3D", "Wight & Zhao"),
}
SYM = {"lambda_1": r"$\lambda_1$", "lambda_2": r"$\lambda_2$", "lmbd": r"$\lambda$", "mu": r"$\mu$",
       "lambda_": r"$\lambda$", "diff_coeff": r"$\lambda\varepsilon^2$"}
PAR_COLORS = ["#1f5fa8", "#d1701c", "#2b8a3e"]
WIDTH, ROW_H = 6.5, 2.05        # inches: a Word page's text width


def _sci(x):
    m, e = f"{x:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def plot(keys, path, inverse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(len(keys), 3, figsize=(WIDTH, ROW_H * len(keys) + 0.35))
    for r, key in enumerate(keys):
        d = np.load(os.path.join(DATA, key + ".npz"))
        meta = json.loads(str(d["meta"]))
        P, R = d["pinn"], d["ref"]
        lo, hi = np.nanmin(R), np.nanmax(R)
        ext = meta["ext"]
        for c, (A, title) in enumerate([(P, "PINNStudio"), (R, "Reference")]):
            ax = axes[r, c]
            im = ax.imshow(A, origin="lower", extent=ext, aspect="auto", cmap="jet", vmin=lo, vmax=hi,
                           interpolation="bilinear")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6.5)
            if r == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
        name, sub = ROWS[key]
        l2 = meta.get("l2_all", meta["l2"])
        axes[r, 0].set_ylabel(meta["ylabel"])
        axes[r, 0].annotate(f"{name}\n{sub}", xy=(-0.42, 0.5), xycoords="axes fraction", ha="right", va="center",
                            fontsize=8.5, fontweight="bold", rotation=90)
        for c in range(2):
            axes[r, c].set_xlabel(meta["xlabel"], labelpad=1)
        ax = axes[r, 2]
        if not inverse:
            E = np.abs(P - R)
            im = ax.imshow(E, origin="lower", extent=ext, aspect="auto", cmap="jet", interpolation="bilinear")
            fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format=fmt)
            cb.ax.tick_params(labelsize=6.5)
            cb.ax.yaxis.get_offset_text().set_fontsize(6.5)
            ax.set_xlabel(meta["xlabel"], labelpad=1)
            lab = (r"$L_2$ = $" + _sci(l2) + "$" + (" (all times)" if "l2_all" in meta else ""))
            ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            if r == 0:
                ax.set_title("|Error|", fontsize=9, fontweight="bold")
        else:
            names = [k[3:] for k in d.files if k.startswith("it_")]
            for i, n_ in enumerate(names):
                it, val = d["it_" + n_], d["val_" + n_]
                tv = meta["true"][n_]
                ax.semilogy(it / 1000, np.maximum(np.abs(val - tv) / abs(tv), 1e-6), lw=1.3, color=PAR_COLORS[i],
                            label=f"{SYM.get(n_, n_)}: {val[-1]:.4g} (true {tv:g})")
            ax.set_xlabel("Iteration (×1000)", labelpad=1)
            ax.set_ylabel("Relative error", labelpad=1)
            ax.grid(color="#e4e7eb", lw=0.5)
            ax.legend(fontsize=6.3, frameon=True, framealpha=0.9, edgecolor="none", loc="lower left",
                      handlelength=1.2, borderaxespad=0.2)
            ax.set_ylim(1e-6, 3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title("Unknown parameters", fontsize=9, fontweight="bold")
        axes[r, 0].text(0.03, 0.97, meta["field"], transform=axes[r, 0].transAxes, ha="left", va="top",
                        fontsize=6.5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        for c in range(3):
            axes[r, c].tick_params(labelsize=7, length=2)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.06, wspace=0.55, hspace=0.55)
    fig.savefig(path, dpi=300)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print("wrote", path)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "export":
        export(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(FORWARD, os.path.join(FIGS, "paper_forward.png"), False)
        plot(INVERSE, os.path.join(FIGS, "paper_inverse.png"), True)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

"""Radius of the shrinking drop in the Allen-Cahn 3D (Wight & Zhao) template, PINN against the reference.

The drop's radius is where phi = 0 along the six half-axes from the centre (0.5, 0.5, 0.5), averaged;
each crossing is the root of a cubic through the four grid values around the sign change (the reference's
51-point grid, and the same grid for the PINN at the reference times). The PINN curve is drawn over the
whole time interval (on a 401-point line), the reference at its saved times, and the sharp-interface
limit for comparison:
phi_t = lambda (eps^2 lap phi - (phi^3 - phi)) moves the interface by mean curvature with speed
lambda eps^2 * 2/R, so R(t)^2 = R0^2 - 4 lambda eps^2 t.

    python tools/plot_3d_allen_cahn_radius.py --model forward.pt [--inverse inverse.pt] \
        --out assets/Images/3D_AllenCahn-Wight/radius.png

The models are the ones PINNStudio saves in solution_results/ (see plot_3d_allen_cahn_isosurface.py).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_3d_allen_cahn_isosurface import ROOT, TIMES, load_net, load_ref  # noqa: E402

LAM_EPS2 = 10 * 0.05 ** 2        # lambda eps^2 of the template
N_T = 41                         # PINN times
C_PINN, C_INV, C_REF, C_THEORY = "#1f5fa8", "#d1701c", "#1f2933", "#8a94a3"


def crossing(r, f):
    """Distance r where f (positive at r = 0) first changes sign: root of the cubic through 4 points."""
    k = int(np.argmax(f <= 0))
    i = min(max(k - 2, 0), len(r) - 4)
    c = np.polyfit(r[i:i + 4], f[i:i + 4], 3)
    roots = [x.real for x in np.roots(c) if abs(x.imag) < 1e-9 and r[i] - 1e-9 <= x.real <= r[i + 3] + 1e-9]
    return min(roots, key=lambda x: abs(x - r[k])) if roots else np.interp(0, -f[k - 1:k + 1], r[k - 1:k + 1])


def radius_on_axes(field, r):
    """Mean phi = 0 crossing along +-x, +-y, +-z from the centre; field(points) -> phi."""
    out = []
    for ax in range(3):
        for sgn in (1, -1):
            P = np.full((len(r), 3), 0.5)
            P[:, ax] = 0.5 + sgn * r
            out.append(crossing(r, field(P)))
    return float(np.mean(out))


def ref_field(g, ref):
    m = len(g) // 2
    def f(P):       # P lies on a grid line through the centre
        idx = np.rint(P / (g[1] - g[0])).astype(int)
        return ref[idx[:, 0], idx[:, 1], idx[:, 2]]
    return f, g[m:] - g[m]


def pinn_radius(net, ts, r):
    return np.array([radius_on_axes(lambda P: net(np.column_stack([P, np.full(len(P), t)])), r) for t in ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the forward run's model")
    ap.add_argument("--inverse", help="an inverse run's model, drawn too")
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "Images", "3D_AllenCahn-Wight", "radius.png"))
    a = ap.parse_args()

    t_ref = np.array([float(t) for t in TIMES])
    r_ref, r_grid = [], None
    for t in TIMES:
        g, _, ref = load_ref(t)
        f, r_grid = ref_field(g, ref)
        r_ref.append(radius_on_axes(f, r_grid))
    r_ref = np.array(r_ref)
    ts = np.linspace(0, 1, N_T)
    fine = np.linspace(0, 0.5, 401)
    nets = [load_net(a.model)] + ([load_net(a.inverse)] if a.inverse else [])
    curves = [pinn_radius(n, ts, fine) for n in nets]          # the curves
    at_ref = [pinn_radius(n, t_ref, r_grid) for n in nets]     # on the reference's grid, for the error
    r_fwd, r_inv = curves[0], (curves[1] if a.inverse else None)
    theory = np.sqrt(np.maximum(r_ref[0] ** 2 - 4 * LAM_EPS2 * ts, 0))

    # radius error at the reference times, both measured on the reference's grid
    e_fwd = np.max(np.abs(at_ref[0] - r_ref) / r_ref)
    e_inv = np.max(np.abs(at_ref[1] - r_ref) / r_ref) if a.inverse else None
    for t, rr, rf in zip(TIMES, r_ref, at_ref[0]):
        print(f"t = {t}: R_ref = {rr:.4f}, R_PINN = {rf:.4f}, theory = {np.sqrt(max(r_ref[0]**2 - 4*LAM_EPS2*float(t), 0)):.4f}")
    print(f"largest relative radius error: forward {e_fwd:.2e}" + (f", inverse {e_inv:.2e}" if e_inv is not None else ""))

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(ts, theory, ls="--", lw=1.0, color=C_THEORY, label=r"Sharp-interface limit $R^2 = R_0^2 - 4\lambda\varepsilon^2 t$")
    ax.plot(ts, r_fwd, lw=1.8, color=C_PINN, label=f"PINN, forward (max error {100 * e_fwd:.2f}%)")
    if r_inv is not None:
        ax.plot(ts, r_inv, lw=1.4, ls=(0, (4, 2)), color=C_INV, label=f"PINN, inverse (max error {100 * e_inv:.2f}%)")
    ax.plot(t_ref, r_ref, "o", ms=5, mfc="white", mec=C_REF, mew=1.2, label="Reference", zorder=5)
    ax.set_xlabel("$t$")
    ax.set_ylabel("Drop radius $R$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.4)
    ax.grid(color="#e4e7eb", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf", bbox_inches="tight", pad_inches=0.03)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

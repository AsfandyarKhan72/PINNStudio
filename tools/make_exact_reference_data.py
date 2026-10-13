#!/usr/bin/env python3
"""
Rewrites reference_data/1D/burgers and reference_data/1D/heat from the two
problems' exact solutions, on the same x grid, times and file names as
before (the earlier files were numerical exports about 1-2% off the exact
solution, which put a floor under every Error Analysis result).

Burgers 1D (Raissi et al. 2019):  u_t + u u_x = (0.01/pi) u_xx on [-1, 1],
    u(x, 0) = -sin(pi x), u(+-1, t) = 0.  Exact solution by the Cole-Hopf
    transformation,
        u(x, t) = - int sin(pi (x - s)) f(x - s) G(s) ds / int f(x - s) G(s) ds,
        f(y) = exp(-cos(pi y) / (2 pi nu)),  G = heat kernel of width sqrt(4 nu t),
    evaluated with Gauss-Hermite quadrature (the same formula behind
    DeepXDE's and Raissi's Burgers datasets; agrees with DeepXDE's
    examples/dataset/Burgers.npz to ~1e-12).
Heat 1D:  u_t = 0.4 u_xx on [0, 1], u(x, 0) = sin(pi x), u(0, t) = u(1, t) = 0.
    Exact solution u = sin(pi x) exp(-0.4 pi^2 t).

Run from the repository root:
    python3 tools/make_exact_reference_data.py
"""
import os

import numpy as np
from numpy.polynomial.hermite import hermgauss

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_data", "1D")

NU = 0.01 / np.pi
_Z, _W = hermgauss(250)


def burgers_exact(x, t):
    x = np.asarray(x, dtype=float)
    if t == 0:
        return -np.sin(np.pi * x)
    y = x[:, None] - np.sqrt(4 * NU * t) * _Z[None, :]
    # f(y) / f_max, to keep the exponentials in range (cancels in the ratio)
    f = np.exp(-(np.cos(np.pi * y) + 1.0) / (2 * np.pi * NU))
    return -(np.sin(np.pi * y) * f) @ _W / (f @ _W)


def heat_exact(x, t):
    return np.sin(np.pi * np.asarray(x, dtype=float)) * np.exp(-0.4 * np.pi ** 2 * t)


# (folder, exact solution, x grid, {file name: t})
SETS = [
    ("burgers", burgers_exact, np.round(np.arange(-0.999, 1.0, 0.002), 6),
     {"t_0.txt": 0.0, "t_0.25.txt": 0.25, "t_0.50.txt": 0.5, "t_0.75.txt": 0.75, "t_1.0.txt": 1.0}),
    ("heat", heat_exact, np.round(np.arange(0.0005, 1.0, 0.001), 6),
     {"t_0.txt": 0.0, "t_0.25.txt": 0.25, "t_0.5.txt": 0.5, "t_0.75.txt": 0.75, "t_1.0.txt": 1.0}),
]


def main():
    for folder, exact, x, files in SETS:
        d = os.path.join(ROOT, folder)
        os.makedirs(d, exist_ok=True)
        for name, t in files.items():
            u = exact(x, t)
            with open(os.path.join(d, name), "w") as fh:
                for xi, ui in zip(x, u):
                    fh.write(f"{xi:.6f}\t{t:.4f}\t{ui:.12e}\n")
            print(f"{folder}/{name}: {len(x)} points, t = {t}")


if __name__ == "__main__":
    main()

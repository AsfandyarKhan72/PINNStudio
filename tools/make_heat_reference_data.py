#!/usr/bin/env python3
"""
Reference data for the 2D Heat and 3D Heat templates (reference_data/2D/heat/, reference_data/3D/heat/), from the
exact solution, replacing files of unknown provenance (most likely a finite-element export) that differ from it by
up to 3e-3 (2D, t = 0.25) and 9e-4 (3D, t = 0.4) in relative L2.

    u_t = 0.4 lap u  on the unit square / cube, t in [0, 1],  u = 0 at t = 0,
    u = 1 on x = 1, du/dn = 0 on the other sides (PINNStudio's templates "2D Heat" and "3D Heat").

The solution depends on x and t only:
    u(x, t) = 1 - sum_{n>=0} 4 (-1)^n / ((2n+1) pi) cos((2n+1) pi x / 2) exp(-0.4 ((2n+1) pi / 2)^2 t).
At t = 0 it is written as the initial condition, 0, with the boundary value 1 on x = 1 (the series converges
slowly there). For t > 0 the series is summed to 2000 terms; the terms left out are below 1e-300 for t >= 0.2.

The values are written at the points of the existing files, in their order and with their names (2D: t_0,
t_0.25, t_0.5, t_0.75, t_1.0; 3D: t_0.0, t_0.2, ..., t_1.0 and the thinned t_1.0_LessData), columns x y [z] t u.

    python tools/make_heat_reference_data.py
"""
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 0.4


def exact(x, t, terms=2000):
    if t == 0:
        return np.where(np.isclose(x, 1.0), 1.0, 0.0)
    n = np.arange(terms)
    m = (2 * n + 1) * np.pi / 2
    c = 4 * (-1.0) ** n / ((2 * n + 1) * np.pi) * np.exp(-K * m * m * t)
    return 1.0 - np.cos(np.outer(x, m)) @ c


def main():
    for d in ("2D/heat", "3D/heat"):
        for f in sorted(glob.glob(os.path.join(ROOT, "reference_data", d, "t_*.txt"))):
            a = np.loadtxt(f)
            t = float(a[0, -2])
            assert np.allclose(a[:, -2], t)
            u = exact(a[:, 0], t)
            old = a[:, -1]
            print(f"{d}/{os.path.basename(f):22s} t = {t:<4g} relative L2 change "
                  f"{np.linalg.norm(u - old) / max(np.linalg.norm(u), 1e-300):.2e}, max {np.abs(u - old).max():.2e}")
            np.savetxt(f, np.column_stack([a[:, :-1], u]), fmt="%.10g", delimiter="  ")


if __name__ == "__main__":
    main()

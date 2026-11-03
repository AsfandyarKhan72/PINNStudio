#!/usr/bin/env python3
"""Reference solutions of the three Poisson templates without COMSOL:

    -Laplace(u) = 1 in the domain,  u = 0 on its boundary,

  disk      the unit disk                                    reference_data/2D/poisson_disk
  lshape    [-1, 1]^2 without the quadrant (0, 1] x (0, 1]   reference_data/2D/poisson_lshape
  sphere    the unit ball                                    reference_data/3D/poisson_sphere

Disk and ball have the exact solutions u = (1 - r^2) / 4 and u = (1 - r^2) / 6; they are written from these
formulas. The L-shaped domain has no closed-form solution (u behaves like r^(2/3) sin(2 theta / 3) at the
re-entrant corner), so it is solved with the five-point finite-difference scheme on uniform grids of spacing
h = 1/N aligned with the corner and every 0.01 grid point (N a multiple of 100), with an algebraic multigrid
solver (pyamg; scipy's sparse direct solver if pyamg is missing, for small N). Several N are solved; the
differences between successive grids show the convergence, and the finest grid is written. Richardson
extrapolation is not applied: the corner singularity makes the order point-dependent (about 4/3 near the
corner, 2 away from it), and the finest-grid differences already sit well below the template's tolerances.

The points are defined here, not read from earlier files (POINTS below), and they are the points of the files
in reference_data/:
  2D (disk, L-shape)  solution.txt on x, y = numpy.linspace(-1, 1, 201), x running fastest, columns x y u;
  3D (ball)           solution.txt on x, y, z = numpy.round(numpy.linspace(-1, 1, 41), 2), z running fastest,
                      columns x y z u;
  points outside the domain get NaN (written NaN in 2D, nan in 3D): r^2 > 1 for the disk and the ball, x > 0
  and y > 0 for the L-shape (its edges x = 0 and y = 0 are in the domain, with u = 0);
  u_obs.txt (CSV, header x,y[,z],u) at the 300 measurement points listed in tools/reference_points/<name>_obs.csv,
  which are points of the grid above.

    python tools/make_poisson_reference_data.py --out DIR                       # all three, L-shape N=1600
    python tools/make_poisson_reference_data.py --out DIR --only lshape --levels 100,200,400,800,1600

Deterministic. Needs NumPy and SciPy (and pyamg for large N).
"""
import argparse
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = {"disk": ("2D", "poisson_disk"), "lshape": ("2D", "poisson_lshape"), "sphere": ("3D", "poisson_sphere")}


POINTS = os.path.join(ROOT, "tools", "reference_points")


def grid_points(name):
    """The evaluation points of solution.txt, in file order, and the mask of the points outside the domain."""
    if SRC[name][0] == "2D":
        g = np.linspace(-1.0, 1.0, 201)
        Y, X = np.meshgrid(g, g, indexing="ij")             # x fastest
        P = np.column_stack([X.ravel(), Y.ravel()])
    else:
        g = np.round(np.linspace(-1.0, 1.0, 41), 2)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")       # z fastest
        P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    if name == "lshape":
        out = (P[:, 0] > 0) & (P[:, 1] > 0)
    else:
        out = (P ** 2).sum(1) > 1.0 + 1e-12
    return P, out


def obs_points(name):
    """The measurement points of u_obs.txt (tools/reference_points/<folder>_obs.csv)."""
    return np.genfromtxt(os.path.join(POINTS, f"{SRC[name][1]}_obs.csv"), delimiter=",", skip_header=1)


# --- L-shape: five-point finite differences ------------------------------------------------------------------
def lshape_fd(N, log=print):
    """u on the grid x_i = -1 + i/N (i = 0..2N), rows y, columns x; NaN outside the L."""
    import scipy.sparse as sp
    m = 2 * N + 1
    g = np.linspace(-1.0, 1.0, m)
    X, Y = np.meshgrid(g, g)
    inside = ~((X > 0) & (Y > 0))                                       # closed L (boundary included)
    interior = inside.copy()
    interior[[0, -1], :] = interior[:, [0, -1]] = False
    interior &= ~(((X == 0) & (Y >= 0)) | ((Y == 0) & (X >= 0)))       # the two inner edges are boundary
    ix = np.flatnonzero(interior.ravel())
    idx = -np.ones(m * m, dtype=np.int64)
    idx[ix] = np.arange(ix.size)
    rows, cols, vals = [ix * 0 + np.arange(ix.size)], [np.arange(ix.size)], [np.full(ix.size, 4.0)]
    for off in (1, -1, m, -m):
        j = idx[ix + off]
        k = j >= 0                                                     # neighbours on the boundary: u = 0
        rows.append(np.arange(ix.size)[k]); cols.append(j[k]); vals.append(np.full(k.sum(), -1.0))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ix.size, ix.size))
    b = np.full(ix.size, 1.0 / N ** 2)                                   # h^2 * f
    t0 = time.time()
    try:
        import pyamg
        ml = pyamg.ruge_stuben_solver(A)
        res = []
        x = ml.solve(b, tol=1e-12, residuals=res, maxiter=500, accel="cg")
        how = f"pyamg CG-AMG, {len(res) - 1} iterations, relative residual {res[-1] / res[0]:.1e}"
    except ImportError:
        from scipy.sparse.linalg import spsolve
        x = spsolve(A.tocsc(), b)
        how = "scipy spsolve"
    u = np.full(m * m, np.nan)
    u[inside.ravel()] = 0.0
    u[ix] = x
    log(f"  L-shape N={N}: {ix.size} unknowns, {how}, {time.time() - t0:.1f} s, max u = {np.nanmax(u):.10f}")
    return g, u.reshape(m, m)


def sample(g, U, pts):
    """U (rows y, columns x on grid g) at points that are grid nodes."""
    N = (len(g) - 1) // 2
    i = np.rint((pts[:, 0] + 1) * N).astype(int)
    j = np.rint((pts[:, 1] + 1) * N).astype(int)
    assert np.allclose(g[i], pts[:, 0], atol=1e-9) and np.allclose(g[j], pts[:, 1], atol=1e-9), "not grid nodes"
    return U[j, i]


def lshape(levels, log=print):
    xy, nan = grid_points("lshape")
    obs = obs_points("lshape")
    out, prev, coarse = {}, None, xy                                    # compare all levels on the 0.01 grid
    for N in levels:
        if N % 100:
            raise SystemExit("N must be a multiple of 100 (the output grid is 0.01)")
        g, U = lshape_fd(N, log)
        v = sample(g, U, coarse)
        out[N] = (v, sample(g, U, obs))
        if prev is not None:
            d = np.nanmax(np.abs(v - prev))
            log(f"    max |u_N - u_N/2| on the 0.01 grid: {d:.3e}")
        prev = v
    v, vo = out[levels[-1]]
    v = v.copy()
    v[nan] = np.nan
    return xy, v, obs, vo, out


# --- disk and ball: the exact solutions ----------------------------------------------------------------------
def exact(name):
    xyz, nan = grid_points(name)
    obs = obs_points(name)
    dim = xyz.shape[1]
    f = lambda p: (1.0 - (p ** 2).sum(1)) / (2 * dim)
    v = f(xyz)
    v[nan] = np.nan
    return xyz, v, obs, f(obs)


def _num(c):
    """A coordinate as the shortest text that reads back as the same number ("-1", "0.010000000000000009")."""
    t = repr(float(c))
    return t[:-2] if t.endswith(".0") else t


def write(dirname, name, xyz, v, obs, vo):
    os.makedirs(dirname, exist_ok=True)
    nan_text = "NaN" if SRC[name][0] == "2D" else "nan"
    with open(os.path.join(dirname, "solution.txt"), "w") as fh:
        for p, val in zip(xyz, v):
            fh.write("".join(f"{_num(c):<25}" for c in p) + (f"{val:.16E}" if np.isfinite(val) else nan_text) + "\n")
    cols = "xyz"[: obs.shape[1]]
    with open(os.path.join(dirname, "u_obs.txt"), "w") as fh:
        fh.write(",".join(cols) + ",u\n")
        for p, val in zip(obs, vo):
            fh.write(",".join(f"{c:.6f}" for c in p) + f",{val:.10e}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder for the candidates (one subfolder per problem)")
    ap.add_argument("--only", choices=list(SRC), action="append")
    ap.add_argument("--levels", default="100,200,400,800,1600", help="L-shape grids N (h = 1/N), coarse to fine")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in a.only or list(SRC):
        print(name, flush=True)
        if name == "lshape":
            xyz, v, obs, vo, levels = lshape([int(s) for s in a.levels.split(",")])
            np.savez_compressed(os.path.join(a.out, "lshape_levels.npz"),
                                **{f"N{N}": lv[0] for N, lv in levels.items()},
                                **{f"obs_N{N}": lv[1] for N, lv in levels.items()})
        else:
            xyz, v, obs, vo = exact(name)
        write(os.path.join(a.out, *SRC[name]), name, xyz, v, obs, vo)


if __name__ == "__main__":
    main()

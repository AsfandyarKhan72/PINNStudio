#!/usr/bin/env python3
"""Numbers of the validation report (README.md next to this file): the regenerated reference solutions in
reference_data/ against the COMSOL files they replaced, and their convergence.

    python benchmarks/reference_regeneration/validate.py

The COMSOL files are read from the repository history (commit COMSOL_COMMIT, the last one with them), so this
needs a git checkout. convergence/ holds the solutions on each grid, from tools/make_burgers2d_reference_data.py
and tools/make_poisson_reference_data.py.
"""
import io
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REF = os.path.join(ROOT, "reference_data")
CAND = REF                      # the regenerated files
COMSOL_COMMIT = "1adc3446fd232dca3e8668420e4cea1516624780"


def comsol(*parts):
    rel_path = "/".join(["reference_data", *parts]).replace(os.sep, "/")
    data = subprocess.run(["git", "-C", ROOT, "show", f"{COMSOL_COMMIT}:{rel_path}"], check=True,
                          capture_output=True).stdout
    return io.StringIO(data.decode())
CONV = os.path.join(HERE, "convergence")
TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
NAMES = {0.0: "0", 0.25: "0.25", 0.5: "0.50", 0.75: "0.75", 1.0: "1.0"}


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def e(x):
    return f"{x:.2e}"


def poisson():
    print("## Poisson\n")
    print("| Problem | points | max abs(new − COMSOL) | relative L2 (new vs COMSOL) | u_obs: max abs difference |")
    print("|---|---|---|---|---|")
    for d in ("2D/poisson_disk", "2D/poisson_lshape", "3D/poisson_sphere"):
        a = np.genfromtxt(os.path.join(CAND, d, "solution.txt"))
        b = np.genfromtxt(comsol(d, "solution.txt"))
        assert np.array_equal(a[:, :-1], b[:, :-1]) and np.array_equal(np.isnan(a[:, -1]), np.isnan(b[:, -1]))
        m = np.isfinite(b[:, -1])
        oa = np.genfromtxt(os.path.join(CAND, d, "u_obs.txt"), delimiter=",", skip_header=1)
        ob = np.genfromtxt(comsol(d, "u_obs.txt"), delimiter=",", skip_header=1)
        assert np.allclose(oa[:, :-1], ob[:, :-1])
        print(f"| {d} | {m.sum()} | {e(np.abs(a[m, -1] - b[m, -1]).max())} | {e(rel(a[m, -1], b[m, -1]))} | "
              f"{e(np.abs(oa[:, -1] - ob[:, -1]).max())} |")
    L = np.load(os.path.join(CONV, "lshape_levels.npz"))
    b = np.genfromtxt(comsol("2D/poisson_lshape/solution.txt"))
    m = np.isfinite(b[:, -1])
    r = np.hypot(b[m, 0], b[m, 1])
    Ns = sorted(int(k[1:]) for k in L.files if k.startswith("N"))
    fin = L[f"N{Ns[-1]}"][m]
    print("\nL-shape, five-point finite differences, h = 1/N, on the 0.01 output grid:\n")
    print("| N | max abs(u_N − u_N/2) | max abs(u_N − u_finest) | relative L2 vs COMSOL | max abs vs COMSOL (r < 0.05) |"
          " max abs vs COMSOL (r ≥ 0.2) |")
    print("|---|---|---|---|---|---|")
    prev = None
    for N in Ns:
        v = L[f"N{N}"][m]
        d = "–" if prev is None else e(np.abs(v - prev).max())
        print(f"| {N} | {d} | {e(np.abs(v - fin).max())} | {e(rel(v, b[m, -1]))} | "
              f"{e(np.abs(v - b[m, -1])[r < 0.05].max())} | {e(np.abs(v - b[m, -1])[r >= 0.2].max())} |")
        prev = v
    print()


def burgers():
    print("## Burgers 2D (Mathias)\n")
    def key(d):                      # burgers_N<n>_dt<dt>.npz
        n, dt = d[len("burgers_N"):-4].split("_dt")
        return int(n), -float(dt)
    levels = sorted((d for d in os.listdir(CONV) if d.startswith("burgers_N")), key=key)
    G = {d: np.load(os.path.join(CONV, d)) for d in levels}
    fin = G[levels[-1]]
    print(f"Finest: {levels[-1][:-4]}.\n")
    com = {(c, t): np.loadtxt(comsol("2D/burgers_mathias", f"t_{NAMES[t]}_{c}.txt"))[:, 3].reshape(101, 101)
           for c in "uv" for t in TIMES}
    print("Relative L2 difference to the finest solution, per output time (u / v):\n")
    print("| solution | " + " | ".join(f"t = {t}" for t in TIMES[1:]) + " |")
    print("|---|" + "---|" * (len(TIMES) - 1))
    for d in levels:
        row = [f"{e(rel(G[d][f'u_{t}'], fin[f'u_{t}']))} / {e(rel(G[d][f'v_{t}'], fin[f'v_{t}']))}" for t in TIMES[1:]]
        print(f"| {d[:-4]} | " + " | ".join(row) + " |")
    row = [f"{e(rel(com[('u', t)], fin[f'u_{t}']))} / {e(rel(com[('v', t)], fin[f'v_{t}']))}" for t in TIMES[1:]]
    print("| **COMSOL** | " + " | ".join(row) + " |")
    print("\nCOMSOL against the finest solution: max abs difference and max abs of each (u / v):\n")
    print("| t | max abs diff u | max abs diff v | max abs u (new, COMSOL) | max abs v (new, COMSOL) |")
    print("|---|---|---|---|---|")
    for t in TIMES:
        print(f"| {t} | {e(np.abs(com[('u', t)] - fin[f'u_{t}']).max())} | {e(np.abs(com[('v', t)] - fin[f'v_{t}']).max())} | "
              f"{np.abs(fin[f'u_{t}']).max():.5f}, {np.abs(com[('u', t)]).max():.5f} | "
              f"{np.abs(fin[f'v_{t}']).max():.5f}, {np.abs(com[('v', t)]).max():.5f} |")
    allu = np.concatenate([fin[f"{c}_{t}"].ravel() for c in "uv" for t in TIMES[1:]])
    allc = np.concatenate([com[(c, t)].ravel() for c in "uv" for t in TIMES[1:]])
    print(f"\nAll times t > 0 and both outputs together: relative L2 (COMSOL vs new) {e(rel(allc, allu))}.\n")
    fd = os.path.join(CONV, "burgers_fd400_t0.25.npy")
    if os.path.exists(fd):
        F = np.load(fd)
        print("Independent check at t = 0.25 (second-order finite differences, h = 1/400, RK4): relative L2 to the "
              f"finest Chebyshev solution u {e(rel(F[0], fin['u_0.25']))}, v {e(rel(F[1], fin['v_0.25']))}; to COMSOL "
              f"u {e(rel(F[0], com[('u', 0.25)]))}, v {e(rel(F[1], com[('v', 0.25)]))}.\n")


if __name__ == "__main__":
    poisson()
    burgers()

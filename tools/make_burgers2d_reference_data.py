#!/usr/bin/env python3
"""Reference solution of the 2D Burgers template (Burgers 2D – Mathias et al.) without COMSOL.

The problem, as the template defines it (pinnstudio/ui/main_window.py, "2D Burgers (Mathias)"):

    u_t + u u_x + v u_y = nu (u_xx + u_yy)
    v_t + u v_x + v v_y = nu (v_xx + v_yy),          nu = 0.01 / pi,
    (x, y) in [0, 1]^2, t in [0, 1],
    u = v = 0 on the four sides,
    u(x, y, 0) = sin(2 pi x) sin(2 pi y),   v(x, y, 0) = sin(pi x) sin(pi y).

Method: Chebyshev collocation in x and y (the Gauss-Lobatto points x_j = (1 - cos(pi j / N)) / 2, j = 0..N; the
unknowns are the values at the interior points, the boundary values are zero), and the integrating-factor
fourth-order Runge-Kutta method in time: the diffusion term is integrated exactly with the matrix exponential
exp(nu dt/2 D2) applied along each axis (D2 the Chebyshev second-derivative matrix with the Dirichlet rows and
columns removed), the convection terms with RK4. The time step is fixed and hits the output times exactly. The
solution is written on the 101 x 101 grid of the reference files (x, y = 0, 0.01, ..., 1) by barycentric
(spectral) interpolation of the Chebyshev polynomial, not by grid interpolation.

(A sine series cannot be used here: with u and v extended oddly across the walls, v u_y and u v_x are even, not
odd, across them, so the extended problem is not the original one.)

    python tools/make_burgers2d_reference_data.py --n 256 --dt 2e-5 --out DIR
    python tools/make_burgers2d_reference_data.py --n 512 --dt 5e-6 --out DIR --device cuda   # with PyTorch

Writes DIR/t_0_u.txt, t_0_v.txt, t_0.25_u.txt, ..., t_1.0_v.txt in the format of reference_data/2D/burgers_mathias
(columns x y t value, x fastest) and DIR/grid.npz with the 101 x 101 values at each output time (for the
convergence study). Needs NumPy and SciPy (PyTorch for --device); deterministic for a given device.
"""
import argparse
import os
import time

import numpy as np
from scipy.linalg import expm

NU = 0.01 / np.pi
TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
NAMES = {0.0: "0", 0.25: "0.25", 0.5: "0.50", 0.75: "0.75", 1.0: "1.0"}     # the existing files' names
GRID = np.round(np.linspace(0.0, 1.0, 101), 10)


def cheb(N):
    """Gauss-Lobatto points on [0, 1] and the first-derivative matrix (Trefethen, Spectral Methods in MATLAB)."""
    j = np.arange(N + 1)
    xc = np.cos(np.pi * j / N)                          # on [-1, 1], from 1 to -1
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** j
    dX = xc[:, None] - xc[None, :]
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(1))
    x = (1.0 - xc) / 2.0                                # on [0, 1], from 0 to 1
    return x, -2.0 * D                                  # d/dx = -2 d/dxc


def interp_matrix(x, pts):
    """Barycentric interpolation from the Chebyshev points x (all N + 1) to pts."""
    N = len(x) - 1
    w = (-1.0) ** np.arange(N + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    d = pts[:, None] - x[None, :]
    hit = np.isclose(d, 0.0, atol=1e-14)
    d[hit] = 1.0
    P = w[None, :] / d
    P /= P.sum(1, keepdims=True)
    r, k = np.nonzero(hit)
    P[r] = 0.0
    P[r, k] = 1.0
    return P


class Burgers2D:
    def __init__(self, N, dt, device=None):
        x, D = cheb(N)
        self.x = x
        D1 = D[1:-1, 1:-1]                              # boundary values are zero: drop their columns
        D2 = (D @ D)[1:-1, 1:-1]
        E = expm(NU * dt / 2 * D2)                      # exp(nu dt/2 d_xx) on the interior values
        P = interp_matrix(x, GRID)[:, 1:-1]
        self.dt = dt
        if device:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)
            self.np = lambda a: a.cpu().numpy()
        else:
            T = lambda a: a
            self.np = lambda a: a
        self.T = T
        self.D1, self.D1t, self.E, self.Et, self.E2, self.E2t, self.P, self.Pt = (
            T(D1), T(D1.T), T(E), T(E.T), T(E @ E), T((E @ E).T), T(P), T(P.T))

    def rhs(self, u, v):
        """-(u u_x + v u_y), -(u v_x + v v_y); arrays rows y, columns x (interior points)."""
        ux, uy = u @ self.D1t, self.D1 @ u
        vx, vy = v @ self.D1t, self.D1 @ v
        return -(u * ux + v * uy), -(u * vx + v * vy)

    def half(self, f):                                  # exp(nu dt/2 Laplacian) f
        return self.E @ f @ self.Et

    def full(self, f):
        return self.E2 @ f @ self.E2t

    def step(self, u, v):
        dt, H = self.dt, self.half
        a1, b1 = self.rhs(u, v)
        a2, b2 = self.rhs(H(u + dt / 2 * a1), H(v + dt / 2 * b1))
        a3, b3 = self.rhs(H(u) + dt / 2 * a2, H(v) + dt / 2 * b2)
        a4, b4 = self.rhs(self.full(u) + dt * H(a3), self.full(v) + dt * H(b3))
        u = self.full(u) + dt / 6 * (self.full(a1) + 2 * H(a2 + a3) + a4)
        v = self.full(v) + dt / 6 * (self.full(b1) + 2 * H(b2 + b3) + b4)
        return u, v

    def on_grid(self, f):
        return self.np(self.P @ f @ self.Pt)


def solve(N, dt, out, device=None, log=print):
    b = Burgers2D(N, dt, device)
    xi = b.x[1:-1]
    X, Y = np.meshgrid(xi, xi)
    u = b.T(np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y))
    v = b.T(np.sin(np.pi * X) * np.sin(np.pi * Y))
    os.makedirs(out, exist_ok=True)
    grids, t, t0 = {}, 0.0, time.time()
    for target in TIMES:
        for _ in range(int(round((target - t) / dt))):
            u, v = b.step(u, v)
        t = target
        for name, f in (("u", u), ("v", v)):
            G = b.on_grid(f)
            G[0, :] = G[-1, :] = G[:, 0] = G[:, -1] = 0.0
            if not np.all(np.isfinite(G)):
                raise SystemExit(f"N={N} dt={dt:g}: the solution blew up before t={t} (time step too large?)")
            grids[f"{name}_{target}"] = G
            write(os.path.join(out, f"t_{NAMES[target]}_{name}.txt"), G, target)
        log(f"N={N} dt={dt:g}: t={t:.2f} after {time.time() - t0:.0f} s; max|u|={np.abs(grids[f'u_{target}']).max():.6f} "
            f"max|v|={np.abs(grids[f'v_{target}']).max():.6f}", flush=True)
    np.savez_compressed(os.path.join(out, "grid.npz"), N=N, dt=dt, **grids)


def write(path, G, t):
    """Columns x y t value, x fastest, as in reference_data/2D/burgers_mathias."""
    with open(path, "w") as f:
        for iy, y in enumerate(GRID):
            for ix, x in enumerate(GRID):
                f.write(f"{x:<15.4f} {y:<15.4f} {t:<15.5f} {G[iy, ix]:.10E}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=256, help="Chebyshev polynomial degree per axis")
    ap.add_argument("--dt", type=float, default=2e-5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, help="a PyTorch device (cuda) instead of NumPy")
    a = ap.parse_args()
    solve(a.n, a.dt, a.out, a.device)


if __name__ == "__main__":
    main()

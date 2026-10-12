#!/usr/bin/env python3
"""Generates the reference / measured data used by the DeepXDE Examples
Quick Examples (reference_data/deepxde/), with our own solvers -- no data
is copied from the DeepXDE repository (LGPL-2.1; PINNStudio is MIT).

Problems with a closed-form solution use the "Reference from formula"
Error Analysis option instead and need no files here; the DeepXDE Burgers
and L-shape examples reuse PINNStudio's existing reference_data/1D/burgers
and reference_data/2D/poisson_lshape (same problems).

Run from the repository root:
    python tools/make_deepxde_reference_data.py
"""
import os

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_data", "deepxde")


def _laplacian(n, dx):
    """Second-difference operator on the interior points (Dirichlet ends handled by the caller)."""
    from scipy.sparse import diags
    return diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / dx ** 2


def allen_cahn(n=2001):
    """u_t = 0.001 u_xx + 5(u - u^3), x in [-1, 1], u(+-1, t) = -1, u(x, 0) = x^2 cos(pi x)."""
    x = np.linspace(-1, 1, n)
    dx = x[1] - x[0]
    L = _laplacian(n - 2, dx).tocsr()
    bc = np.zeros(n - 2)
    bc[0] = bc[-1] = -1.0 / dx ** 2

    def rhs(_t, u):
        return 0.001 * (L @ u + bc) + 5.0 * (u - u ** 3)

    u0 = (x ** 2 * np.cos(np.pi * x))[1:-1]
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    sol = solve_ivp(rhs, (0, 1), u0, t_eval=times, method="BDF", rtol=1e-8, atol=1e-10,
                    jac=lambda _t, u: 0.001 * L + np.diag(5.0 * (1 - 3 * u ** 2)))
    xs = np.linspace(-1, 1, 201)
    out = {}
    for k, t in enumerate(times):
        full = np.concatenate([[-1.0], sol.y[:, k], [-1.0]])
        out[t] = np.interp(xs, x, full)
    return xs, out


def reaction(D=2.0, kf=0.1, n=801):
    """DeepXDE's reaction-diffusion inverse problem with its true parameters:
    ca_t = 1e-3 D ca_xx - kf ca cb^2,  cb_t = 1e-3 D cb_xx - 2 kf ca cb^2,
    x in [0, 1], t in [0, 10], ca = cb = 1 - x on the boundary, ca = cb = exp(-20 x) at t = 0."""
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    m = n - 2
    L = _laplacian(m, dx).tocsr()
    bc = np.zeros(m)
    bc[0] = 1.0 / dx ** 2      # value 1 at x = 0; value 0 at x = 1
    d = 1e-3 * D

    def rhs(_t, y):
        ca, cb = y[:m], y[m:]
        r = kf * ca * cb ** 2
        return np.concatenate([d * (L @ ca + bc) - r, d * (L @ cb + bc) - 2 * r])

    y0 = np.concatenate([np.exp(-20 * x[1:-1])] * 2)
    t = np.linspace(0, 10, 201)
    sol = solve_ivp(rhs, (0, 10), y0, t_eval=t, method="BDF", rtol=1e-8, atol=1e-10)
    xs = np.linspace(0, 1, 201)
    Ca = np.empty((len(t), len(xs)))
    Cb = np.empty_like(Ca)
    for k in range(len(t)):
        Ca[k] = np.interp(xs, x, np.concatenate([[1.0], sol.y[:m, k], [0.0]]))
        Cb[k] = np.interp(xs, x, np.concatenate([[1.0], sol.y[m:, k], [0.0]]))
    return xs, t, Ca, Cb


def reaction_rate_bvp():
    """DeepXDE's diffusion-reaction inverse problem with a space-dependent rate:
    0.01 u'' = k(x) u + sin(2 pi x), u(0) = u(1) = 0, k(x) = 0.1 + exp(-0.5 (x - 0.5)^2 / 0.15^2)."""
    l = 0.01
    k = lambda x: 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)
    xs = np.linspace(0, 1, 100)
    res = solve_bvp(lambda x, y: np.vstack((y[1], (k(x) * y[0] + np.sin(2 * np.pi * x)) / l)),
                    lambda ya, yb: np.array([ya[0], yb[0]]), xs, np.zeros((2, xs.size)), tol=1e-8, max_nodes=100000)
    assert res.success, res.message
    return xs, res.sol(xs)[0]


def _save(rel, data, header=""):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, data, fmt="%.8g", header=header)
    print("wrote", os.path.relpath(path, os.path.dirname(OUT)), data.shape)


def main():
    xs, snaps = allen_cahn()
    for t, u in snaps.items():
        _save(f"allen_cahn/t_{t:g}.txt", np.column_stack([xs, np.full_like(xs, t), u]))

    xs, t, Ca, Cb = reaction()
    X, T = np.meshgrid(xs, t)
    _save("reaction_inverse/ca_obs.txt", np.column_stack([X.ravel(), T.ravel(), Ca.ravel()]))
    _save("reaction_inverse/cb_obs.txt", np.column_stack([X.ravel(), T.ravel(), Cb.ravel()]))

    xs, u = reaction_rate_bvp()
    _save("diffusion_reaction_rate/u_obs.txt", np.column_stack([xs, u]))

    # Measured data generated from exact solutions (as DeepXDE's own scripts do):
    xo = np.linspace(-1, 1, 10)
    _save("diffusion_inverse/u_obs_t1.txt", np.column_stack([xo, np.ones_like(xo), np.sin(np.pi * xo) * np.exp(-1.0)]))
    xo = np.linspace(-1, 1, 100)
    _save("elliptic_inverse_field/u_obs.txt", np.column_stack([xo, np.sin(np.pi * xo)]))
    xo = np.linspace(-1, 1, 10000)   # the batched version samples 100 of these per step
    _save("elliptic_inverse_field/u_obs_10000.txt", np.column_stack([xo, np.sin(np.pi * xo)]))
    # Brinkman-Forchheimer: u = 1 - cosh(20 (x - 1/2)) / cosh(10) at 5 interior points.
    xo = np.linspace(1 / 6, 1, 5, endpoint=False)
    _save("brinkman_forchheimer/u_obs.txt", np.column_stack([xo, 1 - np.cosh(20 * (xo - 0.5)) / np.cosh(10)]))
    # Point Set Operator BC target: du/dx = 2(x + 1) = 4 at x = 1.
    _save("poisson_pointset_operator/dudx_at_x1.txt", np.array([[1.0, 4.0]]))


if __name__ == "__main__":
    main()

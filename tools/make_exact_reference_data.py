#!/usr/bin/env python3
"""
Rewrites PINNStudio's 1D reference data (burgers, heat, schrodinger,
allen_cahn in reference_data/1D) from exact or high-accuracy solutions, on
the same x grid, times and file names as before (the earlier files were
numerical exports 1-13% off, which put a floor under every Error Analysis
result).

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

Schrödinger 1D (Raissi et al. 2019; DeepXDE's Schrödinger example):
    i h_t + 0.5 h_xx + |h|^2 h = 0 on [-5, 5] (periodic), h(x, 0) = 2 sech(x);
    the files hold |h|. No closed form: computed with a Fourier pseudo-spectral
    method (1024 modes) and Strang split-step time stepping (dt = 1e-5),
    converged to ~1e-6, and evaluated on the files' grid by Fourier
    interpolation.
Allen-Cahn 1D (Wight & Zhao 2021): u_t = 1e-4 u_xx + 5u - 5u^3 on [-1, 1]
    (periodic), u(x, 0) = x^2 cos(pi x). Computed with a Fourier
    pseudo-spectral method (1024 modes) and ETDRK4 (dt = 1e-4), converged
    to ~1e-9.
The earlier Schrödinger and Allen-Cahn files were 5-13% and ~1% off.
Cahn-Hilliard 1D (Wight & Zhao 2021): u_t = (g2 (u^3 - u) - g1 u_xx)_xx on
    [-1, 1] (periodic), g2 = 0.01, g1 = 1e-6, u(x, 0) = -cos(2 pi x); written
    as u and the chemical potential mu = g2 (u^3 - u) - g1 u_xx
    (t_<T>_u.txt, t_<T>_mu.txt). Computed with a Fourier pseudo-spectral
    method (2048 modes) and a stabilized semi-implicit scheme (linear
    4th-order term implicit, nonlinear term explicit with a 2*g2 k^2
    stabilizer, dt = 2.5e-6); 1024 modes / dt = 1e-5 agree to <1e-4.

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


def _fourier_eval(coef, period, x0, x):
    """Values at points x of the trigonometric interpolant with FFT
    coefficients `coef` on [x0, x0 + period)."""
    n = len(coef)
    k = np.fft.fftfreq(n, d=1.0 / n)
    return np.real(np.exp(2j * np.pi * np.outer(np.asarray(x) - x0, k) / period) @ coef / n)


def schrodinger_abs_h(times, n=1024, dt=1e-5):
    """|h| at each time in `times` (ascending), as Fourier coefficients."""
    L, x0 = 10.0, -5.0
    x = x0 + L * np.arange(n) / n
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    h = (2 / np.cosh(x)).astype(complex)
    out, t = {}, 0.0
    for T in times:
        steps = int(round((T - t) / dt))
        if steps:
            d = (T - t) / steps
            half = np.exp(-0.5j * k ** 2 * d / 2)
            for _ in range(steps):
                h = np.fft.ifft(half * np.fft.fft(h))
                h = h * np.exp(1j * np.abs(h) ** 2 * d)
                h = np.fft.ifft(half * np.fft.fft(h))
        t = T
        # |h| is smooth, so interpolate it (not h) with its own Fourier series
        out[T] = np.fft.fft(np.abs(h))
    return out, L, x0


def allen_cahn_wz(times, n=1024, dt=1e-4):
    """Wight & Zhao's 1D Allen-Cahn at each time, as Fourier coefficients (ETDRK4)."""
    L, x0 = 2.0, -1.0
    x = x0 + L * np.arange(n) / n
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    lin = -1e-4 * k ** 2
    E, E2 = np.exp(dt * lin), np.exp(dt * lin / 2)
    r = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    LR = dt * lin[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

    def nl(v):
        u = np.real(np.fft.ifft(v))
        return np.fft.fft(5 * u - 5 * u ** 3)
    v = np.fft.fft(x ** 2 * np.cos(np.pi * x))
    out, t = {}, 0.0
    for T in times:
        for _ in range(int(round((T - t) / dt))):
            Nv = nl(v)
            a = E2 * v + Q * Nv
            Na = nl(a)
            b = E2 * v + Q * Na
            Nb = nl(b)
            c = E2 * a + Q * (2 * Nb - Nv)
            Nc = nl(c)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        t = T
        out[T] = v.copy()
    return out, L, x0


def cahn_hilliard_wz(times, n=2048, dt=2.5e-6, g1=1e-6, g2=0.01):
    """Wight & Zhao's 1D Cahn-Hilliard: {t: (u coefficients, mu coefficients)}."""
    L, x0 = 2.0, -1.0
    x = x0 + L * np.arange(n) / n
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    k2 = k * k
    S = 2 * g2
    den = 1 + dt * (g1 * k2 * k2 + S * k2)
    u = -np.cos(2 * np.pi * x)
    v = np.fft.fft(u)
    out, t = {}, 0.0

    def both(v):
        u = np.real(np.fft.ifft(v))
        return v.copy(), np.fft.fft(g2 * (u ** 3 - u)) + g1 * k2 * v
    for T in times:
        for _ in range(int(round((T - t) / dt))):
            u = np.real(np.fft.ifft(v))
            v = (v + dt * (-k2 * np.fft.fft(g2 * (u ** 3 - u)) + S * k2 * v)) / den
        t = T
        out[T] = both(v)
    return out, L, x0


CAHN_HILLIARD_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def write_cahn_hilliard():
    d = os.path.join(ROOT, "cahn_hilliard")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.startswith("t_") and f.endswith(".txt"):
            os.remove(os.path.join(d, f))
    coefs, L, x0 = cahn_hilliard_wz(CAHN_HILLIARD_TIMES)
    x = np.round(np.arange(-0.999, 1.0, 0.002), 6)
    for t in CAHN_HILLIARD_TIMES:
        for name, c in zip(("u", "mu"), coefs[t]):
            vals = _fourier_eval(c, L, x0, x)
            with open(os.path.join(d, f"t_{t:g}_{name}.txt"), "w") as fh:
                for xi, vi in zip(x, vals):
                    fh.write(f"{xi:.6f}\t{t:.4f}\t{vi:.12e}\n")
        print(f"cahn_hilliard/t_{t:g}_u.txt, _mu.txt: {len(x)} points")


_INITIAL = {"schrodinger": lambda x: 2 / np.cosh(x), "allen_cahn": lambda x: x ** 2 * np.cos(np.pi * x)}


def _spectral_set(folder, solver, files):
    times = sorted(set(files.values()))
    coefs, L, x0 = solver(times)
    # t = 0 is the initial condition itself (x^2 cos(pi x) has a kink across
    # the periodic boundary, which a Fourier series only approximates).
    return lambda x, t: _INITIAL[folder](np.asarray(x)) if t == 0 else _fourier_eval(coefs[t], L, x0, x)


# (folder, exact solution, x grid, {file name: t})
SETS = [
    ("burgers", burgers_exact, np.round(np.arange(-0.999, 1.0, 0.002), 6),
     {"t_0.txt": 0.0, "t_0.25.txt": 0.25, "t_0.50.txt": 0.5, "t_0.75.txt": 0.75, "t_1.0.txt": 1.0}),
    ("heat", heat_exact, np.round(np.arange(0.0005, 1.0, 0.001), 6),
     {"t_0.txt": 0.0, "t_0.25.txt": 0.25, "t_0.5.txt": 0.5, "t_0.75.txt": 0.75, "t_1.0.txt": 1.0}),
]


SPECTRAL = [
    ("schrodinger", schrodinger_abs_h, np.round(np.arange(-4.999, 5.0, 0.002), 6),
     {"t_0.txt": 0.0, "t_0.25.txt": 0.25, "t_0.50.txt": 0.5, "t_0.75.txt": 0.75, "t_1.0.txt": 1.0,
      "t_1.25.txt": 1.25, "t_1.50.txt": 1.5}),
    ("allen_cahn", allen_cahn_wz, np.round(np.arange(-0.999, 1.0, 0.002), 6),
     {"t_0.txt": 0.0, "t_0.25.txt": 0.25, "t_0.50.txt": 0.5, "t_0.75.txt": 0.75, "t_1.0.txt": 1.0}),
]


FISHER_KPP_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def fisher_kpp(times, n=4001, D=0.01, r=1.0):
    """Fisher-KPP u_t = D u_xx + r u (1 - u) on [0, 1], u_x = 0 at both ends, u(x, 0) = exp(-50 (x - 0.5)^2):
    the README's "Solving Your Own Problem" walkthrough. No closed form, so a fine finite-difference solve
    (n points, second order, ghost points for the no-flux ends) integrated with Radau to rtol 1e-11. Halving
    the grid spacing (2001 -> 4001 points) changes u by at most 3e-7."""
    from scipy.integrate import solve_ivp
    from scipy.sparse import diags
    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    up, lo = np.ones(n - 1), np.ones(n - 1)
    up[0] = lo[-1] = 2.0
    A = (diags([lo, -2.0 * np.ones(n), up], [-1, 0, 1]) * (D / h ** 2)).tocsr()
    sol = solve_ivp(lambda t, u: A @ u + r * u * (1 - u), (0.0, max(times)), np.exp(-50 * (x - 0.5) ** 2),
                    method="Radau", t_eval=list(times), jac=lambda t, u: A + diags(r * (1 - 2 * u)),
                    rtol=1e-11, atol=1e-13)
    return x, {t: sol.y[:, k] for k, t in enumerate(times)}


def write_fisher_kpp():
    d = os.path.join(ROOT, "fisher_kpp")
    os.makedirs(d, exist_ok=True)
    x, u = fisher_kpp(FISHER_KPP_TIMES)
    keep = slice(None, None, 4)                 # every 4th point: x = 0, 0.001, ..., 1
    for t in FISHER_KPP_TIMES:
        with open(os.path.join(d, f"t_{t:g}.txt"), "w") as fh:
            for xi, ui in zip(x[keep], u[t][keep]):
                fh.write(f"{xi:.6f}\t{t:.4f}\t{ui:.12e}\n")
        print(f"fisher_kpp/t_{t:g}.txt: {len(x[keep])} points")


def main():
    sets = list(SETS) + [(f, _spectral_set(f, solver, files), x, files) for f, solver, x, files in SPECTRAL]
    for folder, exact, x, files in sets:
        d = os.path.join(ROOT, folder)
        os.makedirs(d, exist_ok=True)
        for name, t in files.items():
            u = exact(x, t)
            with open(os.path.join(d, name), "w") as fh:
                for xi, ui in zip(x, u):
                    fh.write(f"{xi:.6f}\t{t:.4f}\t{ui:.12e}\n")
            print(f"{folder}/{name}: {len(x)} points, t = {t}")
    write_cahn_hilliard()
    write_fisher_kpp()


if __name__ == "__main__":
    main()

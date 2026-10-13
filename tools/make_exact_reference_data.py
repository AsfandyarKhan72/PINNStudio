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


if __name__ == "__main__":
    main()

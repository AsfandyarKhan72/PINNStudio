#!/usr/bin/env python3
"""
Reference data for the 2D Allen-Cahn template of Wight & Zhao (2021) (reference_data/2D/allen_cahn_wight/),
replacing files of unknown provenance (most likely a finite-element export).

    phi_t = 0.00625 lap phi - 10 (phi^3 - phi)  on [0, 1]^2 (periodic), t in [0, 10],
    phi(x, y, 0) = tanh((0.35 - sqrt((x - 0.5)^2 + (y - 0.5)^2 + 1e-6)) / 0.05)

(PINNStudio's template "2D Allen-Cahn (Wight & Zhao)": "dphi_t - 0.00625*(dphi_xx + dphi_yy) + 10*(phi**3 - phi)",
periodic on all sides; the circle shrinks and disappears just before t = 10.)

Method: Fourier pseudo-spectral in space, ETDRK4 in time (Kassam & Trefethen 2005) in float64, with the diffusion in
the linear part and the whole reaction term in the nonlinear part (putting its +10 phi in the linear part made the
scheme blow up once the circle disappears). The solution is written at the points of the existing files (101 x 101,
spacing 0.01) by evaluating its Fourier series there. Three solutions are computed -- N = 256, dt = 1e-3;
N = 256, dt = 5e-4; N = 384, dt = 1e-3 -- the differences printed (time step and grid), and the N = 256, dt = 5e-4
one written.

    python tools/make_allen_cahn_wight_reference_data.py [OUT_DIR]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reference_data", "2D", "allen_cahn_wight")
EPS2, C = 0.00625, 10.0
TIMES = [(0.0, "0"), (2.5, "2.5"), (5.0, "5.0"), (10.0, "10.0")]


def u0(X, Y):
    return np.tanh((0.35 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + 1e-6)) / (2 * 0.025))


def solve(N, dt):
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi * np.fft.fftfreq(N, 1 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    L = -EPS2 * (KX ** 2 + KY ** 2)
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    r = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    LR = dt * L[..., None] + r
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=-1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=-1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=-1))

    def g(v):
        u = np.real(np.fft.ifft2(v))
        return np.fft.fft2(-C * (u ** 3 - u))
    v, t, out = np.fft.fft2(u0(X, Y)), 0.0, {}
    for T, _ in TIMES:
        for _ in range(int(round((T - t) / dt))):
            Nv = g(v); a = E2 * v + Q * Nv; Na = g(a); b = E2 * v + Q * Na; Nb = g(b)
            c = E2 * a + Q * (2 * Nb - Nv); Nc = g(c)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        t = T
        out[T] = (k, v.copy())
    return out


def at(kv, px, py):
    k, V = kv
    N = len(k)
    return np.real(np.einsum("pi,ij,pj->p", np.exp(1j * np.outer(px, k)), V / N ** 2, np.exp(1j * np.outer(py, k))))


def main():
    g = np.round(np.linspace(0, 1, 101), 10)
    PX, PY = np.meshgrid(g, g, indexing="xy")             # x fastest, then y: the order of the existing files
    px, py = PX.ravel(), PY.ravel()
    a, b, c = solve(256, 1e-3), solve(256, 5e-4), solve(384, 1e-3)
    os.makedirs(OUT, exist_ok=True)
    for T, name in TIMES:
        ua, ub, uc = at(a[T], px, py), at(b[T], px, py), at(c[T], px, py)
        print(f"t = {T:<4g} max |dt 1e-3 - dt 5e-4| = {np.abs(ua - ub).max():.1e}   "
              f"max |N 256 - N 384| = {np.abs(ua - uc).max():.1e}", flush=True)
        np.savetxt(os.path.join(OUT, f"t_{name}.txt"), np.column_stack([px, py, np.full(px.size, T), ub]),
                   fmt="%.10g", delimiter="  ")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

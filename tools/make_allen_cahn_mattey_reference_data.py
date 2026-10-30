#!/usr/bin/env python3
"""
Reference data for the 2D Allen-Cahn template of Mattey & Ghosh (2022) (reference_data/2D/allen_cahn_mattey/),
replacing files of unknown provenance that differ from the exact initial condition by 2e-5 and from a converged
solution by 2.9e-4 (relative L2) at t = 1 -- about the error of the trained PINNs themselves.

    h_t = 1e-4 lap h - (h^3 - h)  on [0, 1]^2 (periodic), t in [0, 1],  h(x, y, 0) = sin(4 pi x) cos(4 pi y)

(PINNStudio's template: "dh_t - 0.0001*(dh_xx + dh_yy) + 1.0*(h**3 - h)", periodic on all sides.)

Method: Fourier pseudo-spectral in space, ETDRK4 in time (Kassam & Trefethen 2005, contour integral with 32
points), in float64. The solution is written at the points of the existing files (the 101 x 101 grid with
spacing 0.01, both ends included), by evaluating its Fourier series there -- exact for the discrete solution,
no interpolation error. Computed twice, on N = 256 with dt = 1e-3 and on N = 512 with dt = 2.5e-4; the largest
difference between the two is printed and must be below 1e-7 (it is about 1e-12), and the finer one is written.

Same file names and layout as before: t_0.txt, t_0.25.txt, t_0.5.txt, t_0.75.txt, t_1.0.txt, columns x y t h.

    python tools/make_allen_cahn_mattey_reference_data.py [OUT_DIR]     (default: reference_data/2D/allen_cahn_mattey)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reference_data", "2D", "allen_cahn_mattey")
EPS2, C = 1e-4, 1.0
TIMES = [(0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")]


def solve(N, dt):
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi * np.fft.fftfreq(N, 1 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    L = -EPS2 * (KX ** 2 + KY ** 2) + C
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    r = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    LR = dt * L[..., None] + r
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=-1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=-1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=-1))
    g = lambda v: -C * np.fft.fft2(np.real(np.fft.ifft2(v)) ** 3)
    v, t, out = np.fft.fft2(np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)), 0.0, {}
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
    PX, PY = np.meshgrid(g, g, indexing="xy")            # x fastest, then y: the order of the existing files
    px, py = PX.ravel(), PY.ravel()
    coarse, fine = solve(256, 1e-3), solve(512, 2.5e-4)
    os.makedirs(OUT, exist_ok=True)
    worst = 0.0
    for T, name in TIMES:
        hc, hf = at(coarse[T], px, py), at(fine[T], px, py)
        d = np.abs(hc - hf).max()
        worst = max(worst, d)
        print(f"t = {T:<5g} max |h(N=256, dt=1e-3) - h(N=512, dt=2.5e-4)| = {d:.1e}")
        np.savetxt(os.path.join(OUT, f"t_{name}.txt"), np.column_stack([px, py, np.full(px.size, T), hf]),
                   fmt="%.10g", delimiter="  ")
    assert worst < 1e-7, worst
    print("wrote", OUT)


if __name__ == "__main__":
    main()

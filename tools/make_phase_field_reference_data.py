#!/usr/bin/env python3
"""
Reference data for the 2D Cahn-Hilliard and 3D Allen-Cahn benchmarks of
Wight & Zhao (2021), "Solving Allen-Cahn and Cahn-Hilliard equations using
the adaptive physics informed neural networks", Commun. Comput. Phys. 29,
930-954 (arXiv:2007.04542), Sections 3.2.3 and 3.3.

2D Cahn-Hilliard (eqs. 3.20-3.21): phi_t = lambda * lap(-eps^2 lap phi + phi^3 - phi)
    on [-1, 1]^2 (periodic), lambda = 1, eps = 0.05, t in [0, 1],
    phi(x, y, 0) = max(tanh((r - R1)/(2 eps)), tanh((r - R2)/(2 eps))),
    r = 0.4, R1 = sqrt((x - 0.7 r)^2 + y^2), R2 = sqrt((x + 0.7 r)^2 + y^2)
    (two bubbles that merge) -- written as tanh((r - min(R1, R2))/(2 eps)) with
    the min smoothed, (R1 + R2 - sqrt((R1 - R2)^2 + 0.0004))/2, as in
    PINNStudio's template (see ch2d below). Written as phi (u) and the chemical potential
    mu = phi^3 - phi - eps^2 lap phi. Fourier pseudo-spectral in space, a
    stabilized semi-implicit step in time (the 4th-order term implicit, the
    nonlinear term explicit with a 2 k^2 stabilizer).
3D Allen-Cahn (eq. 3.13): phi_t = lambda * (eps^2 lap phi - (phi^3 - phi)) on
    [0, 1]^3 (periodic), lambda = 10, eps = 0.05, t in [0, 1],
    phi(x, y, z, 0) = tanh((0.35 - sqrt((x-.5)^2 + (y-.5)^2 + (z-.5)^2)) / (2 eps))
    (a shrinking sphere). Fourier pseudo-spectral in space, ETDRK4 in time.

Each solution is computed twice (finer grid / smaller time step) and the
difference printed, as a convergence check. Needs PyTorch; uses the GPU if
there is one (the 3D case is slow on a CPU).

Run from the repository root:
    python3 tools/make_phase_field_reference_data.py [ch2d] [ac3d]
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("REF_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_data")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64


def _wavenumbers(n, length, dim):
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    ks = np.meshgrid(*([k] * dim), indexing="ij")
    return torch.tensor(sum(kk ** 2 for kk in ks), dtype=DT, device=DEV)


def _grid(n, lo, length, dim):
    x = lo + length * np.arange(n) / n
    return [torch.tensor(g, dtype=DT, device=DEV) for g in np.meshgrid(*([x] * dim), indexing="ij")]


# ── 2D Cahn-Hilliard ────────────────────────────────────────────────────
CH_EPS, CH_LAM, CH_DELTA = 0.05, 1.0, 0.02
CH_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def ch2d(n, dt, times=CH_TIMES, S=2.0):
    """{t: (phi, mu)} on the n x n grid of [-1, 1)^2."""
    X, Y = _grid(n, -1.0, 2.0, 2)
    r = 0.4
    R1 = torch.sqrt((X - 0.7 * r) ** 2 + Y ** 2)
    R2 = torch.sqrt((X + 0.7 * r) ** 2 + Y ** 2)
    # max(tanh((r - R1)/2eps), tanh((r - R2)/2eps)) = tanh((r - min(R1, R2))/2eps);
    # the min is smoothed, (R1 + R2 - sqrt((R1 - R2)^2 + CH_DELTA^2))/2: the plain
    # min has a corner along x = 0 (the neck where the bubbles meet), whose
    # curvature a PINN can't see (autograd gives no second derivative in a
    # corner) -- with it the network learned an almost frozen solution. The
    # smoothed min differs by at most CH_DELTA/2 = 0.01, on x = 0 only (phi
    # changes by 0.5% relative L2, up to 0.1 where the neck crosses the interface).
    D = (R1 + R2 - torch.sqrt((R1 - R2) ** 2 + CH_DELTA ** 2)) / 2
    phi = torch.tanh((r - D) / (2 * CH_EPS))
    k2 = _wavenumbers(n, 2.0, 2)
    denom = 1 + dt * CH_LAM * (CH_EPS ** 2 * k2 ** 2 + S * k2)
    ph = torch.fft.fft2(phi)
    out, t, step = {}, 0.0, 0
    nsteps = {tt: int(round(tt / dt)) for tt in times}

    def both(ph):
        p = torch.fft.ifft2(ph).real
        mu = p ** 3 - p + CH_EPS ** 2 * torch.fft.ifft2(k2 * ph).real
        return p, mu
    for tt in sorted(times):
        while step < nsteps[tt]:
            p = torch.fft.ifft2(ph).real
            nl = torch.fft.fft2(p ** 3 - p)
            ph = (ph + dt * CH_LAM * (-k2 * nl + S * k2 * ph)) / denom
            step += 1
        out[tt] = both(ph)
    return out


def write_ch2d():
    t0 = time.time()
    fine = ch2d(400, 1e-6)
    ref = ch2d(200, 1e-6)
    coarse = ch2d(200, 2e-6)
    d = os.path.join(ROOT, "2D", "cahn_hilliard_wight")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.endswith(".txt"):
            os.remove(os.path.join(d, f))
    xs = np.round(-1.0 + 0.01 * np.arange(201), 6)            # the 200-point grid plus x = 1 (= -1, periodic)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    for tt in CH_TIMES:
        for j, name in enumerate(("u", "mu")):
            a = ref[tt][j].cpu().numpy()
            dn = np.abs(a - fine[tt][j].cpu().numpy()[::2, ::2]).max()
            dtm = np.abs(a - coarse[tt][j].cpu().numpy()).max()
            print(f"CH 2D t={tt:g} {name}: max |N200 - N400| = {dn:.2e}, max |dt - 2dt| = {dtm:.2e}, "
                  f"range [{a.min():.4f}, {a.max():.4f}]", flush=True)
            a = np.concatenate([a, a[:1]], 0)
            a = np.concatenate([a, a[:, :1]], 1)
            # x varies fastest, like the other 2D files
            rows = np.column_stack([X.T.ravel(), Y.T.ravel(), np.full(X.size, tt), a.T.ravel()])
            np.savetxt(os.path.join(d, f"t_{tt:g}_{name}.txt"), rows, fmt="%.10g", delimiter="  ")
        m = float(ref[tt][0].mean())
        print(f"   mean of phi (conserved): {m:.8f}", flush=True)
    print(f"CH 2D done in {time.time() - t0:.0f} s", flush=True)


# ── 3D Allen-Cahn ───────────────────────────────────────────────────────
AC_EPS, AC_LAM = 0.05, 10.0
AC_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def ac3d(n, dt, times=AC_TIMES):
    """{t: phi} on the n^3 grid of [0, 1)^3 (ETDRK4, Kassam & Trefethen)."""
    X, Y, Z = _grid(n, 0.0, 1.0, 3)
    phi = torch.tanh((0.35 - torch.sqrt((X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2)) / (2 * AC_EPS))
    k2 = _wavenumbers(n, 1.0, 3)
    L = AC_LAM * (-AC_EPS ** 2 * k2 + 1.0)                    # linear part: lambda (eps^2 lap + 1)
    E, E2 = torch.exp(dt * L), torch.exp(dt * L / 2)
    # phi-functions by contour integral (16 points), elementwise
    M = 16
    rts = torch.exp(1j * torch.pi * (torch.arange(1, M + 1, dtype=DT, device=DEV) - 0.5) / M)
    Q = torch.zeros_like(L); f1 = torch.zeros_like(L); f2 = torch.zeros_like(L); f3 = torch.zeros_like(L)
    for rr in rts:
        z = dt * L + rr
        Q += (dt * ((torch.exp(z / 2) - 1) / z)).real
        f1 += (dt * ((-4 - z + torch.exp(z) * (4 - 3 * z + z ** 2)) / z ** 3)).real
        f2 += (dt * ((2 + z + torch.exp(z) * (-2 + z)) / z ** 3)).real
        f3 += (dt * ((-4 - 3 * z - z ** 2 + torch.exp(z) * (4 - z)) / z ** 3)).real
    Q, f1, f2, f3 = Q / M, f1 / M, f2 / M, f3 / M

    def N(vh):
        v = torch.fft.ifftn(vh).real
        return torch.fft.fftn(-AC_LAM * v ** 3)
    vh = torch.fft.fftn(phi)
    out, step = {}, 0
    nsteps = {tt: int(round(tt / dt)) for tt in times}
    for tt in sorted(times):
        while step < nsteps[tt]:
            Nv = N(vh)
            a = E2 * vh + Q * Nv; Na = N(a)
            b = E2 * vh + Q * Na; Nb = N(b)
            c = E2 * a + Q * (2 * Nb - Nv); Nc = N(c)
            vh = E * vh + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
            step += 1
        out[tt] = torch.fft.ifftn(vh).real
    return out


def write_ac3d():
    t0 = time.time()
    ref = ac3d(100, 1e-4)
    fine = ac3d(200, 1e-4)
    coarse = ac3d(100, 2e-4)
    d = os.path.join(ROOT, "3D", "allen_cahn_wight")
    os.makedirs(d, exist_ok=True)
    xs = np.round(0.02 * np.arange(51), 6)                    # every 2nd point of the 100-grid, plus x = 1 (= 0)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    for tt in AC_TIMES:
        a = ref[tt].cpu().numpy()
        dn = np.abs(a - fine[tt].cpu().numpy()[::2, ::2, ::2]).max()
        dtm = np.abs(a - coarse[tt].cpu().numpy()).max()
        vol = float((ref[tt] > 0).double().mean())
        print(f"AC 3D t={tt:g}: max |N100 - N200| = {dn:.2e}, max |dt - 2dt| = {dtm:.2e}, "
              f"equivalent radius {(3 * vol / (4 * np.pi)) ** (1 / 3):.4f}", flush=True)
        a = a[::2, ::2, ::2]
        a = np.concatenate([a, a[:1]], 0); a = np.concatenate([a, a[:, :1]], 1); a = np.concatenate([a, a[:, :, :1]], 2)
        # x fastest, then y, then z, like reference_data/3D/heat
        rows = np.column_stack([X.transpose(2, 1, 0).ravel(), Y.transpose(2, 1, 0).ravel(),
                                Z.transpose(2, 1, 0).ravel(), np.full(X.size, tt), a.transpose(2, 1, 0).ravel()])
        np.savetxt(os.path.join(d, f"t_{tt:g}.txt"), rows, fmt="%.8g", delimiter="  ")
    print(f"AC 3D done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["ch2d", "ac3d"]
    print("device:", DEV, flush=True)
    if "ch2d" in which:
        write_ch2d()
    if "ac3d" in which:
        write_ac3d()

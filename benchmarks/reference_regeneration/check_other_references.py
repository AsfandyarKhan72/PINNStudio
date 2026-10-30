"""Independent checks of the four reference datasets that have no generator script in tools/:
reference_data/2D/heat, 3D/heat, 2D/allen_cahn_mattey and 2D/allen_cahn_wight.

2D and 3D Heat: u_t = 0.4 lap u on the unit square/cube, u = 0 at t = 0, u = 1 on x = 1, insulated elsewhere.
The solution depends on x and t only:
    u = 1 - sum_n 4 (-1)^n / ((2n+1) pi) cos((2n+1) pi x / 2) exp(-0.4 ((2n+1) pi / 2)^2 t).
2D Allen-Cahn: u_t = eps2 lap u - c (u^3 - u), periodic on [0, 1]^2, solved here with a Fourier pseudo-spectral
method and ETDRK4 time stepping (Kassam & Trefethen 2005) on N x N points, evaluated on the points of each file by
spectral interpolation.
    Mattey & Ghosh: eps2 = 1e-4, c = 1, u0 = sin(4 pi x) cos(4 pi y), t in [0, 1].
    Wight & Zhao:   eps2 = 0.00625, c = 10, u0 = tanh((0.35 - r) / 0.05), r = |(x, y) - (0.5, 0.5)|, t in [0, 10].

Usage: python benchmarks/reference_regeneration/check_other_references.py [heat|mattey|wight] [N] [dt]
Prints the relative L2 and the largest difference between each file and the independent solution.
Nothing is written.
"""
import glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(ROOT, "reference_data")


def heat_exact(x, t, k=0.4, terms=400):
    u = np.ones_like(x)
    for n in range(terms):
        m = (2 * n + 1) * np.pi / 2
        u -= 4 * (-1) ** n / ((2 * n + 1) * np.pi) * np.cos(m * x) * np.exp(-k * m * m * t)
    return u


def allen_cahn(eps2, c, u0, times, N, dt):
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi * np.fft.fftfreq(N, 1 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    L = -eps2 * (KX ** 2 + KY ** 2) + c                  # linear part, with the +c u of the reaction term
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    r = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    LR = dt * L[..., None] + r
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=-1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=-1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=-1))
    g = lambda v: -c * np.fft.fft2(np.real(np.fft.ifft2(v)) ** 3)
    v, t, out = np.fft.fft2(u0(X, Y)), 0.0, {}
    for T in sorted(times):
        for _ in range(int(round((T - t) / dt))):
            Nv = g(v); a = E2 * v + Q * Nv; Na = g(a); b = E2 * v + Q * Na; Nb = g(b)
            cc = E2 * a + Q * (2 * Nb - Nv); Nc = g(cc)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        t = T
        out[T] = v.copy()
    return k, out


def spectral_at(k, V, px, py):
    N = len(k)
    return np.real(np.einsum("pi,ij,pj->p", np.exp(1j * np.outer(px, k)), V / N ** 2, np.exp(1j * np.outer(py, k))))


def files(d):
    fs = [f for f in glob.glob(os.path.join(REF, d, "t_*.txt")) if "LessData" not in f]
    return sorted(fs, key=lambda f: float(os.path.basename(f)[2:-4]))


def report(name, t, u, e):
    print(f"{name:28s} t = {t:<5g} relative L2 {np.linalg.norm(u - e) / np.linalg.norm(e):.2e}   "
          f"max |diff| {np.abs(u - e).max():.2e}")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "heat"
    if what == "heat":
        for d in ("2D/heat", "3D/heat"):
            for f in files(d):
                t = float(os.path.basename(f)[2:-4])
                if t > 0:
                    A = np.loadtxt(f)
                    report(d, t, A[:, -1], heat_exact(A[:, 0], t))
        return
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    if what == "mattey":
        d, eps2, c, dt = "2D/allen_cahn_mattey", 1e-4, 1.0, 1e-3
        u0 = lambda X, Y: np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)
    else:
        d, eps2, c, dt = "2D/allen_cahn_wight", 0.00625, 10.0, 2e-4
        u0 = lambda X, Y: np.tanh((0.35 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + 1e-6)) / (2 * 0.025))
    dt = float(sys.argv[3]) if len(sys.argv) > 3 else dt
    fs = files(d)
    ts = [float(os.path.basename(f)[2:-4]) for f in fs]
    k, out = allen_cahn(eps2, c, u0, ts, N, dt)
    for f, t in zip(fs, ts):
        A = np.loadtxt(f)
        report(f"{d} (N={N}, dt={dt:g})", t, A[:, -1], spectral_at(k, out[t], A[:, 0], A[:, 1]))


if __name__ == "__main__":
    main()

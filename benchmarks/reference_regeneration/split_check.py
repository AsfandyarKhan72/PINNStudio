"""Independent check of the regenerated Allen-Cahn references: second-order Strang splitting (exact Fourier
diffusion half-steps, exact pointwise reaction step) instead of ETDRK4, compared with the committed files.

    u_t = eps2 lap u + c (u - u^3)  on [0, 1]^2 periodic.  Reaction u' = c (u - u^3) has the exact solution
    u(t) = u0 / sqrt(u0^2 + (1 - u0^2) exp(-2 c t)).

    python split_check.py REPO {mattey|wight} N DT
"""
import glob, os, sys
import numpy as np

repo, which, N, dt = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
if which == "mattey":
    eps2, c, folder = 1e-4, 1.0, "allen_cahn_mattey"
    u0f = lambda X, Y: np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)
else:
    eps2, c, folder = 0.00625, 10.0, "allen_cahn_wight"
    u0f = lambda X, Y: np.tanh((0.35 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + 1e-6)) / 0.05)
files = {}
for f in glob.glob(os.path.join(repo, "reference_data", "2D", folder, "t_*.txt")):
    t = float(os.path.basename(f)[2:-4])
    files[t] = np.loadtxt(f)
x = np.arange(N) / N
X, Y = np.meshgrid(x, x, indexing="ij")
k = 2 * np.pi * np.fft.fftfreq(N, 1 / N)
KX, KY = np.meshgrid(k, k, indexing="ij")
Dh = np.exp(-eps2 * (KX ** 2 + KY ** 2) * dt / 2)
E = np.exp(-2 * c * dt)
u = u0f(X, Y)


def at(u, px, py):
    V = np.fft.fft2(u) / N ** 2
    return np.real(np.einsum("pi,ij,pj->p", np.exp(1j * np.outer(px, k)), V, np.exp(1j * np.outer(py, k))))


t = 0.0
for T in sorted(files):
    for _ in range(int(round((T - t) / dt))):
        u = np.real(np.fft.ifft2(Dh * np.fft.fft2(u)))
        u = u / np.sqrt(u * u + (1 - u * u) * E)
        u = np.real(np.fft.ifft2(Dh * np.fft.fft2(u)))
    t = T
    a = files[T]
    p = at(u, a[:, 0], a[:, 1])
    print(f"{which} t = {T:g}: relative L2 (Strang N={N}, dt={dt:g}) vs committed file = "
          f"{np.linalg.norm(p - a[:, -1]) / np.linalg.norm(a[:, -1]):.2e}", flush=True)

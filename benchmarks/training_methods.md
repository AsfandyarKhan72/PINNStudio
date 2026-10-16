# Training methods study (October 2026)

Which published PINN training techniques actually help, tested with PINNStudio's own
templates on one GPU (RTX 4090). About 85 runs on 9 problems. Every number is the
relative L2 error of the main output against the reference data the template ships with
(PINNStudio's Error Analysis), at the template's own network and training settings unless
the row says otherwise.

**Run-to-run noise.** Changing only the random seed moves the error by about ±20%
(1D Cahn-Hilliard, three seeds), so only differences of about 1.5x or more are real.

**Terms used below**

| Term | Meaning | Where in PINNStudio |
|---|---|---|
| Scaled | each equation written so its residual is O(1) (see 1D Cahn-Hilliard) | the equations / Output transform |
| Periodic | input embedding `cos(pi*x); sin(pi*x); t` (2D: also in y) -- periodic by construction | Boundary conditions → Exact periodicity (or Network → Input transform → Formula) |
| Hard IC | output transform `IC(x) + t*u` -- the initial condition holds exactly | Initial conditions → Exact (or Network → Output transform → Formula) |
| RBA | residual-based attention, per-point weights (Anagnostopoulos et al., CMAME 2024) | Loss weights → Per-point weights |
| SA-PINN | self-adaptive per-point weights (McClenny & Braga-Neto, JCP 2023) | Loss weights → Per-point weights |
| keep / plain L-BFGS | per-point weights kept during L-BFGS, or L-BFGS on the plain loss (the default) | Loss weights → During L-BFGS |
| Causal | causal training (Wang, Sankaran & Perdikaris, CMAME 2024) | Loss weights → Causal training |
| MsFFN | multi-scale Fourier feature network, scales `1, 5` | Network → Network type |
| Time-adaptive | time-marching in windows (Wight & Zhao) | Adaptive training → Method |

## Summary

| Problem | Plain PINN | Best | How |
|---|---|---|---|
| 1D Cahn-Hilliard | 28–42% unscaled; 0.4–3% scaled | 0.11–0.6% | scaling, then MsFFN + RBA, or RBA + periodic + hard IC |
| 1D Allen-Cahn, no time windows | 10% (t = 1) | 0.27% (37x) | RBA + periodic + hard IC |
| 2D Allen-Cahn, Mattey & Ghosh (t ≤ 1) | 1.3% | 0.054% (25x) | RBA + periodic + hard IC (time-adaptive template: 0.67%) |
| 2D Allen-Cahn, Wight & Zhao (t ≤ 10) | 200% (fails) | 0.16–2.0% | time-adaptive + RBA + periodic (template alone 0.79–2.1%); best on the whole domain: RBA, 3.5–19% |
| Klein-Gordon | 8% | 0.3% (25x) | MsFFN; RBA or SA-PINN ~12x |
| Kovasznay flow | 0.02–0.09% | 0.008–0.04% (2x) | MsFFN |
| 1D Burgers | 0.04–0.13% | same | nothing helps; several hurt |
| Schrödinger 1D | 0.18% | same | nothing helps |
| Helmholtz 2D | 0.19% | same | nothing helps; MsFFN fails |

What worked: **scaling** the equations (the actual fix for Cahn-Hilliard); **hard IC +
periodic + RBA** on phase-field problems; **MsFFN** (large gains, but problem-dependent);
**RBA** with plain L-BFGS (helps hard problems, safe on easy ones); **time-adaptive**
training for long time ranges.
What did not: gradient-norm loss balancing (worse everywhere, removed); causal training
(never a reliable gain); SA-PINN (large gain on Klein-Gordon, unstable elsewhere);
keeping per-point weights during L-BFGS on already-accurate problems (2–10x worse);
separate u/mu networks; learning-rate decay.

## 1D Cahn-Hilliard (u, t = 0.25 / 1)

Wight & Zhao (2021), mixed form u_t = mu_xx, mu = g2 (u^3 - u) - g1 u_xx, g1 = 1e-6, g2 = 0.01,
on [-1, 1] x [0, 1], periodic, 20k Adam + 20k L-BFGS, no time windows.

| Method | t = 0.25 | t = 1 |
|---|---|---|
| Unscaled template | 28% | 42% |
| Unscaled + causal (eps 1 / 10 / 100) | 25–30% | 42–69% |
| Unscaled + gradient-norm balancing | 28% | 41% |
| **Scaled template** (3 seeds) | **2.2–3.3%** | **0.40–0.52%** |
| + causal eps 1 (32 / 16 chunks) | 2.7–4.7% | 0.43–0.93% |
| + gradient-norm balancing | 19% | 6.5% |
| + time-adaptive (4 windows) | 3.4% | 0.41% |
| + separate u/mu networks (PFNN) | 2.5% | 0.39% |
| + periodic | 2.0% | 0.46% |
| + SA-PINN, mean-1 (keep) | 1.8% | 0.40% |
| + RBA (keep) | 1.1% | 0.30% |
| + SA-PINN, original (keep) | 0.91% | 0.24% |
| + RBA + periodic (plain L-BFGS) | 1.2% | 0.26% |
| + RBA + periodic + LR decay (keep) | 1.1% | 0.29% |
| + RBA + periodic (keep) | 0.78% | 0.17% |
| + MsFFN (2 seeds) | 1.4–1.5% | 0.17–0.25% |
| + MsFFN + SA-PINN (keep) | 0.97% | 0.21% |
| + MsFFN + RBA (plain L-BFGS) | 1.1% | 0.11% |
| **+ RBA + periodic + hard IC (keep)** | **0.60%** | **0.12%** |
| **+ MsFFN + RBA (keep)** (2 seeds) | **0.54–0.57%** | **0.11–0.14%** |

Physics check (runs with the mass/energy check): mass conserved to ~1e-4 and the free
energy decreasing at all 40 time samples in every run; E(1) = 5.80e-4 against the
reference 5.797e-4.

## 1D Allen-Cahn (u, t = 0.25 / 1)

Template settings with time-adaptive training switched off unless stated.

| Method | t = 0.25 | t = 1 |
|---|---|---|
| **Plain** (2 seeds) | **0.69–0.85%** | **10%** |
| + gradient-norm balancing | 74% | 99% |
| + SA-PINN, original | ~100% (diverged) | ~100% |
| + causal eps 100 (plain L-BFGS) | 15% | 67% |
| + causal eps 1 | 1.1% | 13% |
| + causal eps 10 | 0.24% | 12% |
| + RBA (keep) | 1.0% | 11% |
| + periodic | 0.54% | 9.8% |
| + SA-PINN mean-1 + periodic | 0.29% | 6.1% |
| + RBA + periodic (plain L-BFGS) | 0.40% | 7.0% |
| + RBA + periodic + LR decay | 0.26% | 5.7% |
| + SA-PINN mean-1 | 0.22% | 4.0% |
| + RBA + periodic (keep) | 0.22% | 4.2% |
| + MsFFN + SA-PINN (keep) | 1.4% | 13% |
| + MsFFN (2 seeds) | 0.08–0.12% | 1.8–2.3% |
| + MsFFN + RBA (plain L-BFGS) | 0.14% | 2.5% |
| + MsFFN + RBA (keep) (2 seeds) | 0.051–0.055% | 0.74–0.89% |
| **+ RBA + periodic + hard IC (keep)** | **0.050%** | **0.27%** |
| Time-adaptive template (4 windows) | 0.05% | 0.48% |
| Time-adaptive + causal eps 1 / 10 | 0.07% | 0.57% / 0.81% |
| Time-adaptive + causal eps 100 | 25% | 86% |
| Time-adaptive + gradient-norm balancing | 0.18% | 1.4% |

## 2D Allen-Cahn, Mattey & Ghosh (u, t = 0.25 / 1)

| Method | t = 0.25 | t = 1 |
|---|---|---|
| Time-adaptive template (4 windows) | 0.56% | 0.67% |
| Whole domain, plain | 1.2% | 1.3% |
| + RBA (plain L-BFGS) | 1.6% | 1.7% |
| + MsFFN + RBA (keep) | 0.78% | 1.7% |
| **+ RBA + periodic + hard IC (keep)** | **0.034%** | **0.054%** |

## 2D Allen-Cahn, Wight & Zhao (u, t = 2.5 / 5 / 10)

| Method | t = 2.5 | t = 5 | t = 10 |
|---|---|---|---|
| **Time-adaptive template (10 windows)** | **0.93%** | **2.1%** | **0.79%** |
| Whole domain, plain | 177% | 187% | 200% |
| + MsFFN + RBA (keep) | 81% | 100% | 110% |
| + RBA + periodic + hard IC (keep) | 17% | 19% | 33% |
| + RBA (plain L-BFGS) | 3.5% | 5.7% | 19% |
| **Time-adaptive + RBA + periodic** | **0.92%** | **2.0%** | **0.16%** |

## Klein-Gordon (t = 2.5 / 5 / 7.5 / 10)

| Method | 2.5 | 5 | 7.5 | 10 |
|---|---|---|---|---|
| **Plain** | **1.6%** | **8.5%** | **6.5%** | **8.1%** |
| + RBA (keep) | 1.2% | 6.4% | 4.7% | 6.1% |
| + SA-PINN mean-1 (keep) | 0.15% | 0.55% | 0.41% | 0.61% |
| + RBA (plain L-BFGS) | 0.10% | 0.67% | 0.41% | 0.68% |
| **+ MsFFN** | **0.08%** | **0.43%** | **0.28%** | **0.31%** |

## 1D Burgers (t = 0.25 / 0.5 / 0.75 / 1)

| Method | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|
| **Plain** | **0.04%** | **0.07%** | **0.12%** | **0.13%** |
| + RBA (keep) | 0.30% | 1.1% | 1.6% | 1.0% |
| + SA-PINN (keep) | 0.22% | 0.76% | 2.0% | 1.3% |
| + MsFFN | 0.42% | 0.19% | 0.25% | 0.44% |
| + SA-PINN (plain L-BFGS) | 0.07% | 0.15% | 0.09% | 0.12% |
| + RBA (plain L-BFGS) | 0.05% | 0.04% | 0.08% | 0.05% |

## Steady and other problems

| Problem | Plain | + RBA (keep) | + RBA (plain L-BFGS) | + SA-PINN (keep) | + MsFFN |
|---|---|---|---|---|---|
| Schrödinger 1D (worst of 6 times) | 0.18% | 0.19% | – | 0.88% | – |
| Helmholtz 2D (Adam only) | 0.19% | 0.41% | 0.41% | 5.2% | ~100% |
| Kovasznay u / v / p | 0.021 / 0.090 / 0.054% | 0.054 / 0.23 / 0.13% | 0.016 / 0.086 / 0.048% | 0.032 / 0.12 / 0.059% | 0.0077 / 0.040 / 0.036% |

## Recommendations for a hard problem

1. Scale the equations so every residual is O(1).
2. Enforce what is known exactly: the initial condition (hard IC) and periodicity.
3. Add RBA per-point weights (L-BFGS on the plain loss unless the problem is clearly hard).
4. Try MsFFN for sharp or multi-scale solutions -- and compare, it can hurt.
5. For long time ranges, use time-adaptive training.
6. Always compare against a plain run: on problems a plain PINN already solves well,
   most of these add nothing or make the result worse.

## References

- **RBA** — Anagnostopoulos, Toscano, Stergiopulos & Karniadakis (2024), Residual-based attention in physics-informed neural networks, *CMAME* 421, 116805, [doi:10.1016/j.cma.2024.116805](https://doi.org/10.1016/j.cma.2024.116805), [arXiv:2307.00379](https://arxiv.org/abs/2307.00379)
- **SA-PINN** — McClenny & Braga-Neto (2023), Self-adaptive physics-informed neural networks, *J. Comput. Phys.* 474, 111722, [doi:10.1016/j.jcp.2022.111722](https://doi.org/10.1016/j.jcp.2022.111722), [arXiv:2009.04544](https://arxiv.org/abs/2009.04544)
- **Causal training** — Wang, Sankaran & Perdikaris (2024), Respecting causality for training physics-informed neural networks, *CMAME* 421, 116813, [doi:10.1016/j.cma.2024.116813](https://doi.org/10.1016/j.cma.2024.116813), [arXiv:2203.07404](https://arxiv.org/abs/2203.07404)
- **Gradient-norm loss balancing** (tested, removed) — Wang, Teng & Perdikaris (2021), Understanding and mitigating gradient flow pathologies in physics-informed neural networks, *SIAM J. Sci. Comput.* 43(5), A3055–A3081, [doi:10.1137/20M1318043](https://doi.org/10.1137/20M1318043)
- **Hard initial condition** — Lagaris, Likas & Fotiadis (1998), Artificial neural networks for solving ordinary and partial differential equations, *IEEE Trans. Neural Netw.* 9(5), 987–1000, [doi:10.1109/72.712178](https://doi.org/10.1109/72.712178)
- **Exact periodicity** — Dong & Ni (2021), A method for representing periodic functions and enforcing exactly periodic boundary conditions with deep neural networks, *J. Comput. Phys.* 435, 110242, [doi:10.1016/j.jcp.2021.110242](https://doi.org/10.1016/j.jcp.2021.110242)
- **MsFFN** — Wang, Wang & Perdikaris (2021), On the eigenvector bias of Fourier feature networks, *CMAME* 384, 113938, [doi:10.1016/j.cma.2021.113938](https://doi.org/10.1016/j.cma.2021.113938)
- **Time-adaptive training and the Allen-Cahn / Cahn-Hilliard problems** — Wight & Zhao (2021), *Commun. Comput. Phys.* 29(3), 930–954, [doi:10.4208/cicp.OA-2020-0086](https://doi.org/10.4208/cicp.OA-2020-0086); Mattey & Ghosh (2022), *CMAME* 390, 114474, [doi:10.1016/j.cma.2021.114474](https://doi.org/10.1016/j.cma.2021.114474)

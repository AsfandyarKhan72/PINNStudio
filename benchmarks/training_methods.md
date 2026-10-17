# Training methods study

*PINNStudio, October 2026*

Which published PINN training techniques actually help, tested on PINNStudio's own templates with one GPU (RTX 4090): about 150 runs on 14 problems. Every number is the relative $L_2$ error of the main output against the reference data the template ships with, as computed by PINNStudio's Error Analysis. Unless a row says otherwise, each run uses its template's own network and training budget.

**Run-to-run variation.** Changing only the random seed moves the error by about ±20% (1D Cahn-Hilliard, three seeds), so only differences of about 1.5× or more should be read as real.

**Reading the tables and figures.**

- A row starting with "+" adds that method to the setup named in the section's text (usually the template, or the row in bold above it).
- Per-point weights (RBA, SA-PINN) are held fixed during the L-BFGS phase, as in the papers. Rows marked *unweighted L-BFGS* run L-BFGS on the ordinary, unweighted loss instead. In the GUI this is Loss weights → *During L-BFGS*: *Keep the weights* or *Plain loss*.
- *Causal* means causal training with the paper's $\varepsilon$ schedule; a fixed $\varepsilon$ is written out.
- The figures show the error against time on a log scale. The dashed gray line is the reference setup (usually a plain PINN), and the thick line is the best one. $t = 0$ is left out because with an exact initial condition the error there is zero. Each figure is also available as a vector PDF in [`figures/pdf/`](figures/pdf/), and all of them are drawn from [`figures/runs.json`](figures/runs.json) by [`make_figures.py`](make_figures.py).

## Methods

| Method | What it does | Where in PINNStudio |
|---|---|---|
| Scaling | each equation written so that its residual is of order 1 (see 1D Cahn-Hilliard) | the equations, and Network → *Output transform* |
| Exact periodicity | the inputs enter as $\cos(2\pi x/L)$, $\sin(2\pi x/L)$, so the solution is periodic by construction ([Dong & Ni, 2021](https://doi.org/10.1016/j.jcp.2021.110242)) | Boundary conditions → *Exact periodicity* |
| Exact IC | the output is $u = \text{IC}(x) + \tau N$, so the initial condition holds exactly ([Lagaris et al., 1998](https://doi.org/10.1109/72.712178)) | Initial conditions → *Exact (built into the network)* |
| RBA | residual-based attention: per-point weights that grow where the residual stays large ([Anagnostopoulos et al., 2024](https://doi.org/10.1016/j.cma.2024.116805)) | Loss weights → *Per-point weights* |
| SA-PINN | self-adaptive per-point weights, trained by gradient ascent ([McClenny & Braga-Neto, 2023](https://doi.org/10.1016/j.jcp.2022.111722)) | Loss weights → *Per-point weights* |
| Causal training | later times count only once earlier ones are fitted ([Wang, Sankaran & Perdikaris, 2024](https://doi.org/10.1016/j.cma.2024.116813)) | Loss weights → *Causal training* |
| MsFFN | multi-scale Fourier feature network, scales $\sigma = 1, 5$ ([Wang, Wang & Perdikaris, 2021](https://doi.org/10.1016/j.cma.2021.113938)) | Network → *Network type* |
| Time-adaptive training | the time interval is solved in windows, each started from the previous one ([Wight & Zhao, 2021](https://doi.org/10.4208/cicp.OA-2020-0086)) | Adaptive training → *Method* |

## Summary

| Problem | Plain PINN | Best | Setup |
|---|---|---|---|
| 1D Cahn-Hilliard (Wight & Zhao) | 28–42% unscaled, 0.4–3% scaled | 0.05–0.06% at $t = 1$ | scaling, exact periodicity, exact IC, SA-PINN (the template) |
| 1D Allen-Cahn, no time windows | 10% at $t = 1$ | 0.065–0.13% at $t = 1$ | exact periodicity, exact IC, causal training, MsFFN |
| 2D Allen-Cahn (Mattey & Ghosh), $t \le 1$ | 1.3% | 0.054–0.057% (25× lower) | exact periodicity, exact IC, RBA or causal training (time-adaptive template: 0.67%) |
| 2D Allen-Cahn (Wight & Zhao), $t \le 10$ | 200% (fails) | 0.16–2.0% | time-adaptive training, RBA, exact periodicity (template alone: 0.79–2.1%) |
| 2D Cahn-Hilliard (Wight & Zhao) | 16–30% (frozen) | 1.2% / 3.8% at $t = 0.25$ / 1 | smoothed initial condition, exact periodicity, exact IC, SA-PINN (the template) |
| 1D Cahn-Hilliard (Mattey & Ghosh) | 84% at $t = 1$ | 0.29% / 0.48% at $t = 0$ / 1 | time-adaptive training in 19 windows, IC as a loss term, exact periodicity (the template); every whole-domain setup failed |
| 2D Cahn-Hilliard (Mattey & Ghosh) | 0.35% at $t' = 1$ | 0.20% at $t' = 1$ | exact periodicity, exact IC, SA-PINN (the template) |
| 3D Allen-Cahn (Wight & Zhao) | – | 0.03% | exact periodicity, IC as a loss term with weight 100 (the template); 0.7–0.8% with the exact IC, whatever the method |
| Klein-Gordon | 8% | 0.2–0.35% (25× lower) | MsFFN, or MsFFN with causal training |
| Kovasznay flow | 0.02–0.09% | 0.008–0.04% (2× lower) | MsFFN |
| 1D Burgers | 0.04–0.13% | same | nothing helps; several methods make it worse |
| Diffusion-reaction 1D | 1.4–11% | same | causal training makes late times worse |
| Schrödinger 1D | 0.18% | same | nothing helps |
| Helmholtz 2D | 0.19% | same | nothing helps; MsFFN fails |

**What worked.** Scaling the equations was the actual fix for Cahn-Hilliard. On the phase-field problems, exact periodicity and an exact initial condition came first, with one weighting method on top: SA-PINN on Cahn-Hilliard, causal training with an MsFFN network on Allen-Cahn, and RBA as a safe general choice. MsFFN gave large gains on some problems and hurt on others. Time-adaptive training helped on long time ranges and on fast early transients.

**What did not.** Gradient-norm loss balancing was worse everywhere and has been removed. Causal training with one fixed $\varepsilon$ barely acted, and causal training combined with per-point weights did worse than either alone. SA-PINN with normalized weights was unstable across seeds. Holding per-point weights during L-BFGS made already-accurate problems 2–10× worse. Separate networks for $u$ and $\mu$, and learning-rate decay, made no difference.

**Following the papers closely mattered.** The first versions of causal training and SA-PINN here left out the $\varepsilon$ schedule and normalized SA-PINN's weights, and both looked useless until they were implemented as published (see [Checked against the papers](#checked-against-the-papers)).

## 1D Cahn-Hilliard (Wight & Zhao)

Mixed form after [Wight & Zhao (2021)](https://doi.org/10.4208/cicp.OA-2020-0086): $u_t = \mu_{xx}$, $\mu = \gamma_2(u^3 - u) - \gamma_1 u_{xx}$ with $\gamma_1 = 10^{-6}$ and $\gamma_2 = 0.01$, on $x \in [-1, 1]$, $t \in [0, 1]$, periodic. Adam 20,000 → L-BFGS 20,000, no time windows.

![Relative L2 error against time for 1D Cahn-Hilliard](figures/ch1d.png)

**Figure 1.** 1D Cahn-Hilliard (Wight & Zhao), error in $u$. All runs except the dashed line use the scaled equations; runs starting with "+" also have exact periodicity and an exact initial condition.

| Method | $t = 0.25$ | $t = 1$ |
|---|---|---|
| Unscaled template | 28% | 42% |
| Unscaled + causal, fixed $\varepsilon$ = 1 / 10 / 100 | 25–30% | 42–69% |
| Unscaled + gradient-norm balancing | 28% | 41% |
| **Scaled template** (3 seeds) | **2.2–3.3%** | **0.40–0.52%** |
| + causal, fixed $\varepsilon = 1$ (32 / 16 chunks) | 2.7–4.7% | 0.43–0.93% |
| + gradient-norm balancing | 19% | 6.5% |
| + time-adaptive training (4 windows) | 3.4% | 0.41% |
| + separate $u$ and $\mu$ networks (PFNN) | 2.5% | 0.39% |
| + exact periodicity | 2.0% | 0.46% |
| + SA-PINN, normalized weights | 1.8% | 0.40% |
| + RBA | 1.1% | 0.30% |
| + SA-PINN (first implementation) | 0.91% | 0.24% |
| + RBA + exact periodicity, unweighted L-BFGS | 1.2% | 0.26% |
| + RBA + exact periodicity + learning-rate decay | 1.1% | 0.29% |
| + RBA + exact periodicity | 0.78% | 0.17% |
| + MsFFN (2 seeds) | 1.4–1.5% | 0.17–0.25% |
| + MsFFN + SA-PINN | 0.97% | 0.21% |
| + MsFFN + RBA, unweighted L-BFGS | 1.1% | 0.11% |
| **+ RBA + exact periodicity + exact IC** | **0.60%** | **0.12%** |
| **+ MsFFN + RBA** (2 seeds) | **0.54–0.57%** | **0.11–0.14%** |

A second round used the methods exactly as their papers describe them, all with exact periodicity and an exact initial condition:

| Method | $t = 0.25$ | $t = 1$ |
|---|---|---|
| + causal | 0.50% | 0.10% |
| + causal + MsFFN | 1.0% | 0.22% |
| + SA-PINN, normalized weights | 0.69% | 0.19% |
| **+ SA-PINN** (3 seeds) | **0.21–0.23%** | **0.051–0.060%** |

SA-PINN is now the template's setting; the error in $\mu$ is 0.25–0.28% at $t = 0.25$ and 0.15–0.18% at $t = 1$. In every run with the physics check, mass was conserved to about $10^{-4}$ and the free energy decreased at all 40 time samples; $E(1)$ = 5.80 × 10⁻⁴ against 5.797 × 10⁻⁴ in the reference.

## 1D Allen-Cahn (Wight & Zhao)

The template's settings with time-adaptive training switched off, unless the row says otherwise.

![Relative L2 error against time for 1D Allen-Cahn](figures/ac1d.png)

**Figure 2.** 1D Allen-Cahn, error in $u$ over the whole time interval (seed 2026). Runs starting with "+" add one method to exact periodicity and an exact initial condition.

| Method | $t = 0.25$ | $t = 1$ |
|---|---|---|
| **Plain** (2 seeds) | **0.69–0.85%** | **10%** |
| + gradient-norm balancing | 74% | 99% |
| + SA-PINN (first implementation) | ~100% (diverged) | ~100% |
| + causal, fixed $\varepsilon = 100$, unweighted L-BFGS | 15% | 67% |
| + causal, fixed $\varepsilon = 1$ | 1.1% | 13% |
| + causal, fixed $\varepsilon = 10$ | 0.24% | 12% |
| + RBA | 1.0% | 11% |
| + exact periodicity | 0.54% | 9.8% |
| + SA-PINN, normalized weights + exact periodicity | 0.29% | 6.1% |
| + RBA + exact periodicity, unweighted L-BFGS | 0.40% | 7.0% |
| + RBA + exact periodicity + learning-rate decay | 0.26% | 5.7% |
| + SA-PINN, normalized weights | 0.22% | 4.0% |
| + RBA + exact periodicity | 0.22% | 4.2% |
| + MsFFN + SA-PINN | 1.4% | 13% |
| + MsFFN (2 seeds) | 0.08–0.12% | 1.8–2.3% |
| + MsFFN + RBA, unweighted L-BFGS | 0.14% | 2.5% |
| + MsFFN + RBA (2 seeds) | 0.051–0.055% | 0.74–0.89% |
| **+ RBA + exact periodicity + exact IC** | **0.050%** | **0.27%** |

Second round, the methods as their papers describe them, three seeds each (2026 / 1 / 2), all with exact periodicity and an exact initial condition, no time windows:

| Method | $t = 0.25$ | $t = 1$ |
|---|---|---|
| Exact periodicity + exact IC only | 0.27% | 4.3% |
| + RBA | 0.053 / 0.11 / 0.070% | 0.43 / 0.78 / 0.50% |
| + causal, fixed $\varepsilon = 1$ | 0.061% | 0.65% |
| + causal | 0.049 / 0.056 / 0.056% | 0.18 / 0.64 / 0.45% |
| + causal + RBA | 0.18% | 1.8% |
| + SA-PINN, normalized weights | 0.053 / 0.31 / 0.14% | 0.10 / 2.7 / 1.05% |
| + SA-PINN | 0.046 / 0.061 / 0.049% | 0.18 / 0.20 / 0.13% |
| **+ causal + MsFFN** | **0.040 / 0.041 / 0.042%** | **0.13 / 0.12 / 0.065%** |

Causal training with the paper's schedule and a Fourier-feature network is the most accurate and the most repeatable here, while causal training together with per-point weights is worse than either alone. (The RBA row gave 0.27% at $t = 1$ in the first round, with an earlier version of the code; the current code gives 0.43% at seed 2026, identical to the digit in two repeats.)

With time-adaptive training, the template's own setting:

| Method | $t = 0.25$ | $t = 1$ |
|---|---|---|
| Time-adaptive template (4 windows) | 0.05% | 0.48% |
| + causal, fixed $\varepsilon$ = 1 / 10 | 0.07% | 0.57% / 0.81% |
| + causal, fixed $\varepsilon = 100$ | 25% | 86% |
| + causal | 0.034% | 0.48% |
| + gradient-norm balancing | 0.18% | 1.4% |

## 2D Allen-Cahn (Mattey & Ghosh)

IBVP-1 of [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474), $t \in [0, 1]$.

![Relative L2 error against time for 2D Allen-Cahn, Mattey & Ghosh](figures/ac2d_mg.png)

**Figure 3.** 2D Allen-Cahn (Mattey & Ghosh), error in $u$. All runs are on the whole time interval except the time-adaptive template.

| Method | $t = 0.25$ | $t = 1$ |
|---|---|---|
| Time-adaptive template (4 windows) | 0.56% | 0.67% |
| Whole domain, plain | 1.2% | 1.3% |
| + RBA, unweighted L-BFGS | 1.6% | 1.7% |
| + MsFFN + RBA | 0.78% | 1.7% |
| **+ RBA + exact periodicity + exact IC** | **0.034%** | **0.054%** |
| + causal + exact periodicity + exact IC | 0.037% | 0.057% |

## 2D Allen-Cahn (Wight & Zhao)

A shrinking circle over a long time range, $t \in [0, 10]$ ([Wight & Zhao, 2021](https://doi.org/10.4208/cicp.OA-2020-0086)).

![Relative L2 error against time for 2D Allen-Cahn, Wight & Zhao](figures/ac2d_wz.png)

**Figure 4.** 2D Allen-Cahn (Wight & Zhao), error in $u$. Runs not marked time-adaptive train on the whole time interval.

| Method | $t = 2.5$ | $t = 5$ | $t = 10$ |
|---|---|---|---|
| **Time-adaptive template (10 windows)** | **0.93%** | **2.1%** | **0.79%** |
| Whole domain, plain | 177% | 187% | 200% |
| + MsFFN + RBA | 81% | 100% | 110% |
| + RBA + exact periodicity + exact IC | 17% | 19% | 33% |
| + RBA, unweighted L-BFGS | 3.5% | 5.7% | 19% |
| **Time-adaptive + RBA + exact periodicity** | **0.92%** | **2.0%** | **0.16%** |

## 2D Cahn-Hilliard (Wight & Zhao)

Two bubbles that merge, $u_t = \nabla^2\mu$, $\mu = u^3 - u - 0.0025\nabla^2 u$ on $[-1, 1]^2$ (periodic), $t \in [0, 1]$; 4 × 128 network, Adam 20,000 → L-BFGS 20,000, exact periodicity and exact IC, no time windows. Leaving $u$ at its initial condition would give 16% at $t = 0.25$ and 30% at $t = 1$.

![Relative L2 error against time for 2D Cahn-Hilliard, Wight & Zhao](figures/ch2d_wz.png)

**Figure 5.** 2D Cahn-Hilliard (Wight & Zhao), error in $u$. "Smoothed IC" is the template's smoothed initial condition (see below).

| Method | $t = 0.25$ | $t = 1$ |
|---|---|---|
| Paper's initial condition, any of plain, RBA, causal, SA-PINN with normalized weights | 13–15% | 27–31% |
| Smoothed initial condition, plain | 16% | 30% |
| Smoothed + causal | 2.9% | 4.2% |
| **Smoothed + SA-PINN (the template)** | **1.2%** | **3.8%** |
| Smoothed + SA-PINN, 6 × 128 network (the paper's size) | 3.5% | 3.3% |
| Smoothed + SA-PINN, 40,000 Adam steps | 1.9% | 3.5% |
| Smoothed + SA-PINN, time-adaptive (5 windows of Adam 10,000 → L-BFGS 10,000), IC as a loss term | 0.42% | 1.7% |

A larger network or twice the Adam steps did not help, so the template keeps 4 × 128 and 20,000 + 20,000 steps. Time-adaptive training, the paper's own method here, is more accurate after the start (1.2–1.7% from $t = 0.5$ on), but it fits the initial condition only to 3.2% and takes about 25 minutes instead of 15. The template stays on the whole domain; time-adaptive training is one setting away (Adaptive training → *Time Adaptive Training*).

With the paper's initial condition every setup learned an almost frozen solution. That condition is a max of two tanh profiles, which has a corner where the bubbles touch, and the corner's curvature is what drives the merging; a network built on the exact initial condition sees no curvature in a corner, because automatic differentiation gives no second derivative there. The template writes the max as $\tanh\big((r - \min(R_1, R_2))/2\varepsilon\big)$ with a smoothed minimum, $\big(R_1 + R_2 - \sqrt{(R_1 - R_2)^2 + 0.0004}\big)/2$. This changes $u$ by at most 0.5%, along the line where the bubbles meet, and the reference data uses the same formula.

## 1D Cahn-Hilliard (Mattey & Ghosh)

$u_t = (u^3 - u - 0.02 u_{xx})_{xx}$ on $[-1, 1]$ (periodic), $u(x, 0) = \cos(\pi x) - \exp(-4(\pi x)^2)$, $t \in [0, 1]$ ([Mattey & Ghosh, 2022](https://doi.org/10.1016/j.cma.2021.114474), eq. 25). The narrow Gaussian in the initial condition makes the solution change very fast at first (it is reshaped well before $t = 0.05$) and then settle. The paper's own standard PINN fails here too, with 86% error over the whole domain; the authors solve it with their bc-PINN, which trains time segments one after another.

![Relative L2 error against time for 1D Cahn-Hilliard, Mattey & Ghosh](figures/ch1d_mg.png)

**Figure 6.** 1D Cahn-Hilliard (Mattey & Ghosh), error in $u$. Whole-domain runs use exact periodicity and an exact IC; time-adaptive runs use 19 windows with the IC as a loss term.

| Method | $t = 0.05$ | $t = 0.25$ | $t = 1$ |
|---|---|---|---|
| Whole domain, plain | 68% | 93% | 84% |
| + RBA | 71% | 81% | 90% |
| + causal | 66% | 93% | 77% |
| + causal + MsFFN | 57% | 90% | 25% |
| + SA-PINN | 93% | 91% | 51% |
| **Time-adaptive, 19 windows (the template)** | **0.35%** | **0.51%** | **0.48%** |
| Time-adaptive, 19 windows, + SA-PINN | 1.5% | 0.27% | 0.39% |

The whole-domain runs use Adam 20,000 → L-BFGS 20,000. Time-adaptive training plays the role of the paper's bc-PINN: 10 windows of 0.01 cover the fast start and 9 windows of 0.1 the rest, each started from the previous window's network, with Adam 5,000 → L-BFGS 5,000 per window and the initial condition (and each window's starting state) as a loss term with weight 100. SA-PINN is slightly better later on but fits the start 4× worse (9.4% at $t = 0$, against 0.29%), so the template uses plain windows. The relative error in $\mu$ is larger, 3–13% after the merge, because $\mu$ is nearly zero once the two phases have settled.

## 2D Cahn-Hilliard (Mattey & Ghosh)

Stripes that sharpen into two phases: $u_{t'} = 0.005\nabla^2\mu$, $\mu = u^3 - u - 0.0002\nabla^2 u$ on $[0, 1]^2$ (periodic), $u(x, y, 0) = 0.4\cos(4\pi x)\cos(4\pi(x + y))$, with time rescaled to $t' = t/0.005 \in [0, 1]$ (IBVP-2 of [Mattey & Ghosh, 2022](https://doi.org/10.1016/j.cma.2021.114474)). $\alpha = 0.0002$ reproduces the paper's figures, while the stated $\alpha = 0.02$ makes the pattern decay. 4 × 128 network, 20,000 points, Adam 20,000 → L-BFGS 20,000, exact periodicity and exact IC.

![Relative L2 error against time for 2D Cahn-Hilliard, Mattey & Ghosh](figures/ch2d_mg.png)

**Figure 7.** 2D Cahn-Hilliard (Mattey & Ghosh), error in $u$; all runs on the whole time interval.

| Method | $t' = 0.2$ | $t' = 0.6$ | $t' = 1$ |
|---|---|---|---|
| Plain | 0.87% | 0.64% | 0.35% |
| + RBA | 0.97% | 0.73% | 0.42% |
| + causal | 0.83% | 0.70% | 0.39% |
| **+ SA-PINN (the template)** | **0.52%** | **0.40%** | **0.20%** |

## 3D Allen-Cahn (Wight & Zhao)

A shrinking sphere, $u_t = 0.025\nabla^2 u - 10(u^3 - u)$ on $[0, 1]^3$ (periodic), $t \in [0, 1]$; 4 × 128 network, 30,000 points, Adam 20,000 → L-BFGS 20,000, exact periodicity.

| Method | $t = 0.25$ | $t = 1$ |
|---|---|---|
| Exact IC, $u = \text{IC} + tN$ | 0.81% | 0.72% |
| Exact IC + RBA | 0.81% | 0.72% |
| Exact IC + causal | 0.81% | 0.72% |
| Exact IC + SA-PINN | 0.81% | 0.72% |
| **IC as a loss term, weight 100 (the template)** | **0.030%** | **0.027%** |

With the exact initial condition every method lands on the same error to three digits (the logs confirm that RBA and the causal schedule were active), so the limit is the $tN$ form itself: the tanh profile relaxes quickly at the start, and the network has to produce that change through $tN$. Training the initial condition as a loss term instead gives 25 times less error. On the slice $z = 0.5$ the sphere stays round, and the largest pointwise error, about 0.013, sits on the interface.

## Klein-Gordon

DeepXDE's Klein-Gordon example, $t \in [0, 10]$.

![Relative L2 error against time for Klein-Gordon](figures/kg.png)

**Figure 8.** Klein-Gordon, error in $u$.

| Method | $t = 2.5$ | $t = 5$ | $t = 7.5$ | $t = 10$ |
|---|---|---|---|---|
| **Plain** | **1.6%** | **8.5%** | **6.5%** | **8.1%** |
| + RBA | 1.2% | 6.4% | 4.7% | 6.1% |
| + SA-PINN, normalized weights | 0.15% | 0.55% | 0.41% | 0.61% |
| + RBA, unweighted L-BFGS | 0.10% | 0.67% | 0.41% | 0.68% |
| **+ MsFFN** | **0.08%** | **0.43%** | **0.28%** | **0.31%** |
| + causal | 1.5% | 8.7% | 11% | 8.4% |
| + SA-PINN | 1.5% | 23% | 3.9% | 13% |
| **+ causal + MsFFN** | **0.08%** | **0.35%** | **0.24%** | **0.22%** |

## 1D Burgers

The Burgers example of [Raissi et al. (2019)](https://doi.org/10.1016/j.jcp.2018.10.045), a problem the plain PINN already solves well.

![Relative L2 error against time for 1D Burgers](figures/burgers.png)

**Figure 9.** 1D Burgers, error in $u$.

| Method | $t = 0.25$ | $t = 0.5$ | $t = 0.75$ | $t = 1$ |
|---|---|---|---|---|
| **Plain** | **0.04%** | **0.07%** | **0.12%** | **0.13%** |
| + RBA | 0.30% | 1.1% | 1.6% | 1.0% |
| + SA-PINN, normalized weights | 0.22% | 0.76% | 2.0% | 1.3% |
| + MsFFN | 0.42% | 0.19% | 0.25% | 0.44% |
| + SA-PINN, normalized weights, unweighted L-BFGS | 0.07% | 0.15% | 0.09% | 0.12% |
| + RBA, unweighted L-BFGS | 0.05% | 0.04% | 0.08% | 0.05% |
| + causal | 0.05% | 0.05% | 0.07% | 0.08% |
| + SA-PINN | 0.95% | 0.66% | 0.51% | 0.69% |

## Diffusion-reaction 1D

DeepXDE's diffusion-reaction example.

| Method | $t = 0.25$ | $t = 0.5$ | $t = 0.75$ | $t = 1$ |
|---|---|---|---|---|
| Plain | 1.4% | 3.4% | 6.7% | 11% |
| + causal | 0.98% | 2.5% | 10% | 32% |

## Steady and other problems

| Problem | Plain | + RBA | + RBA, unweighted L-BFGS | + SA-PINN, normalized weights | + MsFFN |
|---|---|---|---|---|---|
| Schrödinger 1D (worst of 6 times) | 0.18% | 0.19% | – | 0.88% | – |
| Helmholtz 2D (Adam only) | 0.19% | 0.41% | 0.41% | 5.2% | ~100% |
| Kovasznay flow, $u$ / $v$ / $p$ | 0.021 / 0.090 / 0.054% | 0.054 / 0.23 / 0.13% | 0.016 / 0.086 / 0.048% | 0.032 / 0.12 / 0.059% | 0.0077 / 0.040 / 0.036% |

## Recommendations for a hard problem

1. Scale the equations so that every residual is of order 1.
2. Enforce what is known exactly: periodicity, and the initial condition. Write the initial condition as a smooth formula, since a corner (a `max`, an `abs`) carries curvature the network cannot see. Then compare the exact initial condition with the initial condition as a loss term (weight 100): on 3D Allen-Cahn the loss term was 25 times more accurate.
3. Add one weighting method: RBA (cheap and safe), SA-PINN with its weights held during L-BFGS (best on Cahn-Hilliard), or causal training with the $\varepsilon$ schedule (best on Allen-Cahn, with MsFFN). Do not combine causal training with per-point weights.
4. Try MsFFN for sharp or multi-scale solutions, and compare: it can also hurt.
5. For long time ranges, or a very fast change right after the start, use time-adaptive training, with short windows over the fast part (as the 1D Cahn-Hilliard template after Mattey & Ghosh does).
6. Always compare against a plain run. On problems a plain PINN already solves well, most of these methods add nothing or make the result worse.

## Checked against the papers

Each method was compared, step by step, with the paper it comes from.

**RBA** ([Anagnostopoulos et al., 2024](https://doi.org/10.1016/j.cma.2024.116805)). The same update, $\lambda \leftarrow \gamma\lambda + \eta |r| / \max|r|$ (their eq. 12), with the same $\gamma = 0.999$ and $\eta = 0.01$, applied to the PDE residual only and with no gradient through $\lambda$. Differences: our weights start at 1 (theirs at 0), and we offer L-BFGS after Adam, where the paper uses Adam only. The paper's Allen-Cahn errors (5.2 × 10⁻³ with RBA alone, 5.8 × 10⁻⁵ with all of its improvements) come from 300,000 Adam steps on 25,600 points with a modified MLP; our runs use Adam 20,000 → L-BFGS 20,000, so the lower accuracy here reflects the budget, not the method.

**Exact periodicity** ([Dong & Ni, 2021](https://doi.org/10.1016/j.jcp.2021.110242)). The $C^\infty$ periodic layer: each first-layer neuron sees $\cos(\omega x)$ and $\sin(\omega x)$ with $\omega = 2\pi/L$, which spans the paper's $\sigma\big(A\cos(\omega x + \phi) + c\big)$. One difference: the paper builds a separate layer for each axis (its eq. 11), while ours feeds all axes' features into one shared first layer. The $C^k$ version (Hermite) is not implemented.

**Causal training** ([Wang, Sankaran & Perdikaris, 2024](https://doi.org/10.1016/j.cma.2024.116813)). The same weights, $w_i = \exp\big(-\varepsilon \sum_{k<i} \mathcal{L}_k\big)$, on the mean squared residual of each time chunk (their eq. 3.2), with no gradient through $w$. The first version here used one fixed $\varepsilon$; the paper's Algorithm 1 instead raises $\varepsilon$ through 0.01, 0.1, 1, 10 and 100, moving on once all weights are above 0.99, and this is now the default (*Raise ε step by step*). With a fixed $\varepsilon = 1$ the weights on 1D Allen-Cahn stayed at 0.98, so causal training barely acted, which is why it looked useless in the first round. Remaining differences: the paper also puts the initial-condition loss at the start of the chain (chunk 0), which does not matter when the initial condition is exact, and we hold the weights during L-BFGS, where the paper uses Adam only.

**SA-PINN** ([McClenny & Braga-Neto, 2023](https://doi.org/10.1016/j.jcp.2022.111722)). One trainable weight per point, trained by gradient ascent with Adam at a learning rate of $5 \times 10^{-3}$ (their Section 4.5), with the mask $\lambda^2$ (one of their example masks), and the weights held during L-BFGS (*Keep the weights*). Differences: the paper also weights the initial-condition points (starting them at $U(0, 100)$ for Allen-Cahn), while here only the PDE points are weighted, which again does not matter with an exact initial condition; our weights start at 1, where the paper draws them from $[0, 1]$; and the option *Keep the mean weight at 1* is ours, not the paper's (unticked is the paper's method).

## References

- **RBA**: S. J. Anagnostopoulos, J. D. Toscano, N. Stergiopulos and G. E. Karniadakis (2024). Residual-based attention in physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering* 421, 116805. [doi:10.1016/j.cma.2024.116805](https://doi.org/10.1016/j.cma.2024.116805), [arXiv:2307.00379](https://arxiv.org/abs/2307.00379)
- **SA-PINN**: L. D. McClenny and U. M. Braga-Neto (2023). Self-adaptive physics-informed neural networks. *Journal of Computational Physics* 474, 111722. [doi:10.1016/j.jcp.2022.111722](https://doi.org/10.1016/j.jcp.2022.111722), [arXiv:2009.04544](https://arxiv.org/abs/2009.04544)
- **Causal training**: S. Wang, S. Sankaran and P. Perdikaris (2024). Respecting causality for training physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering* 421, 116813. [doi:10.1016/j.cma.2024.116813](https://doi.org/10.1016/j.cma.2024.116813), [arXiv:2203.07404](https://arxiv.org/abs/2203.07404)
- **Gradient-norm loss balancing** (tested, removed): S. Wang, Y. Teng and P. Perdikaris (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing* 43(5), A3055–A3081. [doi:10.1137/20M1318043](https://doi.org/10.1137/20M1318043)
- **Exact initial condition**: I. E. Lagaris, A. Likas and D. I. Fotiadis (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks* 9(5), 987–1000. [doi:10.1109/72.712178](https://doi.org/10.1109/72.712178)
- **Exact periodicity**: S. Dong and N. Ni (2021). A method for representing periodic functions and enforcing exactly periodic boundary conditions with deep neural networks. *Journal of Computational Physics* 435, 110242. [doi:10.1016/j.jcp.2021.110242](https://doi.org/10.1016/j.jcp.2021.110242)
- **MsFFN**: S. Wang, H. Wang and P. Perdikaris (2021). On the eigenvector bias of Fourier feature networks: from regression to solving multi-scale PDEs with physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering* 384, 113938. [doi:10.1016/j.cma.2021.113938](https://doi.org/10.1016/j.cma.2021.113938)
- **Time-adaptive training, Allen-Cahn and Cahn-Hilliard problems**: C. L. Wight and J. Zhao (2021). Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive physics informed neural networks. *Communications in Computational Physics* 29(3), 930–954. [doi:10.4208/cicp.OA-2020-0086](https://doi.org/10.4208/cicp.OA-2020-0086)
- **Sequential training, Allen-Cahn and Cahn-Hilliard problems**: R. Mattey and S. Ghosh (2022). A novel sequential method to train physics informed neural networks for Allen-Cahn and Cahn-Hilliard equations. *Computer Methods in Applied Mechanics and Engineering* 390, 114474. [doi:10.1016/j.cma.2021.114474](https://doi.org/10.1016/j.cma.2021.114474)
- **1D Burgers**: M. Raissi, P. Perdikaris and G. E. Karniadakis (2019). Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics* 378, 686–707. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

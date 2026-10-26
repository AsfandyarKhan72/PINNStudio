# Comparison with the papers behind the templates

*PINNStudio, October 2026 — results below (single runs, seed 2026, one RTX 4090); the same table is kept up to date in the [benchmarks page](README.md)*

Twelve PINNStudio templates come from five papers. For each problem we compare four results on the paper's own error measure:

1. **Paper** — the error the paper reports.
2. **Base PINN** — the baseline the paper compares its method against (Wight & Zhao's "base PINN", Mattey & Ghosh's "std-PINN"): soft initial and boundary conditions, the whole time interval at once, equal loss weights, no adaptive technique, Adam then L-BFGS. It uses the network and points the paper gives for its baseline: 4×200 with 512 + 201 + 20,000 points and 100,000 Adam iterations for Mattey & Ghosh; 10,000 residual and 200 + 200 initial/boundary points for Wight & Zhao (who do not give its network: 4×128, the network of their next trial, in 1D and their 6×128 in 2D and 3D); for Raissi et al. the PINN of the paper is the base PINN, so the two rows are the same run; for Mathias et al. their 4×20 network with the conditions as loss terms.
3. **Paper's method in PINNStudio** — the paper's network, points, training scheme and optimizers, as far as the paper states them.
4. **PINNStudio best** — the best setup of the [training methods study](training_methods.md), tuned as in the [tuning study](tuning_study.md).

If (3) lands near (1), PINNStudio reproduces the paper; (2) and (4) then show what the paper's method adds over a base PINN and how PINNStudio's best setup compares with it.

## The problems

The templates' equations, coefficients, domains and initial conditions were checked against the papers: all match. Two notes: Mattey & Ghosh's 2D Cahn-Hilliard runs to t = 0.005 in the paper; the template rescales time to [0, 1] (t' = t/0.005), the same problem. The 1D Allen-Cahn problem (u_t = 10⁻⁴u_xx − 5(u³ − u), u₀ = x² cos πx, periodic) is in both Wight & Zhao and Mattey & Ghosh, so it is compared with both.

| Problem (template) | Paper | Paper's method | Reported error (relative L₂ over space and time) |
|---|---|---|---|
| Allen-Cahn 1D | Wight & Zhao (2021), Sec. 3.1 | 4×128 tanh, weighted loss (C₀ = 100), adaptive resampling | 2.33 × 10⁻² (base PINN 0.99) |
| | Mattey & Ghosh (2022), Sec. 5.1 | bc-PINN, 6×128, 4 segments of 0.25, ICGL + transfer learning | 1.68 × 10⁻² (std-PINN 0.99, XPINN 0.96) |
| Cahn-Hilliard 1D – Wight & Zhao | Wight & Zhao, Sec. 3.2 | 4×128, (u, μ), time-adaptive approach I + resampling | 9.51 × 10⁻³ |
| Cahn-Hilliard 1D – Mattey & Ghosh | Mattey & Ghosh, Sec. 5.2 | bc-PINN, 6×128, (h, μ), segments of 0.05–0.1, ICGL + TL | 1.86 × 10⁻² (std-PINN 0.86) |
| Allen-Cahn 2D – Wight & Zhao | Wight & Zhao, Sec. 3.1 | approach II (a network per time step Δt = 1), 6×128, C₀ = 10³, batch 32 | error plot only |
| Allen-Cahn 2D – Mattey & Ghosh | Mattey & Ghosh, Sec. 6.1 (IBVP-1) | bc-PINN, 6×128, 5 segments of 0.2, ICGL 20%, 15,000 Adam + 10,000 L-BFGS per segment | 2.5 × 10⁻² (> 95% without ICGL) |
| Cahn-Hilliard 2D – Wight & Zhao | Wight & Zhao, Sec. 3.2 | approach II, Δt = 0.2, 6×128, C₀ = 10³ | error plot only |
| Cahn-Hilliard 2D – Mattey & Ghosh | Mattey & Ghosh, Sec. 6.2 (IBVP-2) | bc-PINN, 5×128, 4 segments, transfer learning with frozen layers | figures only |
| Allen-Cahn 3D – Wight & Zhao | Wight & Zhao, Sec. 3.1 | approach II, Δt = 0.5, 6×128, C₀ = 10³ | figures only |
| Burgers 1D | Raissi et al. (2019), Sec. 2.1 | 8×20 tanh, 100 IC/BC points, 10,000 LHS points, L-BFGS only | 6.7 × 10⁻⁴ |
| Schrödinger 1D | Raissi et al., Sec. 2.2 | 4×100, 50 + 50 + 20,000 points, Adam 50,000 then L-BFGS (code) | 1.97 × 10⁻³ (of \|h\|) |
| Burgers 2D | Mathias et al. (2022) | hard IC/BC, 4×20 with skip connections, MAE loss, solution data points | figures only (absolute RMS) |

## Results

Each row runs in PINNStudio: the paper's **base PINN**, the **paper's method** set up as the paper describes it (`pinnstudio/core/paper_settings.py`, one comment per problem with the section and table it follows), and **PINNStudio tuned**, the best setup found in the [training methods](training_methods.md) and [tuning](tuning_study.md) studies. "Last time" is the relative L₂ error at the last reference time, "all times" the error over all reference snapshots together, the measure the papers report.

| Problem | Paper's method | Base PINN, L₂ (last time) | Paper's method in PINNStudio, L₂ (last time) | Paper's method in PINNStudio, L₂ (all times) | Reported in the paper (all times) | PINNStudio tuned, L₂ (last time) |
|---|---|---|---|---|---|---|
| Allen-Cahn 1D – Wight & Zhao | Wight & Zhao's method | 7.6 × 10<sup>-1</sup> | 4.6 × 10<sup>-2</sup> | 2.9 × 10<sup>-2</sup> | 2.3 × 10<sup>-2</sup> (Wight & Zhao (2021), Table 3.1) | **6.3 × 10<sup>-4</sup>** |
| Allen-Cahn 1D – Wight & Zhao | Mattey & Ghosh's bc-PINN | 7.6 × 10<sup>-1</sup> | 2.4 × 10<sup>-3</sup> | 1.9 × 10<sup>-3</sup> | 1.7 × 10<sup>-2</sup> (Mattey & Ghosh (2022), Table 2) | **6.3 × 10<sup>-4</sup>** |
| Cahn-Hilliard 1D – Wight & Zhao | Wight & Zhao's method | 3.5 × 10<sup>-3</sup> | 8.6 × 10<sup>-3</sup> | 4.3 × 10<sup>-2</sup> | 9.5 × 10<sup>-3</sup> (Wight & Zhao (2021), Sec. 3.2) | **6.0 × 10<sup>-4</sup>** |
| Cahn-Hilliard 1D – Mattey & Ghosh | Mattey & Ghosh's bc-PINN | 1.9 × 10<sup>-3</sup> | 5.7 × 10<sup>-3</sup> | 5.5 × 10<sup>-3</sup> | 1.9 × 10<sup>-2</sup> (Mattey & Ghosh (2022), Sec. 5.2) | **5.7 × 10<sup>-3</sup>** |
| Burgers 1D – Raissi et al. | Raissi et al.'s settings | 3.3 × 10<sup>-4</sup> | 3.3 × 10<sup>-4</sup> | 4.0 × 10<sup>-4</sup> | 6.7 × 10<sup>-4</sup> (Raissi et al. (2019), Sec. 2.1) | **3.3 × 10<sup>-4</sup>** |
| Allen-Cahn 2D – Wight & Zhao | Wight & Zhao's method | 1.0 × 10<sup>0</sup> | 2.8 × 10<sup>-3</sup> | 1.2 × 10<sup>-2</sup> | figures only (Wight & Zhao (2021), Fig. 3.8) | **1.6 × 10<sup>-3</sup>** |
| Allen-Cahn 2D – Mattey & Ghosh | Mattey & Ghosh's bc-PINN | 2.8 × 10<sup>-2</sup> | 2.5 × 10<sup>-3</sup> | 1.8 × 10<sup>-3</sup> | 2.5 × 10<sup>-2</sup> (Mattey & Ghosh (2022), Sec. 6.1.1) | **3.6 × 10<sup>-4</sup>** |
| Cahn-Hilliard 2D – Wight & Zhao | Wight & Zhao's method | 1.1 × 10<sup>-1</sup> | 2.4 × 10<sup>-1</sup> | 7.5 × 10<sup>-1</sup> | figures only (Wight & Zhao (2021), Fig. 3.13) | **1.1 × 10<sup>-2</sup>** |
| Cahn-Hilliard 2D – Mattey & Ghosh | Mattey & Ghosh's bc-PINN | 8.2 × 10<sup>-2</sup> | 2.0 × 10<sup>-2</sup> | 1.9 × 10<sup>-2</sup> | figures only (Mattey & Ghosh (2022), Sec. 6.2.1) | **1.1 × 10<sup>-3</sup>** |
| Burgers 2D – Mathias et al. | Mathias et al.'s settings | 1.6 × 10<sup>-1</sup> | 1.7 × 10<sup>-1</sup> | 1.8 × 10<sup>-1</sup> | figures only (Mathias et al. (2022)) | **8.4 × 10<sup>-3</sup>** |
| Allen-Cahn 3D – Wight & Zhao | Wight & Zhao's method | 9.9 × 10<sup>-1</sup> | 4.6 × 10<sup>-4</sup> | 8.1 × 10<sup>-4</sup> | figures only (Wight & Zhao (2021), Fig. 3.9) | **1.9 × 10<sup>-4</sup>** |

**Reading the table.** Where the paper reports a number, the paper's method run in PINNStudio lands close to it or below it (1D Allen-Cahn of Wight & Zhao 2.9% against their 2.3%; bc-PINN 0.19% against 1.7%, 0.55% against 1.9% and 0.18% against 2.5% on Mattey & Ghosh's three problems; Burgers 1D 0.040% against 0.067%). The exception is Wight & Zhao's 1D Cahn-Hilliard method (4.3% over all times against 0.95%), whose time-step, resampling schedule and iterations the paper does not give. PINNStudio's tuned settings are at least as accurate as the paper's method on every problem; on 2D Cahn-Hilliard of Wight & Zhao their method (approach II with mini-batches, run full-batch here) is worse than the base PINN.

## What PINNStudio needed

- **bc-PINN (Mattey & Ghosh)** — *added* (Adaptive training → time windows with *Backward compatibility (bc-PINN)* and *IC-guided learning*): one network on a growing time range, segment by segment; in each segment the residual and boundary terms on the new segment, the start of the segment against the previous solution, and a backward-compatibility term holding the network to its stored solution on the earlier segments. Their initial-condition guided learning (ICGL: first fit the segment's starting profile, constant in time) and transfer learning with frozen layers go with it.
- **Wight & Zhao's approach II** is PINNStudio's time-adaptive training (a network per time window, each starting from the previous one's prediction). Their **approach I** (one network on [0, tᵢ], tᵢ growing) is close to bc-PINN without the backward-compatibility term; *added* as *Growing time range*. Adaptive resampling and the weighted initial-condition loss exist.
- **Mini-batch training** (batch 32, 100 epochs, Wight & Zhao) — *not added*: PINNStudio trains on all points at once, so their runs are made full-batch (a difference to their method).
- **Mathias et al.**: the MAE loss and solution data points exist; the skip connections are not available in PINNStudio (their effect on a 4×20 network is small; noted as a difference).
- **The error measure**: the papers report one relative L₂ over space and time; PINNStudio's Error Analysis gives it per reference time. *Added*: the Training Log also prints the error over all reference times together ("All N reference times together"), used in the *all times* column.

## What the papers leave out

Each paper leaves some settings unstated (point counts for the 2D/3D examples, loss weights, L-BFGS settings, seeds, boundary conditions of some 2D examples). Where the authors' code exists (Raissi et al., Mattey & Ghosh's 1D Allen-Cahn, Mathias et al.) it fills these gaps; otherwise the choice made is listed with the result. Where paper and code disagree (e.g. Raissi's Schrödinger trains Adam first in the code, not in the paper) the code is followed and the difference noted.

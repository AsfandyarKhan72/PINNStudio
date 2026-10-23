# Comparison with the papers behind the templates

*PINNStudio, October 2026 — plan; results will be added as the runs finish*

Twelve PINNStudio templates come from five papers. For each problem we compare four results on the paper's own error measure:

1. **Paper** — the error the paper reports.
2. **Plain PINN** — soft initial and boundary conditions, the whole time interval at once, equal loss weights, Adam then L-BFGS.
3. **Paper's method in PINNStudio** — the paper's network, points, training scheme and optimizers, as far as the paper states them.
4. **PINNStudio best** — the best setup of the [training methods study](training_methods.md), tuned as in the [tuning study](tuning_study.md).

If (3) lands near (1), PINNStudio reproduces the paper; (2) and (4) then show what the paper's method adds over a plain PINN and how PINNStudio's best setup compares with it.

## The problems

The templates' equations, coefficients, domains and initial conditions were checked against the papers: all match. Two notes: Mattey & Ghosh's 2D Cahn-Hilliard runs to t = 0.005 in the paper; the template rescales time to [0, 1] (t' = t/0.005), the same problem. The 1D Allen-Cahn problem (u_t = 10⁻⁴u_xx − 5(u³ − u), u₀ = x² cos πx, periodic) is in both Wight & Zhao and Mattey & Ghosh, so it is compared with both.

| Problem (template) | Paper | Paper's method | Reported error (relative L₂ over space and time) |
|---|---|---|---|
| Allen-Cahn 1D | Wight & Zhao (2021), Sec. 3.1 | 4×128 tanh, weighted loss (C₀ = 100), adaptive resampling | 2.33 × 10⁻² (plain PINN 0.99) |
| | Mattey & Ghosh (2022), Sec. 5.1 | bc-PINN, 6×128, 4 segments of 0.25, ICGL + transfer learning | 1.68 × 10⁻² (plain PINN 0.99, XPINN 0.96) |
| Cahn-Hilliard 1D – Wight & Zhao | Wight & Zhao, Sec. 3.2 | 4×128, (u, μ), time-adaptive approach I + resampling | 9.51 × 10⁻³ |
| Cahn-Hilliard 1D – Mattey & Ghosh | Mattey & Ghosh, Sec. 5.2 | bc-PINN, 6×128, (h, μ), segments of 0.05–0.1, ICGL + TL | 1.86 × 10⁻² (plain PINN 0.86) |
| Allen-Cahn 2D – Wight & Zhao | Wight & Zhao, Sec. 3.1 | approach II (a network per time step Δt = 1), 6×128, C₀ = 10³, batch 32 | error plot only |
| Allen-Cahn 2D – Mattey & Ghosh | Mattey & Ghosh, Sec. 6.1 (IBVP-1) | bc-PINN, 6×128, 5 segments of 0.2, ICGL 20%, 15,000 Adam + 10,000 L-BFGS per segment | 2.5 × 10⁻² (> 95% without ICGL) |
| Cahn-Hilliard 2D – Wight & Zhao | Wight & Zhao, Sec. 3.2 | approach II, Δt = 0.2, 6×128, C₀ = 10³ | error plot only |
| Cahn-Hilliard 2D – Mattey & Ghosh | Mattey & Ghosh, Sec. 6.2 (IBVP-2) | bc-PINN, 5×128, 4 segments, transfer learning with frozen layers | figures only |
| Allen-Cahn 3D – Wight & Zhao | Wight & Zhao, Sec. 3.1 | approach II, Δt = 0.5, 6×128, C₀ = 10³ | figures only |
| Burgers 1D | Raissi et al. (2019), Sec. 2.1 | 8×20 tanh, 100 IC/BC points, 10,000 LHS points, L-BFGS only | 6.7 × 10⁻⁴ |
| Schrödinger 1D | Raissi et al., Sec. 2.2 | 4×100, 50 + 50 + 20,000 points, Adam 50,000 then L-BFGS (code) | 1.97 × 10⁻³ (of \|h\|) |
| Burgers 2D | Mathias et al. (2022) | hard IC/BC, 4×20 with skip connections, MAE loss, solution data points | figures only (absolute RMS) |

## What PINNStudio needs

- **bc-PINN (Mattey & Ghosh)** — new: one network on a growing time range, segment by segment; in each segment the residual and boundary terms on the new segment, the start of the segment against the previous solution, and a backward-compatibility term holding the network to its stored solution on the earlier segments. Their initial-condition guided learning (ICGL: first fit the segment's starting profile, constant in time) and transfer learning with frozen layers go with it.
- **Wight & Zhao's approach II** is PINNStudio's time-adaptive training (a network per time window, each starting from the previous one's prediction). Their **approach I** (one network on [0, tᵢ], tᵢ growing) is close to bc-PINN without the backward-compatibility term; it comes with the same work. Adaptive resampling and the weighted initial-condition loss exist.
- **Mini-batch training** (batch 32, 100 epochs, Wight & Zhao) — not in PINNStudio, which trains on all points at once. It can be added as a training option; otherwise their runs are made full-batch and the difference noted.
- **Mathias et al.**: the MAE loss and solution data points exist; the skip connections are not available in PINNStudio (their effect on a 4×20 network is small; noted as a difference).
- **The error measure**: the papers report one relative L₂ over space and time; PINNStudio's Error Analysis gives it per reference time. A space-time error over all reference snapshots will be added for this comparison.

## What the papers leave out

Each paper leaves some settings unstated (point counts for the 2D/3D examples, loss weights, L-BFGS settings, seeds, boundary conditions of some 2D examples). Where the authors' code exists (Raissi et al., Mattey & Ghosh's 1D Allen-Cahn, Mathias et al.) it fills these gaps; otherwise the choice made is listed with the result. Where paper and code disagree (e.g. Raissi's Schrödinger trains Adam first in the code, not in the paper) the code is followed and the difference noted.

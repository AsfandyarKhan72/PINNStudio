---
search:
  boost: 2
---

# Time windows and bc-PINN

## What it does

Time-adaptive training splits the time interval into consecutive windows and trains them one after
another; each window starts from the previous window's solution, which serves as its initial condition
(Wight & Zhao, [2021](../about/references.md#wight2021)). Options:

- **Transfer learning**: each window's network starts from the previous window's weights (otherwise each
  window trains a new network from random weights).
- **Growing time range** (Wight & Zhao's "approach I"): each window trains on the whole range so far,
  \( [t_{\text{start}}, t_1] \), with the true initial condition.
- **Backward compatibility** (bc-PINN, Mattey & Ghosh, [2022](../about/references.md#mattey2022)): the
  solution on the windows already trained is stored and later windows are held to it.
- **IC-guided learning**: each window first fits its starting profile (see
  [IC pre-training and guided learning](pretraining.md)).
- **Resampling rounds** (Wight & Zhao's approach I + resampling): after a window's training, points with
  large residual are added and training continues.

## When to use it

Long time intervals and fast-changing or stiff solutions (phase fields, fronts) that one network over the
whole interval follows poorly.

## How to enable it

**Training** → **Adaptive training** page → **Method:** **Time Adaptive Training** (greyed out for a
stationary problem and removed for an inverse problem). For an ODE the choice is offered but refused when you
solve (an ODE is solved as a problem in \( t \) without a separate time axis); before v1.6.0 it was switched off
silently. **About time-adaptive training** opens a description.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Time step groups:** rows **Start time:** → **End time:**, **Steps (n):** | Each group splits \( [t_s, t_e] \) into *n* equal windows; **➕ Add step group** adds a group | one group 0 → 1, 10 steps | steps 1–500 | `ta_step_groups` (JSON list of `{t_start, t_end, steps}`) |
| **IC grid resolution:** | Points per axis of the grid that carries the solution to the next window (squared in 2D, cubed in 3D) | 201 (21 when switching to 3D) | 201, 101, 51, 21, 11, or any integer typed in | `ta_grid_size` |
| **Enable transfer learning (warm start from previous step)** | Start each window from the previous window's weights | off | — | `ta_transfer_learning` |
| **Transfer optimizer:** | Which state is carried over: **Adam** (after the window's last Adam phase and its resampling rounds) or **L-BFGS** (as the window ended) | Adam | Adam, L-BFGS | `ta_transfer_optimizer` |
| **Growing time range: each window trains on [start, t] (one network)** | Approach I | off | needs transfer learning | `ta_growing` |
| **Backward compatibility (bc-PINN): keep the solution on earlier windows** | bc-PINN | off | needs transfer learning | `ta_backward` |
| **Weight:** | Loss weight of the backward-compatibility terms | 100 | 0–10⁶ | `ta_backward_weight` |
| **Times/window:** | Stored times per window | 10 | 1–1000 | `ta_backward_times` |
| **Points:** | Stored points in space (a random subset of the window grid when it has more) | 2000 | 1–1,000,000 | `ta_backward_points` |
| **IC-guided learning:** | Fraction of the first Adam phase's iterations | 0 (off) | 0–0.9 | `ta_icgl_fraction` |
| **Resampling rounds:** | Rounds per window (0 = off) | 0 | 0–50 | `ta_resample_rounds` |
| **Points per round:** | Points added per round | 200 | 1–100,000 | `ta_resample_points` |
| **Candidates per round:** | Random candidate points drawn in the window each round, ranked by residual | 10,000 | 100–1,000,000 (at least **Points per round:**) | `ta_resample_candidates` |
| **Adam iterations per round:** | Training after each round | 2000 | 1–1,000,000 | `ta_resample_iters` |

Configuration-only field (no control in the window): `ta_backward_seed`, the seed of bc-PINN's choice of
stored points (−1, the default: follow the run's seed; the benchmark configurations set 1234). A value
kept in an opened problem file is dropped when an example is loaded.

## Formulation

For window \( [t_0, t_1] \) (from the step groups):

1. **Domain.** Collocation and boundary points on \( [t_0, t_1] \), or \( [t_{\text{start}}, t_1] \) with a
   growing range. Every window builds its own `dde.data.TimePDE` with the **Collocation points** counts.
2. **Starting condition.** Window 1 (and every window with a growing range) uses the true initial
   condition. From window 2 on (no growing range), each output is held to the previous window's
   prediction \( \hat u(\mathbf{x}, t_0) \) on the IC grid (one `PointSetBC` per output; in 2D/3D these grid
   points are also anchors). On a disk, polygon or custom shape only the grid points in the domain are used.
3. **Network.** A new network; with transfer learning, the previous window's weights (and an MsFFN's Fourier
   matrices) are loaded into it, from memory.
4. **Backward compatibility.** After each window except the last, the newest network predicts the solution
   at up to **Points:** spatial points (a random subset of the window grid without replacement, drawn with a
   generator seeded by `ta_backward_seed`, else the run's seed, else unseeded) at the stored times — all
   earlier stored times plus **Times/window:** equally spaced times in the window (and \( t_1 \)). Later
   windows add, per output \( k \), the term
   \( w_{\mathrm{bc}}\,\ell\big(u_{\theta,k}(\mathbf{x}_s, t_s) - \hat u_k(\mathbf{x}_s, t_s)\big) \).
5. **IC-guided learning**, if on (see [pretraining](pretraining.md)).
6. **Training phases**, with the loss weights mapped onto this window's terms: the equation and boundary
   weights as set, each output's previous-solution term the weight of its initial condition (1 for an output
   without one), and the backward-compatibility terms **Weight:**.
7. **Resampling rounds**, if on: draw **Candidates per round:** random points in the window, rank them by
   \( \sum_k |r_k| \), add the **Points per round:** with the largest residual, and train for
   **Adam iterations per round:** with the optimizer, learning rate and loss function of the last
   first-order phase.
8. The window's prediction at \( t_1 \) on the grid becomes the next window's starting condition.

Each window saves its models under `time_adaptive_steps/step_NNN_t…_to_t…/` (and its `step_config.json`),
which Restore & visualize and Error Analysis read.

## Defaults and limitations

- **Resampling rounds and IC-guided learning use a constant learning rate.**
- IC-guided learning uses **10 time levels** per window.
- In 3D the IC grid is cubed: 101 per axis is over a million points per window; the page warns from 51 up
  and switches 201, 101 or 51 to 21 when the dimension changes to 3D.
- Time windows are not available for inverse problems.

!!! info "New in v1.6.0 (in development)"
    - Growing time range, backward compatibility (bc-PINN), IC-guided learning and resampling rounds are new.
    - Loss weights are mapped onto the later windows' terms (every weight used to become 1 from the second
      window on; this changes the Cahn-Hilliard time-window results).
    - Transfer learning keeps the network in memory (*Adam*: after the last Adam phase; *L-BFGS*: as the
      window ended); before, it depended on whether a save folder was set.
    - Only grid points inside a non-box domain carry the solution.
    - bc-PINN's stored points follow the run's seed (they used a fixed seed 1234).
    - IC-guided learning uses the first Adam phase's learning rate (with the mean squared error); resampling uses
      the optimizer, learning rate and loss function of the last first-order phase.
    - With resampling rounds, each window's network after the rounds is saved (`model_resampled-….pt`) and Error
      Analysis and Restore use it; before, they used the network saved before the rounds.
    - With resampling rounds and Transfer optimizer *Adam*, the next window starts from the network after the
      rounds; before, it started from the one before them while its initial condition came from the one after
      them (this changes the Cahn-Hilliard 1D – Wight & Zhao paper-method runs).
    - With these options, the exported script is Solve's own script (see [Generated scripts](scripts.md)).

## Supported problem types

Time-dependent 1D, 2D and 3D forward problems.

## Compatibility with other features

Refused by `validate()`:

> Time-adaptive training: Growing time range and Backward compatibility train one network through all the
> time windows, so they need 'transfer learning' (each window starts from the previous one's weights) --
> tick it, or untick ….

> Time-adaptive training: backward compatibility needs at least 1 time per window and 1 point.

> Time-adaptive training: resampling needs at least 1 point per round, at least as many candidates as
> points, and at least 1 Adam iteration per round.

> Time-adaptive training: resampling adds collocation points, which per-point weights (…) can't follow --
> set Per-point weights to None, or resampling rounds to 0.

> Exact initial condition can't be combined with time-adaptive training yet …

> 'Scale by initial losses' isn't available with Time-Adaptive training -- turn one of them off.

> frac(...) problems can't use RAR, Time-Adaptive training or IC pre-training …

Refused by the code generator: extra training points (anchors), and an Operator BC applied at the initial
time. Causal training and per-point weights (without resampling) are not refused; causal weights restart in
each window. Which of these combinations a shipped example or a test uses is on
[Compatibility of methods](compatibility.md).

## Common mistakes and troubleshooting

- **GPU out of memory in 3D**: lower **IC grid resolution:** (11 or 21).
- **The solution drifts window by window**: errors accumulate from window to window; use transfer learning
  with backward compatibility, more iterations per window, or fewer, longer windows.
- **"Sweep" of the number of windows**: each step group's **Steps (n):** can be swept (see
  [Sweeps](sweeps.md)).

## Example

**Allen-Cahn 2D – Wight & Zhao** with **Settings: Default** (10 windows, a new network per window),
**Cahn-Hilliard 1D – Wight & Zhao** with **Settings: Default** (growing range with resampling rounds),
**Cahn-Hilliard 1D – Mattey & Ghosh** (bc-PINN with transfer learning and IC-guided learning), and
**Allen-Cahn 2D – Mattey & Ghosh** and **Cahn-Hilliard 2D – Mattey & Ghosh** with **Settings: Default**
(bc-PINN) in **Examples**. See [Tutorial 12: Causal and time-window training](../tutorials/12-causal-windows.md).

## Related

- **Tutorial:** [12. Causal and time-window training](../tutorials/12-causal-windows.md)
- **Examples:** [Allen-Cahn 2D – Wight & Zhao](../examples/2d/allen-cahn-2d-wight-zhao.md), [Cahn-Hilliard 1D – Wight & Zhao](../examples/1d/cahn-hilliard-1d-wight-zhao.md), [Cahn-Hilliard 1D – Mattey & Ghosh](../examples/1d/cahn-hilliard-1d-mattey-ghosh.md), [Allen-Cahn 2D – Mattey & Ghosh](../examples/2d/allen-cahn-2d-mattey-ghosh.md), [Cahn-Hilliard 2D – Mattey & Ghosh](../examples/2d/cahn-hilliard-2d-mattey-ghosh.md); more in the gallery under [Advanced training strategies](../examples/index.md#advanced-training)
- **Theory:** [Time-dependent training](../theory/time-dependent.md)
- **Method references:** [Wight & Zhao 2021](../about/references.md#wight2021) (time windows, growing range, resampling); [Mattey & Ghosh 2022](../about/references.md#mattey2022) (bc-PINN)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/time-windows.md"

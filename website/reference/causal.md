---
search:
  boost: 2
---

# Causal training

## What it does

Causal training (Wang, Sankaran & Perdikaris, [2024](../about/references.md#wang2024causal)) splits the
time interval into chunks and weights each chunk's PDE residual by how well all *earlier* chunks are
already fitted, so later times only start to count once earlier times are solved. PINNStudio applies it to
the PDE residual at the training points, with the ε schedule of the paper's Algorithm 1 by default.

## When to use it

Time-dependent problems where a plain PINN converges to a wrong solution that violates causality
(Allen–Cahn, other stiff evolution problems). It acts on the residual loss itself, so it is most useful
when the residual term is the one that misleads training.

## How to enable it

**Training** → **Loss weights** page → **Automatic weighting** → **Causal training**. The options appear
under it.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Causal training** | Turn the method on | off | — | `causal_training` |
| **Time chunks:** | Number of equal time chunks \( M \) | 32 | 2–1000 | `causal_chunks` |
| **Raise ε step by step (0.01 → 100)** | Use the ε schedule (Algorithm 1) | on | — | `causal_anneal` |
| **Fixed ε:** | Causality parameter when the schedule is off (disabled while it is on) | 1.0 | 0.001–10⁶ | `causal_eps` |

## Formulation

The training points of the current training range \( [t_0, t_1] \) (the whole interval, or one time window)
are split into \( M \) equal chunks by their time. With \( L_k \) the mean squared residual (summed over the
equations) of the points in chunk \( k \), computed without gradient, chunk \( i \) gets the weight

\[
w_i = \exp\!\Big(-\varepsilon \sum_{k<i} L_k\Big), \qquad w_1 = 1,
\]

and each residual in chunk \( i \) is multiplied by \( \sqrt{w_i} \), so that the mean-squared PDE term
becomes \( \frac{1}{N}\sum_i w_i \sum_{n\in i} r(\mathbf{x}_n)^2 \).

**ε schedule** (box ticked): ε starts at 0.01 and moves to 0.1, 1, 10 and 100, each time once every chunk
weight is above 0.99, i.e. \( \min_i w_i > 0.99 \). It stays at 100 after that. With the box unticked, ε
is the **Fixed ε:** value throughout.

**L-BFGS**: the weights are held fixed at their last values (L-BFGS needs a fixed objective), and the
schedule does not advance.

The schedule starts over only for a new time range (the next time window); further training phases and
RAR rounds on the same range continue with it. Only the training points are weighted: the test points and
RAR or resampling candidates get the plain residual (the training points are recognised by a checksum of the
point set, refreshed before every step, so a point resampler's new points are weighted and a candidate set of the
same size is not). The Training Log prints, every 1000 iterations, the
current ε, the smallest weight and how far in time the weights have settled (all above 0.99).

## Defaults and limitations

- The schedule values (0.01, 0.1, 1, 10, 100) and the threshold 0.99 are fixed.
- The chunks are equal in time; a chunk with no training point has \( L_k = 0 \).
- With a fixed ε, the right value depends on the size of the residuals (the tool tip notes that on 1D
  Allen-Cahn a fixed ε = 1 left the weights near 0.98, so causal training barely acted).

## Supported problem types

Time-dependent 1D, 2D and 3D PDEs, forward and inverse; with RAR (script generated in the tests) and with time
windows (not refused, untested; see [Compatibility](compatibility.md)).

## Compatibility with other features

Refused by `validate()`:

> Causal training needs a time-dependent problem (it weights the residual by time).

> Causal training isn't available for an ODE (PINNStudio solves it as a problem in t without a separate
> time axis); turn it off.

> Causal training: the number of time chunks must be at least 2.

> Causal training: the causality parameter epsilon must be positive.

Combining it with per-point weights is allowed, but the page warns that the two reweight the same
residuals and together did worse than either alone in PINNStudio's benchmarks.

## Common mistakes and troubleshooting

- **The weights never settle** ("settled up to t = …" stays small): train longer, or start with a smaller
  fixed ε.
- **Later times are never trained**: ε is too large for the size of the residuals; use the schedule.

## Example

**Allen-Cahn 1D – Wight & Zhao** with **Settings: Tuned** in **Examples** (causal training with the
schedule, an MsFFN network, an exact initial condition and exact periodicity). See
[Tutorial 12: Causal and time-window training](../tutorials/12-causal-windows.md).

## Related

- **Tutorial:** [12. Causal and time-window training](../tutorials/12-causal-windows.md)
- **Examples:** [Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md)
- **Theory:** [Adaptive loss weighting › Causal training](../theory/loss-weighting.md#causal-training)
- **Method reference:** [Wang, Sankaran & Perdikaris 2024](../about/references.md#wang2024causal)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/causal.md"

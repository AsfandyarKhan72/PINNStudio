---
search:
  boost: 2
---

# Compatibility of methods

Most training methods and network options in PINNStudio can be switched on independently, so many combinations
are possible. This page records, for each combination, what is known about it: whether a shipped example or a
test uses it, whether PINNStudio refuses it, or whether nothing has been checked. It describes the development
version that will become v1.6.0.

The status of a combination is one of

| Status | Meaning |
|---|---|
| **S**, supported | A shipped example uses the combination (its gallery page is linked), or a test in `tests/` generates the training script for it and, where noted, trains a few iterations. |
| **R**, refused | `PINNConfig.validate()`, the code generator or the window blocks it. Solve, Export and every run of a sweep stop with a message that says what to change; the summaries are below. |
| **U**, untested | Not refused, but no example or test uses it. It may work; check the result yourself. |
| **—** | Not a combination: the two are alternatives of one setting. |

"Supported" means only that the combination runs as designed in the example or test named. It says nothing
about whether the combination improves a result on your problem; see
[Choosing a training strategy](../guide/choosing-a-strategy.md) for that, and
[Scope and limitations](../theory/limitations.md). Being able to tick two controls at the same time does not
make a combination supported. The *Supported problem types* sections of the other reference pages list what is
not refused; this page adds which of those cases have been exercised.

## Methods by problem type

| Method | ODE | Time-dependent PDE | Stationary PDE | Inverse problem | Non-box geometry |
|---|---|---|---|---|---|
| [Time windows](time-windows.md) | R | S (1D, 2D, 3D) | R | R | S (disk, test) |
| [Causal training](causal.md) | R | S (1D); 2D, 3D U | R | S | U |
| [RAR](rar.md) | U | S (1D); 2D, 3D U | S (test) | S (test) | U |
| [RBA](point-weights.md) | U | S (1D test, 2D); 3D U | U | U | U |
| [SA-PINN](point-weights.md) | U | S (1D, 2D); 3D U | U | S | U |
| [Exact initial condition](exact-constraints.md) | R | S (1D, 2D); 3D U | R | S | U |
| [Exact periodicity](exact-constraints.md) | R | S (1D, 2D, 3D) | U | S | R |
| [Output transform (formula)](networks.md) | S | S | S | S | S |
| [Input transform (features)](networks.md) | S | S | S | S | S |
| [MsFFN](networks.md) | U | S (1D) | S (1D) | S | U |
| [STMsFFN](networks.md) | R | S (1D) | R | S | U |
| [PFNN](networks.md) | U | U | S | S | U |
| [IC pre-training](pretraining.md) | R | S (tests, 1D and 2D) | R | U | U |
| [Scale by initial losses](loss-weights.md) | U | S (1D) | U | S | U |
| [`frac(...)` problems](equations.md) | R | S (1D) | S (1D, 2D, 3D) | S | S (disk, sphere) |
| [Point Resampler callback](optimizers.md) | U | S | S | S | U |
| [Point Set BC with a batch size](boundary-conditions.md) | U | U | S | U | U |
| [Inverse parameters and measured data](inverse.md) | S | S | S | — | S |

"Non-box geometry" means a Disk, Ellipse, Triangle, Polygon, Sphere, a combination of shapes (**Custom**) or a
**Sketch**; the box shapes are Interval, Rectangle and Cuboid.

### Evidence and refusal messages, by method

**Time windows.** Examples: [Cahn-Hilliard 1D – Mattey & Ghosh](../examples/1d/cahn-hilliard-1d-mattey-ghosh.md),
[Allen-Cahn 2D – Mattey & Ghosh](../examples/2d/allen-cahn-2d-mattey-ghosh.md),
[Allen-Cahn 3D – Wight & Zhao](../examples/3d/allen-cahn-3d-wight-zhao.md). Non-box: two tests in
`tests/test_audit_v160.py` generate time-window scripts on a disk
(`test_time_windows_on_a_disk_keep_only_grid_points_in_the_domain`,
`test_time_windows_on_a_disk_keep_the_domain_grid_after_the_first_window`).
Stationary: the window disables **Time Adaptive Training**, and `validate()` refuses a problem file that has
both ("Time-adaptive training needs a time-dependent problem"). Inverse: the window removes the choice in
**Inverse** mode, and `validate()` refuses it ("Time-adaptive training is available for forward problems only").
ODE: the window still offers the choice, and `validate()` refuses it (an ODE is solved as a problem in \( t \)
without a separate time axis, so there are no windows to step through). Before v1.6.0 the scripts switched time
windows off silently in the stationary and ODE cases.

**Causal training.** Examples: [Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md)
(Settings: Tuned, and Inverse mode). Refused for a stationary problem ("Causal training needs a time-dependent
problem") and for an ODE ("Causal training isn't available for an ODE").

**RAR.** Examples: [Burgers 1D – RAR](../examples/1d/burgers-1d-rar.md),
[Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md) (Settings: Default). Stationary:
`tests/test_steady_state_rar_gating.py` builds and runs a stationary RAR problem. Inverse:
`tests/test_lbfgs_steps.py::test_training_method_checks` generates RAR rounds that train the unknowns.

**RBA and SA-PINN.** Examples: [Allen-Cahn 2D – Wight & Zhao](../examples/2d/allen-cahn-2d-wight-zhao.md)
(RBA, Settings: Tuned), [Cahn-Hilliard 1D – Wight & Zhao](../examples/1d/cahn-hilliard-1d-wight-zhao.md) and
[Allen-Cahn 2D – Mattey & Ghosh](../examples/2d/allen-cahn-2d-mattey-ghosh.md) (SA-PINN, Settings: Tuned; the
latter also in Inverse mode). RBA in 1D: `tests/test_training_methods.py::test_generated_code_runs` trains a
small 1D heat problem with it.

**Exact initial condition.** Examples: [Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md)
(Inverse mode), [Cahn-Hilliard 1D – Wight & Zhao](../examples/1d/cahn-hilliard-1d-wight-zhao.md) and
[Cahn-Hilliard 2D – Mattey & Ghosh](../examples/2d/cahn-hilliard-2d-mattey-ghosh.md) (Settings: Tuned).
Refused on a stationary problem or an ODE ("needs a time-dependent PDE problem"), for an initial condition
loaded from a file, and for an initial condition that is not a formula in x, y, z.

**Exact periodicity.** Examples: the same phase-field examples, and
[Allen-Cahn 3D – Wight & Zhao](../examples/3d/allen-cahn-3d-wight-zhao.md) (Settings: Tuned, and Inverse mode). Refused for an ODE,
without a periodic boundary condition, and when an output has no periodic condition on a periodic axis
("Exact periodicity makes every output periodic"). Non-box: it needs a **Periodic BC** row, and the code
generator refuses a Periodic BC on any shape other than Interval, Rectangle or Cuboid ("Periodic BC requires an
Interval (1D), Rectangle (2D), or Cuboid (3D) geometry"), including combined shapes.

**Output and input transforms.** Examples: [Lotka-Volterra ODE](../examples/ode/lotka-volterra-ode.md) (both,
ODE), [Diffusion 1D – hard IC and BCs](../examples/1d/diffusion-1d-hard-ic-and-bcs.md) (output transform,
time-dependent), [Brinkman-Forchheimer 1D – unknown ν_e, K](../examples/1d/brinkman-forchheimer-1d-unknown-nu-e-k.md)
(output transform, inverse), [Wave 1D – space-time multi-scale Fourier features](../examples/1d/wave-1d-space-time-multi-scale-fourier-features.md)
(input features, also inverse), [Laplace 2D – disk](../examples/2d/laplace-2d-disk.md) (input features,
stationary, disk) and [Fractional Poisson 2D – disk](../examples/2d/fractional-poisson-2d-disk.md) (output
transform, disk).

**Fourier-feature networks.** Examples: [Poisson 1D – multi-scale Fourier features](../examples/1d/poisson-1d-multi-scale-fourier-features.md)
(MsFFN, stationary), [Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md) (MsFFN,
Inverse mode and Settings: Tuned), [Wave 1D – space-time multi-scale Fourier features](../examples/1d/wave-1d-space-time-multi-scale-fourier-features.md)
(STMsFFN, forward and inverse). STMsFFN is refused for an ODE ("STMsFFN needs space and time") and for a
stationary problem. Both need an even number of neurons (at least 2) in the first hidden layer, and refuse a
weight decay greater than 0.

**PFNN.** Examples: [Linear elasticity 2D – plate](../examples/2d/linear-elasticity-2d-plate.md) (also in
Inverse mode), [Diffusion-reaction 1D – unknown rate field](../examples/1d/diffusion-reaction-1d-unknown-rate-field.md),
[Poisson 1D – unknown forcing field (batches)](../examples/1d/poisson-1d-unknown-forcing-field-batches.md); all
are stationary. Refuses a weight decay greater than 0.

**IC pre-training.** No shipped example uses it. `tests/test_round47_ic_pretrain_shared_block.py` trains and
restores a pre-training on a 1D time-dependent problem, with and without time windows; `tests/test_audit_v160.py`
generates it together with time windows on a 2D problem and with scaling by the initial losses. Refused on a stationary problem or an ODE,
and when no output has its initial condition as a loss term ("an exact initial condition needs no
pre-training").

**Scale by initial losses.** Example: [Wave 1D – space-time multi-scale Fourier features](../examples/1d/wave-1d-space-time-multi-scale-fourier-features.md)
(forward and inverse). Refused with time windows ("'Scale by initial losses' isn't available with
Time-Adaptive training").

**`frac(...)` problems.** Examples: [Fractional diffusion 1D](../examples/1d/fractional-diffusion-1d.md)
(time-dependent), [Fractional Poisson 1D – unknown order α](../examples/1d/fractional-poisson-1d-unknown-order-alpha.md)
(inverse), [Fractional Poisson 2D – disk](../examples/2d/fractional-poisson-2d-disk.md),
[Fractional Poisson 3D – sphere](../examples/3d/fractional-poisson-3d-sphere.md). Refused for an ODE
("frac(...) (fractional operator) needs a 1D, 2D or 3D problem"); a static fractional mesh is refused except
on a 1D Interval.

**Point Resampler callback.** Examples: [Diffusion 1D – resampling](../examples/1d/diffusion-1d-resampling.md),
[Heat 1D – resampling](../examples/1d/heat-1d-resampling.md), [Black-Scholes 1D – American put](../examples/1d/black-scholes-1d-american-put.md),
[Poisson 1D – multi-scale Fourier features](../examples/1d/poisson-1d-multi-scale-fourier-features.md)
(stationary), [Wave 1D – space-time multi-scale Fourier features](../examples/1d/wave-1d-space-time-multi-scale-fourier-features.md)
(forward and inverse).

**Point Set BC with a batch size.** Example: [Poisson 1D – unknown forcing field (batches)](../examples/1d/poisson-1d-unknown-forcing-field-batches.md)
(stationary; it learns an unknown field as a second output and runs as a forward problem).

**Inverse problems.** Examples on every problem type, for instance
[Lorenz ODE – unknown C₁, C₂, C₃](../examples/ode/lorenz-ode-unknown-c1-c2-c3.md),
[Euler beam 1D](../examples/1d/euler-beam-1d.md) (stationary) and [Poisson 2D – L-shape](../examples/2d/poisson-2d-l-shape.md)
(polygon), in Inverse mode. An inverse problem needs at least one measured-data file.

## Method pairs

Abbreviations: **TW** time windows, **CA** causal training, **RAR**, **PW** per-point weights (RBA or SA-PINN),
**EIC** exact initial condition, **EP** exact periodicity, **OT** output transform formula, **IT** input
transform, **FF** Fourier-feature network (MsFFN or STMsFFN), **PF** PFNN, **IP** IC pre-training, **SI** scale
by initial losses, **FR** `frac(...)` problem, **RS** Point Resampler callback, **BP** Point Set BC with a
batch size.

| | TW | CA | RAR | PW | EIC | EP | OT | IT | FF | PF | IP | SI | FR | RS | BP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TW** | | U | — | S | R | S | U | U | U | U | S | R | R | U | U |
| **CA** | U | | S | S | S | S | U | U | S | U | U | U | U | U | U |
| **RAR** | — | S | | U | U | U | U | U | U | U | U | U | R | U | U |
| **PW** | S | S | U | | S | S | U | U | U | U | U | U | U | R | R |
| **EIC** | R | S | U | S | | S | R | U | S | U | R | U | U | U | U |
| **EP** | S | S | U | S | S | | U | R | S | U | U | U | U | U | U |
| **OT** | U | U | U | U | R | U | | S | U | S | U | U | S | U | U |
| **IT** | U | U | U | U | U | R | S | | S | U | U | S | U | S | U |
| **FF** | U | S | U | U | S | S | U | S | | — | U | S | U | S | U |
| **PF** | U | U | U | U | U | U | S | U | — | | U | U | U | U | S |
| **IP** | S | U | U | U | R | U | U | U | U | U | | S | R | U | U |
| **SI** | R | U | U | U | U | U | U | S | S | U | S | | U | S | U |
| **FR** | R | U | R | U | U | U | S | U | U | U | R | U | | U | U |
| **RS** | U | U | U | R | U | U | U | S | S | U | U | S | U | | U |
| **BP** | U | U | U | R | U | U | U | U | U | S | U | U | U | U | |

Of the 105 pairs, 24 are supported, 10 refused, 2 are alternatives and 69 are not refused but untested.

### Supported pairs

| Pair | Example or test |
|---|---|
| TW + PW | [Allen-Cahn 2D – Wight & Zhao](../examples/2d/allen-cahn-2d-wight-zhao.md), Settings: Tuned (RBA). SA-PINN with time windows is untested. |
| TW + EP | [Allen-Cahn 2D – Wight & Zhao](../examples/2d/allen-cahn-2d-wight-zhao.md), Settings: Tuned |
| TW + IP | `tests/test_round47_ic_pretrain_shared_block.py` (1D, trains); `tests/test_audit_v160.py::test_time_window_ic_pretraining_fits_only_outputs_with_an_ic_term` (2D, the script compiles) |
| CA + RAR | `tests/test_audit_v160.py::test_causal_weights_follow_the_training_points_only` (script generated) |
| CA + PW | `tests/test_training_methods.py::test_generated_code_runs` (RBA, trains a few iterations). The **Loss weights** page shows a warning for this pair: in PINNStudio's own study the two together did worse than either alone. |
| CA + EIC, CA + EP, CA + FF | [Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md) (MsFFN), Inverse mode and Settings: Tuned |
| PW + EIC, PW + EP | [Cahn-Hilliard 1D – Wight & Zhao](../examples/1d/cahn-hilliard-1d-wight-zhao.md), [Allen-Cahn 2D – Mattey & Ghosh](../examples/2d/allen-cahn-2d-mattey-ghosh.md), Settings: Tuned (SA-PINN); EP also with RBA in [Allen-Cahn 2D – Wight & Zhao](../examples/2d/allen-cahn-2d-wight-zhao.md) |
| EIC + EP | the same examples; `tests/test_training_methods.py::test_exact_constraints` |
| EIC + FF, EP + FF | [Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md) (MsFFN) |
| OT + IT | [Lotka-Volterra ODE](../examples/ode/lotka-volterra-ode.md) |
| OT + PF | [Linear elasticity 2D – plate](../examples/2d/linear-elasticity-2d-plate.md) |
| OT + FR | [Fractional diffusion 1D](../examples/1d/fractional-diffusion-1d.md), [Fractional Poisson 2D – disk](../examples/2d/fractional-poisson-2d-disk.md) |
| IT + FF, IT + SI, IT + RS, FF + SI, FF + RS, SI + RS | [Wave 1D – space-time multi-scale Fourier features](../examples/1d/wave-1d-space-time-multi-scale-fourier-features.md) (STMsFFN); FF + RS also [Poisson 1D – multi-scale Fourier features](../examples/1d/poisson-1d-multi-scale-fourier-features.md) (MsFFN) |
| PF + BP | [Poisson 1D – unknown forcing field (batches)](../examples/1d/poisson-1d-unknown-forcing-field-batches.md), Settings: Tuned |
| IP + SI | `tests/test_audit_v160.py::test_initial_loss_weights_follow_ic_pretraining_in_solve`; `tests/test_audit_v160_runs.py::test_initial_loss_weights_with_ic_pretraining_agree` trains it |

### Refused pairs

| Pair | Refused by | What the message says |
|---|---|---|
| TW + EIC | `validate()` | An exact initial condition can't be combined with time-adaptive training: each window starts from the previous window's solution, not the formula. |
| TW + SI | `validate()` | 'Scale by initial losses' isn't available with Time-Adaptive training. |
| TW + FR, RAR + FR, IP + FR | `validate()` | `frac(...)` problems can't use RAR, Time-Adaptive training or IC pre-training: DeepXDE's fractional data keeps its own collocation points. |
| PW + RS | `validate()` | Per-point weights belong to fixed training points, but the Point Resampler replaces them; turn one off. |
| PW + BP | `validate()` | A Point Set condition with a batch size resamples its points every 100 steps; set its batch size to 0, or the per-point weights to None. |
| EIC + OT | `validate()` | An exact initial condition can't be combined with your own output transform formula; write the condition into the formula yourself (IC + t·u). |
| EIC + IP | `validate()` | Refused when every output's initial condition is exact, because pre-training then has nothing to fit. With at least one initial condition as a loss term the pair is not refused (untested). |
| EP + IT | `validate()` | Exact periodicity can't be combined with the input transform; write the cos/sin features into the input formula yourself. |

### Conditions on supported pairs

- **IT + STMsFFN**: an input-feature formula must end with a feature of \( t \) alone (STMsFFN takes the last
  input as time); otherwise `validate()` refuses it.
- **TW + PW**: refused when time-window resampling rounds are on ("resampling adds collocation points, which
  per-point weights can't follow").
- **RAR + PW** (untested): not refused; the weights start again at 1 after every round, because they belong to
  fixed points.

## Time-window options

The options on the **Adaptive training** page under **Time Adaptive Training** depend on each other:

| Option | Needs | Example or test |
|---|---|---|
| **Enable transfer learning** | — | [Cahn-Hilliard 1D – Wight & Zhao](../examples/1d/cahn-hilliard-1d-wight-zhao.md), [Allen-Cahn 2D – Mattey & Ghosh](../examples/2d/allen-cahn-2d-mattey-ghosh.md) |
| **Growing time range** | transfer learning (else refused) | [Cahn-Hilliard 1D – Wight & Zhao](../examples/1d/cahn-hilliard-1d-wight-zhao.md); 2D: `tests/test_ta_growing_2d.py` trains past the first window |
| **Backward compatibility (bc-PINN)** | transfer learning (else refused); at least 1 time and 1 point | [Cahn-Hilliard 1D – Mattey & Ghosh](../examples/1d/cahn-hilliard-1d-mattey-ghosh.md) (Tuned), [Allen-Cahn 2D – Mattey & Ghosh](../examples/2d/allen-cahn-2d-mattey-ghosh.md), [Cahn-Hilliard 2D – Mattey & Ghosh](../examples/2d/cahn-hilliard-2d-mattey-ghosh.md) |
| **IC-guided learning** | a fraction from 0 to below 1 | the three Mattey & Ghosh examples above, with bc-PINN |
| **Resampling rounds** | per-point weights None; points ≥ 1, candidates ≥ points, iterations ≥ 1 | [Cahn-Hilliard 1D – Wight & Zhao](../examples/1d/cahn-hilliard-1d-wight-zhao.md), with the growing time range |

Growing time range and bc-PINN in the same run are not refused; `tests/test_export_time_window_options.py`
exports and runs each option on its own, not the two together. With time windows, the code generator also
refuses **Extra training points** (anchors) and an **Operator BC** applied at the initial time (both need the
whole time range; use no adaptive method or RAR instead).

## Other refused settings

These do not involve two methods but stop Solve in the same way:

- **Weight decay** greater than 0 with a PFNN, MsFFN or STMsFFN network (`validate()`), or in a run that uses
  L-BFGS or NNCG anywhere, including RAR's L-BFGS step (the window's optimizer check). AdamW needs a weight
  decay greater than 0; NNCG needs DeepXDE 1.13.0 or later.
- **Interface 2D BC** outside a 2D problem with exactly 2 outputs on a Rectangle or Polygon (code generator).
- **Operator BC at the initial time** on a stationary problem (code generator).
- **`integral(...)`** terms outside an ODE or a stationary 1D problem (`validate()`).
- **Loss function** other than MSE or MAE in any phase (`validate()`).

The complete list of checks is `PINNConfig.validate()` in `pinnstudio/core/config.py`; see also
[Troubleshooting](../help/troubleshooting.md#diagnosing-poor-results) for what to do when a run is refused.

## How this page is kept

The statuses come from reading `PINNConfig.validate()`, the checks in `pinnstudio/core/codegen.py`, the
window's disabling of choices in `pinnstudio/ui/main_window.py`, the settings of the shipped examples (as
shown on their gallery pages) and the tests in `tests/`. A new example or test that uses an untested
combination, or a new check, should be recorded here.

## Related

- **Guide:** [Choosing a training strategy](../guide/choosing-a-strategy.md)
- **Settings:** [All settings](settings-index.md)
- **Help:** [Troubleshooting](../help/troubleshooting.md)
- **Known limitations:** [Scope and limitations](../theory/limitations.md)

### Implementation and source code

--8<-- "_snippets/source/compatibility.md"

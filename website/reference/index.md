# Feature reference

This section describes each feature of PINNStudio precisely: what it does, where it is set in the window,
its settings and defaults, the formulation the generated DeepXDE script uses, the combinations PINNStudio
refuses, and a shipped example that uses it. The [User guide](../guide/index.md) explains the window page
by page; the [Tutorials](../tutorials/index.md) walk through complete problems; the
[Theory](../theory/index.md) section gives the background.

These pages describe the development version that will become **v1.6.0**. Behaviour that exists only in
that version is marked with a "New in v1.6.0 (in development)" box. The released version on PyPI is
v1.5.0.

!!! note "Scope"
    PINNStudio helps you set up, train and evaluate physics-informed neural networks. It does not
    guarantee that training converges, or that a result is physically valid or accurate, for an arbitrary
    problem. Every method on these pages can make a result worse on a problem that a plain PINN already
    solves well; compare with a plain run.

## How the pages are organised

Each feature page uses the same headings: *What it does*, *When to use it*, *How to enable it* (the
model-tree page and the exact labels), *Settings* (a table with the label, its meaning, the default, the
allowed range and the configuration field it is saved as), *Formulation*, *Defaults and limitations*,
*Supported problem types*, *Compatibility with other features*, *Common mistakes and troubleshooting*, and
*Example*. Each ends with *Related*: the tutorial, example problems from the gallery, the theory page, the original
method's reference and a pointer to the page's limitations, then *Implementation and source code*, the functions
and tests behind the feature with links to the code on GitHub (written by `tools/docs/make_source_links.py` from
`tools/docs/source_map.py`).

The configuration fields are those of `PINNConfig` in `pinnstudio/core/config.py`, the content of a saved
`.pinn.json` problem file (see [Configuration schema](../developer/config.md)). Messages quoted under
*Compatibility* are the ones `PINNConfig.validate()` and the code generator print when Solve or a sweep
refuses a combination.

## Problems

| Page | Content |
|---|---|
| [Forward problems](forward.md) | Solving for the field from the equations and conditions |
| [Inverse problems](inverse.md) | Unknown constants and unknown fields learned from measured data |
| [ODEs and PDEs](equations.md) | Dimensions, steady and time-dependent problems, expression syntax, derivative names, `integral(...)`, `frac(...)` |
| [Coupled systems](coupled.md) | Several outputs and several equations |

## Domain and conditions

| Page | Content |
|---|---|
| [Geometry](geometry.md) | Interval, Rectangle, Cuboid, Disk, Ellipse, Sphere, Triangle, Polygon, combined shapes, Sketch |
| [Boundary conditions](boundary-conditions.md) | Dirichlet, Neumann, Robin, Periodic, Point Set, Point Set Operator, Operator, Interface 2D |
| [Initial conditions and imported data](initial-conditions.md) | Initial-condition formulas, initial condition from a file |
| [Exact constraints](exact-constraints.md) | Exact initial condition, exact periodicity |

## Network and training

| Page | Content |
|---|---|
| [Networks and Fourier features](networks.md) | FNN, PFNN, MsFFN, STMsFFN, activations, initializers, transforms, weight decay |
| [Optimizers and training phases](optimizers.md) | Adam, AdamW, L-BFGS, NNCG, learning-rate decay, L-BFGS options, precision, callbacks |
| [Loss weights](loss-weights.md) | Weight per term and per phase, scaling by the initial losses |
| [IC pre-training and guided learning](pretraining.md) | Initial-condition pre-training, IC-guided learning in time windows |

## Adaptive methods

| Page | Content |
|---|---|
| [RAR](rar.md) | Residual-based adaptive refinement |
| [RBA and SA-PINN](point-weights.md) | Per-point residual weights |
| [Causal training](causal.md) | Time-chunk weights and the ε schedule |
| [Time windows and bc-PINN](time-windows.md) | Time-adaptive training, transfer learning, growing range, backward compatibility, resampling |

Which of these methods to try for a given problem is discussed in [Choosing a training strategy](../guide/choosing-a-strategy.md).

## Data and results

| Page | Content |
|---|---|
| [Observational data](observations.md) | Measured-data files, weights, custom expressions, reduction, anchors |
| [Error metrics](metrics.md) | Relative L2, MSE, maximum error, parameter error |
| [Plots and data export](plots.md) | Plot types, plot settings, solution data files |
| [Checkpoints and restoring](checkpoints.md) | Model Checkpoint, saved models, Restore & visualize |
| [Generated scripts](scripts.md) | Solve's script and the exported script |
| [Sweeps and Bayesian optimization](sweeps.md) | Parameter sweeps and Bayesian search |

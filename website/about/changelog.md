# Release history

A readable summary of PINNStudio's releases, newest first. Every change, with its details, is in the repository's
[CHANGELOG.md](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/CHANGELOG.md), which follows the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Release dates are those given there.

## Versions

Releases from v1.0.0 on are numbered versions with a git tag (`v1.5.0`) and a GitHub release; PyPI has every
release from 1.1.0 on. Between releases, the code in the GitHub repository can be identified by its git commit.

| Version | Date |
|---|---|
| v1.6.0 | YYYY-MM-DD |
| v1.5.0 | 2026-10-21 |
| v1.4.0 | 2026-10-05 |
| v1.3.0 | 2026-09-30 |
| v1.2.2 | 2026-09-07 |
| v1.2.1 | 2026-09-07 |
| v1.2.0 | 2026-09-07 |
| v1.1.2 | 2026-09-01 |
| v1.1.1 | 2026-09-01 |
| v1.1.0 | 2026-08-29 |
| v1.0.0 | 2026-08-28 |
| 0.1.0 | 2026-08-27 |

The dates are those of the release headings in `CHANGELOG.md`; for v1.0.0 to v1.1.2, which have no CHANGELOG
entries, the dates of their GitHub releases. 0.1.0 was not tagged or published on PyPI. How a release is made is described in
[Releases](../developer/releases.md).

## v1.6.0 (YYYY-MM-DD)

The main changes, by theme:

**Examples**

- A **Settings** list next to the **Examples** list: **Default** loads an example as its source solves it (DeepXDE's
  settings, or the paper's method for Wight & Zhao, Mattey & Ghosh's bc-PINN and Raissi et al.), **Tuned** loads
  the run selected in the benchmarks (the most accurate, or the fastest within 25% of the lowest error). The Training Log says what the chosen settings change.
- Inverse versions of 12 more DeepXDE forward examples, each finding physical constants from measured data.
- A new example, *Poisson 3D – cube with spherical cut-out*; 64 examples in all.
- Six phase-field examples name their outputs as their papers do (`phi`, `h`, `mu` instead of `u`); problems saved
  before still open with their files found.

**Training methods**

- Time-adaptive training can train one network through all the time windows: a **growing time range**
  (Wight & Zhao), **backward compatibility** (bc-PINN, Mattey & Ghosh), **initial-condition guided learning**, and
  **resampling** in each window.
- Error Analysis also gives the relative \( L_2 \) error over all reference times together, and a 3D view for 3D
  problems on a box.
- **Parameter sweep** can vary the random seed, a final L-BFGS phase, per-point weights, causal training, the network
  architecture, exact periodicity and exact initial conditions, time-adaptive training, the measured-data times of
  inverse problems, and every named constant of the equations. Each run's settings are checked before it trains.

**Correctness (pre-release audit)**

A review of the generated code found and fixed a series of problems, with regression tests. Among them: the weights
of later time windows were all replaced by 1; RAR rounds and time windows used hidden optimizer settings instead of
the **Training phases**; causal weights could go stale; with transfer learning the result depended on the save
folder; several training-method combinations that silently did the wrong thing are now refused with an explanation
(per-point weights with point resampling, IC pre-training without any initial-condition loss, exact periodicity on an
output without a periodic condition); output names are checked; a boundary condition that selects no training point
is reported before training instead of producing a NaN loss. The time-window weight fix changes the results of the
Cahn-Hilliard time-window runs. See [Troubleshooting](../help/troubleshooting.md) for the new messages.

**Export, saving and platforms**

- **Export script** agrees with **Solve** in more cases (time-window options, the first window's random weights,
  Model Checkpoint, the GPU settings).
- Saved problems store paths into PINNStudio's `reference_data` folder as `@REF/...`, so a problem saved from an
  example finds its data on another computer.
- **Train on: CPU** now works on Windows with an NVIDIA GPU; GPU runs no longer print allocator warnings on Windows or
  with PyTorch 2.9 and newer; a run that runs out of GPU memory says so.
- The installers compile PINNStudio when installing (faster first start), print the installed version, and point
  out another PINNStudio that comes first on the PATH. The update notice gives the right update command for each kind
  of installation.
- SciPy is a declared dependency.

**License**

- PINNStudio stays under the MIT License: the interface now uses PySide6 (Qt for Python) under the LGPL-3.0
  instead of PyQt6. See [License](license.md), and [Dependencies and attribution](dependencies.md) for the licenses
  of the libraries.

## v1.5.0 (2026-10-21)

The current release on PyPI, and the first release since 1.4.0 (the 1.5.0 and 1.5.1 entries prepared on 2026-10-07
were not published separately and are included in it). Headline changes:

- **Installation and platforms.** The one-command installers (uv, with the PyTorch build matching the NVIDIA driver),
  `pinnstudio --create-shortcut`, tests on Windows, macOS and Linux on GitHub Actions, **Train on: CPU** and
  **CPU threads** settings, and a timing study of the window, GPU and CPU.
- **A new interface.** The Start tab; a model tree with right-click menus and help on every settings page; equations
  typeset as you type; the Script tab; results while training (**Live loss**, a status bar with progress and time
  left); **Results › History** to compare runs; copying and saving figures and tables.
- **Domains.** Sketching 2D and 3D domains, picking boundary conditions on the sketch, DXF/SVG import, a Geometry tab
  and a 3D view of 3D domains.
- **Problem types.** ODE problems, fractional equations (`frac(...)`), integro-differential equations
  (`integral(...)`), named constants, the coordinates in equations, formula input/output transforms, and more
  boundary-condition options.
- **Training methods.** Exact initial conditions and exact periodicity, per-point weights (RBA, SA-PINN), causal
  training, loss weights from the initial losses, multi-scale Fourier feature networks (MsFFN, STMsFFN), PFNN,
  training monitors, RAR with a residual tolerance and on steady problems, extra training points.
- **Examples.** DeepXDE's documented forward and inverse examples in the Examples list, and new Allen-Cahn and
  Cahn-Hilliard templates after Wight & Zhao and Mattey & Ghosh.
- **Analysis.** Error Analysis against an exact-solution formula, a Plot settings dialog for every figure,
  parameter-convergence plots, Bayesian search and a summary plot in **Parameter sweep**, and
  **Restore & visualize** from the model file alone.

## v1.4.0 (2026-10-05)

- **Parameter Sweep**: a batch of training runs from one set-up (one at a time, all combinations or paired values),
  driven by **Solve** and **Stop**.
- **Custom geometry**: domains built from 2D and 3D primitives with union, subtraction and intersection.
- A figure-size option for the result plots.
- Fixes to Restore for time-adaptive and steady problems, among others. The 1D and 2D Cahn-Hilliard templates were
  removed until they were numerically reliable (they returned in 1.5.0).

## v1.3.0 (2026-09-30)

- A problem built from scratch with the default optimizer phases no longer fails validation with
  "Phase 1 iterations must be positive".
- Panel dividers easier to move, and cross-platform fonts.

## v1.2.2 (2026-09-07)

- Training no longer crashes on DeepXDE 1.10.0 for single-output equations with a second derivative.

## v1.2.1 (2026-09-07)

- A check for a newer release on PyPI at start-up, with a banner showing the update command.

## v1.2.0 (2026-09-07)

- Reference data is included in the installed package (earlier releases shipped without it, so Error Analysis had
  no data for the templates).
- Friendlier optimizer names, time-adaptive training on by default for the 2D Allen-Cahn templates and restricted to
  forward problems, corrected coefficients of two templates, and fixes to time-adaptive Error Analysis and Restore
  plots.

## v0.1.0 (2026-08-27)

The first public release: 1D and 2D forward and inverse problems, a free-form equation editor for coupled systems,
Dirichlet, Neumann and periodic boundary conditions, Adam and L-BFGS training phases, IC-guided pre-training, RAR and
time-adaptive training, error analysis against reference data, plotting and model restore, with seven templates.

The CHANGELOG has no entries for v1.0.0, v1.1.0, v1.1.1 and v1.1.2; their notes are on the repository's
[GitHub releases page](https://github.com/AsfandyarKhan72/PINNStudio/releases).

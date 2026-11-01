# Tutorials

The tutorials are step-by-step exercises in the PINNStudio window. Each one states what you will learn,
what you need, the exact clicks and values, what you should see, and what to do when it does not work.
None of them requires editing Python.

Every tutorial was run in the application before it was written. The numbers quoted under
"Expected results" come from those runs, usually with fewer iterations than the tutorial recommends so that
they finish in minutes on a CPU. Your numbers will differ with hardware, random seed and iteration counts.
PINNStudio helps you set up, train and evaluate PINNs; it does not guarantee convergence or accuracy for an
arbitrary problem (see [Scope and limitations](../theory/limitations.md)).

!!! info "New in v1.6.0 (in development)"
    The tutorials describe the development version that will become v1.6.0. Labels and defaults may
    differ in the released v1.5.0 (`pip install pinnstudio`).

## Where to start

If you are new to PINNStudio, do tutorials 1 and 2 first. They introduce the model tree, the editors,
**Solve** and **Error Analysis**, which the other tutorials take for granted.

| # | Tutorial | Starts from | You learn |
|---|---|---|---|
| 1 | [First forward ODE](01-ode.md) | blank ODE system | outputs, constants, equations, initial conditions, training phases, exact-solution error |
| 2 | [1D Burgers equation](02-burgers.md) | example | the model tree of a time-dependent PDE, boundary-condition rows, reference files |
| 3 | [2D heat equation](03-heat-2d.md) | example | a 2D domain, mixed Dirichlet/Neumann conditions, the geometry view |
| 4 | [Coupled PDE system](04-coupled.md) | blank 1D problem | two outputs and equations, one boundary row per output, per-term loss weights |
| 5 | [Phase-field (Allen–Cahn)](05-phase-field.md) | example | a stiff phase-field equation, periodic boundaries, a wider network |
| 6 | [Inverse parameter estimation](06-inverse.md) | example | inverse mode, unknown parameters, measured data, parameter convergence |
| 7 | [Importing observational data](07-data.md) | blank 1D problem + data file | your own data file, two unknown coefficients, the file format |
| 8 | [Custom geometry](08-geometry.md) | blank 2D problem | a polygon domain, conditions selected by a location expression |
| 9 | [Exact constraints](09-exact.md) | example + blank problem | initial and boundary conditions built into the network |
| 10 | [RAR](10-rar.md) | example | residual-based adaptive refinement |
| 11 | [RBA and SA-PINN](11-point-weights.md) | example | per-point loss weights |
| 12 | [Causal and time-window training](12-causal-windows.md) | example | causal weighting and time-adaptive (windowed) training |
| 13 | [Comparing configurations](13-compare.md) | tutorial 1 | Default vs Tuned settings, a parameter sweep |
| 14 | [Bayesian hyperparameter search](14-bayesian.md) | tutorial 1 | Bayesian search over a learning rate |
| 15 | [Restoring a trained model](15-restore.md) | a finished run | Restore & visualize, what can and cannot be restored |
| 16 | [Standalone scripts](16-scripts.md) | tutorial 1 | the Script tab, Export script, running a script without the GUI |
| 17 | [Error analysis](17-error-analysis.md) | tutorials 1–2 | reference files vs formulas, metrics, output files |

## Conventions

- GUI labels are written in **bold** exactly as they appear; pages of the model tree are named as in the tree
  (**Equations**, **Boundary conditions**, …); menus as **Problem › New › 1D problem**.
- Text you type is shown in `code`. Expressions use Python syntax: `**` for powers, `pi`, `sin`, `exp`, and
  derivatives such as `du_t`, `du_xx`.
- "Run folder" means the timestamped folder that each **Solve** creates under **Save to:** on the **Results**
  page (by default `~/PINNStudio_Results`).
- Screenshots are of the real application. Result figures are the PNG files PINNStudio wrote during our
  verification runs; their captions give the iteration counts used.

For a description of every page and control, see the [User guide](../guide/index.md); for what each method
does, see the [Feature reference](../reference/index.md) and [Theory](../theory/index.md).

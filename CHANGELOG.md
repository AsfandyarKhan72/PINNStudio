# Changelog

All notable changes to PINNStudio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.5.0] - 2026-10-07

### Added

- **Network Type (FNN / PFNN)**: an independent dropdown in the Neural Network panel. FNN (the existing default) is a single shared trunk; PFNN (`dde.nn.PFNN`, a genuine DeepXDE built-in) gives each output its own parallel sub-network merging only at the final layer -- useful for multi-output problems (e.g. 1D Schrödinger's real/imaginary parts) where the outputs have different scales/behavior. A single shared network-construction helper is used everywhere a network gets built (training scripts, exported scripts, all Restore/Error-Analysis script builders), so a training run and the script that later restores its checkpoint can never disagree on how the network was shaped.
- **Training Monitors**: track an arbitrary expression of the solution and its derivatives (`u`, `du_x`, `du_xx`, `du_xy`, ...) during training, logged alongside the loss curve, instead of only being able to inspect a quantity after training finishes. Built on DeepXDE's own `dde.callbacks.OperatorPredictor`; reuses the Custom PDE editor's own derivative-vocabulary builder, so monitor expressions use the identical `d<output>_<vars>` syntax.
- **Derivative-aware "Custom..." plotting**: the Custom expression field (previously algebraic combinations of raw outputs only, e.g. `sqrt(u**2+v**2)`) can now also reference derivatives (`du_x`, `du_xx`, ...), evaluated through DeepXDE's own `dde.Model.predict(x, operator=...)` built-in -- the same mechanism Training Monitors uses, and the same derivative-vocabulary builder, reused rather than duplicated. Wired into every place a Custom expression can be configured: the live Results panel, the Restore & Visualize tab (Standard and Time-Adaptive), the "Export as DeepXDE Script" exporter, and Error Analysis's own per-group custom-expression selector.

### Changed

- Training Monitors' and the Custom PDE editor's own in-app example text ("mu", "dmu_x", "dmu_xx") now uses "u, du_x, du_xx" -- "u" is the actual output name in nearly every template, where "mu" never appears.

### Fixed

- **A plain Adam-only training run (Optimizer Scheduler off, no second optimizer phase) with Save enabled never actually saved a model checkpoint** -- it wrote the run's config JSON but never called `model.save()`, so Model Restore would find nothing to load for a run trained exactly this way. For an Inverse problem in this same configuration, nothing was saved at all, not even the config JSON. Every other configuration (scheduler phases, a second optimizer phase, Time-Adaptive) already saved correctly; this was a narrow, specific gap, now fixed for both Forward and Inverse problems.
- **The "Export as DeepXDE Script" exporter's Time-Adaptive loss plot showed only the last optimizer phase of the last time sub-domain**, not the full run -- a known limitation carried unfixed since 1.1.0. It now stitches the loss history across every phase of every time sub-domain, with a continuously increasing iteration axis, matching how the GUI's own live Time-Adaptive loss plot (and its clean-script non-Time-Adaptive counterpart) already worked.

## [1.4.0] - 2026-10-05

### Added

- **Parameter Sweep**: run a batch of training configurations from one setup instead of clicking Solve repeatedly and tracking results by hand. Enable it from the left panel (below Adaptive Training) and the shared Solve/Stop controls drive it, the Training Log streams every run in order, and the right panel shows the last completed run's own figure once the sweep finishes. Three ways to combine multiple swept parameters: one-at-a-time (vary each separately against a shared baseline), all combinations (full grid/cross product), and specified combinations (pair up each parameter's *i*-th value, COMSOL-style). Each run gets its own results subfolder (model checkpoint, plots, Error Analysis metrics) under the same Save to: location as a normal Solve, plus an auto-generated `sweep_manifest.json`/`sweep_summary.csv` summarizing every run.
  - Sweepable parameters span network architecture (hidden layers, neurons per layer, activation, kernel initializer, weight decay), collocation point counts and sampling distribution, every Optimizer Scheduler phase's iterations/learning rate/optimizer, every PDE/BC/IC loss weight, Inverse mode's trainable-variable initial guesses and observed-data loss weights, RAR's own parameters (training rounds, sampling points, points added per cycle, Adam/L-BFGS iterations), Time Adaptive's own parameters (per-step-group step count, IC grid resolution), and Input/Output Transform's per-axis/per-output scale — gated to only appear when the relevant mode/feature is actually selected.
  - See the [Parameter Sweep](README.md#parameter-sweep) section of the README for the full walkthrough.
- **Custom Geometry**: build a domain from 2D primitives and boolean (CSG) operations instead of being limited to the built-in template shapes, extended to 3D with Cuboid and Sphere primitives plus the same CSG operations. Works anywhere a geometry is configured; train normally once a shape is built.
- **Plot Settings figure-size option** (Default / Square / Wide / Custom) for every single-panel results plot -- Loss, Line, Surface, both Animation GIF types, Parameter Convergence, the Parametric-Sweep summary bar chart, and all of their Time-Adaptive equivalents -- configurable from both the Setup tab's Plot Settings dialog and the Restore tab's own viz-settings dialog. "Default" renders identically to before this option existed.
- README: 1D Burgers and 2D Allen-Cahn (Wight & Zhao) results sections now include a "Restore & Visualize" animation (the trained model restored from its checkpoint and re-animated without retraining); the top example-gallery captions for Burgers and Allen-Cahn (Wight & Zhao) now link to their source papers; Custom Geometry is documented; Fisher-KPP and 3D Poisson (Sphere) sections gained their result images (surface/line comparisons, inverse convergence animations).
- CI smoke-test workflow now also runs the Parameter Sweep codegen-integration test (`test_sweep_param_expansion.py`), the Training Callbacks visibility test (`test_training_callbacks_visibility.py`), the Custom Geometry test (`test_custom_geometry.py`), the Plot Settings figure-size test (`test_plot_figsize_settings.py`), and the Time-Adaptive Restore + Error Analysis regression test (`test_restore_ta_error_analysis.py`), alongside the existing sweep registry/panel tests.

### Changed

- The Parameter Sweep panel's own configuration (mode, sweep-parameter rows, add/refresh controls) now stays hidden until "Enable Parameter Sweep" is checked, instead of always taking up panel space for a feature that's off by default.
- IC loss-weight rows are now hidden for steady-state problems, where they don't apply.
- Removed "1D Surface Animation (GIF)" as a selectable plot type: a 1D animation frame only has one real spatial axis, and this type faked a second axis by stretching the solution across a cosmetic width rather than showing real data. Line Animation already covers 1D's time-animated case, and Surface (static) still shows the full x-t field.
- Time-Adaptive's loss plot now shows the full run (every sub-domain and every optimizer-scheduler phase within it) instead of only the last phase of the last sub-domain.
- Restore Model's prediction is now masked to the problem's real domain instead of its bounding box, matching how the main Setup tab already renders curved/non-rectangular geometries (e.g. Sphere).

### Fixed

- Restoring a combined Time-Adaptive model with Error Analysis enabled crashed with a `NameError`.
- Restore Model crashed on steady-state problems.
- The Solve button's readiness check incorrectly blocked training when a Custom Geometry was configured.
- 3D surface comparison plots showed only a sparse handful of points for curved geometries (e.g. Sphere) instead of the full point cloud.
- The Restore panel could briefly show the previous run's plot/animation before the new one loaded, instead of clearing it first.

## [1.3.0] - 2026-09-30

### Added

- Fisher-KPP walkthrough in the README now includes the generated solution figure, and the hero tagline calls out that solving and visualization both happen entirely within PINNStudio.
- CI smoke-test workflow now also runs the from-scratch (no-template) config validation regression test.

### Changed

- Widened both panel divider handles (control panel / plots, and Training Log / plots) from 2px to 4px, and reduced the Loss/Solution plot boxes' minimum size, so the dividers have real room to move on laptop-sized windows instead of running out of slack almost immediately.
- Cross-platform font stylesheets now use a full fallback chain (Helvetica Neue, Ubuntu, Noto Sans, DejaVu Sans, Arial, sans-serif) instead of a single Windows-specific font name, avoiding unpredictable substitute-font metrics on macOS/Linux.

### Fixed

- **A custom PDE built from scratch (no template selected) with the default Optimizer Scheduler enabled failed validation with "Phase 1 iterations must be positive"**, even though the visible scheduler phases had valid iteration counts. Config validation was checking the legacy (hidden, unused-when-scheduler-is-on) iteration fields instead of the active scheduler phases. Validation now checks whichever iteration source is actually active, and separately flags a scheduler with no phases or with all-zero-iteration phases.
- Both panel splitters could snap fully closed on a small drag past their minimum size instead of stopping there, because `childrenCollapsible` was left at Qt's default of `True`. Both splitters now set `childrenCollapsible(False)`.

## [1.2.2] - 2026-09-07

### Fixed

- **Training crashed immediately on deepxde==1.10.0 (the minimum version this package requires) for every single-output PDE with a second derivative** -- 1D Heat, 1D Allen-Cahn, 2D Heat, both 2D Allen-Cahn templates, and any custom single-output PDE using u_xx/u_yy/u_tt/u_xy/u_xt/u_yt, with `ValueError: Do not use component for 1D y.`. That deepxde version raises this error whenever `component` is passed to `dde.grad.hessian()` for a single-output model, even when it is explicitly 0; newer deepxde releases dropped this restriction, which is why it was not caught earlier. The PDE builder now omits `component` for single-output models and keeps passing it for multi-output models, matching what each deepxde version expects. Existing 1.2.0/1.2.1 users on an older or minimum-pinned deepxde should upgrade.

## [1.2.1] - 2026-09-07

### Added

- In-app update checker: on launch, PINNStudio checks PyPI in the background for a newer release and shows a dismissible banner with the upgrade command if one is available. Runs off the GUI thread and fails silently when offline or if PyPI is unreachable.

## [1.2.0] - 2026-09-07

### Added

- "Show inverse reference" guidance box in the Inverse PINN panel, explaining the trainable/unknown parameter and how to name it in a custom PDE.
- `Problem Type` and `Reference` columns in the README's Built-in Templates table, linking each template to its source paper.
- Mahmood Mamivand added as a co-author in the project citation.
- Time Adaptive training now defaults on for both 2D Allen-Cahn templates, with a tuned IC grid resolution and transfer-learning optimizer.

### Changed

- Optimizer and Adaptive Training dropdowns now show friendlier names ("Adam", "L-BFGS", "Residual-based Adaptive Refinement (RAR)") without changing the underlying config values.
- Time Adaptive training is now restricted to Forward problems -- it never correctly optimized the inverse parameter, so it no longer appears when Inverse is selected.
- 2D Allen-Cahn (Wight & Zhao) Inverse mode now trains over t in [0, 2.5] instead of [0, 10]; the full time domain made the inverse problem far harder to converge. Forward mode is unchanged.
- Default Adam iterations increased to 20000 for all 2D templates.
- README overhauled with badges, quick links, a table of contents, reorganized template sections, and a note that finer Time Adaptive steps or more collocation points can improve accuracy.

### Fixed

- **Reference data was not bundled into the installed package.** A packaging bug meant a fresh `pip install` of 1.1.2 or earlier silently shipped without `reference_data/`, so Error Analysis and auto-loaded observation files were missing for every template. Existing 1.1.2 users should upgrade.
- 2D geometry reconstruction in Time-Adaptive Error Analysis was hardcoded to a 1D interval, producing incorrect error metrics for 2D templates.
- Inverse-parameter convergence plot's x-axis double-counted iterations across training phases.
- Optimizer Scheduler now honors each template's configured iteration counts instead of always defaulting to 10000/10000 Adam/L-BFGS.
- Corrected coefficients/domain in the 2D Allen-Cahn (Mattey & Ghosh) and 2D Cahn-Hilliard (Wight) equations to match their source papers.
- "Line (time steps)" and "Animation Surface (GIF)" plots in Model Restore no longer crash on newer matplotlib versions (removed use of the deprecated/removed `matplotlib.cm.get_cmap` and `QuadContourSet.collections` APIs).
- Fixed a Qt ampersand-mnemonic bug that mangled the "Boundary & Initial Conditions" and "Restore & Visualize" labels, plus README math rendering and Error Analysis colormap consistency.

### Removed

- 1D and 2D Cahn-Hilliard Quick Example templates removed from the GUI and README -- they are not yet numerically reliable. Their reference data remains in the repository for a future fix.
- Leftover one-off patch scripts removed from the repo root.

## [0.1.0] - 2026-08-27

### Added

Initial public release. PINNStudio is a no-code PyQt6 GUI for building, training, and visualizing Physics-Informed Neural Networks on top of DeepXDE.

- 1D `(x, t)` and 2D `(x, y, t)` problem definitions, forward and inverse.
- Free-form PDE residual editor supporting multi-output, coupled PDE systems.
- Per-side, per-output boundary condition configuration (Dirichlet, Neumann, Periodic) and initial conditions from an expression or a data file.
- Collocation point controls (domain/boundary/initial/test counts, point distribution) with a 2D domain preview.
- Neural network configuration (hidden layers, neurons per layer, activation).
- Two-stage optimization (Adam + L-BFGS) with detailed L-BFGS settings and configurable float precision.
- Multi-phase optimizer scheduling and optional IC-guided pre-training.
- Residual-based adaptive refinement (RAR) and time-adaptive stepping with transfer learning between time windows.
- Parametric studies over a chosen parameter.
- Inverse-problem support for estimating unknown PDE parameters from observation data.
- Seven built-in Quick Example templates: 1D Heat, 1D Allen-Cahn, 1D Cahn-Hilliard, 2D Heat, 2D Allen-Cahn (Mattey & Wight forms), and 2D Cahn-Hilliard (Wight).
- Bundled FEM reference data (`reference_data/`) for these templates, so Error Analysis auto-configures against real ground truth out of the box on any machine.
- Error analysis against reference/ground-truth data (L2, MSE, max error; line and surface comparison plots).
- Configurable result plotting (colormap, resolution, DPI, colorbar, snapshot count) and solution data export.
- Live training log streaming with a Stop control.
- Model restore: reload a saved checkpoint to regenerate plots and re-run error analysis without retraining.

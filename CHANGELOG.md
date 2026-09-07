# Changelog

All notable changes to PINNStudio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- Eight built-in Quick Example templates: 1D Heat, 1D Allen-Cahn, 1D Cahn-Hilliard, 2D Heat, 2D Allen-Cahn (Mattey & Wight forms), 2D Cahn-Hilliard (Wight), and an FeCr spinodal-decomposition PINN. - Bundled FEM reference data (`reference_data/`) for seven of the eight templates, so Error Analysis auto-configures against real ground truth out of the box on any machine.
- Error analysis against reference/ground-truth data (L2, MSE, max error; line and surface comparison plots).
- Configurable result plotting (colormap, resolution, DPI, colorbar, snapshot count) and solution data export.
- Live training log streaming with a Stop control.
- Model restore: reload a saved checkpoint to regenerate plots and re-run error analysis without retraining.

# Changelog

All notable changes to PINNStudio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- Seven built-in Quick Example templates: 1D Heat, 1D Allen-Cahn, 1D Cahn-Hilliard, 2D Heat, 2D Allen-Cahn (Mattey & Wight forms), 2D Cahn-Hilliard (Wight), and an FeCr spinodal-decomposition PINN.
- Error analysis against reference/ground-truth data (L2, MSE, max error; line and surface comparison plots).
- Configurable result plotting (colormap, resolution, DPI, colorbar, snapshot count) and solution data export.
- Live training log streaming with a Stop control.
- Model restore: reload a saved checkpoint to regenerate plots and re-run error analysis without retraining.

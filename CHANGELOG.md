# Changelog

All notable changes to PINNStudio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Standardized Plot Settings between "Plot output" (Setup tab) and Restore & Visualize**: both panels now open the exact same single settings dialog for a given plot type, showing the exact same set of fields -- previously Output and Restore had grown independently (Output used two separate dialogs, one of them reached only through a standalone "⚙" button; Restore already had one unified, auto-popping dialog) and had quietly drifted apart in which fields each one offered.
  - **Restore & Visualize's line plots ("Line (time steps)" and "Animation Line (GIF)") now have their own configurable y/z slice**, matching the slice control "Plot output" already had (see the 2D/3D slice feature from the previous round). Previously, restoring a saved model and viewing a line plot always silently showed whatever slice the model happened to be solved at, read back from that run's own saved `model_config.json`, with no way to look at a different slice without re-solving. The new control defaults to the domain midpoint (unchanged behavior) and can be set to any other value, independently of what the model was originally solved with -- lets you compare different slices of an already-restored model without retraining. Threaded through every Restore script builder, including the Time-Adaptive path and the inline Error Analysis line-comparison.
  - **"Plot output"'s settings dialog gained title/x-label/y-label override fields**, which Restore's dialog already had -- previously only Restore let you override the auto-generated plot title and axis labels. Now available identically in both the live Solve path and "Export as DeepXDE Script", across every plot type (static, animation, multi-snapshot) where it makes sense; left blank (the default) keeps the existing auto-generated text exactly as before.
  - **Removed the "⚙" Plot Settings button beside "Plot output"'s plot-type dropdown.** Selecting a plot type now always pops open its (now-unified) settings dialog immediately, for every plot type, exactly like Restore & Visualize's dropdown already did -- there's only ever one settings dialog to find per plot type, not a separate button whose relationship to the dropdown wasn't obvious.
  - While unifying the two dialogs, found and fixed a few smaller pre-existing issues along the way: the 2D/3D "time snapshots" count control was missing entirely for a non-steady-state 3D Surface plot (only ever shown for 2D); the y/z slice control was shown (but silently unused) for "Surface Animation (GIF)", a plot type whose own script-generation code never reads a slice value at all; and the "Plot output" settings dialog could pop open unexpectedly while silently reloading a saved project or toggling Inverse Problem mode on/off, since restoring the plot-type dropdown's saved value now always triggers the same auto-pop as a real user selection -- all three fixed.
  - **Note on scope** (as of that round): Output's Surface plot for a non-steady 2D/3D problem showed several evenly-spaced time snapshots side by side in one figure (count configurable), while Restore's Surface plot showed only a single snapshot at one chosen time -- real multi-snapshot support for Restore's Surface plot was left for a follow-up round. See "Restore & Visualize's Surface plot now supports multiple time snapshots" below -- that follow-up.
- **Configurable line-plot slice for 2D/3D problems**: every "Line (time steps)" and "Line Animation (GIF)" plot necessarily collapses a 2D/3D spatial domain down to a single x-axis curve, which requires picking a fixed value for the dimension(s) not shown (y for 2D; y and z for 3D) -- previously always the domain midpoint, silently, with no way to change it and no indication in the plot of which slice was shown. The "Line Plot Settings" dialog now exposes this value (gated to only appear for 2D/3D problems), defaulting to "Auto" (the domain midpoint, i.e. unchanged prior behavior) with an option to type an explicit value instead. The chosen slice is shown directly in the plot's title/y-axis label, e.g. `u(x, y=0.5, t)` for 2D or `u(x, y=0.5, z=0.3, t)` for 3D, instead of leaving the reader to guess. Applies everywhere a 2D/3D line plot or line animation can be produced: the live Solve path, "Export as DeepXDE Script", and Restore & Visualize (both plain and Time-Adaptive) -- including each of those paths' own Error Analysis line-comparison plot, which now also filters the reference data down to the same slice instead of comparing against every reference point regardless of its y/z coordinate.
  - While building this, discovered that 2D/3D "Line (time steps)" had no real implementation at all in either `generate_script()` or `generate_clean_script()` -- the plot-type selector was never actually checked for 2D/3D at all, so choosing it silently fell back to a Surface heatmap. Both now produce a genuine line plot (single curve for a steady-state problem; one curve per time step, with a legend, for a time-dependent one) at the configured slice.
- **RAR per-round diagnostics**: every Residual-based Adaptive Refinement round now saves, under this run's own `solution_results/rar_rounds/round_NN/`:
  - `solution_plot.png` -- a snapshot of whatever field is configured (raw output or a derivative-aware Custom expression -- the same field every other plot already shows). For **1D**, this is the x-t plane; axis orientation follows the same Plot Settings "Swap axes" choice every other 1D plot uses (t on the x-axis, x on the y-axis, by default). For **2D/3D**, this file is now the solution *and* the error comparison merged into one: with at least one Error Analysis reference file configured, it's a 3-panel PINN | Reference | |Error| figure sharing one color scale between PINN and Reference (same convention the full Inline Error Analysis uses); with no reference, it's just the PINN prediction, same as before. Compares against the single reference closest to `t_max`; the full multi-file/multi-time Inline Error Analysis sweep still runs once, unchanged, at the very end.
  - `collocation_points.png` -- a 3-color scatter: original domain points (gray), points added in *earlier* rounds (orange), points added *this* round (red). 1D: x-t plane (same axis-orientation fix as above). **2D now shows two panels side by side**: the existing 3D (x, y, t) view, plus a plain 2D (x, y) view collapsing time, so the spatial concentration of refinement is readable without rotating a 3D plot. 3D: unchanged, a 3D (x, y, z) scatter with time folded into per-category transparency (its 4 real coordinate dimensions can't all be plot axes at once).
  - `error_compare.png` as a *separate* file only remains for 1D (no merge there); for 2D/3D it's folded into `solution_plot.png` as described above. The "L2 rel error = ..., MSE = ..." log line is still printed every round, regardless.
  - This makes it possible to see, after training, exactly where RAR concentrated its effort round over round (sharp gradients, moving fronts, etc.) instead of only ever seeing the final model's plot.
- **RAR "Points from:"**: for a multi-equation PDE (`model.predict(x, operator=pde)` returns one residual array per governing equation), RAR's point selection can now be restricted to one equation/output's residual instead of always combining every equation (the original, still-default behavior). Equation order is treated as output order, matching every current template (one equation per output, same order).
- Currently wired into the live "Solve" path (`generate_script()`); "Export as DeepXDE Script" (`generate_clean_script()`) parity is planned as a follow-up.
- **Restore & Visualize's Surface plot now supports multiple time snapshots**, closing the gap the previous round's "Note on scope" above left open. Its Plot Settings dialog's old single "Plot at time t =" field is now a "2D/3D time snapshots:" count control, identical to the one "Plot output" already has -- for every non-steady 2D/3D case: plain restore, and Time-Adaptive combined restore (each snapshot resolves its own per-step model independently, since different snapshot times can fall in different TA steps, each trained separately). Restore's existing true-3D rendering for a 3D Surface plot (six flat, geometry-masked faces blended into one continuous-looking colored cube -- nicer than Output's own z-mid-plane 2D heatmap fallback for 3D) is preserved and extended to multiple side-by-side panels rather than replaced by Output's simpler style. Steady-state problems are unaffected: the snapshot count control stays hidden and the plot stays single-panel, exactly as before.
- **1D Schrödinger template now defaults L-BFGS's floating-point type to float64** (previously float32, the overall app default, still used by every other template). A controlled side-by-side run on this exact equation/architecture showed float32 L-BFGS plateauing at a noticeably higher final loss than float64 (~4.6e-4 vs ~1.8e-4) -- a known float32/L-BFGS numerical-noise-floor limitation (the `lbfgs_float_type` setting's own tooltip already recommends float64 generally); this template is simply the first to default to it, since this equation's dispersive, multi-time-step nature benefits from it more than most.

### Fixed

- **"Export as DeepXDE Script"'s "Line Animation (GIF)" for a 2D/3D problem silently produced a Surface Animation GIF instead**, diverging from the live Solve path (which already animated the correct line-at-a-slice view) -- a leftover guard excluded 2D/3D from the "line" animation branch entirely. Now produces a real line animation at the configured slice (see the new configurable-slice feature above), matching the live Solve path.
- **"Export as DeepXDE Script"'s Error Analysis line-comparison, for a 3D problem, computed its y-slice/y-tolerance from the x-domain's bounds instead of the y-domain's** (a copy-paste mistake in how a shared template string's positional placeholders were filled in) -- found and fixed while extending this same code to use the new configurable slice value above; harmless when `x_min`/`x_max` happened to equal `y_min`/`y_max` (the common case for a square/cubic domain), but silently wrong whenever they didn't.
- **2D/3D comparison plots silently didn't share a color scale when built with `matplotlib`'s `contourf(..., levels=<int>, vmin=..., vmax=...)`**: passing an integer level *count* alongside `vmin`/`vmax` does not actually constrain `contourf`'s auto-selected levels to that range -- each panel still silently auto-ranges to its own data's min/max, so two panels meant to share one color scale (e.g. a PINN-prediction panel and a Reference panel) could show visually different scales despite identical `vmin`/`vmax` arguments. Fixed in the RAR 2D solution-vs-reference plot above by passing explicit level *boundaries* (`np.linspace(vmin, vmax, N+1)`) instead of a count.
- **The same `contourf(levels=<int>, vmin=..., vmax=...)` color-scale bug, in the full Inline/Error Analysis 2D comparison plots (`surface_comparison.png`)**: flagged as a follow-up in the note above, now fixed the same way, at all four sites it occurs (Standard-path 2D and 1D, Time-Adaptive-path 2D and 1D).
- **Legacy per-side Boundary Conditions (configs saved before the current Boundary Conditions panel existed): a bottom-edge Periodic BC's derivative-continuity loss weight was hardcoded to `1.0`** instead of reading the next GUI-provided weight, silently discarding whatever weight the user actually set for that term (and failing to advance the shared weight-list cursor, which would misalign any further such weight read afterward). Only reachable when reopening a very old saved problem file predating the Boundary Conditions panel; unreachable from a newly-created problem in the current GUI.
- **Time-Adaptive training with "Forward IC from file" enabled dropped every time-step group's own continuity anchor points after the first step.** Each step after the first is supposed to anchor its initial condition to the *previous* step's own predicted solution (`anchors=` in `dde.data.TimePDE`), independent of whether the very first step's IC came from a file or an expression -- but the code gated this on the bare "IC from file" flag instead of "first step AND IC-from-file," so every later step silently lost those anchor points whenever "Forward IC from file" was on. The underlying loss-constraint term was still added either way, so training wasn't broken, but later steps were weakened in how strongly they were pulled to match their own true starting point.
- **Custom plot-field expression (e.g. `sqrt(u**2+v**2)`) crashed with `NameError: name 'np' is not defined`** on the live Solve path (`generate_script()`), any time a plot or GIF that reads it actually ran -- e.g. exporting a solution GIF for the 1D Schrödinger template's default `|h| = sqrt(u**2+v**2)` field. Root cause: the expression was being run through the same "np."-prefixing conversion used for PDE expressions, but it's evaluated at plot time in its own small, isolated namespace that maps bare function names (`sqrt`, `sin`, `pi`, ...) straight to their `torch` equivalents, with no `np` binding in it at all. "Export as DeepXDE Script" and every Restore & Visualize script builder already passed this value through unconverted and were never affected -- `generate_script()` now matches them.
- **Legacy per-side Boundary Conditions (a problem file saved before the current Boundary Conditions panel existed), combined with a steady-state problem and an active Initial Condition on some output, crashed with `'Rectangle' object has no attribute 'on_initial'`.** The modern Boundary-Conditions-panel path already correctly skips building an IC constraint for a steady-state problem (no initial time slice for `dde.icbc.IC`'s `on_initial()` to match against once geomtime is just a plain spatial geometry); the legacy fallback path had no equivalent check, in both where it builds the IC constraint itself and where it computes that constraint's own loss weight (which, left unfixed on its own, would have desynced every loss weight after it from its matching constraint, once the constraint itself stopped being built). Found and fixed during this round's own testing -- not reachable from a problem built fresh in the current GUI, which always sends the Boundary Conditions panel's data (even when empty); only from reopening a pre-panel saved problem file.

## [1.5.1] - 2026-10-07

### Fixed

- **The Results panel could show a stale plot left over from a completely different, earlier run** instead of the just-finished run's own loss/solution figure (e.g. a 1D Heat run's surface plot looking like a leftover 3D Heat plot). Root cause: after training finished, `_on_done()` re-derived "where to look" from scratch instead of reusing the run it just finished -- it rebuilt a fresh config (computing a brand-new, never-actually-used timestamped folder name) and read the Setup tab's raw "Save to:" text (the shared *parent* folder, not this run's own unique subfolder), so a plot sitting directly in that parent folder from an unrelated earlier run could win over the real one. Now reuses the exact `PINNConfig` object `SolverThread` actually trained with (and that config's own `save_dir`), the same per-run folder the training subprocess really wrote its files into -- applies to both the solution/surface plot and the loss plot, for whatever output (raw network output or a derivative-aware Custom expression) was configured. This is the same stale-plot bug class the Restore tab already had and fixed (see `_last_restore_save_dir`), just never applied to the plain Solve path.

### Changed

- **Activation function list**: now offers every identifier DeepXDE's own `deepxde.nn.activations` module supports -- tanh, sin, Sigmoid, ReLU, SiLU, Swish, ELU, GELU, SELU -- instead of only tanh/relu/sigmoid/swish, and every label is spelled exactly the way DeepXDE's own documentation capitalizes it (e.g. `ReLU`, `SiLU`, not `relu`/`silu`). `sin` in particular is a well-known good fit for PINN problems with smooth/periodic solutions (SIREN-style sinusoidal activations). A problem saved before this change (with a stored lowercase name like `"relu"`) still reopens showing the matching renamed item (`ReLU`), not a silently different activation.

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

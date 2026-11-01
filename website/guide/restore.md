# Restore & visualize

The **Restore & visualize** page (under **Results** in the tree) loads a trained model from a run folder
and plots it again — another plot type, another output, a derived field, finer plot settings — without
training again.

![The Restore & visualize page with a trained 1D Burgers model chosen: Model file …/solution_results/model_lbfgs-phase1-5021.pt, Config …/solution_results/model_config.json found next to it, the summary line 1D · time-dependent · forward · 1 output, Plot type Surface, Output 1 (u) and the Restore & Visualize button; the Training Log reports the auto-detected config and optimizer](../assets/screenshots/guide_restore_model.png){ .screenshot loading=lazy }

Background: [Checkpoints and restoring](../reference/checkpoints.md).

## Restore a model

1. **Model file:** — **Browse** for a `.pt` file in a run folder's `solution_results/`, or in
   `time_adaptive_steps/step_…/` of a time-adaptive run.
2. **Config:** — the model's settings are found next to it (a `.json` of the same name,
   `model_config.json`, or a time window's `step_config.json`); otherwise **Browse** for them. A summary
   line shows what the config describes, for example *1D · time-dependent · forward · 1 output*.
3. **Plot type:** — Surface, Line (time steps), Animation Line (GIF), Animation Surface (GIF), as the
   problem allows (Line for an ODE or a stationary 1D problem).
4. **Output:** — one output, or **Custom...** with an expression of the outputs and their derivatives
   (for example `sqrt(u**2+v**2)` or `du_x`) and a label.
5. Click **Restore & Visualize**.

The figure appears in the Overview tab, across both panels; **Show loss plot** above it opens the loss
plot saved when the model was trained. The look of the figure (colours, sizes, time steps, the cut
through a 2D/3D domain) comes from the **plot settings** link, the same Plot Settings as the Results page;
each restore is saved in its own folder in the Results page's **Save to** folder.

If [Error Analysis](error-analysis.md) is set up, the restore also compares the model with the reference
files whose times lie in the model's time range, for the output being plotted.

## Time-adaptive runs

When the model file lies in a time window's folder, PINNStudio finds the other windows of the run and
says how many (*N time-adaptive steps, t = … → …*). Surface and line plots then use, for each time, the
window that covers it. For the animations, **Combine all detected steps into one full-domain animation**
(on) makes one animation over the whole time range; unticked, only the chosen window is animated.

## Inverse models

For an inverse model, PINNStudio adds **Parameter Convergence** and **Parameter Convergence (GIF)** to the
plot types (and selects the first) and finds the run's `<name>_convergence.txt` files under
**Convergence files and true values:**, with each parameter's true value when the config records it.
**Add parameter file** adds another file. These plots read only the convergence files, not the model.
For an older inverse model whose config does not list its parameters, type them in
**Parameter names:** (for example `D` or `D,k`).

## Settings on this page

These settings are not part of the saved problem.

| GUI label | Meaning | Default |
|---|---|---|
| **Model file:** | The trained network (`.pt`) | blank |
| **Config:** | The model's settings (`.json`) | found next to the model |
| **Combine all detected steps into one full-domain animation** | Time-adaptive runs: one animation over all windows | on |
| **Parameter names:** | Inverse models whose config lacks them | blank |
| **Plot type:** | See above | Surface (Parameter Convergence for an inverse model) |
| **Output:** | Output or Custom... | Output 1 |
| **Convergence files and true values:** | Inverse: `<name>_convergence.txt` files and true values | found in the run folder |

# Plots and data export

## What it does

At the end of a run the script saves a loss plot and a solution plot (or animation) of the type chosen on
the **Results** page, and for an inverse problem the parameter-convergence plot. **Plot settings**
(also **Settings › Plot Settings...**) sets how every saved figure looks, in Solve, the exported script and
Restore & visualize. **Export Solution** writes the solution on a grid as text files.

## When to use it

To choose what a run saves and how the figures look, and to get the predicted solution as numbers for use
elsewhere.

## How to enable it

**Results** page → **Plots & Output**: **Outputs to plot:**, **Plot type:**, **Plot settings**,
**Error Analysis**, **Export Solution**, and for an inverse problem **Save parameters:**. Choosing a plot
type opens **Plot settings** on that type's tab.

## Settings

### Plot type and outputs

| Label | Meaning | Choices | Config field |
|---|---|---|---|
| **Outputs to plot:** | **All outputs** (with more than one), one output, or **Custom...** (an expression of the outputs and their derivatives, e.g. `sqrt(u**2+v**2)` or `du_x`, with a label) | — | `plot_output_idx`, `plot_all_outputs`, `plot_custom_expr`, `plot_custom_label` |
| **Plot type:** | The solution figure | see the table below | `plot_type` |
| **Save parameters:** (inverse) | How often the unknowns are recorded | No, Every 100 iters, Every 1000 iters, or a number | `inv_param_save` |

| Problem | **Plot type:** choices |
|---|---|
| ODE, stationary 1D | Line |
| time-dependent 1D | Surface, Line (time steps), Line Animation (GIF) |
| stationary 2D/3D | Surface, Line (time steps) |
| time-dependent 2D/3D | Surface, Line (time steps), Line Animation (GIF), Surface Animation (GIF) |
| any inverse problem | the above plus Parameter Convergence, Parameter Convergence (GIF) |

*Surface* is an x–t map in 1D (t on the horizontal axis by default, **1D surface:**), and snapshots of the
spatial field at a few times in 2D/3D (**Time snapshots:**; 3D at a z slice). *Line (time steps)* draws
\( u \) along \( x \) at several times; in 2D/3D the other axes are cut at **y =** / **z =** (the middle of
the domain by default). *Parameter Convergence* draws each unknown against the iteration, with dashed
**True value** and dotted **Final value** lines.

### Plot settings tabs

| Tab | Main settings (labels) | Config fields |
|---|---|---|
| General | **Resolution (DPI):** (300), **Figure size:** (Default, Square, Wide, Custom; **Width × height:**), **Title:**, **X-axis label:**, **Y-axis label:**, **Title font size:** (13), **Panel title font size:** (10), **Axis label font size:** (10), **Legend font size:**, **Grid lines:**, **Significant digits:** (4), **Scientific notation:** (auto, always, never), **Written as:** | `plot_dpi`, `plot_figsize_mode`, `plot_figsize_w/_h`, `plot_title_override`, `plot_xlabel_override`, `plot_ylabel_override`, `plot_title_fontsize`, `plot_subplot_title_fontsize`, `plot_axis_label_fontsize`, `plot_legend_fontsize`, `plot_grid`, `plot_sig_digits`, `plot_sci_notation`, `plot_sci_style` |
| Surface | **Colour map:** (jet), **Error colour map:**, **Contour levels:** (100), **Grid resolution:** (200), **3D grid resolution:** (40), **3D view surface level:**, **Colour range:** (automatic or fixed), colour bar, **Time snapshots:** (2), **1D surface:** | `plot_colormap`, `plot_err_colormap`, `plot_levels`, `plot_resolution`, `plot_resolution_3d`, `plot_iso_level`, `plot_auto_range`, `plot_vmin`, `plot_vmax`, `plot_colorbar`, `plot_n_2d_snapshots`, `plot_swap_xt` |
| Line | **Time steps to show:** (5), **Line width:** (2), **y =** / **z =** slice, **Reference in line comparisons:** (Line, Dots), **PINN line colour:**, **Reference line colour:**, **Error Analysis panels per row:** (4), parameter-convergence plot settings | `num_timesteps_line`, `plot_linewidth`, `line_slice_y(_auto)`, `line_slice_z(_auto)`, `plot_ref_style`, `plot_pinn_color`, `plot_ref_color`, `plot_ea_cols`, `param_plot_*` |
| Animation | **Frames:** (20), **Frame rate (fps):** (10), **Resolution (DPI):** (150), **Line width:** (3), parameter-convergence animation settings | `num_timesteps_anim`, `plot_fps`, `plot_dpi_anim`, `plot_linewidth_anim`, `param_anim_*` |
| Loss | **Show:** (Train + Test totals, Each loss term, Totals and each term), **Line width:** (2), **Loss axis:** (logarithmic), **Record every:** (1000 iterations) | `loss_plot_mode`, `loss_plot_linewidth`, `loss_plot_log_y`, `loss_display_every` |

**Record every:** is DeepXDE's `display_every` for every training call: it sets how often the loss is
printed in the Training Log and recorded for the loss plot.

### Export Solution (solution data)

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Enable solution data export** | Write the solution as text files after a forward Solve | off | — | `export_enabled` |
| **Grid size (points per axis):** | Points along each axis of the bounding box | 101 | 101, 51, 21, 11, or any integer | `export_grid_size` |
| **Number of time snapshots:** | Equally spaced times over \( [t_{\min}, t_{\max}] \) | 11 | 2–50 | `export_t_steps` |

The files go to `solution_results/solution_data/` in the run folder: one file `solution_t<time>.txt` per
time (four decimals in the name), or one `solution.txt` for a stationary problem. Each is comma-separated
with a header line naming the columns: the coordinates (`x[,y[,z]][,t]`; `t` for an ODE) followed by
**every output** by name, on the full grid of the domain's bounding box. For a disk, polygon or other
non-box shape, grid points outside the domain (or in a hole) have the value `nan` in every output column.

## Formulation

The plotted fields are the network's predictions on regular grids (`model.predict`); a **Custom...** field
with derivatives is evaluated with `model.predict(x, operator=...)`, the same derivative names as in the
equations.

## Defaults and limitations

- Solution data export is for forward problems and is written by Solve only; the exported script prints a
  note saying so.
- The export grid covers the bounding box of a non-box shape; the points outside the domain are written
  as `nan` (they used to carry the network's extrapolation without a mark).

!!! info "New in v1.6.0 (in development)"
    Exported solution data of a non-box shape is `nan` outside the domain, and an ODE's file names its
    variable `t` in the header (it said `x`).
- The **Export Solution Data** dialog's note shows the column order of 1D and 2D files; 3D files are
  `x, y, z, t, …` and every file has one column per output.

## Supported problem types

All; the plot types available depend on the problem (table above).

## Compatibility with other features

A saved problem whose plot type does not apply to the current problem is switched to one that does, with a
note in the Training Log. With time windows the surface and line plots are stitched from the windows.

## Common mistakes and troubleshooting

- **Animations are slow to write**: lower **Frames:** or **Resolution (DPI):** on the Animation tab, or the
  **3D grid resolution:** for 3D surfaces.
- **The colour range hides detail**: switch **Colour range:** to a fixed range.

## Example

**Schrödinger 1D** (a **Custom...** field \( |h| = \sqrt{u^2 + v^2} \)) and any 2D time-dependent example
such as **Heat 2D** (Surface Animation (GIF)) in **Examples**. See the
[Results and plots guide page](../guide/results.md).

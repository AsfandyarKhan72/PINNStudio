# Configuration schema

Every problem in PINNStudio is one `PINNConfig` object, a Python dataclass defined in
`pinnstudio/core/config.py`. The window builds it from its widgets (`MainWindow._build_config()`), a saved
problem (`.pinn.json`) is this object written to JSON, and both code generators read nothing else
(see [Architecture and code generation](architecture.md)).

## How the configuration is organised

- **Flat fields.** Most settings are plain scalars (`num_domain`, `learning_rate`, `causal_eps`, ...).
- **Per-output strings.** Settings that exist once per output are comma- or `|`-separated strings, in output
  order: `output_names = "u,v"`, `pde_expressions = "eq1|eq2"`, `ic_expressions = "ic1|ic2"`,
  `ic_active = "True,False"`, `ic_hard = "False,True"`.
- **JSON-encoded lists.** Variable-length structures are JSON strings: the boundary-condition rows
  (`custom_bc_json`), the Training Phases (`scheduler_phases`), the unknown parameters
  (`inverse_variables_json`), the measured-data files (`inverse_obs_files_json`), the Custom geometry
  (`geom_custom_shapes_json`), the swept parameters (`sweep_parameters`), the training monitors
  (`training_monitors`) and the named constants (`constants`). The Error Analysis reference files
  (`ea_files`) are the one exception: they are stored with `repr()` and read with `ast.literal_eval`.
- **Legacy fields.** Some fields are kept only so that problems saved by older versions still open, for
  example the single-optimizer fields (`optimizer`, `iterations`, `optimizer2`, `iterations2`), the per-side
  boundary fields (`bc_left_types`, ...), and `parametric_study` / `parametric_param` /
  `parametric_values` (superseded by the Parameter Sweep fields). The comments in `config.py` say which
  field is read when; for training, the Training Phases (`optimizer_scheduler`, `scheduler_phases`) are what
  every current example uses.

## Validation

`PINNConfig.validate()` returns a list of human-readable problems (an empty list means the configuration
can be handed to the generators). The window calls it before every **Solve** and the sweep runner calls it
before every run of a sweep, so a combination that cannot work is refused with a reason instead of failing
inside the training script. It is deliberately not exhaustive: it targets settings that would silently
produce a degenerate run or crash deep inside DeepXDE or PyTorch, including values a hand-edited
`.pinn.json` file could contain but the widgets would not allow.

Two helpers in the same file are used by both generators:

- `use_training_phases(config)` and `phases_from_legacy(config)` decide whether the Training Phases or the
  legacy single-optimizer fields drive training.
- `LBFGS_RECOMMENDED` holds the L-BFGS options that **Use recommended L-BFGS settings** applies
  (DeepXDE's own defaults).

## Adding a field

A new setting needs a field here with a default that reproduces the previous behaviour, so that problems
saved before the field existed load unchanged (a missing key keeps the dataclass default). Write the comment
above the field: the table below is generated from it. The complete procedure is in
[Adding a training method](add-method.md).

## Field reference

The table is generated from `pinnstudio/core/config.py` by `tools/docs/make_config_reference.py`; do not edit
it by hand. After changing `config.py`, run

```bash
python tools/docs/make_config_reference.py          # rewrite the table below
python tools/docs/make_config_reference.py --check  # exit status 1 if the table is out of date
```

The description is the first sentence of the developer comment in the source. Some of these comments
record why a field was introduced rather than what it means to a user; the meaning of each setting in
the window is described in the [User guide](../guide/index.md) and the [Feature reference](../reference/index.md).

<!-- BEGIN GENERATED CONFIG TABLE -->

301 fields, in the order they are defined in `pinnstudio/core/config.py`.

### General

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `save_dir` | `str` | `""` |  | -- |
| `adapt_method` | `str` | `"None"` | **Method:** | -- |
| `rar_cycles` | `int` | `3` | **RAR training rounds:** | -- |
| `rar_candidates` | `int` | `50000` | **Residual sampling points:** | -- |
| `rar_add_points` | `int` | `500` | **Points added per round:** | -- |
| `rar_adam_iters` | `int` | `20000` | **Adam iterations:** | -- |
| `rar_lbfgs_iters` | `int` | `10000` | **L-BFGS iterations:** | -- |
| `rar_tolerance` | `float` | `0.0` |  | Stop refining early once the mean \|residual\| over the candidates is below this (0 = off; rar_cycles is then a cap) -- DeepXDE's Burgers-RAR example loops until it's below 0.005.. |
| `rar_output_selector` | `int` | `-1` |  | Which equation's residual to rank candidate points by when the PDE has more than one governing equation (model.predict(x, operator=pde) returns a list, one entry per equation, in the same order the equations were entered in the PDE editor ... |
| `time_adaptive` | `bool` | `False` | **Method:** | -- |
| `ta_num_steps` | `int` | `5` |  | -- |
| `ta_grid_size` | `int` | `101` |  | -- |
| `ta_step_groups` | `str` | `""` |  | -- |
| `ta_transfer_learning` | `bool` | `False` | **Enable transfer learning (warm start from previous step)** | -- |
| `ta_transfer_optimizer` | `str` | `"adam"` | **Transfer optimizer:** | -- |
| `ta_resample_rounds` | `int` | `0` | **Resampling rounds:** | ta_resample_rounds: adaptive resampling in each window (Wight & Zhao's approach I + resampling): after the window's training, ta_resample_points of ta_resample_candidates random points with the largest PDE residual are added to its ... |
| `ta_resample_points` | `int` | `200` | **Points per round:** | ta_resample_rounds: adaptive resampling in each window (Wight & Zhao's approach I + resampling): after the window's training, ta_resample_points of ta_resample_candidates random points with the largest PDE residual are added to its ... |
| `ta_resample_candidates` | `int` | `10000` | **Candidates per round:** | ta_resample_rounds: adaptive resampling in each window (Wight & Zhao's approach I + resampling): after the window's training, ta_resample_points of ta_resample_candidates random points with the largest PDE residual are added to its ... |
| `ta_resample_iters` | `int` | `2000` | **Adam iterations per round:** | ta_resample_rounds: adaptive resampling in each window (Wight & Zhao's approach I + resampling): after the window's training, ta_resample_points of ta_resample_candidates random points with the largest PDE residual are added to its ... |
| `ta_growing` | `bool` | `False` | **Growing time range: each window trains on [start, t] (one network)** | ta_growing: each window's residual and boundary points cover the whole time range so far [t_start, t1], with the true initial condition at t_start (Wight & Zhao 2021, "approach I"). |
| `ta_backward` | `bool` | `False` | **Backward compatibility (bc-PINN): keep the solution on earlier windows** | ta_backward: backward-compatible PINN (Mattey & Ghosh 2022): after each window the network's prediction on the time range so far is stored (ta_backward_points spatial points x ta_backward_times times per window) and later windows are held ... |
| `ta_backward_weight` | `float` | `100.0` | **Weight:** | ta_backward: backward-compatible PINN (Mattey & Ghosh 2022): after each window the network's prediction on the time range so far is stored (ta_backward_points spatial points x ta_backward_times times per window) and later windows are held ... |
| `ta_backward_times` | `int` | `10` | **Times/window:** | ta_backward: backward-compatible PINN (Mattey & Ghosh 2022): after each window the network's prediction on the time range so far is stored (ta_backward_points spatial points x ta_backward_times times per window) and later windows are held ... |
| `ta_backward_points` | `int` | `2000` | **Points:** | ta_backward: backward-compatible PINN (Mattey & Ghosh 2022): after each window the network's prediction on the time range so far is stored (ta_backward_points spatial points x ta_backward_times times per window) and later windows are held ... |
| `ta_backward_seed` | `int` | `-1` |  | Seed for choosing bc-PINN's stored points: -1 (default) follows the run's seed setting; the benchmark configurations set 1234 explicitly (the value their recorded runs used). |
| `ta_icgl_fraction` | `float` | `0.0` | **IC-guided learning:** | ta_icgl_fraction: initial-condition guided learning (Mattey & Ghosh): before each window's training, fit the window's starting profile, held constant over the window, for this fraction of the first Adam phase's iterations (0 = off). |
| `parametric_study` | `bool` | `False` |  | -- |
| `parametric_param` | `str` | `"none"` |  | -- |
| `parametric_values` | `str` | `""` |  | -- |
| `learning_rate` | `float` | `0.001` |  | -- |
| `loss_type` | `str` | `"MSE"` |  | -- |

### Problem dimension

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `problem_dim` | `str` | `"1D"` |  | "1D" or "2D" |

### PDE settings

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `pde_expression` | `str` | `"du_t - 0.4 * du_xx"` |  | -- |

### Domain settings

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `x_min` | `float` | `0.0` |  | -- |
| `x_max` | `float` | `1.0` |  | -- |
| `y_min` | `float` | `0.0` |  | -- |
| `y_max` | `float` | `1.0` |  | -- |
| `t_min` | `float` | `0.0` |  | -- |
| `t_max` | `float` | `1.0` |  | -- |
| `num_domain` | `int` | `2000` | **Domain points:** | -- |
| `num_boundary` | `int` | `200` | **Boundary points:** | -- |
| `num_initial` | `int` | `200` | **Initial points:** | -- |
| `num_test` | `int` | `1000` | **Test points:** | -- |
| `point_distribution` | `str` | `"Hammersley"` | **Point distribution:** | -- |
| `anchors_mode` | `str` | `"none"` | **Extra training points:** | Extra training points ("anchors", DeepXDE's anchors=) "none", "file" (a points file: one column per input, x[, y[, z]][, t]) or "box" (anchors_count random points; anchors_box gives a "lo,hi" range per input column, ';'-separated, blank = ... |
| `anchors_file` | `str` | `""` |  | -- |
| `anchors_count` | `int` | `1000` | **Count:** | Extra training points ("anchors", DeepXDE's anchors=) "none", "file" (a points file: one column per input, x[, y[, z]][, t]) or "box" (anchors_count random points; |
| `anchors_box` | `str` | `""` | **Ranges:** | anchors_box gives a "lo,hi" range per input column, ';'-separated, blank = that column's domain range). |
| `plot_type` | `str` | `"Surface"` | **Plot type:** | -- |
| `num_timesteps_line` | `int` | `5` |  | Split from a single shared "num_timesteps" field: the static "Line (time steps)" plot and the GIF animations (Line/Surface Animation) want different defaults -- fewer steps keeps the static overlay readable, while more frames makes the ... |
| `num_timesteps_anim` | `int` | `20` |  | -- |

### Initial condition

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `ic_type` | `str` | `"sin"` |  | -- |
| `ic_expression` | `str` | `"np.sin(np.pi * x[:, 0:1])"` |  | -- |

### Boundary conditions (1D)

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `bc_left` | `float` | `0.0` |  | -- |
| `bc_right` | `float` | `0.0` |  | -- |
| `bc_left_type` | `str` | `"Dirichlet"` |  | -- |
| `bc_right_type` | `str` | `"Dirichlet"` |  | -- |

### Neural network

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `layers` | `List[int]` | `[2, 64, 64, 64, 1]` |  | -- |
| `activation` | `str` | `"tanh"` | **Activation:** | -- |
| `kernel_initializer` | `str` | `"Glorot uniform"` | **Kernel initializer:** | -- |
| `network_type` | `str` | `"FNN"` | **Network type:** | Network architecture: "FNN" (plain fully-connected, shared across all outputs) or "PFNN" (parallel/independent sub-network per output, merging only at the final layer -- DeepXDE's dde.nn.PFNN). layers[0] (the raw input dimension) and ... |
| `msffn_sigmas` | `str` | `"1, 10"` | **Fourier scales (σ):** | Multi-scale Fourier feature networks: network_type "MsFFN" (all inputs) or "STMsFFN" (space and time separately); the scales sigma of the random Fourier features, comma-separated. |
| `msffn_sigmas_t` | `str` | `"1, 10"` | **Time scales (σₜ):** | -- |
| `input_transform_enabled` | `bool` | `False` | **Enable input transform** | Optional input/output transform: x_transformed = x_raw * scale + shift (input, one entry per input dimension) and y_transformed = y_raw * scale + shift (output, one entry per output component). |
| `input_transform_scale` | `List[float]` | `[1.0, 1.0]` |  | -- |
| `input_transform_shift` | `List[float]` | `[0.0, 0.0]` |  | -- |
| `output_transform_enabled` | `bool` | `False` | **Enable output transform** | -- |
| `output_transform_scale` | `List[float]` | `[1.0]` |  | -- |
| `output_transform_shift` | `List[float]` | `[0.0]` |  | -- |
| `input_transform_mode` | `str` | `"scale"` |  | Formula mode -- DeepXDE's apply_feature_transform / apply_output_transform with an arbitrary function, as its examples use them: "scale" keeps the scale/shift rows above, "formula" uses: input_feature_formula -- the network's input ... |
| `input_feature_formula` | `str` | `""` |  | Formula mode -- DeepXDE's apply_feature_transform / apply_output_transform with an arbitrary function, as its examples use them: "scale" keeps the scale/shift rows above, "formula" uses: input_feature_formula -- the network's input ... |
| `output_transform_mode` | `str` | `"scale"` |  | -- |
| `output_transform_formula` | `str` | `""` |  | output_transform_formula -- one expression per output, '\|'-separated, in the coordinates and the raw network outputs by name (a hard constraint, e.g. |

### Training

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `optimizer` | `str` | `"adam"` |  | -- |
| `iterations` | `int` | `10000` |  | -- |
| `optimizer2` | `str` | `"none"` |  | -- |
| `iterations2` | `int` | `5000` |  | -- |
| `loss_weights` | `List[float]` | `[1.0, 1.0, 1.0, 1.0]` |  | -- |
| `loss_weight_obs` | `float` | `1.0` |  | -- |
| `inv_param_log_scale` | `bool` | `False` | **Log scale** | -- |
| `inv_param_save` | `str` | `"Every 100 iters"` |  | -- |

### Inverse PINN

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `problem_type` | `str` | `"Forward"` |  | -- |

### Multi-output

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `num_outputs` | `int` | `1` |  | layers[0] (the raw input dimension) and layers[-1] (num_outputs) are unaffected; |
| `num_equations` | `int` | `0` |  | Number of PDE residuals; 0 = one per output, as before. |
| `output_names` | `str` | `"u"` |  | -- |
| `pde_expressions` | `str` | `"du_t - 0.4 * du_xx"` |  | -- |
| `constants` | `str` | `""` |  | Named constants: JSON list of {"name", "value"}; each value a number or an expression in math functions and earlier constants. |

### BCs left/right (all dims)

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `bc_left_types` | `str` | `"Dirichlet"` |  | -- |
| `bc_right_types` | `str` | `"Dirichlet"` |  | -- |
| `bc_left_values` | `str` | `"0.0"` |  | -- |
| `bc_right_values` | `str` | `"0.0"` |  | -- |
| `bc_left_active` | `str` | `"True"` |  | -- |
| `bc_right_active` | `str` | `"True"` |  | -- |
| `bc_left_deriv` | `str` | `"False"` |  | -- |
| `bc_right_deriv` | `str` | `"False"` |  | -- |

### BCs bottom/top (2D only)

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `bc_bottom_types` | `str` | `"Dirichlet"` |  | -- |
| `bc_top_types` | `str` | `"Dirichlet"` |  | -- |
| `bc_bottom_values` | `str` | `"0.0"` |  | -- |
| `bc_top_values` | `str` | `"0.0"` |  | -- |
| `bc_bottom_active` | `str` | `"True"` |  | -- |
| `bc_top_active` | `str` | `"True"` |  | -- |
| `bc_bottom_deriv` | `str` | `"False"` |  | -- |
| `bc_top_deriv` | `str` | `"False"` |  | -- |

### ICs

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `ic_expressions` | `str` | `"np.sin(np.pi * x[:, 0])"` |  | -- |
| `ic_active` | `str` | `"True"` |  | -- |
| `ic_hard` | `str` | `""` |  | Exact (hard) initial conditions, one "True"/"False" per output like ic_active: output = IC(x) + tau * network output, tau = (t - t_min) / (t_max - t_min), so the condition holds exactly and needs no loss term (its IC row is switched off). |
| `periodic_inputs` | `bool` | `False` | **Exact periodicity (periodic inputs)** | Exact periodicity: every axis with a periodic boundary condition enters the network as cos(2 pi x / L), sin(2 pi x / L) instead of x (L = the domain's length on that axis), so the solution is periodic by construction. |
| `loss_weights_multi` | `str` | `""` |  | -- |
| `loss_weights_from_initial` | `float` | `0.0` |  | &gt; 0 -&gt; before training, each loss weight becomes c / (that term's initial loss), as in DeepXDE's wave example (loss_weights = 5 / initial_losses); the weights above are ignored. |
| `integral_points` | `int` | `20` |  | Gauss-Legendre points for integral(...) terms in the equations (integro-differential ODEs) -- DeepXDE's quad_deg. |
| `fractional_alpha` | `str` | `"1.5"` |  | Fractional equations -- frac(u) in an equation is DeepXDE's discretized fractional operator applied to output u (1D: the sum of the left and right Riemann-Liouville derivatives of order alpha; 2D/3D: the integral over directions of the ... |
| `fractional_resolution` | `str` | `"100"` | **Resolution:** | -- |
| `fractional_meshtype` | `str` | `"dynamic"` | **Mesh:** | -- |

### Inverse settings

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `inverse_param_name` | `str` | `"trainable_variable"` |  | -- |
| `inverse_param_init` | `float` | `1.0` |  | -- |
| `inverse_data_file` | `str` | `""` |  | -- |
| `inverse_ic_type` | `str` | `"expression"` |  | -- |
| `inverse_ic_file` | `str` | `""` |  | -- |
| `forward_ic_from_file` | `bool` | `False` |  | -- |
| `forward_ic_file` | `str` | `""` |  | -- |
| `ic_file_anchors` | `bool` | `False` |  | Also check the equation at the IC file's points (anchors). |
| `template_type` | `str` | `""` |  | -- |
| `inverse_variables_json` | `str` | `""` |  | Multiple trainable variables (Inverse). |
| `inverse_obs_output_idx` | `int` | `0` |  | Which model output column the (legacy, single) measured-data file corresponds to (0-based index into output_names). |
| `inverse_obs_files_json` | `str` | `""` |  | Multiple measured-data files (Inverse). |
| `inverse_obs_anchors` | `bool` | `True` | **Check the equation at the measured points too (anchors)** | Every measured point is also an anchor (the equation is checked there too). |
| `inverse_obs_reduce` | `str` | `"all"` |  | Use fewer of the measured points (the files stay as they are): "all", "grid" (N points per space axis, evenly spaced, edges kept; each time in a file reduced the same way) or "random" (N points, with the run's seed). |
| `inverse_obs_grid_n` | `int` | `51` |  | -- |
| `inverse_obs_random_n` | `int` | `1000` |  | -- |
| `export_enabled` | `bool` | `False` | **Enable solution data export** | Export Off by default -- solution data used to be written on every non-Inverse run unconditionally; now it only happens when the user opts in via the Export Solution Data dialog's checkbox. |
| `export_grid_size` | `int` | `101` |  | -- |
| `export_t_steps` | `int` | `11` |  | -- |
| `plot_output_idx` | `int` | `0` |  | -- |
| `plot_custom_expr` | `str` | `""` |  | Optional derived scalar field to plot in the Results panel instead of a single raw output column -- e.g. |
| `plot_custom_label` | `str` | `""` |  | Display label for the field above (colorbar/axis label, and the printed output name) -- falls back to the expression itself when left blank. |
| `plot_swap_xt` | `bool` | `True` |  | 1D "Surface" plot axis orientation (both the Standard and Time-Adaptive final solution plots): True (the default) puts t on the x-axis and the spatial domain x on the y-axis; False keeps the original orientation (x on the x-axis, t on the ... |
| `plot_figsize_mode` | `str` | `"Default"` |  | Figure-size standardization for single-panel results plots (Loss, Line, Surface, the two Animation GIF types, Parameter Convergence, and their Time-Adaptive equivalents) -- deliberately NOT applied to the multi-column Error Analysis ... |
| `plot_figsize_w` | `float` | `7.0` |  | "Custom" uses plot_figsize_w/plot_figsize_h for every in-scope plot regardless of its own default dimensions. |
| `plot_figsize_h` | `float` | `5.0` |  | "Custom" uses plot_figsize_w/plot_figsize_h for every in-scope plot regardless of its own default dimensions. |

### L-BFGS

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `lbfgs_use_default` | `bool` | `True` | **Use recommended L-BFGS settings** | -- |
| `lbfgs_maxcor` | `int` | `100` |  | -- |
| `lbfgs_ftol` | `float` | `0.0` |  | -- |
| `lbfgs_gtol` | `float` | `1e-8` |  | -- |
| `float_type` | `str` | `"float32"` |  | -- |
| `lbfgs_maxiter` | `int` | `15000` |  | -- |
| `lbfgs_maxfun` | `int` | `18750` |  | -- |
| `lbfgs_maxls` | `int` | `50` |  | -- |
| `lbfgs_float_type` | `str` | `"float32"` |  | -- |
| `ic_pretrain` | `bool` | `False` | **Enable IC-guided pre-training** | -- |
| `ic_pretrain_optimizer` | `str` | `"adam"` | **Optimizer:** | -- |
| `ic_pretrain_iterations` | `int` | `20000` | **Iterations:** | -- |
| `ic_pretrain_num_test` | `int` | `10000` | **Test points:** | -- |
| `ic_pretrain_num_initial` | `int` | `1000` | **Initial points:** | -- |
| `ic_pretrain_lr` | `float` | `1e-3` | **Learning rate:** | -- |
| `ic_pretrain_decay_type` | `str` | `"none"` |  | Learning-rate decay for IC pre-training, same choices as a Training Phase: "none", "step", "cosine", "exponential", "inverse time". |
| `ic_pretrain_decay_p1` | `float` | `5000` |  | -- |
| `ic_pretrain_decay_p2` | `float` | `0.9` |  | -- |
| `ic_pretrain_loss` | `str` | `"MSE"` | **Loss function:** | -- |
| `ic_pretrain_restore` | `bool` | `False` |  | -- |
| `ic_pretrain_restore_path` | `str` | `""` |  | -- |
| `optimizer_scheduler` | `bool` | `False` | **Enable Optimizer Scheduler** | -- |
| `scheduler_phases` | `str` | `""` |  | -- |
| `scheduler_same_weights` | `bool` | `True` | **Use same weights for all phases** | -- |
| `weight_decay` | `float` | `0.0` |  | Weight decay (L2 regularization on the network). |
| `cb_early_stopping` | `bool` | `False` | **Stop training when loss stops improving** | Training Callbacks (all opt-in, off by default). |
| `cb_early_stopping_min_delta` | `float` | `0.0` |  | -- |
| `cb_early_stopping_patience` | `int` | `2000` |  | -- |
| `cb_early_stopping_baseline` | `str` | `""` |  | blank = None (no baseline) |
| `cb_early_stopping_monitor` | `str` | `"loss_train"` |  | or "loss_test" |
| `cb_early_stopping_start_from` | `int` | `0` |  | needs deepxde&gt;=1.12.0, guarded at codegen time |
| `cb_point_resampler` | `bool` | `False` | **Periodically resample collocation points** | -- |
| `cb_point_resampler_period` | `int` | `100` |  | -- |
| `cb_point_resampler_pde_points` | `bool` | `True` | **Resample PDE (domain) points** | -- |
| `cb_point_resampler_bc_points` | `bool` | `False` | **Also resample boundary-condition points** | -- |
| `cb_model_checkpoint` | `bool` | `False` | **Periodically save the model during training** | -- |
| `cb_checkpoint_period` | `int` | `1000` |  | -- |
| `cb_checkpoint_save_better_only` | `bool` | `True` | **Only save when the monitored loss improves** | -- |
| `cb_checkpoint_monitor` | `str` | `"train loss"` |  | or "test loss" |
| `cb_timer` | `bool` | `False` | **Stop training after a time budget** | -- |
| `cb_timer_minutes` | `float` | `60.0` |  | -- |
| `causal_training` | `bool` | `False` | **Causal training** | Training methods (adaptive training). |
| `causal_chunks` | `int` | `32` | **Time chunks:** | -- |
| `causal_eps` | `float` | `1.0` | **Fixed ε:** | -- |
| `causal_anneal` | `bool` | `True` | **Raise ε step by step (0.01 → 100)** | The paper's schedule (Algorithm 1): epsilon steps through 0.01, 0.1, 1, 10, 100, moving on once every chunk weight is above 0.99. |
| `point_weights` | `str` | `"None"` | **Per-point weights:** | "None", "RBA" or "SA-PINN" |
| `rba_gamma` | `float` | `0.999` | **Decay γ:** | -- |
| `rba_eta` | `float` | `0.01` | **Step η:** | -- |
| `sa_lr` | `float` | `0.005` | **Weights' learning rate:** | Per-point weights scale each training point's residual by its own weight: "RBA" (residual-based attention, Anagnostopoulos et al., CMAME 2024: lambda &lt;- gamma*lambda + eta*\|r\|/max\|r\|) or "SA-PINN" (self-adaptive weights, McClenny & ... |
| `sa_normalize` | `bool` | `False` | **Keep the mean weight at 1** | SA-PINN: rescale the weights to mean 1 after every ascent step (not in the paper, whose weights only grow by ~sa_lr per Adam step). |
| `point_weights_lbfgs` | `str` | `"plain"` | **During L-BFGS:** | During L-BFGS: "plain" drops the point weights (L-BFGS refines on the plain loss) or "keep" holds them at their last values. |
| `training_monitors_enabled` | `bool` | `False` | **Watch expressions during training (diagnostic only)** | Training Monitors: optional, pure-observability DeepXDE dde.callbacks.OperatorPredictor instances. |
| `training_monitors` | `str` | `"[]"` |  | JSON-encoded list of dicts (plain json.dumps/json.loads -- the convention sweep_parameters/geom_custom_shapes_json use, NOT the repr()/ast.literal_eval convention ea_files uses): {"name": str, -- label; used in the log filename and as the ... |
| `plot_colormap` | `str` | `"jet"` |  | -- |
| `plot_all_outputs` | `bool` | `True` | **All outputs** | Multi-output problems: also save every output's own plot, as solution_plot_&lt;name&gt;.&lt;ext&gt;, besides the selected "Plot output". |
| `plot_levels` | `int` | `100` |  | -- |
| `plot_resolution` | `int` | `200` |  | -- |
| `plot_dpi` | `int` | `300` |  | -- |
| `plot_n_2d_snapshots` | `int` | `2` |  | -- |
| `plot_colorbar` | `bool` | `True` |  | -- |
| `plot_auto_range` | `bool` | `True` |  | -- |
| `plot_vmin` | `float` | `-1.0` |  | -- |
| `plot_vmax` | `float` | `1.0` |  | -- |
| `plot_linewidth` | `float` | `2.0` |  | -- |
| `plot_linewidth_anim` | `float` | `3.0` |  | Line Animation (GIF)'s own line width -- kept separate from the static plots' plot_linewidth above (see num_timesteps_line/_anim comment) so the two can default differently. |
| `plot_fps` | `int` | `10` |  | -- |
| `plot_title_fontsize` | `float` | `13.0` |  | Font-size standardization: previously every plot across the app hardcoded its own title/axis-label font sizes ad hoc, with real inconsistencies even within the same plot type (e.g. |
| `plot_subplot_title_fontsize` | `float` | `10.0` |  | -- |
| `plot_axis_label_fontsize` | `float` | `10.0` |  | plot_axis_label_fontsize covers x/y axis label text and tick labels. |
| `param_plot_color` | `str` | `"#e03131"` |  | Parameter convergence plot and animation (inverse problems; Plot settings -&gt; Line / Animation): the same style in Solve, the exported script and Restore. |
| `param_plot_linewidth` | `float` | `3.0` |  | -- |
| `param_plot_markers` | `bool` | `False` |  | -- |
| `param_plot_show_true` | `bool` | `True` |  | -- |
| `param_true_color` | `str` | `"#212529"` |  | -- |
| `param_true_style` | `str` | `"--"` |  | -- |
| `param_plot_show_final` | `bool` | `True` |  | -- |
| `param_final_color` | `str` | `"#868e96"` |  | -- |
| `param_final_style` | `str` | `":"` |  | -- |
| `param_anim_color` | `str` | `"#e03131"` |  | -- |
| `param_anim_linewidth` | `float` | `3.0` |  | -- |
| `param_anim_frames` | `int` | `60` |  | -- |
| `param_plot_figures` | `str` | `"combined"` |  | "combined": one figure with a panel per parameter; "separate": also one figure per parameter (&lt;name&gt;_convergence_plot.png / _animation.gif) |
| `param_anim_figures` | `str` | `"combined"` |  | -- |
| `param_plot_draw` | `str` | `"line"` |  | How the curve is drawn: "line", "line_markers" or "markers" (dots only), the marker's shape and size; the animation can also mark the current value with a dot at the curve's tip. param_plot_markers above is the older on/off for markers ... |
| `param_plot_marker` | `str` | `"o"` |  | -- |
| `param_plot_marker_size` | `float` | `4.0` |  | -- |
| `param_anim_draw` | `str` | `"line"` |  | -- |
| `param_anim_head` | `bool` | `True` |  | -- |
| `plot_grid` | `bool` | `True` |  | Grid lines on every line-type figure (line plots and animations, line comparisons, loss, parameter convergence); Plot settings -&gt; General. |
| `plot_ref_style` | `str` | `"line"` |  | The reference in line comparisons (Error Analysis): "line" or "dots". |
| `plot_sig_digits` | `int` | `4` |  | Numbers written in figures (Plot settings -&gt; General -&gt; Numbers): significant digits (parameter values get two more), and scientific notation "auto" (below 0.01 or from 100,000 up), "always" or "never", written as "1.2 x 10^-4" ("times") ... |
| `plot_pinn_color` | `str` | `"#ff6b6b"` |  | Line colours of the PINN and the reference in every line figure, the \|Error\| panels' colormap ("" = the solution's), how many panels a row of an Error Analysis grid has, and the animations' resolution. |
| `plot_ref_color` | `str` | `"#4dabf7"` |  | -- |
| `plot_err_colormap` | `str` | `""` |  | -- |
| `plot_ea_cols` | `int` | `4` |  | -- |
| `plot_dpi_anim` | `int` | `150` |  | -- |
| `plot_legend_fontsize` | `float` | `0.0` |  | Legend font size (0 = each figure's own, 7-10) and the grid of the 3D surfaces (faces get slow to draw above about 40 points per side). |
| `plot_resolution_3d` | `int` | `40` |  | -- |
| `plot_iso_level` | `str` | `""` |  | The level of the surface drawn in Error Analysis' 3D view of a 3D box problem ("" = the middle of the reference's range, e.g. 0 for a phase field between -1 and 1). |
| `plot_sci_notation` | `str` | `"auto"` |  | -- |
| `plot_sci_style` | `str` | `"times"` |  | -- |
| `loss_plot_mode` | `str` | `"train_test"` |  | Loss Plot Settings: previously every loss plot across the app hardcoded "semilogy of summed Train/Test loss" with no user control, and model.train()'s own display_every -- which doubles as the loss plot's own sampling granularity, since ... |
| `loss_display_every` | `int` | `1000` |  | loss_display_every: a single value applied to every model.train() call that feeds the plotted loss curve (confirmed with the user: one shared value, not kept split by phase) -- replaces the old scattered 1000/200/500 constants; |
| `loss_plot_linewidth` | `float` | `2.0` |  | -- |
| `loss_plot_log_y` | `bool` | `True` |  | loss_plot_log_y: True (default) matches the always-semilogy behavior every loss plot in the app has always had. |
| `line_slice_y_auto` | `bool` | `True` | **Auto (domain midpoint)** | Line (time steps) and Line Animation (GIF) plot a 1D curve of u vs x only -- for a 2D/3D problem that leaves y (and z, in 3D) unaccounted for. |
| `line_slice_y` | `float` | `0.0` |  | -- |
| `line_slice_z_auto` | `bool` | `True` | **Auto (domain midpoint)** | -- |
| `line_slice_z` | `float` | `0.0` |  | -- |
| `plot_title_override` | `str` | `""` |  | Optional title/axis-label overrides for the plot built from the Setup tab's "Plot output" settings (live Solve, "Export as DeepXDE Script"). |
| `plot_xlabel_override` | `str` | `""` |  | -- |
| `plot_ylabel_override` | `str` | `""` |  | -- |
| `ea_files` | `str` | `"[]"` |  | -- |
| `ea_do_line` | `bool` | `True` |  | -- |
| `ea_do_surface` | `bool` | `True` |  | -- |
| `ea_metric_l2` | `bool` | `True` |  | Error Metrics to Compute: previously the Error Analysis dialog's "L2 Relative"/"MSE"/"Max Error" checkboxes were pure dead UI -- their state was stored but nothing downstream ever read it, so every metric was always computed and shown ... |
| `ea_metric_mse` | `bool` | `False` |  | -- |
| `ea_metric_max` | `bool` | `False` |  | -- |
| `ea_ref_mode` | `str` | `"files"` |  | Reference solution from a formula instead of files: ea_ref_mode "files" \| "formula"; ea_ref_formula has one expression per output ('\|'-separated, blank = no reference for that output) in x, y, z, t with numpy math names plus erf/norm_cdf ... |
| `ea_ref_formula` | `str` | `""` |  | ea_ref_formula has one expression per output ('\|'-separated, blank = no reference for that output) in x, y, z, t with numpy math names plus erf/norm_cdf; |
| `ea_ref_times` | `str` | `""` |  | ea_ref_times are the comparison times (comma-separated; |
| `ea_ref_grid` | `int` | `0` |  | ea_ref_grid is the points per axis of the reference grid (0 = 201 in 1D, 101 in 2D, 31 in 3D). |

### Geometry type & 3D domain

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `geometry_type` | `str` | `"Rectangle"` |  | 2D: Rectangle\|Disk\|Ellipse\|Triangle\|Polygon\|Custom ; 3D: Cuboid\|Sphere |
| `z_min` | `float` | `0.0` |  | _auto=True (the default, matching the previous hardcoded behavior exactly) means "use (y_min+y_max)/2 / (z_min+z_max)/2 at codegen time"; |
| `z_max` | `float` | `1.0` |  | _auto=True (the default, matching the previous hardcoded behavior exactly) means "use (y_min+y_max)/2 / (z_min+z_max)/2 at codegen time"; |
| `geom_custom_shapes_json` | `str` | `"[]"` |  | Custom geometry (2D only, Phase 1): an ORDERED list of primitive shapes combined with boolean operations -- a small COMSOL-style CSG sequence, mirroring DeepXDE's own CSGUnion/CSGDifference/ CSGIntersection, which this is built directly on ... |
| `geom_sketch` | `bool` | `False` |  | True when the Custom geometry was drawn on the Sketch canvas (Domain -&gt; Shape -&gt; Sketch). |

### Disk / Ellipse / Sphere center

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `geom_center_x` | `float` | `0.5` |  | -- |
| `geom_center_y` | `float` | `0.5` |  | -- |
| `geom_center_z` | `float` | `0.5` |  | -- |
| `geom_radius` | `float` | `0.5` | **Radius:** | -- |

### Ellipse

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `geom_semi_major` | `float` | `0.5` |  | -- |
| `geom_semi_minor` | `float` | `0.3` |  | -- |
| `geom_angle` | `float` | `0.0` |  | -- |

### Triangle / Polygon vertices, "x1,y1;x2,y2;..." format

| Field | Type | Default | Window control | Description (from the source comment) |
|---|---|---|---|---|
| `geom_triangle_vertices` | `str` | `"0,0;1,0;0,1"` |  | -- |
| `geom_polygon_vertices` | `str` | `"0,0;1,0;1,0.6;0.5,1;0,0.6"` |  | -- |
| `bc_boundary_json` | `str` | `""` |  | Shape-aware boundary conditions for non-box geometries (JSON-encoded) |
| `bc_edge_json` | `str` | `""` |  | Triangle/Polygon: one BC group per edge per output |
| `custom_bc_json` | `str` | `""` |  | Fully-customizable BC builder, used when no Quick Example template is selected -- a flat, user-authored list of BCs (any DeepXDE BC class, any location, any output), independent of geometry_type. |
| `steady_state` | `bool` | `False` |  | Steady-state (time-independent) problems, e.g. the Poisson equation. |
| `gpu_device_index` | `int` | `0` |  | GPU device selection & memory reservation for the generated training script. device_index picks which CUDA device to train on (0 = first GPU, matching all prior behavior -- unchanged default); memory_fraction is the maximum share of that ... |
| `gpu_memory_fraction` | `float` | `0.95` |  | -- |
| `train_device` | `str` | `"auto"` |  | Where to train: "auto" = the GPU when there is one (the default), "cpu" = always the CPU. |
| `cpu_threads` | `int` | `0` |  | CPU threads for CPU training: 0 = Automatic (about one per performance core, at most 8), or a number of your own. |
| `use_random_seed` | `bool` | `True` |  | Reproducibility. |
| `random_seed` | `int` | `2026` |  | When on (the default), the generated script calls dde.config.set_random_seed(random_seed) once, right at the start -- this seeds NumPy, PyTorch and Python's own `random` together (that's what DeepXDE's helper does internally), so point ... |
| `sweep_enabled` | `bool` | `False` | **Enable Parameter Sweep** | Parameter Sweep. |
| `sweep_mode` | `str` | `"oat"` | **Mode:** | "oat" (one-at-a-time, every other swept field held at its baseline value), "grid" (every combination of every parameter's values -- COMSOL calls this "All combinations"), or "zip" (every swept parameter's Nth value run together as one ... |
| `sweep_bayes_calls` | `int` | `20` | **Runs:** | Sweep_mode "bayes" -- Bayesian search (DeepXDE's Helmholtz HPO example): the baseline, then sweep_bayes_calls - 1 runs each chosen by a Gaussian-process model of the results so far (pinnstudio/core/bayes_opt.py), minimizing the final ... |
| `sweep_bayes_initial` | `int` | `5` | **Random first:** | The first sweep_bayes_initial runs after the baseline are random. |
| `sweep_bayes_objective` | `str` | `"loss"` | **Minimize:** | -- |
| `sweep_bayes_seed` | `int` | `0` |  | -- |
| `sweep_parameters` | `str` | `"[]"` |  | JSON-encoded list of swept parameters, one dict per added row: {"id": "&lt;sweep_registry id&gt;", "mode": "list", "values": [..]} {"id": "...", "mode": "linear"\|"log", "min": .., "max": .., "n": ..} "id" is a key into ... |
| `sweep_save_dir` | `str` | `""` | **Save to:** | Where a sweep organizes its own output -- distinct from save_dir above (which is whatever the Setup tab's own "Save results to" is set to, and would make every run in a sweep collide into the same folder if reused as-is). |
| `sweep_include_baseline` | `str` | `"auto"` |  | Also run the current settings unchanged (the "baseline") before the swept values: "on", "off", or "auto" (the default) -- on only where it is a needed reference point, One-at-a-time with 2+ parameters and Bayesian search ... |
| `sweep_label` | `str` | `""` |  | Name part of the sweep's folder, "&lt;sweep_label&gt;_sweep__&lt;timestamp&gt;" (set by the GUI from the loaded example / problem, like a normal run's "&lt;label&gt;__&lt;timestamp&gt;" folder). |
| `sweep_export_mode` | `str` | `"same_as_setup"` |  | What each run in a sweep actually saves (model, loss/solution plot, Error Analysis figures if configured) is controlled by the same plot_type/plot_output_idx/plot_custom_expr/plot_custom_label/ export_t_steps fields a normal Solve already ... |
| `sweep_plot_type` | `str` | `"Surface"` |  | -- |
| `sweep_plot_output_idx` | `int` | `0` |  | -- |
| `sweep_plot_custom_expr` | `str` | `""` |  | -- |
| `sweep_plot_custom_label` | `str` | `""` |  | -- |
| `sweep_export_t_steps` | `int` | `11` |  | -- |

<!-- END GENERATED CONFIG TABLE -->

<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

#### Problem type, dimension and ODEs

**Entry points**

- `MainWindow._build_setup_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Problem page: dimension, forward or inverse, steady or in time
- `MainWindow._on_dim_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): switching between ODE, 1D, 2D and 3D
- `generate_script` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the script Solve runs, for every problem type

**Helpers**

- `_ode_to_1d` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): an ODE problem rewritten as a 1D problem in t
- `_effective_config` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the settings as the generators read them

**Tests**

- [`tests/test_round74_ode.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round74_ode.py): ODE problems
- [`tests/test_templates.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_templates.py): every template in Forward and Inverse mode

Described in [Forward problems](../reference/forward.md), [Problem](../guide/problem.md).

#### Equation parsing and residual generation

**Entry points**

- `MainWindow._build_pde_inputs` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Equations page: one expression per equation, outputs, constants
- `_simplify_pde_expr` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): typed expression to the residual's Python code

**Helpers**

- `_simplify_expr` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): math names and powers to NumPy/PyTorch syntax
- `_fold_numeric_calls` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): numeric calls such as sqrt(3) folded to values
- `_pde_coord_names` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): which coordinates (x, y, z, t) the equations use
- `_clean_needed_derivs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): derivatives (du_x, du_xx, ...) the exported residual computes
- `_n_equations` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): number of residuals, independent of the number of outputs
- `_constant_values` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): named constants
- `_integral_text` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): integral(...) terms
- `_frac_data_text` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): frac(...) terms (DeepXDE's fractional PDE data)
- [`pinnstudio/ui/equation.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/equation.py): live preview of the typed equations
- [`pinnstudio/core/constants.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/constants.py): constants table

**Tests**

- [`tests/test_round61_pde_coordinates.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round61_pde_coordinates.py): coordinates and math functions in equations
- [`tests/test_round63_equation_count.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round63_equation_count.py): number of equations and outputs
- [`tests/test_round70_constants.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round70_constants.py): named constants

Described in [ODEs and PDEs](../reference/equations.md), [Coupled systems](../reference/coupled.md), [Equations, outputs and constants](../guide/equations.md).

#### Geometry, the Sketch and boundary edges

**Entry points**

- `MainWindow._build_domain_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Geometry, Collocation points, Initial and Boundary conditions pages
- `MainWindow._on_geometry_type_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the shape and its parameters
- `MainWindow._build_custom_geom_shapes_json` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Custom geometry rows (combined shapes)

**Helpers**

- `_build_custom_geom_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): combined shapes to DeepXDE geometry code
- `_clean_geom_inner` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the geometry constructor in the exported script
- `_shape_bbox` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): bounding box of a non-box shape
- [`pinnstudio/ui/sketch_canvas.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_canvas.py): 2D Sketch
- [`pinnstudio/ui/sketch_edges.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_edges.py): edges of the sketch as boundary parts
- [`pinnstudio/ui/sketch_import.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_import.py): DXF/SVG import
- [`pinnstudio/ui/sketch3d.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch3d.py): 3D Sketch

**Tests**

- [`tests/test_custom_geometry.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_custom_geometry.py): Custom geometry: validation, panel, generated code
- [`tests/test_geometry_box.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_geometry_box.py): bounding box of non-box shapes
- [`tests/test_sketch_geometry.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_sketch_geometry.py): sketched shapes become Custom-geometry rows

Described in [Geometry](../reference/geometry.md), [Geometry and domain](../guide/geometry.md).

#### Collocation points and domain preview

**Entry points**

- `MainWindow._build_domain_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Collocation points page (with the domain pages)
- `MainWindow._preview_domain` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): domain and point preview

**Helpers**

- `_anchor_points_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): extra training points (DeepXDE anchors)
- [`pinnstudio/ui/view3d.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/view3d.py): 3D view of the domain

**Tests**

- [`tests/test_round65_anchors.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round65_anchors.py): extra training points
- [`tests/test_view3d.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_view3d.py): 3D view

Described in [Collocation points](../guide/points.md).

#### Boundary conditions

**Entry points**

- `MainWindow._add_custom_bc_entry` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): one row of the Boundary conditions page
- `MainWindow._build_custom_bc_json` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): rows to the saved settings (custom_bc_json)

**Helpers**

- `_clean_bc_row_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): one row to DeepXDE boundary-condition code
- `_bc_operator_def_lines` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Operator and Point Set Operator conditions
- `_bc_sides` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): side names (left, right, ...) in Where expressions
- `_bc_check_periodic_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): checks before a periodic condition

**Tests**

- [`tests/test_round68_bc_where_names.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round68_bc_where_names.py): Where expressions and side names
- [`tests/test_round71_pointset_points.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round71_pointset_points.py): Point Set conditions without a file
- [`tests/test_sketch_edges.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_sketch_edges.py): sketch edges as boundary conditions

Described in [Boundary conditions](../reference/boundary-conditions.md), [Boundary conditions](../guide/boundary-conditions.md).

#### Initial conditions and imported data

**Entry points**

- `MainWindow._build_domain_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Initial conditions page: one row per output, formula or file

**Helpers**

- `MainWindow._ic_file_anchors_checked` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): an initial condition read from a file: also check the equation there
- `_clean_active_ic_outputs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): which outputs get an initial-condition term
- `_bc_initial_operator_rows` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): operator conditions applied at the initial time
- [`pinnstudio/core/portable_paths.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/portable_paths.py): data paths saved as @REF/... so problems open anywhere

**Tests**

- [`tests/test_ic_file_choice.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ic_file_choice.py): formula or file, never both
- [`tests/test_round64_initial_operator_decay.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round64_initial_operator_decay.py): operator condition at the initial time
- [`tests/test_portable_paths.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_portable_paths.py): portable data paths

Described in [Initial conditions and imported data](../reference/initial-conditions.md), [Initial conditions](../guide/initial-conditions.md).

#### Inverse problems: unknown parameters

**Entry points**

- `MainWindow._build_inverse_panel` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Inverse page
- `MainWindow._add_inverse_var_row` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): one unknown parameter (initial and true value)

**Helpers**

- `_parse_inverse_variables` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): unknown parameters to DeepXDE variables
- `inv_record_period` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): how often the parameter values are recorded
- `_param_axis_helper_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): parameter convergence plots
- `MainWindow.INVERSE_AUTO_VARS` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the templates' unknown parameters in Inverse mode

**Tests**

- [`tests/test_added_inverse_modes.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_added_inverse_modes.py): Inverse modes added to DeepXDE forward examples
- [`tests/test_burgers2d_inverse_setup.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_burgers2d_inverse_setup.py): an inverse template set up end to end
- [`tests/test_round49_param_axis_shared.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round49_param_axis_shared.py): parameter plots

Described in [Inverse problems](../reference/inverse.md), [Inverse problems and measured data](../guide/inverse.md).

#### Observational (measured) data

**Entry points**

- `MainWindow._add_inverse_data_row` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): one measured-data file on the Inverse page
- `MainWindow._build_inverse_obs_files_json` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): data rows to the saved settings

**Helpers**

- `_parse_inverse_obs_files` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): data files to DeepXDE point-set conditions
- `obs_reduce_args` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the Use: choice (all points, grid, random subset)
- [`pinnstudio/core/obs_reduce.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/obs_reduce.py): reading and thinning the data points

**Tests**

- [`tests/test_obs_reduce.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_obs_reduce.py): Use: all points, grid, random subset
- [`tests/test_large_data_note.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_large_data_note.py): note for large data files

Described in [Observational data](../reference/observations.md), [Inverse problems and measured data](../guide/inverse.md).

#### Networks, Fourier features and transforms

**Entry points**

- `MainWindow._build_network_panel` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Neural network page
- `MainWindow._on_network_type_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): FNN, PFNN, MsFFN, STMsFFN

**Helpers**

- `_net_construction_helper_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the network in the generated script
- `_parse_sigmas` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Fourier-feature scales
- `_net_transforms_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): input features and output transforms

**Tests**

- [`tests/test_network_architecture.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_network_architecture.py): network types and sizes
- [`tests/test_round73_msffn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round73_msffn.py): multi-scale Fourier-feature networks
- [`tests/test_round62_transform_formulas.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round62_transform_formulas.py): input and output transform formulas

Described in [Networks and Fourier features](../reference/networks.md), [Neural network](../guide/network.md).

#### Optimizers, training phases and callbacks

**Entry points**

- `MainWindow._build_training_panel` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Training page
- `MainWindow._add_scheduler_phase` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): one row of Training phases

**Helpers**

- `MainWindow._validate_optimizer_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): checks before Solve
- `_clean_sched_phases` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the phases a run actually trains
- `_clean_phase_train_lines` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): compile and train calls for one phase
- `_lbfgs_maxfun` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): L-BFGS iteration limit
- `_build_train_cbs_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): callbacks: early stopping, point resampling, checkpoints, timer
- `phases_from_legacy` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py): older saved problems to training phases

**Tests**

- [`tests/test_round78_example_phases.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round78_example_phases.py): examples train through the phases shown
- [`tests/test_lbfgs_steps.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_lbfgs_steps.py): L-BFGS steps
- [`tests/test_training_callbacks_visibility.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_training_callbacks_visibility.py): callbacks

Described in [Optimizers and training phases](../reference/optimizers.md), [Training phases](../guide/training.md).

#### Loss weights

**Entry points**

- `MainWindow._build_weight_inputs` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Loss weights page: one weight per term

**Helpers**

- `_clean_loss_weights` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the weight list in term order
- `_clean_phase_weights` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): weights per training phase
- `_init_loss_weights_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Scale by initial losses

**Tests**

- [`tests/test_steady_state_ic_weights.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_steady_state_ic_weights.py): no initial-condition weight without time
- [`tests/test_round75_wave_helmholtz_bayes.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round75_wave_helmholtz_bayes.py): weights from the initial losses (wave example)
- [`tests/test_entered_values.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_entered_values.py): weights used as entered

Described in [Loss weights](../reference/loss-weights.md), [Loss weights](../guide/loss-weights.md).

#### IC pre-training and IC-guided learning

**Entry points**

- `MainWindow._build_training_panel` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): IC pre-training options on the Training page
- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): IC-guided learning in the time-window options
- `_ic_pretrain_train_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): pre-training on the initial condition, in both scripts

**Helpers**

- `_icgl_outputs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): outputs IC-guided learning fits in each window
- `_ta_icgl_iterations` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): IC-guided learning iterations per window

**Tests**

- [`tests/test_round47_ic_pretrain_shared_block.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round47_ic_pretrain_shared_block.py): pre-training block shared by both scripts
- [`tests/test_round40_ic_pretrain_restore_parity.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round40_ic_pretrain_restore_parity.py): pre-training and restoring
- [`tests/test_ta_bcpinn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_bcpinn.py): IC-guided learning with bc-PINN windows

Described in [IC pre-training and guided learning](../reference/pretraining.md), [Adaptive training](../guide/adaptive.md).

#### Residual-based adaptive refinement (RAR)

**Entry points**

- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Adaptive training page
- `_gs_rar_loop_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the RAR loop of the Solve script

**Helpers**

- `_rar_phase` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): loss and weights of an RAR round
- `_rar_tolerance_check` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): stop on a residual tolerance
- `_rar_optimizer` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the optimizer and learning rate of each round
- `_clean_rar_diagnostics_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): RAR in the exported script, with per-round figures

**Tests**

- [`tests/test_rar_round_diagnostics.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_rar_round_diagnostics.py): per-round diagnostics
- [`tests/test_round66_rar_tolerance.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round66_rar_tolerance.py): residual tolerance
- [`tests/test_steady_state_rar_gating.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_steady_state_rar_gating.py): RAR for steady problems

Described in [RAR](../reference/rar.md), [Adaptive training](../guide/adaptive.md).

#### RBA and SA-PINN point weights

**Entry points**

- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the controls on the Loss weights page (Automatic weighting), built with the Adaptive training page
- `_training_methods_defs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): runtime code of RBA, SA-PINN and causal training

**Helpers**

- `_point_weights_mode` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): which point weights a run uses
- `_training_methods_cbs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the callbacks that update the weights

**Tests**

- [`tests/test_training_methods.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_training_methods.py): causal training, RBA and SA-PINN
- [`tests/test_ta_bcpinn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_bcpinn.py): point weights inside time windows

Described in [RBA and SA-PINN](../reference/point-weights.md), [Loss weights](../guide/loss-weights.md).

#### Causal training

**Entry points**

- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the controls on the Loss weights page (Automatic weighting), built with the Adaptive training page
- `_training_methods_defs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): runtime code of causal training (and RBA, SA-PINN)

**Helpers**

- `_training_methods_cbs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the callback that updates the causal weights

**Tests**

- [`tests/test_training_methods.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_training_methods.py): causal training, RBA and SA-PINN

Described in [Causal training](../reference/causal.md), [Loss weights](../guide/loss-weights.md).

#### Time windows: transfer learning, bc-PINN, growing range, resampling

**Entry points**

- `MainWindow._add_ta_step_group` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): time-window groups on the Adaptive training page
- `MainWindow._build_ta_step_groups_json` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): groups to the saved settings
- `_gs_time_adaptive_loop_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the time-window loop: transfer learning, bc-PINN, growing range, resampling

**Helpers**

- `_bc_rng_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): random generator for bc-PINN's stored points
- `_ta_clean_unsupported` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): time-window options the exported script does not write

**Tests**

- [`tests/test_ta_bcpinn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_bcpinn.py): one network through all windows, bc-PINN
- [`tests/test_ta_growing_2d.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_growing_2d.py): growing time range in 2D
- [`tests/test_export_time_window_options.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_export_time_window_options.py): export trains the same windows as Solve

Described in [Time windows and bc-PINN](../reference/time-windows.md), [Adaptive training](../guide/adaptive.md).

#### Exact constraints

**Entry points**

- `MainWindow._build_domain_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Exact periodicity and exact initial-condition options
- `_exact_constraints` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): exact initial conditions and periodic input features

**Helpers**

- `MainWindow._update_exact_notes` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the Network page's note when exact constraints build the transform
- `_net_transforms_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): output transform the constraints are built into
- `_periodic_axes` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py): axes that can be made exactly periodic

**Tests**

- [`tests/test_training_methods.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_training_methods.py): exact initial condition and periodicity
- [`tests/test_round62_transform_formulas.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round62_transform_formulas.py): hand-written output transforms

Described in [Exact constraints](../reference/exact-constraints.md), [Loss weights](../guide/loss-weights.md).

#### Parameter sweeps and Bayesian optimization

**Entry points**

- `MainWindow._start_sweep` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Parameter sweep page: start a sweep or a Bayesian search
- `run_sweep` in [`pinnstudio/core/sweep_runner.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_runner.py): runs the configurations one after another

**Helpers**

- `BayesRuns` in [`pinnstudio/core/sweep_runner.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_runner.py): Bayesian search: next run from the results so far
- `BayesOptimizer` in [`pinnstudio/core/bayes_opt.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/bayes_opt.py): Gaussian process with expected improvement
- `available_params` in [`pinnstudio/core/sweep_registry.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_registry.py): the sweepable parameters
- `write_sweep_summary_plot` in [`pinnstudio/core/sweep_runner.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_runner.py): summary table and plot

**Tests**

- [`tests/test_sweep_registry.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_sweep_registry.py): sweepable parameters
- [`tests/test_sweep_execution.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_sweep_execution.py): a sweep run end to end
- [`tests/test_round75_wave_helmholtz_bayes.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round75_wave_helmholtz_bayes.py): Bayesian search (Helmholtz example)

Described in [Sweeps and Bayesian optimization](../reference/sweeps.md), [Parameter sweeps](../guide/sweeps.md), [Bayesian optimization](../theory/bayesian.md).

#### Script generation and export

**Entry points**

- `generate_script` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the script Solve runs (shown in the Script tab)
- `generate_clean_script` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Export as DeepXDE Script: the short standalone script
- `MainWindow._export_deepxde_script` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the export command

**Helpers**

- `MainWindow._refresh_script_tab` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the Script tab
- `run_pinn` in [`pinnstudio/core/runner.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/runner.py): runs a generated script in its own process
- [`tools/check_export_equivalence.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/check_export_equivalence.py): checks that both scripts train the same model

**Tests**

- [`tests/test_export_parity.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_export_parity.py): export and restore parity
- [`tests/test_export_time_window_options.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_export_time_window_options.py): export trains the same model as Solve
- [`tests/test_round55_generator_naming.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round55_generator_naming.py): the two generators

Described in [Generated scripts](../reference/scripts.md), [Exporting a script](../guide/export.md).

#### Checkpoints and restoring

**Entry points**

- `MainWindow._build_restore_panel` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Restore & visualize page
- `MainWindow._on_restore` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Restore & visualize a saved model

**Helpers**

- `MainWindow._build_restore_script` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the restore script
- `MainWindow._build_restore_script_ta` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): restoring time-window models
- `_build_train_cbs_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Model Checkpoint callback

**Tests**

- [`tests/test_restore_steady_state.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_restore_steady_state.py): restoring a steady-state model
- [`tests/test_restore_ta_error_analysis.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_restore_ta_error_analysis.py): restoring time-window models
- [`tests/test_training_callbacks_visibility.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_training_callbacks_visibility.py): Model Checkpoint settings

Described in [Checkpoints and restoring](../reference/checkpoints.md), [Restore & visualize](../guide/restore.md).

#### Visualization, plots and data export

**Entry points**

- `MainWindow._display_run_result_plots` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Results tabs after a run
- `MainWindow._on_plot_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Plot settings dialog

**Helpers**

- `generate_clean_script._result_plot_parts` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): result plots of the exported script
- `_restore_surface_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): surface plots
- `_restore_line_plot_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): line plots over time steps
- `_restore_anim_surface_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): animations
- `_loss_plot_runtime_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): saved loss figure
- `MainWindow._on_export_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Export Solution (solution data)
- `FigureStyleDialog` in [`pinnstudio/ui/figure_style.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/figure_style.py): figure style
- `LiveLossPlot` in [`pinnstudio/ui/results.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/results.py): live loss during training
- `RunCompare` in [`pinnstudio/ui/results.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/results.py): comparing runs
- [`pinnstudio/ui/export_tools.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/export_tools.py): copying and saving figures and tables

**Tests**

- [`tests/test_figure_style.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_figure_style.py): one Plot settings dialog for every figure
- [`tests/test_plot_settings_more.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_plot_settings_more.py): plot settings reach every script
- [`tests/test_round69_output_named_plots.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round69_output_named_plots.py): plots of multi-output problems

Described in [Plots and data export](../reference/plots.md), [Results and plots](../guide/results.md), [Solve and the training log](../guide/solve.md).

#### Error Analysis and metrics

**Entry points**

- `MainWindow._show_ea_dialog` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Error Analysis dialog
- `_gs_ea_setup_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Error Analysis in the Solve script

**Helpers**

- `_ea_metrics_row_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): one row of error_metrics.txt
- `_ea_metrics_csv_header` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): metric columns (L2, MSE, max, mean absolute)
- `_formula_reference_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): reference from an exact-solution formula
- `auto_line_slice` in [`pinnstudio/core/line_slice.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/line_slice.py): where line comparisons cut a 2D/3D domain
- [`pinnstudio/core/ea_view3d.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/ea_view3d.py): 3D view of the error

**Tests**

- [`tests/test_ea_fixes.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ea_fixes.py): reference files and metrics
- [`tests/test_ea_spacetime.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ea_spacetime.py): relative L2 over all reference points
- [`tests/test_round67_formula_reference.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round67_formula_reference.py): reference from a formula

Described in [Error metrics](../reference/metrics.md), [Error Analysis](../guide/error-analysis.md), [Error measures](../theory/metrics.md).

#### Configuration, validation and saved problems

**Entry points**

- `PINNConfig` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py): every setting of a problem (saved as .pinn.json)
- `PINNConfig.validate` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py): checks run before Solve and on a loaded file

**Helpers**

- `MainWindow._build_config` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the window's widgets to a PINNConfig
- `MainWindow._apply_config` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): a PINNConfig back into the widgets
- `MainWindow._save_problem` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Save Problem
- `MainWindow._open_problem` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Open Saved Problem
- `MainWindow._on_gpu_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): GPU / Hardware settings
- [`tools/docs/make_config_reference.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/docs/make_config_reference.py): the configuration reference page

**Tests**

- [`tests/test_config_options.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_config_options.py): GPU and random-seed settings
- [`tests/test_template_copies.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_template_copies.py): saving never changes an example
- [`tests/test_audit_v160.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_audit_v160.py): validation checks of the v1.6.0 audit

Described in [Configuration schema](../developer/config.md), [Save, reopen and export](../getting-started/save-and-export.md).

#### Compatibility of methods: the checks that refuse a combination

**Entry points**

- `PINNConfig.validate` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py): the combinations refused before every Solve and sweep run
- `MainWindow._validate_optimizer_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): weight decay with L-BFGS or NNCG, AdamW, NNCG's DeepXDE version
- `MainWindow._on_steady_state_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): a stationary problem disables Time Adaptive Training
- `MainWindow._on_problem_type_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Inverse mode removes Time Adaptive Training

**Helpers**

- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the warning for causal training with per-point weights
- `_check_anchor_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): extra training points refused with time windows
- `_bc_check_initial_operator_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Operator BC at the initial time
- `_bc_check_periodic_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Periodic BC only on box shapes
- `_bc_check_interface2d_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Interface 2D BC requirements
- `_effective_config` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): settings the generators switch off (time windows on a stationary problem)
- `_ode_to_1d` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): an ODE as a 1D problem in t (time windows switched off)

**Tests**

- [`tests/test_audit_v160.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_audit_v160.py): refused and generated combinations of the v1.6.0 audit
- [`tests/test_training_methods.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_training_methods.py): causal training with RBA, exact constraints, validation
- [`tests/test_ta_bcpinn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_bcpinn.py): time-window options and their checks
- [`tests/test_lbfgs_steps.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_lbfgs_steps.py): network and method checks; RAR in an inverse problem
- [`tests/test_steady_state_rar_gating.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_steady_state_rar_gating.py): RAR on a stationary problem
- [`tests/test_round77_fractional.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round77_fractional.py): frac(...) problems and their checks

Described in [Compatibility of methods](../reference/compatibility.md).

#### Examples and templates

**Entry points**

- `ENTRIES` in [`pinnstudio/core/example_catalog.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/example_catalog.py): the Examples list: names, order, modes
- `EXAMPLES` in [`pinnstudio/core/deepxde_examples.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/deepxde_examples.py): DeepXDE examples as settings
- `MainWindow._on_template_selected` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): PINNStudio templates, set up in the window

**Helpers**

- `PAPER` in [`pinnstudio/core/paper_settings.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/paper_settings.py): the papers' settings (Settings: Default of a paper template)
- `settings_for` in [`pinnstudio/core/example_settings.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/example_settings.py): Settings: Default or Tuned
- [`pinnstudio/resources/tuned_settings.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/resources/tuned_settings.json): the Tuned settings
- [`tools/docs/make_gallery.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/docs/make_gallery.py): the example gallery pages

**Tests**

- [`tests/test_templates.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_templates.py): every template in Forward and Inverse mode
- [`tests/test_deepxde_examples.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_deepxde_examples.py): DeepXDE examples
- [`tests/test_example_settings.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_example_settings.py): Settings: Default and Tuned

Described in [Examples](../examples/index.md), [Adding an example](../developer/add-example.md).

Files on the `main` branch of the repository.

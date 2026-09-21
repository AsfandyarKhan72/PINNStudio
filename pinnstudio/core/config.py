from dataclasses import dataclass, field
from typing import List

@dataclass
class PINNConfig:
    save_dir: str = ""

    adapt_method: str = "None"
    rar_cycles: int = 3
    rar_candidates: int = 50000
    rar_add_points: int = 500
    rar_adam_iters: int = 20000
    rar_lbfgs_iters: int = 10000

    time_adaptive: bool = False
    ta_num_steps: int = 5
    ta_grid_size: int = 101

    ta_step_groups: str = ""
    ta_transfer_learning: bool = False
    ta_transfer_optimizer: str = "adam"

    parametric_study: bool = False
    parametric_param: str = "none"
    parametric_values: str = ""
    learning_rate: float = 0.001
    loss_type: str = "MSE"

    # Problem dimension
    problem_dim: str = "1D"  # "1D" or "2D"

    # PDE settings
    pde_expression: str = "du_t - 0.4 * du_xx"

    # Domain settings
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    num_domain: int = 2000
    num_boundary: int = 200
    num_initial: int = 200
    num_test: int = 1000
    point_distribution: str = "Hammersley"

    plot_type: str = "Surface"
    num_timesteps: int = 4

    # Initial condition
    ic_type: str = "sin"
    ic_expression: str = "np.sin(np.pi * x[:, 0:1])"

    # Boundary conditions (1D)
    bc_left: float = 0.0
    bc_right: float = 0.0
    bc_left_type: str = "Dirichlet"
    bc_right_type: str = "Dirichlet"

    # Neural network
    layers: List[int] = field(default_factory=lambda: [2, 64, 64, 64, 1])
    activation: str = "tanh"
    kernel_initializer: str = "Glorot uniform"

    # Optional input/output transform: x_transformed = x_raw * scale + shift
    # (input, one entry per input dimension) and y_transformed = y_raw *
    # scale + shift (output, one entry per output component). Disabled by
    # default -- identity scale=1/shift=0 either way, so enabling with
    # untouched defaults changes nothing.
    input_transform_enabled: bool = False
    input_transform_scale: List[float] = field(default_factory=lambda: [1.0, 1.0])
    input_transform_shift: List[float] = field(default_factory=lambda: [0.0, 0.0])
    output_transform_enabled: bool = False
    output_transform_scale: List[float] = field(default_factory=lambda: [1.0])
    output_transform_shift: List[float] = field(default_factory=lambda: [0.0])

    # Training
    optimizer: str = "adam"
    iterations: int = 10000
    optimizer2: str = "none"
    iterations2: int = 5000
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    loss_weight_obs: float = 1.0
    inv_param_log_scale: bool = False
    inv_param_save: str = "Every 100 iters"

    # Inverse PINN
    problem_type: str = "Forward"

    # Multi-output
    num_outputs: int = 1
    output_names: str = "u"
    pde_expressions: str = "du_t - 0.4 * du_xx"

    # BCs left/right (all dims)
    bc_left_types: str = "Dirichlet"
    bc_right_types: str = "Dirichlet"
    bc_left_values: str = "0.0"
    bc_right_values: str = "0.0"
    bc_left_active: str = "True"
    bc_right_active: str = "True"
    bc_left_deriv: str = "False"
    bc_right_deriv: str = "False"

    # BCs bottom/top (2D only)
    bc_bottom_types: str = "Dirichlet"
    bc_top_types: str = "Dirichlet"
    bc_bottom_values: str = "0.0"
    bc_top_values: str = "0.0"
    bc_bottom_active: str = "True"
    bc_top_active: str = "True"
    bc_bottom_deriv: str = "False"
    bc_top_deriv: str = "False"

    # ICs
    ic_expressions: str = "np.sin(np.pi * x[:, 0])"
    ic_active: str = "True"
    loss_weights_multi: str = ""

    # Inverse settings
    inverse_param_name: str = "trainable_variable"
    inverse_param_init: float = 1.0
    inverse_data_file: str = ""
    inverse_ic_type: str = "expression"
    inverse_ic_file: str = ""
    forward_ic_from_file: bool = False
    forward_ic_file: str = ""
    template_type: str = ""

    # Multiple trainable variables (Inverse). JSON-encoded list of
    # {"name": str, "init": float} dicts, one per trainable variable, in
    # order (variable 1 first). Empty string means "not set" -- fall back
    # to the single legacy inverse_param_name/inverse_param_init fields
    # above for configs saved before this feature existed. All trainable
    # variables are fit against the same shared measured-data setup below
    # (one or more files) -- variables never get their own private
    # dataset, no matter how many of them are being estimated.
    inverse_variables_json: str = ""
    # Which model output column the (legacy, single) measured-data file
    # corresponds to (0-based index into output_names). Defaults to
    # output 0. Superseded by inverse_obs_files_json below when that is
    # set; kept as the fallback for configs saved before multi-file
    # support existed.
    inverse_obs_output_idx: int = 0
    # Multiple measured-data files (Inverse). JSON-encoded list of
    # {"path": str, "output_idx": int, "weight": float} dicts, one per
    # measured-data file, in order (file 1 first) -- each file becomes
    # its own PointSetBC observation constraint and its own loss-weight
    # term. Empty string means "not set" -- fall back to a single entry
    # built from the legacy inverse_data_file/inverse_obs_output_idx/
    # loss_weight_obs fields above, so configs saved before this feature
    # existed keep loading as exactly one measured-data file, unchanged.
    inverse_obs_files_json: str = ""

    # Export
    export_grid_size: int = 101
    export_t_steps: int = 11
    plot_output_idx: int = 0

    # L-BFGS
    lbfgs_use_default: bool = True
    lbfgs_maxcor: int = 200
    lbfgs_ftol: float = 1e-20
    lbfgs_gtol: float = 1e-15
    float_type: str = "float32"
    lbfgs_maxiter: int = 50000
    lbfgs_maxfun: int = 62500
    lbfgs_maxls: int = 100
    lbfgs_float_type: str = "float32"
    ic_pretrain: bool = False
    ic_pretrain_optimizer: str = "adam"
    ic_pretrain_iterations: int = 20000
    ic_pretrain_num_test: int = 10000
    ic_pretrain_num_initial: int = 1000
    ic_pretrain_lr: float = 1e-3
    ic_pretrain_loss: str = "MSE"
    batch_size: int = 0
    ic_pretrain_restore: bool = False
    ic_pretrain_restore_path: str = ""
    optimizer_scheduler: bool = False
    scheduler_phases: str = ""
    scheduler_same_weights: bool = True

    # Weight decay (L2 regularization on the network). Applies to
    # whichever network is being trained (main model, IC pre-training,
    # RAR refinement, every scheduler phase) since DeepXDE sets this at
    # the network level, not per optimizer call. 0.0 = off (default,
    # matches all pre-existing behavior). Not compatible with L-BFGS or
    # NNCG (DeepXDE raises an error if weight_decay > 0 for either) --
    # validated in the GUI before a run starts, not just left to crash.
    weight_decay: float = 0.0

    # Training Callbacks (all opt-in, off by default). Applied to the
    # live "Training Phases" scheduler (and the legacy single/dual-phase
    # fallback path) -- NOT to IC pre-training or the RAR refinement
    # sub-loop, which are short, purpose-built inner loops of their own
    # where early-stopping/point-resampling/checkpointing would fight
    # their intent rather than help it.
    cb_early_stopping: bool = False
    cb_early_stopping_min_delta: float = 0.0
    cb_early_stopping_patience: int = 2000
    cb_early_stopping_baseline: str = ""  # blank = None (no baseline)
    cb_early_stopping_monitor: str = "loss_train"  # or "loss_test"
    cb_early_stopping_start_from: int = 0  # needs deepxde>=1.12.0, guarded at codegen time

    cb_point_resampler: bool = False
    cb_point_resampler_period: int = 100
    cb_point_resampler_pde_points: bool = True
    cb_point_resampler_bc_points: bool = False

    cb_model_checkpoint: bool = False
    cb_checkpoint_period: int = 1000
    cb_checkpoint_save_better_only: bool = True
    cb_checkpoint_monitor: str = "train loss"  # or "test loss"

    cb_timer: bool = False
    cb_timer_minutes: float = 60.0

    plot_colormap: str = "RdBu_r"
    plot_levels: int = 100
    plot_resolution: int = 200
    plot_dpi: int = 300
    plot_n_2d_snapshots: int = 2
    plot_colorbar: bool = True
    plot_auto_range: bool = True
    plot_vmin: float = -1.0
    plot_vmax: float = 1.0
    plot_linewidth: float = 2.0
    plot_fps: int = 10

    ea_files: str = "[]"
    ea_do_line: bool = True
    ea_do_surface: bool = True

    # Geometry type & 3D domain
    geometry_type: str = "Rectangle"  # 2D: Rectangle|Disk|Ellipse|Triangle|Polygon ; 3D: Cuboid|Sphere
    z_min: float = 0.0
    z_max: float = 1.0

    # Disk / Ellipse / Sphere center
    geom_center_x: float = 0.5
    geom_center_y: float = 0.5
    geom_center_z: float = 0.5
    geom_radius: float = 0.5

    # Ellipse
    geom_semi_major: float = 0.5
    geom_semi_minor: float = 0.3
    geom_angle: float = 0.0

    # Triangle / Polygon vertices, "x1,y1;x2,y2;..." format
    geom_triangle_vertices: str = "0,0;1,0;0,1"
    geom_polygon_vertices: str = "0,0;1,0;1,1;0,1"

    # Shape-aware boundary conditions for non-box geometries (JSON-encoded)
    bc_boundary_json: str = ""  # Disk/Ellipse/Sphere: one BC group per output
    bc_edge_json: str = ""      # Triangle/Polygon: one BC group per edge per output

    # Fully-customizable BC builder, used when no Quick Example template is
    # selected -- a flat, user-authored list of BCs (any DeepXDE BC class,
    # any location, any output), independent of geometry_type. JSON-encoded
    # list of dicts; see MainWindow._build_custom_bc_json().
    custom_bc_json: str = ""


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
    # Split from a single shared "num_timesteps" field: the static "Line
    # (time steps)" plot and the GIF animations (Line/Surface Animation)
    # want different defaults -- fewer steps keeps the static overlay
    # readable, while more frames makes the animation play back slower/
    # smoother -- so each now has its own field instead of fighting over
    # one shared number (see also plot_linewidth / plot_linewidth_anim
    # below for the same split applied to line width).
    num_timesteps_line: int = 5
    num_timesteps_anim: int = 20

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

    # Network architecture: "FNN" (plain fully-connected, shared across all
    # outputs) or "PFNN" (parallel/independent sub-network per output,
    # merging only at the final layer -- DeepXDE's dde.nn.PFNN). layers[0]
    # (the raw input dimension) and layers[-1] (num_outputs) are unaffected;
    # only the hidden-layer entries change shape for PFNN (each becomes a
    # per-branch width repeated num_outputs times) -- computed where the
    # layer list is built (main_window.py._build_config()), never stored
    # here as anything other than the plain FNN-style flat list, to avoid
    # duplicating that shape logic across files. PFNN has no `regularization`
    # kwarg in DeepXDE, so weight_decay is silently not applied when this is
    # "PFNN" (codegen.py suppresses the regularization arg in that case).
    network_type: str = "FNN"

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
    # Optional derived scalar field to plot in the Results panel instead of
    # a single raw output column -- e.g. "sqrt(u**2+v**2)" for the 1D
    # Schrodinger template's |h|, built from its two real-valued outputs
    # u (real part) and v (imaginary part). The expression's variables are
    # this problem's own output_names, so any future multi-output template
    # can define its own derived field the same way. Empty string (the
    # default) keeps the existing plot_output_idx-only behavior exactly as
    # it was -- this is purely additive.
    plot_custom_expr: str = ""
    # Display label for the field above (colorbar/axis label, and the
    # printed output name) -- falls back to the expression itself when left
    # blank. Ignored when plot_custom_expr is empty.
    plot_custom_label: str = ""
    # 1D "Surface" plot axis orientation (both the Standard and
    # Time-Adaptive final solution plots): True (the default) puts t on the
    # x-axis and the spatial domain x on the y-axis; False keeps the
    # original orientation (x on the x-axis, t on the y-axis). Only affects
    # the 1D x-t heatmap -- 2D/3D "Surface" plots show spatial snapshots at
    # fixed times and have no x/t axis choice to make. Defaulting to True
    # (rather than the pre-v32 False) is a deliberate, user-requested
    # change to what every 1D "Surface" plot looks like by default, not
    # just a new opt-in -- unlike plot_custom_expr above, which had to stay
    # inert-by-default for backward compatibility.
    plot_swap_xt: bool = True

    # Figure-size standardization for single-panel results plots (Loss,
    # Line, Surface, the two Animation GIF types, Parameter Convergence,
    # and their Time-Adaptive equivalents) -- deliberately NOT applied to
    # the multi-column Error Analysis comparison grids, which already pick
    # their own width from however many reference files/columns are being
    # compared and would look wrong forced into any of these modes, nor to
    # the Setup tab's domain-sampling preview plots, whose aspect ratio is
    # tied to the problem's own geometry. "Default" (the default) keeps
    # every in-scope figsize=(...) call exactly as it was before this
    # field existed -- fully backward compatible. "Square" forces a 1:1
    # aspect ratio (side length = the larger of that plot's own default
    # width/height). "Wide" keeps that plot's own default height but widens
    # it to a fixed 1.8x ratio -- useful for presentation slides. "Custom"
    # uses plot_figsize_w/plot_figsize_h for every in-scope plot regardless
    # of its own default dimensions.
    plot_figsize_mode: str = "Default"
    plot_figsize_w: float = 7.0
    plot_figsize_h: float = 5.0

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
    # Line Animation (GIF)'s own line width -- kept separate from the
    # static plots' plot_linewidth above (see num_timesteps_line/_anim
    # comment) so the two can default differently.
    plot_linewidth_anim: float = 3.0
    plot_fps: int = 10

    ea_files: str = "[]"
    ea_do_line: bool = True
    ea_do_surface: bool = True

    # Geometry type & 3D domain
    geometry_type: str = "Rectangle"  # 2D: Rectangle|Disk|Ellipse|Triangle|Polygon|Custom ; 3D: Cuboid|Sphere
    z_min: float = 0.0
    z_max: float = 1.0

    # Custom geometry (2D only, Phase 1): an ORDERED list of primitive
    # shapes combined with boolean operations -- a small COMSOL-style CSG
    # sequence, mirroring DeepXDE's own CSGUnion/CSGDifference/
    # CSGIntersection, which this is built directly on top of (no new
    # geometry math of our own). JSON-encoded list of dicts, one per shape,
    # in build order:
    #   {"type": "Rectangle"|"Disk"|"Ellipse"|"Triangle"|"Polygon",
    #    "op": "base"|"union"|"subtract"|"intersect", <shape's own params>}
    # The first shape's "op" is always ignored (it's the starting region);
    # each later shape's "op" combines it against the RUNNING result built
    # so far from every earlier shape -- same order-matters sequential-tree
    # semantics as COMSOL's own Boolean Operations, not a flat set.
    # Per-type params (same fields/format the single-shape geom_* widgets
    # above already use, just scoped to one entry in this list instead of
    # being the problem's only shape):
    #   Rectangle: x_min, x_max, y_min, y_max
    #   Disk:      center_x, center_y, radius
    #   Ellipse:   center_x, center_y, semi_major, semi_minor, angle
    #   Triangle:  vertices ("x1,y1;x2,y2;x3,y3")
    #   Polygon:   vertices ("x1,y1;x2,y2;...", 4+ points)
    # Only read/written when geometry_type == "Custom"; empty/"[]" otherwise.
    geom_custom_shapes_json: str = "[]"

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

    # Steady-state (time-independent) problems, e.g. the Poisson equation.
    # When True, the generated script builds a plain dde.data.PDE over the
    # spatial geometry only -- no GeometryXTime, no time axis on the
    # network input, no Initial Condition, no Time-Adaptive/RAR (both are
    # inherently time-based). x_min/x_max/y_min/y_max/z_min/z_max still
    # define the spatial domain as usual; t_min/t_max/num_initial/
    # ic_expressions/time_adaptive/adapt_method are all ignored.
    steady_state: bool = False

    # GPU device selection & memory reservation for the generated training
    # script. device_index picks which CUDA device to train on (0 = first
    # GPU, matching all prior behavior -- unchanged default); memory_fraction
    # is the maximum share of that device's memory PyTorch is allowed to
    # reserve upfront (0.95 = the prior hardcoded default, also unchanged).
    # Both are ignored when no CUDA device is available -- the generated
    # script always falls back to CPU in that case regardless of these.
    gpu_device_index: int = 0
    gpu_memory_fraction: float = 0.95

    # Reproducibility. When on (the default), the generated script calls
    # dde.config.set_random_seed(random_seed) once, right at the start --
    # this seeds NumPy, PyTorch and Python's own `random` together (that's
    # what DeepXDE's helper does internally), so point sampling and network
    # weight initialization are the same every run. Deliberately NOT paired
    # with torch.backends.cudnn.deterministic=True -- that would make GPU
    # runs bit-for-bit reproducible too, but costs real training speed, and
    # speed was judged more valuable than that last mile of exactness here.
    # 2026 is just a fixed, memorable default -- there's nothing special
    # about the number itself.
    use_random_seed: bool = True
    random_seed: int = 2026

    # Parameter Sweep. Distinct from the older parametric_study/
    # parametric_param/parametric_values fields above: those predate the
    # Optimizer Scheduler and multi-output loss-weights system, and their
    # own code comment in codegen.py already documents that 4 of their 6
    # options are dead (silently do nothing) as a result -- they're left
    # in place rather than removed here to avoid breaking existing script
    # generation, but nothing new should build on them. This sweep system
    # is driven from the GUI/a dedicated sweep runner repeatedly calling
    # the normal, already-correct single-run codegen path (see
    # pinnstudio/core/sweep_registry.py), not from an internal loop
    # inside the generated script, so it has no equivalent dependency on
    # a particular training architecture to go stale over.
    sweep_enabled: bool = False
    # "oat" (one-at-a-time, every other swept field held at its baseline
    # value), "grid" (every combination of every parameter's values --
    # COMSOL calls this "All combinations"), or "zip" (every swept
    # parameter's Nth value run together as one combination -- COMSOL's
    # "Specified combinations"; every parameter must have the same
    # number of values, checked in validate() below).
    sweep_mode: str = "oat"
    # JSON-encoded list of swept parameters, one dict per added row:
    #   {"id": "<sweep_registry id>", "mode": "list", "values": [..]}
    #   {"id": "...", "mode": "linear"|"log", "min": .., "max": .., "n": ..}
    # "id" is a key into sweep_registry.available_params()'s ids for this
    # config. Empty list ("[]", the default) means no sweep configured.
    sweep_parameters: str = "[]"

    # Where a sweep organizes its own output -- distinct from save_dir
    # above (which is whatever the Setup tab's own "Save results to" is
    # set to, and would make every run in a sweep collide into the same
    # folder if reused as-is). Blank -- the default, since this is a new
    # field a hand-edited/old save file won't have -- falls back to
    # sweep_runner.sweep_root_dir()'s own default location. Every run
    # (baseline included) gets its own descriptively-named subfolder
    # under <sweep_save_dir>/sweep_<timestamp>/ -- see sweep_runner.py.
    sweep_save_dir: str = ""
    # What each run in a sweep actually saves (model, loss/solution
    # plot, Error Analysis figures if configured) is controlled by the
    # same plot_type/plot_output_idx/plot_custom_expr/plot_custom_label/
    # export_t_steps fields a normal Solve already uses -- "same_as_setup"
    # (the default, i.e. exactly today's behavior) leaves those alone.
    # "custom" overrides them for every run in the sweep (baseline
    # included) with the sweep_plot_* fields below, without touching
    # what the Setup tab itself shows/uses for a normal Solve.
    sweep_export_mode: str = "same_as_setup"
    sweep_plot_type: str = "Surface"
    sweep_plot_output_idx: int = 0
    sweep_plot_custom_expr: str = ""
    sweep_plot_custom_label: str = ""
    sweep_export_t_steps: int = 11

    def validate(self):
        """Sanity-check the fields that would otherwise either silently
        produce a degenerate run or crash deep inside DeepXDE/PyTorch with a
        traceback that gives no hint it was a config problem -- most
        importantly the ones a hand-edited or corrupted .pinn.json save
        file could set to something the GUI's own spinboxes/combo boxes
        would never allow (a negative iteration count, an empty layers
        list, an inverted domain range, etc.). Returns a list of
        human-readable problem descriptions; an empty list means the config
        is safe to hand to codegen. Deliberately not exhaustive -- this
        catches the failure modes that are easy to hit from a hand-edited
        file and hard to diagnose from the resulting crash, not every
        possible misconfiguration."""
        errors = []

        if not isinstance(self.layers, (list, tuple)) or len(self.layers) < 2:
            errors.append(
                "Network layers must be a list of at least 2 sizes (input and "
                f"output); got {self.layers!r}."
            )
        else:
            for _n in self.layers:
                if not isinstance(_n, int) or _n < 1:
                    errors.append(
                        f"Every network layer size must be a positive integer; "
                        f"found {_n!r} in layers={self.layers!r}."
                    )
                    break

        if self.num_domain <= 0:
            errors.append(f"Domain points (num_domain) must be positive; got {self.num_domain}.")
        if self.num_boundary < 0:
            errors.append(f"Boundary points (num_boundary) cannot be negative; got {self.num_boundary}.")
        if self.num_initial < 0:
            errors.append(f"Initial points (num_initial) cannot be negative; got {self.num_initial}.")
        if self.num_test < 0:
            errors.append(f"Test points (num_test) cannot be negative; got {self.num_test}.")

        # self.iterations / self.iterations2 ("Phase 1"/"Phase 2 iterations")
        # are only actually read at training time when the Optimizer
        # Scheduler is off -- this mirrors codegen.py's own gate for which
        # iteration source is used (config.optimizer_scheduler and
        # len(config.scheduler_phases) > 0, see _sched_active throughout
        # codegen.py). When the scheduler *is* active, those two legacy
        # fields are never read at all, so unconditionally requiring
        # self.iterations > 0 here rejected otherwise-valid configs: the
        # scheduler is ON by default in the GUI (with its own default Adam
        # + L-BFGS phases, both with a real iteration count), while the
        # legacy "Phase 1 iterations" spinbox is hidden and defaults to 0
        # of its own accord. Loading a built-in template happens to also
        # set that hidden spinbox to a real value as a side effect, so this
        # only ever surfaced when building a config from scratch without
        # loading a template first -- PDE/IC/BC filled in, default
        # scheduler phases showing, yet validate() still failed on a field
        # that was never going to be used for that run.
        _sched_active = bool(self.optimizer_scheduler) and bool((self.scheduler_phases or "").strip())
        if not _sched_active:
            if self.iterations <= 0:
                errors.append(f"Phase 1 iterations must be positive; got {self.iterations}.")
            if self.iterations2 < 0:
                errors.append(f"Phase 2 iterations cannot be negative; got {self.iterations2}.")
        if self.learning_rate <= 0:
            errors.append(f"Learning rate must be positive; got {self.learning_rate}.")
        if self.weight_decay < 0:
            errors.append(f"Weight decay cannot be negative; got {self.weight_decay}.")

        if self.problem_dim not in ("1D", "2D", "3D"):
            errors.append(f"problem_dim must be '1D', '2D', or '3D'; got {self.problem_dim!r}.")
        if self.problem_type not in ("Forward", "Inverse"):
            errors.append(f"problem_type must be 'Forward' or 'Inverse'; got {self.problem_type!r}.")

        if self.x_min >= self.x_max:
            errors.append(f"x_min ({self.x_min}) must be less than x_max ({self.x_max}).")
        if self.problem_dim in ("2D", "3D") and self.y_min >= self.y_max:
            errors.append(f"y_min ({self.y_min}) must be less than y_max ({self.y_max}).")
        if self.problem_dim == "3D" and self.z_min >= self.z_max:
            errors.append(f"z_min ({self.z_min}) must be less than z_max ({self.z_max}).")
        if not self.steady_state and self.t_min >= self.t_max:
            errors.append(f"t_min ({self.t_min}) must be less than t_max ({self.t_max}).")

        if self.num_outputs < 1:
            errors.append(f"num_outputs must be at least 1; got {self.num_outputs}.")
        else:
            _names = [n for n in (self.output_names or "").split(",") if n.strip()]
            if len(_names) < self.num_outputs:
                errors.append(
                    f"output_names ({self.output_names!r}) has fewer entries than "
                    f"num_outputs ({self.num_outputs})."
                )

        # JSON-encoded fields: a hand-edited file can put invalid JSON in
        # any of these, which would otherwise only surface as a confusing
        # crash the moment codegen tries to json.loads() it.
        import json as _json_cfg
        for _field in (
            "scheduler_phases", "custom_bc_json", "bc_boundary_json",
            "bc_edge_json", "inverse_variables_json", "inverse_obs_files_json",
            "geom_custom_shapes_json",
        ):
            _raw = getattr(self, _field, "") or ""
            if _raw.strip():
                try:
                    _json_cfg.loads(_raw)
                except (ValueError, TypeError) as _e:
                    errors.append(f"{_field} is not valid JSON: {_e}")

        # Custom geometry: geom_custom_shapes_json must actually describe at
        # least one recognized shape, or there's nothing to build a geometry
        # from (the GUI should never let this happen via normal use, but a
        # hand-edited or old config file could leave it empty). This mirrors
        # the light-touch validation style used elsewhere in this method --
        # it catches the crash-causing case (nothing to build) without
        # trying to deeply validate every shape's numeric parameters, the
        # same way geom_radius/geom_triangle_vertices etc. aren't validated
        # for the other geometry types today.
        if self.geometry_type == "Custom":
            _raw_shapes = getattr(self, "geom_custom_shapes_json", "") or ""
            _KNOWN_SHAPE_TYPES = ("Rectangle", "Disk", "Ellipse", "Triangle", "Polygon", "Cuboid", "Sphere")
            try:
                _shapes_cfg = _json_cfg.loads(_raw_shapes) if _raw_shapes.strip() else []
            except (ValueError, TypeError):
                _shapes_cfg = None  # already reported as invalid JSON above
            if _shapes_cfg is not None:
                if not isinstance(_shapes_cfg, list) or len(_shapes_cfg) == 0:
                    errors.append(
                        "Geometry Type is 'Custom' but no shapes have been "
                        "added; add at least one shape in the Custom "
                        "geometry builder."
                    )
                elif not all(
                    isinstance(_s, dict) and _s.get("type") in _KNOWN_SHAPE_TYPES
                    for _s in _shapes_cfg
                ):
                    errors.append(
                        "geom_custom_shapes_json contains a shape with an "
                        "unrecognized or missing 'type' (expected one of "
                        f"{', '.join(_KNOWN_SHAPE_TYPES)})."
                    )

        # When the scheduler path is what's actually going to train (see
        # _sched_active above), it needs the same "something will actually
        # run" guarantee that self.iterations > 0 gave the legacy path --
        # an empty phase list (every phase removed in the GUI, still
        # possible with the scheduler left on) or every phase sitting at 0
        # iterations would otherwise train for 0 steps with no error at all.
        if _sched_active:
            try:
                _phases_cfg = _json_cfg.loads(self.scheduler_phases)
            except (ValueError, TypeError):
                _phases_cfg = None  # already reported as invalid JSON above
            if _phases_cfg is not None:
                if not isinstance(_phases_cfg, list) or len(_phases_cfg) == 0:
                    errors.append(
                        "Optimizer Scheduler is enabled but has no phases "
                        "configured; add at least one phase, or turn the "
                        "scheduler off and set Phase 1/Phase 2 iterations "
                        "instead."
                    )
                elif not any(
                    isinstance(_p, dict) and isinstance(_p.get("iterations"), (int, float))
                    and _p.get("iterations", 0) > 0
                    for _p in _phases_cfg
                ):
                    errors.append(
                        "Every configured Optimizer Scheduler phase has 0 "
                        "iterations; at least one phase needs a positive "
                        "iteration count for training to do anything."
                    )

        # ea_files is the one exception: it's encoded with repr()/parsed
        # with ast.literal_eval (a list of (time, path[, output_selector])
        # tuples) elsewhere in this codebase, not json.dumps/json.loads --
        # so it must be checked the same way, not lumped in with the
        # JSON-encoded fields above (which would reject every legitimate
        # value, since a Python tuple/None literal isn't valid JSON).
        _raw_ea = getattr(self, "ea_files", "") or ""
        if _raw_ea.strip():
            import ast as _ast_cfg
            try:
                _ast_cfg.literal_eval(_raw_ea)
            except (ValueError, TypeError, SyntaxError) as _e:
                errors.append(f"ea_files is not valid: {_e}")

        # Inverse mode needs at least one real observed-data file path to
        # fit against. Without one, codegen.py's _parse_inverse_obs_files()
        # falls all the way through to an empty inverse_data_file, and the
        # generated script doesn't fail until deep inside its data loader
        # tries to read from "" -- an uncaught FileNotFoundError with no
        # hint the actual problem is an unset observed-data path. This is
        # reachable from the live GUI today: several built-in templates
        # substitute a PDE constant for a trainable variable in Inverse
        # mode (INVERSE_AUTO_CONST) without also auto-seeding a matching
        # observed-data file (INVERSE_AUTO_VARS/OBS) -- e.g. "1D Heat" --
        # since Inverse mode isn't gated per-template and no ground-truth
        # data is bundled for every template. This check mirrors codegen's
        # own fallback order exactly: prefer inverse_obs_files_json's own
        # per-row paths, falling back to the legacy single-file
        # inverse_data_file field only when none of that JSON's rows have
        # a path set -- so it flags exactly the configs that would
        # otherwise crash, not ones that legitimately rely on the legacy
        # field.
        if self.problem_type == "Inverse":
            _has_obs_path = False
            _raw_obs = getattr(self, "inverse_obs_files_json", "") or ""
            if _raw_obs.strip():
                try:
                    _parsed_obs = _json_cfg.loads(_raw_obs)
                except (ValueError, TypeError):
                    _parsed_obs = []
                for _of in (_parsed_obs or []):
                    if str((_of or {}).get("path") or "").strip():
                        _has_obs_path = True
                        break
            if not _has_obs_path and str(self.inverse_data_file or "").strip():
                _has_obs_path = True
            if not _has_obs_path:
                errors.append(
                    "Inverse mode needs at least one observed-data file to fit "
                    "against, but no observation row has a file selected."
                )

        if self.gpu_device_index < 0:
            errors.append(f"GPU device index cannot be negative; got {self.gpu_device_index}.")
        if not (0.0 < self.gpu_memory_fraction <= 1.0):
            errors.append(
                f"GPU memory fraction must be greater than 0 and at most 1; "
                f"got {self.gpu_memory_fraction}."
            )

        if self.use_random_seed and not isinstance(self.random_seed, int):
            errors.append(f"Random seed must be an integer; got {self.random_seed!r}.")

        if self.sweep_enabled:
            if self.sweep_mode not in ("oat", "grid", "zip"):
                errors.append(
                    f"Parameter Sweep mode must be 'oat', 'grid', or 'zip'; got {self.sweep_mode!r}."
                )
            try:
                _sweep_params = _json_cfg.loads(self.sweep_parameters or "[]")
            except (ValueError, TypeError):
                _sweep_params = None
            if not isinstance(_sweep_params, list):
                errors.append(
                    "Parameter Sweep's configured parameters are not valid JSON; "
                    "this usually means a hand-edited save file -- re-add the "
                    "swept parameters from the Parameter Sweep tab."
                )
            elif not _sweep_params:
                errors.append(
                    "Parameter Sweep is enabled but no parameters have been "
                    "added -- add at least one from the Parameter Sweep tab, "
                    "or turn Parameter Sweep off to run a normal single Solve."
                )
            else:
                # Importing here (not at module level) avoids any import-
                # order coupling between config.py and sweep_registry.py --
                # same lazy-import style this method already uses for json.
                from pinnstudio.core import sweep_registry as _sweep_reg
                _available_ids = {p.id for p in _sweep_reg.available_params(self)}
                # Collected alongside the per-entry checks below so the
                # "zip" mode length check after this loop only compares
                # entries that are themselves well-formed -- an entry
                # that already has its own error gets None here and is
                # skipped, rather than piling on a second, confusing
                # error about its (meaningless) length too.
                _entry_lengths = []
                for _si, _sp in enumerate(_sweep_params):
                    if not isinstance(_sp, dict) or "id" not in _sp or "mode" not in _sp:
                        errors.append(
                            f"Parameter Sweep entry #{_si + 1} is missing its "
                            f"'id' or 'mode'; got {_sp!r}."
                        )
                        _entry_lengths.append(None)
                        continue
                    _pid, _pmode = _sp.get("id"), _sp.get("mode")
                    if _pid not in _available_ids:
                        errors.append(
                            f"Parameter Sweep entry #{_si + 1} ('{_pid}') is not "
                            "a sweepable parameter for this problem's current "
                            "setup (e.g. a scheduler-phase parameter when the "
                            "Optimizer Scheduler is off, or that phase no "
                            "longer exists) -- remove and re-add it."
                        )
                        _entry_lengths.append(None)
                        continue
                    if _pmode == "list":
                        _vals = _sp.get("values")
                        if not isinstance(_vals, list) or len(_vals) == 0:
                            errors.append(
                                f"Parameter Sweep entry #{_si + 1} ('{_pid}') "
                                "needs at least one value."
                            )
                            _entry_lengths.append(None)
                        else:
                            _entry_lengths.append(len(_vals))
                    elif _pmode in ("linear", "log"):
                        _pmin, _pmax, _pn = _sp.get("min"), _sp.get("max"), _sp.get("n")
                        _entry_ok = True
                        if not isinstance(_pn, int) or _pn < 2:
                            errors.append(
                                f"Parameter Sweep entry #{_si + 1} ('{_pid}') "
                                f"needs at least 2 steps; got {_pn!r}."
                            )
                            _entry_ok = False
                        if not isinstance(_pmin, (int, float)) or not isinstance(_pmax, (int, float)):
                            errors.append(
                                f"Parameter Sweep entry #{_si + 1} ('{_pid}') "
                                "needs numeric min/max values."
                            )
                            _entry_ok = False
                        elif _pmin >= _pmax:
                            errors.append(
                                f"Parameter Sweep entry #{_si + 1} ('{_pid}') "
                                f"needs min < max; got min={_pmin}, max={_pmax}."
                            )
                            _entry_ok = False
                        elif _pmode == "log" and (_pmin <= 0 or _pmax <= 0):
                            errors.append(
                                f"Parameter Sweep entry #{_si + 1} ('{_pid}') "
                                "uses log spacing, which needs both min and max "
                                f"to be positive; got min={_pmin}, max={_pmax}."
                            )
                            _entry_ok = False
                        _entry_lengths.append(_pn if _entry_ok and isinstance(_pn, int) else None)
                    else:
                        errors.append(
                            f"Parameter Sweep entry #{_si + 1} ('{_pid}') has "
                            f"an unrecognized mode {_pmode!r} -- expected "
                            "'list', 'linear', or 'log'."
                        )
                        _entry_lengths.append(None)

                # "zip" ("Specified combinations", COMSOL's term) runs
                # each parameter's Nth value together as one combination
                # -- e.g. P1=[1,2,3] and P2=[10,20,30] gives exactly 3
                # runs, (1,10)/(2,20)/(3,30), never the 9-run Cartesian
                # product "grid" mode would give. That only makes sense
                # when every parameter has the same number of values.
                if self.sweep_mode == "zip":
                    _valid_lengths = [_l for _l in _entry_lengths if _l is not None]
                    if len(_valid_lengths) > 1 and len(set(_valid_lengths)) > 1:
                        errors.append(
                            "Specified Combinations mode needs every swept parameter to "
                            f"have the same number of values; got {_valid_lengths} -- "
                            "adjust the value lists/ranges so they match."
                        )

        return errors


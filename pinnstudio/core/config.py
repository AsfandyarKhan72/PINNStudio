from dataclasses import dataclass, field
from typing import List

@dataclass
class PINNConfig:
    save_dir: str = ""

    adapt_method: str = "None"
    rar_cycles: int = 3
    rar_candidates: int = 50000
    rar_add_points: int = 500
    rar_adam_iters: int = 5000
    rar_lbfgs_iters: int = 0

    time_adaptive: bool = False
    ta_num_steps: int = 5
    ta_grid_size: int = 101

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

    # Training
    optimizer: str = "adam"
    iterations: int = 10000
    optimizer2: str = "none"
    iterations2: int = 5000
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    loss_weight_obs: float = 1.0
    inv_param_log_scale: bool = False
    inv_param_save: str = "No"

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
    inverse_param_name: str = "beta"
    inverse_param_init: float = 1.0
    inverse_data_file: str = ""
    inverse_ic_type: str = "expression"
    inverse_ic_file: str = ""

    # Export
    export_grid_size: int = 101
    export_t_steps: int = 11
    plot_output_idx: int = 0

    # L-BFGS
    lbfgs_use_default: bool = True
    lbfgs_maxcor: int = 100
    lbfgs_ftol: float = 1e-12
    lbfgs_gtol: float = 1e-8
    lbfgs_maxiter: int = 15000
    lbfgs_maxfun: int = 15000
    lbfgs_maxls: int = 50
"""The settings of the papers PINNStudio's templates come from: Examples ->
Settings: Default loads a template this way (example_settings.py), and the
tuning study runs them as variants for benchmarks/paper_comparison.md:

    python tools/tuning_study.py --gui --templates '1D Allen-Cahn (whole interval)' \
        --variants plain,paper_wz,paper_mg --out runs/papers

Variants (each a changed copy of the template as it loads):
  plain     the base PINN: soft IC and periodic BC, the whole interval, every loss
            weight 1, no adaptive technique, Adam then L-BFGS, with the network and
            points of the paper's own baseline (plain())
  paper_wz  Wight & Zhao (2021), as far as the paper states it
  paper_mg  Mattey & Ghosh (2022, CMAME), bc-PINN, as far as the paper states it
  paper_raissi  Raissi et al. (2019)

Settings the papers leave out are chosen here; each function's comment lists
what the paper states (with its section) and what was chosen.
"""
import copy
import json

ADAM_LR = 0.001


def _n_terms(cfg):
    """(PDE equations, BC entries, active soft IC terms), the order of the loss weights."""
    n_eq = len([e for e in (cfg.pde_expressions or "").split("|") if e.strip()])
    n_bc = len(json.loads(cfg.custom_bc_json or "[]"))
    act = [a.strip() == "True" for a in (cfg.ic_active or "").split(",")]
    hard = [h.strip() == "True" for h in (cfg.ic_hard or "").split(",")]
    n_ic = sum(1 for i, a in enumerate(act) if a and not (i < len(hard) and hard[i]))
    return n_eq, n_bc, n_ic


def _phases(cfg, adam, lbfgs, ic_w, bc_w=1.0):
    n_eq, n_bc, n_ic = _n_terms(cfg)
    w = ",".join(["1.0"] * n_eq + [str(float(bc_w))] * n_bc + [str(float(ic_w))] * n_ic)
    ph = []
    if adam:
        ph.append({"optimizer": "adam", "iterations": int(adam), "lr": ADAM_LR, "loss": "MSE", "weights": w,
                   "decay_type": "none", "decay_p1": 5000.0, "decay_p2": 0.9})
    if lbfgs:
        ph.append({"optimizer": "lbfgs", "iterations": int(lbfgs), "lr": ADAM_LR, "loss": "MSE", "weights": w,
                   "decay_type": "none", "decay_p1": 5000.0, "decay_p2": 0.9})
    cfg.optimizer_scheduler = True
    cfg.scheduler_phases = json.dumps(ph)
    # the same weights in every phase: the window reads them from loss_weights_multi
    # (left as it was, the template's IC weight of 100 came back when the window loaded it)
    cfg.scheduler_same_weights = True
    cfg.loss_weights_multi = w
    cfg.loss_weights = [1.0, float(bc_w), float(bc_w), float(ic_w)]    # the older 1D weight fields, to match


def _soft(cfg, ic_outputs=None):
    """Soft IC (for the given outputs, default those the template has an IC for, as a loss term or built
    in; an IC of 0 counts, e.g. Schrödinger's v(x, 0) = 0) and soft periodic BC."""
    n_out = len(cfg.output_names.split(","))
    ics = (cfg.ic_expressions or "").split("|")
    act = (cfg.ic_active or "").split(",")
    hard = (cfg.ic_hard or "").split(",")
    flag = lambda xs, i: i < len(xs) and xs[i].strip() == "True"
    if ic_outputs is not None:
        on = ic_outputs
    elif act != [""] or hard != [""]:
        on = [i for i in range(n_out) if flag(act, i) or flag(hard, i)]
    else:
        on = [i for i in range(n_out) if i < len(ics) and ics[i].strip() != ""]
    cfg.ic_hard = ",".join(["False"] * n_out)
    cfg.ic_active = ",".join("True" if i in on else "False" for i in range(n_out))
    cfg.periodic_inputs = False
    cfg.point_weights = "None"
    cfg.causal_training = False
    cfg.network_type = "FNN"
    cfg.adapt_method = "None"
    cfg.time_adaptive = False
    cfg.ic_pretrain = False


def _net(cfg, depth, width):
    lay = list(cfg.layers)
    cfg.layers = [lay[0]] + [int(width)] * int(depth) + [lay[-1]]


def _windows(cfg, n, backward=False, growing=False, icgl=0.0, bc_w=100.0, bc_times=10, bc_points=2000, tl=True):
    cfg.time_adaptive = True
    cfg.adapt_method = "Time Adaptive"
    cfg.ta_num_steps = int(n)
    cfg.ta_step_groups = ""
    cfg.ta_transfer_learning = bool(tl)
    cfg.ta_backward = bool(backward)
    cfg.ta_growing = bool(growing)
    cfg.ta_icgl_fraction = float(icgl)
    cfg.ta_backward_weight = float(bc_w)
    cfg.ta_backward_times = int(bc_times)
    cfg.ta_backward_points = int(bc_points)
    cfg.ta_resample_rounds = 0


def plain(cfg, tkey=None):
    """The base PINN, as the template's paper describes its baseline: soft IC and BC, the whole time
    interval, every loss weight 1, no adaptive technique, Adam then L-BFGS; with the paper's network,
    points and iterations for its baseline where it states them."""
    _soft(cfg)
    paper = TEMPLATE_PAPER.get(tkey)
    dim = len(list(cfg.layers)) and int(list(cfg.layers)[0])
    if paper == "paper_mg":
        # M&G Sec. 5.1.1 / 5.2.1: std-PINN 4x200 tanh, 512 IC, 201 BC and 20,000 residual points (1D AC),
        # 100,000 Adam then L-BFGS (50,000 max, Sec. 3.4).
        _net(cfg, 4, 200)
        cfg.num_domain, cfg.num_initial, cfg.num_boundary = 20000, 512, 201
        _phases(cfg, 100000, 50000, ic_w=1.0)
        return
    if paper == "paper_wz":
        # W&Z Sec. 3.1, Table 3.1 "base PINN": N_f = 10,000, N_u = N_b = 200 (Sec. 3.1 defaults), equal
        # weights, full batch, Adam then L-BFGS-B. Network not stated for it: 4x128 (their next trial) in 1D,
        # 6x128 (their benchmarks) in 2D and 3D. Adam iterations not stated: 20,000, then L-BFGS 10,000.
        _net(cfg, 4 if dim == 2 else 6, 128)
        if dim == 2:
            cfg.num_domain, cfg.num_initial, cfg.num_boundary = 10000, 200, 200
        _phases(cfg, 20000, 10000, ic_w=1.0)
        return
    if paper == "paper_raissi":
        # Raissi et al.'s PINN is itself the base PINN: their network, points and L-BFGS (paper_raissi).
        f = PAPER.get((tkey, "paper_raissi"))
        if f:
            f(cfg)
            return
    if paper == "paper_mathias":
        # Mathias et al.'s network and points (4x20, 10,000 physics points), with the IC and BC as MSE loss
        # terms instead of built into the output (u = v = 0 on the whole boundary; 1,000 initial and 1,000
        # boundary points, not stated by them), and Adam lr 1e-3 then L-BFGS.
        _net(cfg, 4, 20)
        n_out = len(cfg.output_names.split(","))
        cfg.output_transform_enabled = False
        cfg.output_transform_formula = ""
        cfg.ic_active = ",".join(["True"] * n_out)
        cfg.custom_bc_json = json.dumps([
            {"type": "dirichlet", "component": i, "location": "True", "value": "0", "axis": "x", "deriv_order": 0,
             "points_file": "", "location2": "", "direction": "normal", "components": "", "batch_size": 0,
             "shuffle": True, "apply_at": "boundary", "points_mode": "file", "points_list": "", "points_count": 10,
             "points_where": "boundary", "values_mode": "file", "values_formula": ""} for i in range(n_out)])
        cfg.num_domain, cfg.num_initial, cfg.num_boundary = 10000, 1000, 1000
        _phases(cfg, 20000, 10000, ic_w=1.0)
        return
    ph = json.loads(cfg.scheduler_phases or "[]") if cfg.optimizer_scheduler else []
    adam = sum(int(p["iterations"]) for p in ph if p["optimizer"] == "adam") or cfg.iterations
    lb = sum(int(p["iterations"]) for p in ph if p["optimizer"].startswith("l")) or 10000
    _phases(cfg, adam, lb, ic_w=1.0)


# --- per template ---------------------------------------------------------------------------------

def _ac1d_wz(cfg):
    # W&Z Sec. 3.1, Table 3.1 ("resampling" trial, their best on this problem, rel. L2 2.33e-2): 4x128 tanh,
    # weighted loss C0 = 100 (Sec. 2.2.1 default), 2,000 LHS collocation points + 200 added per resampling,
    # about six resamplings, Adam then L-BFGS-B. Not stated: Adam lr/iterations, L-BFGS settings, N_u, N_b
    # (Sec. 3.1 defaults N_u = N_b = 200 used). Resampling = PINNStudio's RAR (residual-based refinement).
    _soft(cfg)
    _net(cfg, 4, 128)
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 2000, 200, 200
    cfg.adapt_method = "RAR"
    cfg.rar_cycles, cfg.rar_add_points, cfg.rar_candidates = 6, 200, 20000
    cfg.rar_adam_iters, cfg.rar_lbfgs_iters = 5000, 5000
    _phases(cfg, 20000, 10000, ic_w=100.0)


def _ac1d_mg(cfg):
    # M&G Sec. 5.1, Table 1 [PAR]: bc-PINN 6x128 tanh, N_i = 128, N_b = 50 and N_r = 20,000 per segment,
    # 50 time steps (0.25) per segment -> 4 segments, 10,000 Adam per segment then L-BFGS; with ICGL and
    # transfer learning (rel. L2 1.68e-2). Weights from the authors' code: 64 on the IC and backward terms,
    # 1 on the rest; ICGL 10% of Adam (code). Backward points: 128 x every reference step in code; here 128
    # points x 50 times per segment. L-BFGS: 50,000 max in the paper (Sec. 3.4).
    _soft(cfg)
    _net(cfg, 6, 128)
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 20000, 128, 50
    _windows(cfg, 4, backward=True, icgl=0.1, bc_w=64.0, bc_times=50, bc_points=128)
    cfg.ta_grid_size = 128                      # each segment's start: the previous solution at 128 points
    _phases(cfg, 10000, 50000, ic_w=64.0)


def _ch1d_wz(cfg):
    # W&Z Sec. 3.2: one network (u, mu), 4x128 tanh, time-adaptive approach I + resampling, Adam lr 1e-3
    # (max 20,000) then L-BFGS-B, weighted loss "as for AC" (C0 = 100); rel. L2 9.51e-3. Not stated: the
    # time step of approach I (0.1 used, as their 1D AC), iterations per stage, the resampling schedule
    # (here 2 rounds of 200 points per stage, 2,000 Adam iterations each; their 1D AC trial added 200 points
    # per resampling).
    _soft(cfg, ic_outputs=[0])
    _net(cfg, 4, 128)
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 10000, 512, 200
    _windows(cfg, 10, growing=True)
    cfg.ta_resample_rounds, cfg.ta_resample_points, cfg.ta_resample_iters = 2, 200, 2000
    cfg.ta_resample_candidates = 10000
    _phases(cfg, 5000, 5000, ic_w=100.0)


def _ch1d_mg(cfg):
    # M&G Sec. 5.2, Table 3 [PAR]: bc-PINN, (h, mu), 6x128, N_i = 128, N_b = 20 and N_r = 10,000 per
    # segment, 20 steps (0.1) per segment -> 10 segments, 10,000 Adam per segment then L-BFGS, ICGL + TL;
    # rel. L2 1.86e-2. Weights for CH are not stated: 64 on IC and backward terms as in their AC code.
    _soft(cfg, ic_outputs=[0])
    _net(cfg, 6, 128)
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 10000, 128, 20
    _windows(cfg, 10, backward=True, icgl=0.1, bc_w=64.0, bc_times=20, bc_points=128)
    cfg.ta_grid_size = 128
    _phases(cfg, 10000, 50000, ic_w=64.0)


def _ac2d_mg(cfg):
    # M&G Sec. 6.1.1, Table 4 [PAR] (IBVP-1): bc-PINN 6x128, N_i = 4096 (the 64^2 grid), N_b = 640 and
    # N_r = 20,000 per segment, 20 steps (0.2) per segment -> 5 segments, 15,000 Adam per segment of which
    # 20% ICGL, then 10,000 L-BFGS; rel. L2 2.5e-2 (> 95% without ICGL). Weights not stated: 64 as in
    # their code. Backward points (not stated): 1,000 of the 64^2 grid x 10 times per segment.
    _soft(cfg)
    _net(cfg, 6, 128)
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 20000, 4096, 640
    _windows(cfg, 5, backward=True, icgl=0.25, bc_w=64.0, bc_times=10, bc_points=1000)
    cfg.ta_grid_size = 64                       # each segment's start on the 64 x 64 grid, N_i = 4096
    _phases(cfg, 12000, 10000, ic_w=64.0)       # 12,000 + 25% of it (3,000) ICGL = 15,000 Adam


def _burgers1d_raissi(cfg):
    # Raissi et al. Sec. 2.1 and their code: 8x20 tanh (3,021 parameters), N_u = 100 IC/BC points,
    # N_f = 10,000 LHS points, L-BFGS-B only (maxiter 50,000), equal weights; rel. L2 6.7e-4.
    # The code normalises inputs to [-1, 1] (not done here; x is already in [-1, 1], t in [0, 1]).
    _soft(cfg)
    _net(cfg, 8, 20)
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 10000, 50, 50
    _phases(cfg, 0, 50000, ic_w=1.0)


def _schrodinger_raissi(cfg):
    # Raissi et al. Sec. 2.2 and their code: 4x100 tanh, (u, v), N_0 = 50, N_b = 50, N_f = 20,000 LHS,
    # Adam 50,000 (code; not in the paper) then L-BFGS-B, equal weights; rel. L2 of |h| 1.97e-3.
    _soft(cfg)
    _net(cfg, 4, 100)
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 20000, 50, 50
    _phases(cfg, 50000, 50000, ic_w=1.0)



def _wz_approach2(cfg, n_windows, outputs_ic=None):
    # Wight & Zhao's benchmarks (Sec. 3.1-3.2, Figs. 3.8, 3.9, 3.13): approach II, a separate network per
    # time window starting from the previous one's prediction, 6x128, weighted loss C0 = 10^3, Adam with
    # mini-batch 32 for 100 epochs per network, then L-BFGS-B. Not stated: the points, the BCs, the Adam
    # learning rate and the L-BFGS settings; whether a network starts from the previous one's weights (here
    # not). PINNStudio trains full batch: 10,000 Adam + 10,000 L-BFGS per window, the template's points and BCs.
    _soft(cfg, ic_outputs=outputs_ic)
    _net(cfg, 6, 128)
    _windows(cfg, n_windows, tl=False)
    _phases(cfg, 10000, 10000, ic_w=1000.0)


def _ac2d_wz(cfg):
    # Sec. 3.1, Fig. 3.8: Delta t = 1 on t in [0, 10] (the caption also says 0.2; the text says 1) -> 10 windows.
    _wz_approach2(cfg, 10)


def _ch2d_wz(cfg):
    # Sec. 3.2, Fig. 3.13: (u, mu) network, Delta t = 0.2 on [0, 1] -> 5 windows.
    _wz_approach2(cfg, 5, outputs_ic=[0])


def _ac3d_wz(cfg):
    # Sec. 3.1, Fig. 3.9: Delta t = 0.5 on [0, 1] -> 2 windows.
    _wz_approach2(cfg, 2)


def _ch2d_mg(cfg):
    # M&G Sec. 6.2.1, Table 7 (IBVP-2, the template's problem; t in (0, 0.005] in the paper, rescaled to
    # [0, 1] in the template): bc-PINN, (h, mu), 5x128, N_i = 4096 (64^2), N_b = 1600 and N_r = 50,000 per
    # segment, 25 steps per segment -> 4 segments, 50,000 Adam and up to 75,000 L-BFGS per segment, ICGL,
    # transfer learning with one more frozen layer per segment (layer freezing is not in PINNStudio).
    # Not stated: the loss weights (64 as in their AC code), the ICGL fraction (20% as in their 2D AC).
    _soft(cfg, ic_outputs=[0])
    _net(cfg, 5, 128)
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 50000, 4096, 1600
    _windows(cfg, 4, backward=True, icgl=0.2, bc_w=64.0, bc_times=10, bc_points=1000)
    cfg.ta_grid_size = 64
    _phases(cfg, 40000, 75000, ic_w=64.0)       # 40,000 + 20% of it (8,000) ICGL ~ 50,000 Adam


def _burgers2d_mathias(cfg):
    # Mathias et al. Sec. 2 and their code (github.com/C4AI/PINN_for_Burgers): the IC and the zero BCs built
    # into the output, U = d*N + (1 - d)*U0 with d = 16x(1-x)y(1-y)tanh(26.4t); 4x20 tanh with residual
    # connections every two layers (PINNStudio: a plain 4x20 network); mean absolute error loss; n_f =
    # 10,000 uniform physics points and no solution data (their pure-physics case); Adam lr 1e-2 for
    # 99,999 steps, full batch, no L-BFGS.
    _soft(cfg, ic_outputs=[])
    _net(cfg, 4, 20)
    cfg.custom_bc_json = "[]"
    d = "16*x*(1 - x)*y*(1 - y)*tanh(26.4*t)"
    cfg.output_transform_enabled = True
    cfg.output_transform_mode = "formula"
    cfg.output_transform_formula = (f"u*{d} + (1 - {d})*sin(2*pi*x)*sin(2*pi*y)|"
                                    f"v*{d} + (1 - {d})*sin(pi*x)*sin(pi*y)")
    cfg.num_domain, cfg.num_initial, cfg.num_boundary = 10000, 0, 0
    w = ",".join(["1.0"] * len([e for e in (cfg.pde_expressions or "").split("|") if e.strip()]))
    cfg.optimizer_scheduler = True
    cfg.scheduler_phases = json.dumps([{"optimizer": "adam", "iterations": 99999, "lr": 0.01, "loss": "MAE",
                                        "weights": w, "decay_type": "none", "decay_p1": 5000.0, "decay_p2": 0.9}])

PAPER = {
    ("1D Allen-Cahn (whole interval)", "paper_wz"): _ac1d_wz,
    ("1D Allen-Cahn (whole interval)", "paper_mg"): _ac1d_mg,
    ("1D Allen-Cahn", "paper_wz"): _ac1d_wz,
    ("1D Allen-Cahn", "paper_mg"): _ac1d_mg,
    ("1D Cahn-Hilliard", "paper_wz"): _ch1d_wz,
    ("1D Cahn-Hilliard (Mattey & Ghosh)", "paper_mg"): _ch1d_mg,
    ("2D Allen-Cahn (Mattey & Ghosh)", "paper_mg"): _ac2d_mg,
    ("1D Burgers", "paper_raissi"): _burgers1d_raissi,
    ("1D Schrödinger", "paper_raissi"): _schrodinger_raissi,
    ("2D Allen-Cahn (Wight & Zhao)", "paper_wz"): _ac2d_wz,
    ("2D Cahn-Hilliard (Wight & Zhao)", "paper_wz"): _ch2d_wz,
    ("3D Allen-Cahn (Wight & Zhao)", "paper_wz"): _ac3d_wz,
    ("2D Cahn-Hilliard (Mattey & Ghosh)", "paper_mg"): _ch2d_mg,
    ("2D Burgers (Mathias)", "paper_mathias"): _burgers2d_mathias,
}


def apply(cfg, variant, tkey):
    """A changed copy of cfg for a plain / paper_* variant, or None when it does not apply."""
    if variant == "plain":
        cfg = copy.deepcopy(cfg)
        plain(cfg, tkey)
        return cfg
    f = PAPER.get((tkey, variant))
    if f is None:
        return None
    cfg = copy.deepcopy(cfg)
    f(cfg)
    return cfg


PAPER_LABEL = {"paper_wz": "Wight & Zhao 2021", "paper_mg": "Mattey & Ghosh 2022, bc-PINN",
               "paper_raissi": "Raissi et al. 2019", "paper_mathias": "Mathias et al. 2022"}

# The paper each template's Default settings follow (the template's own source).
TEMPLATE_PAPER = {
    "1D Allen-Cahn": "paper_wz", "1D Allen-Cahn (whole interval)": "paper_wz",
    "1D Cahn-Hilliard": "paper_wz", "1D Cahn-Hilliard (Mattey & Ghosh)": "paper_mg",
    "2D Allen-Cahn (Mattey & Ghosh)": "paper_mg", "1D Burgers": "paper_raissi",
    "1D Schrödinger": "paper_raissi", "2D Allen-Cahn (Wight & Zhao)": "paper_wz",
    "2D Cahn-Hilliard (Wight & Zhao)": "paper_wz", "3D Allen-Cahn (Wight & Zhao)": "paper_wz",
    "2D Cahn-Hilliard (Mattey & Ghosh)": "paper_mg", "2D Burgers (Mathias)": "paper_mathias",
}

import os
os.environ["DDE_BACKEND"] = "pytorch"
# ── CUDA performance environment variables ────────────────────
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"        # async CUDA launches
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"      # RTX 4090 = Ada Lovelace = sm_89
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

def _simplify_expr(expr, is_2d=False):
    """Convert user-friendly math syntax to numpy syntax."""
    import re
    e = expr.strip()
    # Replace math functions with numpy equivalents
    for fn in ["sin","cos","tan","sinh","cosh","tanh","arcsin","arccos","arctan",
               "exp","log","log10","sqrt","abs","ceil","floor"]:
        e = re.sub(rf'\b{fn}\(', f'np.{fn}(', e)
    # Replace pi and e constants
    e = re.sub(r'\bpi\b', 'np.pi', e)
    e = re.sub(r'\bexp\b', 'np.e', e)
    # Replace x, y variables — must be done carefully to avoid replacing inside words
    if is_2d:
        e = re.sub(r'\by\b', '__Y__', e)
        e = re.sub(r'\bx\b', 'x[:, 0]', e)
        e = e.replace('__Y__', 'x[:, 1]')
    else:
        e = re.sub(r'\bx\b', 'x[:, 0]', e)
    return e

def _simplify_pde_expr(expr):
    """Convert user-friendly math in PDE — only functions and constants, not variables."""
    import re
    e = expr.strip()
    for fn in ["sin","cos","tan","sinh","cosh","tanh","arcsin","arccos","arctan",
               "exp","log","log10","sqrt","abs","ceil","floor"]:
        e = re.sub(rf'(?<![a-zA-Z_]){fn}\(', f'np.{fn}(', e)
    e = re.sub(r'(?<![a-zA-Z_])pi(?![a-zA-Z_])', 'np.pi', e)
    return e

def generate_script(config):
    is_2d = config.problem_dim == "2D"
    # Convert user-friendly IC expressions
    ic_exprs_raw = config.ic_expressions.split("|")
    ic_exprs_converted = [_simplify_expr(e, is_2d) for e in ic_exprs_raw]
    config_ic_expressions = "|".join(ic_exprs_converted)
    ta_ic_expr = _simplify_expr(config.ic_expression, is_2d)

    # Convert user-friendly PDE expressions
    pde_exprs_raw = config.pde_expressions.split("|")
    pde_exprs_converted = [_simplify_pde_expr(e) for e in pde_exprs_raw]
    config_pde_expressions = "|".join(pde_exprs_converted)

    pde_expr_single = _simplify_pde_expr(config.pde_expression)

    if config.forward_ic_from_file:
        _ta_ic_init = f"""_ic_ta_data = np.loadtxt(r"{config.forward_ic_file}")
    _ic_ta_mask = np.abs(_ic_ta_data[:, 2]) < 1e-10
    prev_u = _ic_ta_data[_ic_ta_mask, 3:4]"""
    else:
        _ta_ic_init = f"prev_u = np.reshape({ta_ic_expr}, (-1, 1))"
    _fecr_pde_block = ""

    script = f"""

import os
os.environ["DDE_BACKEND"] = "pytorch"

# ── Windows has no /tmp by default; make "/tmp/..." paths work there too ──
if os.name == "nt":
    _tmp_root = os.path.splitdrive(os.getcwd())[0] + os.sep + "tmp"
    os.makedirs(_tmp_root, exist_ok=True)

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*cuBLAS.*")

# ── Force GPU initialization ──────────────────────────────────
_effective_float = "{config.float_type}"
if "{config.lbfgs_float_type}" == "float64" and _effective_float == "float32":
    _effective_float = "float64"
    print("  [Info] Using float64 globally (required for L-BFGS float64 mode)")
dde.config.set_default_float(_effective_float)
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)

    # ── Maximize GPU memory usage ─────────────────────────────
    # Reserve as much memory as possible upfront
    torch.cuda.empty_cache()
    total_mem = torch.cuda.get_device_properties(0).total_memory
    # Allow PyTorch to use up to 95% of GPU memory
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)

    # ── Performance settings ──────────────────────────────────
    _is_f64 = "{config.float_type}" == "float64"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    torch.backends.cudnn.benchmark        = False
    torch.backends.cudnn.deterministic    = False

    # ── Warm up CUDA ──────────────────────────────────────────
    _dummy = torch.zeros(10, 10, requires_grad=True, device='cuda')
    _loss = (_dummy ** 2).sum()
    _loss.backward()
    torch.cuda.synchronize()
    del _dummy, _loss
    torch.cuda.empty_cache()

    _free, _total = torch.cuda.mem_get_info(0)
    print(f"✅ GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"   Total memory: {{_total / 1e9:.1f}} GB")
    print(f"   Available:    {{_free / 1e9:.1f}} GB")
    print(f"   TF32:         False")
    print(f"   Float type:   {config.float_type}")
    print(f"   cuDNN bench:  False")
else:
    print("⚠️ No GPU found — running on CPU")

# ── Save directory setup ─────────────────────────────────────
import os as _os
_save_dir = "{config.save_dir}".strip()
_use_save = bool(_save_dir)
if _use_save:
    _os.makedirs(_save_dir, exist_ok=True)
    import json as _json
    _model_config = {{
        "layers": {config.layers},
        "activation": "{config.activation}",
        "num_outputs": {config.num_outputs},
        "output_names": "{config.output_names}",
        "x_min": {config.x_min}, "x_max": {config.x_max},
        "y_min": {config.y_min}, "y_max": {config.y_max},
        "t_min": {config.t_min}, "t_max": {config.t_max},
        "problem_dim": "{config.problem_dim}",
        "pde_expressions": "{config.pde_expressions}",
        "optimizer": "{config.optimizer}",
        "optimizer2": "{config.optimizer2}",
        "loss_type": "{config.loss_type}",
    }}
    _os.makedirs(_os.path.join(_save_dir, "solution_results"), exist_ok=True)
    with open(_os.path.join(_save_dir, "solution_results", "model_config.json"), "w") as _mcf:
        _json.dump(_model_config, _mcf, indent=2)
    print(f"Model config saved to: {{_os.path.join(_save_dir, 'solution_results', 'model_config.json')}}")

# ── Solution results folder ───────────────────────────────────
_sol_dir = _os.path.join(_save_dir, "solution_results") if _use_save else "/tmp"
if _use_save:
    _os.makedirs(_sol_dir, exist_ok=True)
_loss_path     = _os.path.join(_sol_dir, "loss_plot.png")
_solution_path = _os.path.join(_sol_dir, "solution_plot.png")
_log_path      = _os.path.join(_sol_dir, "training_log.txt") if _use_save else None

# ── Problem dimension ─────────────────────────────────────────
_is_2d = "{config.problem_dim}" == "2D"

# ── Parametric study setup ────────────────────────────────────
_parametric = {config.parametric_study}
_param_name  = "{config.parametric_param}"
_param_values_str = "{config.parametric_values}"

if _parametric:
    _param_values = [v.strip() for v in _param_values_str.split(",") if v.strip()]
    print(f"=== Parametric Study: {{_param_name}} = {{_param_values}} ===")
else:
    _param_values = [None]

# ── Inverse PINN setup ───────────────────────────────────────
_problem_type = "{config.problem_type}"

if _problem_type == "Inverse":
    import pandas as _pd

    {config.inverse_param_name} = dde.Variable({config.inverse_param_init})
    print(f"Inverse PINN: inferring {config.inverse_param_name}, init = {config.inverse_param_init}")

    def _load_data(fpath):
        try:
            arr = np.loadtxt(fpath, delimiter=",", skiprows=1)
        except Exception:
            try:
                arr = np.loadtxt(fpath, skiprows=1)
            except Exception:
                arr = np.loadtxt(fpath, delimiter=",")
        return arr

    _obs_data = _load_data("{config.inverse_data_file}")
    if _is_2d:
        _obs_xt = _obs_data[:, 0:3]  # x, y, t
        _obs_u  = _obs_data[:, 3:4]  # u
    else:
        _obs_xt = _obs_data[:, 0:2]  # x, t
        _obs_u  = _obs_data[:, 2:3]  # u
    print(f"Loaded {{len(_obs_xt)}} observation points from data file")

    if "{config.inverse_ic_type}" == "File (x, t, u)":
        _ic_data = _load_data("{config.inverse_ic_file}")
        _ic_xt   = _ic_data[:, 0:2]
        _ic_u    = _ic_data[:, 2:3]
        print(f"Loaded {{len(_ic_xt)}} IC points from file")

    _{config.inverse_param_name}_history = []

    class _PrintParamCallback(dde.callbacks.Callback):
        def __init__(self, var, name, period=1000):
            super().__init__()
            self.var = var
            self.name = name
            self.period = period
            self._last_bucket = 0
        def set_offset(self, offset):
            # No-op, kept so older scheduler-loop code can still call it.
            # Real progress now comes from the model's own train_state.step,
            # which is already cumulative across every phase — no manual
            # offset bookkeeping needed. This also fixes L-BFGS: DeepXDE only
            # calls on_batch_end once per outer L-BFGS step (which can cover
            # hundreds of real iterations at once), not once per real
            # iteration the way Adam does, so counting callback firings
            # (the old approach) almost never reached the print period.
            pass
        def on_batch_end(self):
            cur = self.model.train_state.step
            bucket = cur // self.period
            if bucket > self._last_bucket:
                self._last_bucket = bucket
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                print(f"  [{config.inverse_param_name}] Iter {{cur}}: {{val:.6f}}", flush=True)

    _print_cb = _PrintParamCallback({config.inverse_param_name}, "{config.inverse_param_name}", period=1000)
    
    # Parameter saving to text file
    _param_save_opt = "{config.inv_param_save}"
    _param_save_period = 100 if _param_save_opt == "Every 100 iters" else 1000 if _param_save_opt == "Every 1000 iters" else 0
    _param_save_path = _os.path.join(_save_dir if _use_save else "/tmp", f"{config.inverse_param_name}_convergence.txt") if _param_save_period > 0 else None

    if _param_save_period > 0 and _param_save_path:
        with open(_param_save_path, "w") as _psf:
            _psf.write(f"iteration,{config.inverse_param_name}\\n")
        print(f"Parameter convergence will be saved to: {{_param_save_path}}")

    class _SaveParamCallback(dde.callbacks.Callback):
        def __init__(self, var, name, period, path):
            super().__init__()
            self.var = var
            self.name = name
            self.period = period
            self.path = path
            self._last_bucket = 0
        def set_offset(self, offset):
            # No-op — see _PrintParamCallback.set_offset above.
            pass
        def on_batch_end(self):
            if self.period == 0: return
            cur = self.model.train_state.step
            bucket = cur // self.period
            if bucket > self._last_bucket:
                self._last_bucket = bucket
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                with open(self.path, "a") as _f:
                    _f.write(f"{{cur}},{{val:.8f}}\\n")

    _save_cb = _SaveParamCallback(
        {config.inverse_param_name},
        "{config.inverse_param_name}",
        _param_save_period,
        _param_save_path if _param_save_path else "/tmp/param_save.txt"
    )


{_fecr_pde_block}
_IS_FECR = False
# ── PDE definition (standard) ───────────────────────────────
def _pde_standard(x, y):
    _n_out = {config.num_outputs}
    _out_names = [n.strip() for n in "{config.output_names}".split(",")]
    _is_2d_pde = "{config.problem_dim}" == "2D"

    _dvars = {{}}
    for _oi, _oname in enumerate(_out_names):
        _dvars[_oname] = y[:, _oi:_oi+1]
        if _is_2d_pde:
            # 2D: inputs are (x, y, t) → j=0,1,2
            # Always compute first-order derivatives
            _dvars[f"d{{_oname}}_x"] = dde.grad.jacobian(y, x, i=_oi, j=0)
            _dvars[f"d{{_oname}}_y"] = dde.grad.jacobian(y, x, i=_oi, j=1)
            _dvars[f"d{{_oname}}_t"] = dde.grad.jacobian(y, x, i=_oi, j=2)
            # Only compute higher-order derivatives if used in PDE
            _pde_str_check = "{config_pde_expressions}"
            _oname_check = _oname
            if f"d{{_oname_check}}_xx" in _pde_str_check or f"d{{_oname_check}}_xxxx" in _pde_str_check or f"d{{_oname_check}}_xxyy" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xx"] = dde.grad.hessian(y, x, component=_oi, i=0, j=0)
            if f"d{{_oname_check}}_yy" in _pde_str_check or f"d{{_oname_check}}_yyyy" in _pde_str_check or f"d{{_oname_check}}_xxyy" in _pde_str_check or f"d{{_oname_check}}_yytt" in _pde_str_check:
                _dvars[f"d{{_oname}}_yy"] = dde.grad.hessian(y, x, component=_oi, i=1, j=1)
            if f"d{{_oname_check}}_xy" in _pde_str_check:
                _dvars[f"d{{_oname}}_xy"] = dde.grad.hessian(y, x, component=_oi, i=0, j=1)
            if f"d{{_oname_check}}_tt" in _pde_str_check:
                _dvars[f"d{{_oname}}_tt"] = dde.grad.hessian(y, x, component=_oi, i=2, j=2)
            if f"d{{_oname_check}}_xt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xt"] = dde.grad.hessian(y, x, component=_oi, i=0, j=2)
            if f"d{{_oname_check}}_yt" in _pde_str_check:
                _dvars[f"d{{_oname}}_yt"] = dde.grad.hessian(y, x, component=_oi, i=1, j=2)
            if f"d{{_oname_check}}_xxxx" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxxx"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            if f"d{{_oname_check}}_yyyy" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yyyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=1, j=1)
            if f"d{{_oname_check}}_xxyy" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=1, j=1)
            if f"d{{_oname_check}}_xxtt" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxtt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=2, j=2)
            if f"d{{_oname_check}}_yytt" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yytt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=2, j=2)
        else:
            # 1D: inputs are (x, t) → j=0,1
            _dvars[f"d{{_oname}}_x"] = dde.grad.jacobian(y, x, i=_oi, j=0)
            _dvars[f"d{{_oname}}_t"] = dde.grad.jacobian(y, x, i=_oi, j=1)
            _pde_str_check = "{config_pde_expressions}"
            _oname_check = _oname
            if f"d{{_oname_check}}_xx" in _pde_str_check or f"d{{_oname_check}}_xxxx" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xx"] = dde.grad.hessian(y, x, component=_oi, i=0, j=0)
            if f"d{{_oname_check}}_tt" in _pde_str_check or f"d{{_oname_check}}_tttt" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_tt"] = dde.grad.hessian(y, x, component=_oi, i=1, j=1)
            if f"d{{_oname_check}}_xt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xt"] = dde.grad.hessian(y, x, component=_oi, i=0, j=1)
            if f"d{{_oname_check}}_xxxx" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxxx"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            if f"d{{_oname_check}}_xxtt" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxtt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=1, j=1)
            if f"d{{_oname_check}}_tttt" in _pde_str_check and f"d{{_oname}}_tt" in _dvars:
                _dvars[f"d{{_oname}}_tttt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_tt"], x, i=1, j=1)

    _eval_ns = {{**_dvars}}
    _eval_ns["dde"] = dde
    _eval_ns["np"]  = np
    _eval_ns["x"]   = x
    _eval_ns["y"]   = y
    _eval_ns["torch"] = torch

    if _problem_type == "Inverse":
        _eval_ns["{config.inverse_param_name}"] = {config.inverse_param_name}

    if _n_out == 1:
        return eval("{pde_expr_single}", _eval_ns)
    else:
        _pde_exprs = "{config_pde_expressions}".split("|")
        return [eval(_expr.strip(), _eval_ns) for _expr in _pde_exprs]
if not _IS_FECR:
    pde = _pde_standard

# ── Geometry & Time domain ──────────────────────────────────
if _is_2d:
    geom = dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])
else:
    geom = dde.geometry.Interval({config.x_min}, {config.x_max})
timedomain = dde.geometry.TimeDomain({config.t_min}, {config.t_max})
geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

# ── Boundary & Initial conditions ────────────────────────────
_out_names_list  = "{config.output_names}".split(",")
_bc_left_types   = "{config.bc_left_types}".split(",")
_bc_right_types  = "{config.bc_right_types}".split(",")
_bc_left_values  = "{config.bc_left_values}".split(",")
_bc_right_values = "{config.bc_right_values}".split(",")
_bc_left_active  = "{config.bc_left_active}".split(",")
_bc_right_active = "{config.bc_right_active}".split(",")
_bc_left_deriv_list  = "{config.bc_left_deriv}".split(",")
_bc_right_deriv_list = "{config.bc_right_deriv}".split(",")

_bc_bottom_types  = "{config.bc_bottom_types}".split(",")
_bc_top_types     = "{config.bc_top_types}".split(",")
_bc_bottom_values = "{config.bc_bottom_values}".split(",")
_bc_top_values    = "{config.bc_top_values}".split(",")
_bc_bottom_active = "{config.bc_bottom_active}".split(",")
_bc_top_active    = "{config.bc_top_active}".split(",")
_bc_bottom_deriv_list = "{config.bc_bottom_deriv}".split(",")
_bc_top_deriv_list    = "{config.bc_top_deriv}".split(",")

_ic_expressions = "{config_ic_expressions}".split("|")
_ic_active_list = "{config.ic_active}".split(",")

_constraints = []

if _problem_type == "Inverse" and "{config.inverse_ic_type}" == "File (x, t, u)":
    _constraints.append(dde.icbc.PointSetBC(_ic_xt, _ic_u, component=0))
else:
    for _oi in range({config.num_outputs}):
        _comp = _oi
        _blt = _bc_left_types[_oi].strip()  if _oi < len(_bc_left_types)  else "Dirichlet"
        _brt = _bc_right_types[_oi].strip() if _oi < len(_bc_right_types) else "Dirichlet"
        _blv = float(_bc_left_values[_oi].strip())  if _oi < len(_bc_left_values)  else 0.0
        _brv = float(_bc_right_values[_oi].strip()) if _oi < len(_bc_right_values) else 0.0

        # ── BC left (x = x_min) ──────────────────────────────
        if _oi < len(_bc_left_active) and _bc_left_active[_oi].strip() == "True":
            def _bl_on_boundary(x, on_boundary):
                return on_boundary and dde.utils.isclose(x[0], {config.x_min})
            if _blt == "Dirichlet":
                def _make_dbc_l(v, comp, on_bd):
                    def _val_fn(x): return np.full((len(x), 1), v)
                    return dde.icbc.DirichletBC(geomtime, _val_fn, on_bd, component=comp)
                _constraints.append(_make_dbc_l(_blv, _comp, _bl_on_boundary))
            elif _blt == "Neumann":
                def _make_nbc_l(v, comp, on_bd):
                    def _val_fn(x): return np.full((len(x), 1), v)
                    return dde.icbc.NeumannBC(geomtime, _val_fn, on_bd, component=comp)
                _constraints.append(_make_nbc_l(_blv, _comp, _bl_on_boundary))
            elif _blt == "Periodic":
                _constraints.append(dde.icbc.PeriodicBC(geomtime, 0, _bl_on_boundary, derivative_order=0, component=_comp))
                if _oi < len(_bc_left_deriv_list) and _bc_left_deriv_list[_oi].strip() == "True":
                    _constraints.append(dde.icbc.PeriodicBC(geomtime, 0, _bl_on_boundary, derivative_order=1, component=_comp))

        # ── BC right (x = x_max) ─────────────────────────────
        if _oi < len(_bc_right_active) and _bc_right_active[_oi].strip() == "True":
            def _br_on_boundary(x, on_boundary):
                return on_boundary and dde.utils.isclose(x[0], {config.x_max})
            if _brt == "Dirichlet":
                def _make_dbc_r(v, comp, on_bd):
                    def _val_fn(x): return np.full((len(x), 1), v)
                    return dde.icbc.DirichletBC(geomtime, _val_fn, on_bd, component=comp)
                _constraints.append(_make_dbc_r(_brv, _comp, _br_on_boundary))
            elif _brt == "Neumann":
                def _make_nbc_r(v, comp, on_bd):
                    def _val_fn(x): return np.full((len(x), 1), v)
                    return dde.icbc.NeumannBC(geomtime, _val_fn, on_bd, component=comp)
                _constraints.append(_make_nbc_r(_brv, _comp, _br_on_boundary))
            elif _brt == "Periodic":
                pass  # Periodic BC is handled by left side only — DeepXDE enforces both ends together

        # ── BC bottom (y = y_min) — 2D only ──────────────────
        if _is_2d:
            _bbt = _bc_bottom_types[_oi].strip()  if _oi < len(_bc_bottom_types)  else "Dirichlet"
            _btt = _bc_top_types[_oi].strip()     if _oi < len(_bc_top_types)     else "Dirichlet"
            _bbv = float(_bc_bottom_values[_oi].strip()) if _oi < len(_bc_bottom_values) else 0.0
            _btv = float(_bc_top_values[_oi].strip())    if _oi < len(_bc_top_values)    else 0.0

            if _oi < len(_bc_bottom_active) and _bc_bottom_active[_oi].strip() == "True":
                def _bb_on_boundary(x, on_boundary):
                    return on_boundary and dde.utils.isclose(x[1], {config.y_min})
                if _bbt == "Dirichlet":
                    def _make_dbc_b(v, comp, on_bd):
                        def _val_fn(x): return np.full((len(x), 1), v)
                        return dde.icbc.DirichletBC(geomtime, _val_fn, on_bd, component=comp)
                    _constraints.append(_make_dbc_b(_bbv, _comp, _bb_on_boundary))
                elif _bbt == "Neumann":
                    def _make_nbc_b(v, comp, on_bd):
                        def _val_fn(x): return np.full((len(x), 1), v)
                        return dde.icbc.NeumannBC(geomtime, _val_fn, on_bd, component=comp)
                    _constraints.append(_make_nbc_b(_bbv, _comp, _bb_on_boundary))
                elif _bbt == "Periodic":
                    _constraints.append(dde.icbc.PeriodicBC(geomtime, 1, _bb_on_boundary, derivative_order=0, component=_comp))
                    if _oi < len(_bc_bottom_deriv_list) and _bc_bottom_deriv_list[_oi].strip() == "True":
                        _constraints.append(dde.icbc.PeriodicBC(geomtime, 1, _bb_on_boundary, derivative_order=1, component=_comp))

            # ── BC top (y = y_max) ────────────────────────────
            if _oi < len(_bc_top_active) and _bc_top_active[_oi].strip() == "True":
                def _bt_on_boundary(x, on_boundary):
                    return on_boundary and dde.utils.isclose(x[1], {config.y_max})
                if _btt == "Dirichlet":
                    def _make_dbc_t(v, comp, on_bd):
                        def _val_fn(x): return np.full((len(x), 1), v)
                        return dde.icbc.DirichletBC(geomtime, _val_fn, on_bd, component=comp)
                    _constraints.append(_make_dbc_t(_btv, _comp, _bt_on_boundary))
                elif _btt == "Neumann":
                    def _make_nbc_t(v, comp, on_bd):
                        def _val_fn(x): return np.full((len(x), 1), v)
                        return dde.icbc.NeumannBC(geomtime, _val_fn, on_bd, component=comp)
                    _constraints.append(_make_nbc_t(_btv, _comp, _bt_on_boundary))
                elif _btt == "Periodic":
                    pass  # Periodic BC handled by bottom side only

        # ── IC ────────────────────────────────────────────────
        if {config.forward_ic_from_file} and _oi == 0:
            # Load IC from file (x, y, t, c format) — use PointSetBC at t=0
            _ic_data = np.loadtxt(r"{config.forward_ic_file}")
            _ic_mask = np.abs(_ic_data[:, 2]) < 1e-10  # rows where t≈0
            _ic_xy   = _ic_data[_ic_mask, :2]          # x, y
            _ic_t0   = np.zeros((_ic_mask.sum(), 1))
            _ic_xyt  = np.hstack([_ic_xy, _ic_t0])     # (x, y, 0)
            _ic_vals = _ic_data[_ic_mask, 3:4]          # c values
            _constraints.append(dde.icbc.PointSetBC(_ic_xyt, _ic_vals, component=0))
            print(f"IC loaded from file: {config.forward_ic_file} — {{_ic_mask.sum()}} points")
        elif _oi < len(_ic_active_list) and _ic_active_list[_oi].strip() == "True":
            _ic_expr = _ic_expressions[_oi].strip() if _oi < len(_ic_expressions) else "np.zeros_like(x[:,0])"
            def _make_ic(expr, comp):
                def _ic_fn(x):
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                return dde.icbc.IC(geomtime, _ic_fn, lambda x, on_initial: on_initial, component=comp)
            _constraints.append(_make_ic(_ic_expr, _comp))

# ── Loss weights ─────────────────────────────────────────────
_n_out_w = {config.num_outputs}
_bc_la = "{config.bc_left_active}".split(",")
_bc_ra = "{config.bc_right_active}".split(",")
_bc_ba = "{config.bc_bottom_active}".split(",")
_bc_ta = "{config.bc_top_active}".split(",")
_ic_a  = "{config.ic_active}".split(",")

_wm_list = [float(v) for v in "{config.loss_weights_multi}".split(",") if v.strip()] if "{config.loss_weights_multi}".strip() else []

_pde_w = []; _bcl_w = []; _bcr_w = []; _bcb_w = []; _bct_w = []; _ic_w = []
_wi = 0
for _oi_w in range(_n_out_w):
    _pde_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
    if _oi_w < len(_bc_la) and _bc_la[_oi_w].strip() == "True":
        _bcl_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
    else:
        _bcl_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0
    _brt_check = "{config.bc_right_types}".split(",")
    _blt_check = "{config.bc_left_types}".split(",")
    _brt_is_periodic = _oi_w < len(_brt_check) and _brt_check[_oi_w].strip() == "Periodic"
    _blt_is_periodic = _oi_w < len(_blt_check) and _blt_check[_oi_w].strip() == "Periodic"
    if _oi_w < len(_bc_ra) and _bc_ra[_oi_w].strip() == "True" and not _brt_is_periodic and not _blt_is_periodic:
        _bcr_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
    else:
        _bcr_w.append(None)  # do NOT advance _wi — GUI sends no value for this slot
    _btt_check = "{config.bc_top_types}".split(",")
    _bbt_check = "{config.bc_bottom_types}".split(",")
    if _is_2d:
        if _oi_w < len(_bc_ba) and _bc_ba[_oi_w].strip() == "True":
            _bcb_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _bcb_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0
        _btt_is_periodic = _oi_w < len(_btt_check) and _btt_check[_oi_w].strip() == "Periodic"
        _bbt_is_periodic = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"
        if _oi_w < len(_bc_ta) and _bc_ta[_oi_w].strip() == "True" and not _btt_is_periodic and not _bbt_is_periodic:
            _bct_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _bct_w.append(None)  # do NOT advance _wi — GUI sends no value for this slot
    else:
        _bcb_w.append(None); _bct_w.append(None)
    if (_oi_w == 0 and {config.forward_ic_from_file}) or (_oi_w < len(_ic_a) and _ic_a[_oi_w].strip() == "True"):
        _ic_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
    else:
        _ic_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0

_bc_left_deriv_active  = "{config.bc_left_deriv}".split(",")
_bc_bottom_deriv_active = "{config.bc_bottom_deriv}".split(",")
_multi_weights = []
for _oi_w in range(_n_out_w):
    _multi_weights.append(_pde_w[_oi_w])
for _oi_w in range(_n_out_w):
    if _bcl_w[_oi_w] is not None:
        _multi_weights.append(_bcl_w[_oi_w])
        # Add extra weight for derivative periodic BC if enabled
        _bbt_is_per = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"
        _bbd_is_active = _oi_w < len(_bc_bottom_deriv_active) and _bc_bottom_deriv_active[_oi_w].strip() == "True"
        if _bbt_is_per and _bbd_is_active:
            _multi_weights.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
    if _bcr_w[_oi_w] is not None: _multi_weights.append(_bcr_w[_oi_w])
    if _bcb_w[_oi_w] is not None:
        _multi_weights.append(_bcb_w[_oi_w])
        # Add extra weight for derivative periodic BC if enabled
        _bbt_is_per = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"
        _bbd_is_active = _oi_w < len(_bc_bottom_deriv_active) and _bc_bottom_deriv_active[_oi_w].strip() == "True"
        if _bbt_is_per and _bbd_is_active:
            _multi_weights.append(1.0)  # derivative periodic BC weight
    if _bct_w[_oi_w] is not None: _multi_weights.append(_bct_w[_oi_w])
    if _ic_w[_oi_w]  is not None: _multi_weights.append(_ic_w[_oi_w])

if _problem_type == "Inverse":
    _multi_weights = _multi_weights + [{config.loss_weight_obs}]

print(f"Loss weights: {{_multi_weights}} ({{len(_multi_weights)}} terms for {{len(_constraints)}} constraints)")
print(f"  PDE weights: {{_pde_w}}")
print(f"  BC left: {{[w for w in _bcl_w if w is not None]}}")
print(f"  BC right: {{[w for w in _bcr_w if w is not None]}}")
if _is_2d:
    print(f"  BC bottom: {{[w for w in _bcb_w if w is not None]}}")
    print(f"  BC top: {{[w for w in _bct_w if w is not None]}}")
print(f"  IC weights: {{[w for w in _ic_w if w is not None]}}")

# ── Data ─────────────────────────────────────────────────────
if _problem_type == "Inverse":
    obs_bc = dde.icbc.PointSetBC(_obs_xt, _obs_u, component=0)
    _constraints.append(obs_bc)
    data = dde.data.TimePDE(
        geomtime, pde, _constraints,
        num_domain={config.num_domain}, num_boundary={config.num_boundary},
        num_initial={config.num_initial}, num_test={config.num_test},
        train_distribution="{config.point_distribution}",
        anchors=_obs_xt
    )
else:
    data = dde.data.TimePDE(
        geomtime, pde, _constraints,
        num_domain={config.num_domain}, num_boundary={config.num_boundary},
        num_initial={config.num_initial}, num_test={config.num_test},
        train_distribution="{config.point_distribution}",
        anchors=None
    )

# ── Parametric loop ──────────────────────────────────────────
_summary = []

for _pval in _param_values:
    if _parametric:
        print(f"\\n===== Running: {{_param_name}} = {{_pval}} =====")

    _lr           = {config.learning_rate}
    _loss_weights = list({config.loss_weights})
    _layers       = list({config.layers})

    if _parametric and _pval is not None:
        if _param_name == "learning_rate":
            _lr = float(_pval)
        elif _param_name == "ic_weight":
            _loss_weights[3] = float(_pval)
        elif _param_name == "pde_weight":
            _loss_weights[0] = float(_pval)
        elif _param_name == "bc_left_weight":
            _loss_weights[1] = float(_pval)
        elif _param_name == "bc_right_weight":
            _loss_weights[2] = float(_pval)
        elif _param_name == "hidden_layers":
            n = int(_pval)
            _layers = [{config.layers[0]}] + [{config.layers[1]}] * n + [{config.layers[-1]}]
        elif _param_name == "neurons_per_layer":
            _layers = [{config.layers[0]}] + [int(_pval)] * {len(config.layers) - 2} + [{config.layers[-1]}]

    net = dde.nn.FNN(_layers, "{config.activation}", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile(
        "{config.optimizer}", lr=_lr, loss="{config.loss_type}",
        loss_weights=_multi_weights,
        external_trainable_variables=[{config.inverse_param_name}] if _problem_type == "Inverse" else None
    )
    if {config.batch_size} > 0:
        print(f"Mini-batch training enabled: batch_size={config.batch_size}")

    _iters = int(_pval) if (_parametric and _param_name == "phase1_iterations" and _pval is not None) else {config.iterations}

    # ── IC Pre-Training ───────────────────────────────────────
    if {config.ic_pretrain} and not {config.time_adaptive}:
        print(f"\\n=== IC Pre-Training: {config.ic_pretrain_iterations} iterations (IC loss only) ===")
        # Build IC-only geometry and data — no domain/BC points, only IC
        if _is_2d:
            _ic_geom_pre = dde.geometry.Rectangle([{config.x_min},{config.y_min}],[{config.x_max},{config.y_max}])
        else:
            _ic_geom_pre = dde.geometry.Interval({config.x_min}, {config.x_max})
        _ic_td_pre  = dde.geometry.TimeDomain({config.t_min}, {config.t_max})
        _ic_gt_pre  = dde.geometry.GeometryXTime(_ic_geom_pre, _ic_td_pre)
        _ic_ics_pre = []
        if {config.forward_ic_from_file}:
            # Load IC from file for pre-training
            _ic_pre_data = np.loadtxt(r"{config.forward_ic_file}")
            _ic_pre_mask = np.abs(_ic_pre_data[:, 2]) < 1e-10
            _ic_pre_xy   = _ic_pre_data[_ic_pre_mask, :2]
            _ic_pre_xyt  = np.hstack([_ic_pre_xy, np.zeros((_ic_pre_mask.sum(), 1))])
            _ic_pre_vals = _ic_pre_data[_ic_pre_mask, 3:4]
            _ic_ics_pre.append(dde.icbc.PointSetBC(_ic_pre_xyt, _ic_pre_vals, component=0))
        else:
            _ic_exprs_pre = "{config_ic_expressions}".split("|")
            for _oi_pre in range({config.num_outputs}):
                _ic_expr_pre = _ic_exprs_pre[_oi_pre].strip() if _oi_pre < len(_ic_exprs_pre) else "np.zeros_like(x[:,0])"
                def _mk_ic_pre(expr, comp):
                    def _ic_fn(x):
                        return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                    return dde.icbc.IC(_ic_gt_pre, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                _ic_ics_pre.append(_mk_ic_pre(_ic_expr_pre, _oi_pre))
        
        # IC pre-training — dummy PDE, no domain/BC, full batch, anchors only
        def _pde_dummy_pre(x, y):
            return [y[:, _oi_d:_oi_d+1] * 0 for _oi_d in range({config.num_outputs})]
        _bc_constraints_pre = [_c for _c in _constraints if not isinstance(_c, dde.icbc.PointSetBC)]
        _ic_pre_constraints = _bc_constraints_pre + _ic_ics_pre
        _data_pre = dde.data.TimePDE(
            _ic_gt_pre, _pde_dummy_pre, _ic_pre_constraints,
            num_domain=0, num_boundary=0,
            num_initial=0, num_test=10000,
            train_distribution="{config.point_distribution}",
            anchors=None
        )
        _model_pre = dde.Model(_data_pre, net)
        _n_bcs_pre = len(_ic_pre_constraints) - len(_ic_ics_pre)
        _ic_only_weights = [0.0] * {config.num_outputs} + [0.0] * _n_bcs_pre + [1000.0] * len(_ic_ics_pre)
        print(f"  IC-only weights: {{_ic_only_weights}} — dummy PDE, full batch")
        _model_pre.compile("{config.ic_pretrain_optimizer}", lr={config.learning_rate},
                           loss="MSE", loss_weights=_ic_only_weights)
        _ic_pre_save_dir = _os.path.join(r"{config.save_dir}", "ic_pretrain")
        _os.makedirs(_ic_pre_save_dir, exist_ok=True)
        if {config.ic_pretrain_restore} and r"{config.ic_pretrain_restore_path}" and _os.path.exists(r"{config.ic_pretrain_restore_path}"):
            print(f"  Restoring IC pre-train model from: {config.ic_pretrain_restore_path}")
            _ic_ckpt = torch.load(r"{config.ic_pretrain_restore_path}", map_location="cpu")
            _ic_state = _ic_ckpt.get("model_state_dict", _ic_ckpt)
            _model_pre.net.load_state_dict(_ic_state)
            print("  IC pre-train model restored — skipping training.")
        else:
            _ic_lh, _ = _model_pre.train(iterations={config.ic_pretrain_iterations},
                                          display_every=10000, batch_size=None,
                                          model_save_path=_os.path.join(_ic_pre_save_dir, "ic_pretrain_model"))
            print(f"  IC pre-training done. Final IC loss: {{sum(_ic_lh.loss_train[-1]):.4e}}")
            print(f"  IC pre-train model saved to: {{_ic_pre_save_dir}}")
        print("=== Starting Main Training ===\\n")

    if not {config.time_adaptive}:
        _sched_active = {config.optimizer_scheduler} and _os.path.exists('/tmp') and {len(config.scheduler_phases) > 0}
        if _problem_type == "Inverse":
            with open("/tmp/param_history.txt", "w") as _f:
                pass
            if not _sched_active:
                _var_cb = dde.callbacks.VariableValue(
                    [{config.inverse_param_name}], period=1000, filename="/tmp/param_history.txt",
                    precision=6
                )
                loss_history, train_state = model.train(iterations=_iters, display_every=1000, callbacks=[_var_cb, _print_cb, _save_cb])
            else:
                # Scheduler phases below define all training — skip this
                # standalone _iters-iteration pass so training only runs
                # for the iteration counts configured in the scheduler.
                print(f"  (Scheduler enabled — skipping standalone {{_iters}}-iteration pass; "
                      f"training runs only for the phases below)")
        else:
            if {config.batch_size} > 0:
                data.batch_size = {config.batch_size}
                print(f"Batch size set to: {config.batch_size}")
            if _use_save:
                pass  # model saved after scheduler phases
                _adam_cfg_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}.json")
                with open(_adam_cfg_path, "w") as _acf:
                    _json.dump(_model_config, _acf, indent=2)
                print(f"Adam config saved to: {{_adam_cfg_path}}")

        # Optimizer Scheduler or Phase 2
        if _sched_active:
            import json as _json_sched
            _sched_phases = _json_sched.loads({repr(config.scheduler_phases)})
            _sched_cum_iters = 0
            for _sp_i, _sp in enumerate(_sched_phases):
                print(f"  === Scheduler Phase {{_sp_i+1}}: {{_sp['optimizer']}} {{_sp['iterations']}} iters ===")
                _sp_weights = [float(w) for w in _sp['weights'].split(',') if w.strip()]
                _sp_ext_vars = [{config.inverse_param_name}] if _problem_type == "Inverse" else None
                if _sp['optimizer'] == 'lbfgs':
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter=_sp['iterations'],
                        maxfun=int(_sp['iterations']*1.25), maxls={config.lbfgs_maxls})
                    _lbfgs_float = "{config.lbfgs_float_type}"
                    model.compile("L-BFGS", loss=_sp.get('loss', '{config.loss_type}'),
                                  loss_weights=_sp_weights, external_trainable_variables=_sp_ext_vars)
                    if _problem_type == "Inverse":
                        _print_cb.set_offset(_sched_cum_iters)
                        _save_cb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            [{config.inverse_param_name}], period=200,
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            display_every=200, callbacks=[_sp_var_cb, _print_cb, _save_cb])
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _spparts = _spl.replace("[","").replace("]","").split()
                                    if len(_spparts) >= 2:
                                        _fa.write(f"{{_sched_cum_iters + int(_spparts[0])}} [{{_spparts[1]}}]\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(display_every=200)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_lbfgs-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} L-BFGS model saved: {{_nta_save_path}}.pt")
                else:
                    model.compile(_sp['optimizer'], lr=_sp['lr'],
                                  loss=_sp.get('loss', '{config.loss_type}'), loss_weights=_sp_weights,
                                  external_trainable_variables=_sp_ext_vars)
                    if {config.batch_size} > 0:
                        data.batch_size = {config.batch_size}
                    if _problem_type == "Inverse":
                        _print_cb.set_offset(_sched_cum_iters)
                        _save_cb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            [{config.inverse_param_name}], period=1000,
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            iterations=_sp['iterations'], display_every=1000,
                            callbacks=[_sp_var_cb, _print_cb, _save_cb])
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _spparts = _spl.replace("[","").replace("]","").split()
                                    if len(_spparts) >= 2:
                                        _fa.write(f"{{_sched_cum_iters + int(_spparts[0])}} [{{_spparts[1]}}]\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(iterations=_sp['iterations'], display_every=1000)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_adam-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} Adam model saved: {{_nta_save_path}}.pt")
        elif "{config.optimizer2}" != "none":
            _phase2_weights = _multi_weights
            if "{config.optimizer2}" == "lbfgs":
                dde.optimizers.set_LBFGS_options(
                    maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                    gtol={config.lbfgs_gtol}, maxiter={config.iterations2},
                    maxfun={config.lbfgs_maxfun}, maxls={config.lbfgs_maxls}
                )
                if "{config.lbfgs_float_type}" == "float64":
                    dde.config.set_default_float("float64")
                    model.net.double()
                    print("  [L-BFGS] Switched to float64")
                if _problem_type == "Inverse":
                    _last_iter = 0
                    try:
                        with open("/tmp/param_history.txt", "r") as _f:
                            _lines = [l.strip() for l in _f if l.strip()]
                            if _lines:
                                _last_iter = int(_lines[-1].replace("[","").replace("]","").split()[0])
                    except Exception:
                        pass
                    _var_cb2 = dde.callbacks.VariableValue(
                        [{config.inverse_param_name}], period=200, filename="/tmp/param_history_phase2.txt",
                        precision=6
                    )
                    model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_phase2_weights,
                                  external_trainable_variables=[{config.inverse_param_name}])
                    _print_cb.set_offset(_iters)
                    loss_history, train_state = model.train(display_every=200, callbacks=[_var_cb2, _print_cb])
                    # Append L-BFGS history to convergence file
                    if _param_save_period > 0 and _param_save_path:
                        try:
                            with open("/tmp/param_history_phase2.txt", "r") as _lf:
                                for _ll in _lf:
                                    _ll = _ll.strip()
                                    if not _ll: continue
                                    _lparts = _ll.replace("[","").replace("]","").split()
                                    if len(_lparts) >= 2:
                                        _lbfgs_iter = int(_lparts[0])
                                        _lbfgs_val = float(_lparts[1])
                                        with open(_param_save_path, "a") as _pf:
                                            _pf.write(f"{{_lbfgs_iter}},{{_lbfgs_val:.8f}}\\n")
                        except Exception as _le:
                            print(f"Could not save L-BFGS history: {{_le}}")
                    try:
                        with open("/tmp/param_history_phase2.txt", "r") as _f2:
                            _p2_lines = [l.strip() for l in _f2 if l.strip()]
                        with open("/tmp/param_history.txt", "a") as _fa:
                            for _p2l in _p2_lines:
                                _p2parts = _p2l.replace("[","").replace("]","").split()
                                if len(_p2parts) >= 2:
                                    _new_iter = _last_iter + int(_p2parts[0])
                                    _fa.write(f"{{_new_iter}} [{{_p2parts[1]}}]\\n")
                    except Exception as _ae:
                        print(f"Could not append phase 2 history: {{_ae}}")
                else:
                    model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_phase2_weights)
                    loss_history, train_state = model.train(display_every=200)
                    if "{config.lbfgs_float_type}" == "float64":
                        dde.config.set_default_float("float32")
                        model.net.float()
                        print("  [L-BFGS] Restored to float32")
                    if _use_save:
                        _lbfgs_model_path = _os.path.join(_sol_dir, "model_lbfgs")
                        model.save(_lbfgs_model_path)
                        _lbfgs_iter = loss_history.steps[-1] if loss_history.steps else _iters
                        _lbfgs_cfg_path = _os.path.join(_sol_dir, f"model_lbfgs-{{_lbfgs_iter}}.json")
                        with open(_lbfgs_cfg_path, "w") as _lcf:
                            _json.dump(_model_config, _lcf, indent=2)
                        print(f"L-BFGS config saved to: {{_lbfgs_cfg_path}}")
            else:
                if _problem_type == "Inverse":
                    model.compile("{config.optimizer2}", lr=_lr, loss="{config.loss_type}",
                                  loss_weights=_phase2_weights,
                                  external_trainable_variables=[{config.inverse_param_name}])
                else:
                    model.compile("{config.optimizer2}", lr=_lr, loss="{config.loss_type}",
                                  loss_weights=_phase2_weights)
                loss_history, train_state = model.train(iterations={config.iterations2}, display_every=1000)

    # ── RAR Loop ─────────────────────────────────────────────
    if "{config.adapt_method}" == "RAR" and not {config.time_adaptive}:
        print("\\n=== Starting RAR Adaptive Refinement ===")
        for rar_cycle in range({config.rar_cycles}):
            print(f"\\n--- RAR Cycle {{rar_cycle+1}}/{config.rar_cycles} ---")
            if _is_2d:
                x_cand = np.random.uniform({config.x_min}, {config.x_max}, {config.rar_candidates})
                y_cand = np.random.uniform({config.y_min}, {config.y_max}, {config.rar_candidates})
                t_cand = np.random.uniform({config.t_min}, {config.t_max}, {config.rar_candidates})
                xt_cand = np.column_stack([x_cand, y_cand, t_cand])
            else:
                x_cand  = np.random.uniform({config.x_min}, {config.x_max}, {config.rar_candidates})
                t_cand  = np.random.uniform({config.t_min}, {config.t_max}, {config.rar_candidates})
                xt_cand = np.column_stack([x_cand, t_cand])
            _rar_res = model.predict(xt_cand, operator=pde)
            if isinstance(_rar_res, list):
                residuals = np.sum([np.abs(r).flatten() for r in _rar_res], axis=0)
            else:
                residuals = np.abs(_rar_res).flatten()
            top_idx    = np.argsort(residuals)[-{config.rar_add_points}:]
            new_points = xt_cand[top_idx]
            print(f"Max residual: {{residuals.max():.4e}}, Mean: {{residuals.mean():.4e}}")
            data.add_anchors(new_points)
            model.compile("{config.optimizer}", lr=_lr, loss="{config.loss_type}", loss_weights=_multi_weights)
            loss_history, train_state = model.train(iterations={config.rar_adam_iters}, display_every=500)
            if {config.rar_lbfgs_iters} > 0:
                dde.optimizers.set_LBFGS_options(
                    maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                    gtol={config.lbfgs_gtol}, maxiter={config.rar_lbfgs_iters},
                    maxfun={config.lbfgs_maxfun}, maxls={config.lbfgs_maxls}
                )
                if "{config.lbfgs_float_type}" == "float64":
                    dde.config.set_default_float("float64")
                    model.net.double()
                    print("  [L-BFGS] Switched to float64")
                model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_multi_weights)
                loss_history, train_state = model.train(display_every=200)
                if "{config.lbfgs_float_type}" == "float64":
                    dde.config.set_default_float("float32")
                    model.net.float()
                    print("  [L-BFGS] Restored to float32")
        print("\\n=== RAR Complete ===")

    # ── Save paths ────────────────────────────────────────────
    if _parametric and _pval is not None:
        _run_dir = _os.path.join(_save_dir if _use_save else "/tmp", f"{{_param_name}}_{{_pval}}")
        _os.makedirs(_run_dir, exist_ok=True)
        _run_loss_path     = _os.path.join(_run_dir, "loss_plot.png")
        _run_solution_path = _os.path.join(_run_dir, "solution_plot.png")
        _run_log_path      = _os.path.join(_run_dir, "training_log.txt")
    else:
        _run_loss_path     = _loss_path
        _run_solution_path = _solution_path
        _run_log_path      = _log_path

    # ── Plot loss ────────────────────────────────────────────
    if not {config.time_adaptive}:
        dde.saveplot(loss_history, train_state, issave=False, isplot=False)
        train_loss = loss_history.loss_train
        test_loss  = loss_history.loss_test
        steps      = loss_history.steps
        total_train = [sum(l) for l in train_loss]
        total_test  = [sum(l) for l in test_loss]

        plt.figure(figsize=(6, 4))
        plt.semilogy(steps, total_train, label="Train loss", color="#4dabf7")
        plt.semilogy(steps, total_test,  label="Test loss",  color="#ff8787", linestyle="--")
        plt.xlabel("Iteration"); plt.ylabel("Loss")
        title_str = f"Loss — {{_param_name}}={{_pval}}" if _parametric else "Training & Test Loss"
        plt.title(title_str)
        plt.legend(); plt.tight_layout()
        plt.savefig(_run_loss_path, dpi=100); plt.close()

        if _run_log_path:
            with open(_run_log_path, "w") as f:
                for i, l in zip(steps, train_loss):
                    f.write(f"Step {{i}}: {{list(l)}}\\n")

        _final_loss = sum(loss_history.loss_train[-1])

        # Parameter history (inverse only)
        if _problem_type == "Inverse":
            try:
                _ph_iters = []; _ph_vals = []
                with open("/tmp/param_history.txt", "r") as _phf:
                    for _line in _phf:
                        _line = _line.strip()
                        if not _line: continue
                        _parts = _line.replace("[","").replace("]","").split()
                        if len(_parts) >= 2:
                            try:
                                _ph_iters.append(float(_parts[0]))
                                _ph_vals.append(float(_parts[1]))
                            except ValueError:
                                continue
                _ph_iters = np.array(_ph_iters); _ph_vals = np.array(_ph_vals)
                print(f"\\n=== {config.inverse_param_name} final value: {{_ph_vals[-1]:.6f}} ===")
                plt.figure(figsize=(6, 4))
                if {config.inv_param_log_scale} and np.all(np.array(_ph_vals) > 0):
                    plt.semilogy(_ph_iters, _ph_vals, color="#69db7c", linewidth=1.5)
                    plt.axhline(y=_ph_vals[-1], color="#ff8787", linestyle="--", alpha=0.5, label=f"Final = {{_ph_vals[-1]:.6f}}")
                    plt.ylabel("log({config.inverse_param_name})")
                else:
                    plt.plot(_ph_iters, _ph_vals, color="#69db7c", linewidth=1.5)
                    plt.axhline(y=_ph_vals[-1], color="#ff8787", linestyle="--", alpha=0.5, label=f"Final = {{_ph_vals[-1]:.6f}}")
                    plt.ylabel("{config.inverse_param_name}")
                plt.xlabel("Iteration")
                plt.title("Inferred Parameter: {config.inverse_param_name}")
                plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig("/tmp/param_plot.png", dpi=100)
                if _use_save:
                    import shutil as _sp
                    _sp.copy("/tmp/param_plot.png", _os.path.join(
                        _run_dir if (_parametric and _pval is not None) else _save_dir, "param_plot.png"))
                plt.close()
                print(f"Final inferred {config.inverse_param_name}: {{_ph_vals[-1]:.6f}}")
            except Exception as _e:
                print(f"Could not plot parameter history: {{_e}}")

        _summary.append((_pval, _final_loss))
        print(f"Final loss: {{_final_loss:.4e}}")

    # ── Plot solution ─────────────────────────────────────────
    if not {config.time_adaptive}:
        _plot_idx = {config.plot_output_idx}

        if _is_2d:
            # 2D: x-y heatmaps at user-selected number of time snapshots
            _n_snaps = {config.plot_n_2d_snapshots}
            _t_snaps = np.linspace({config.t_min}, {config.t_max}, _n_snaps)
            _res_2d = {config.plot_resolution}
            _xp = np.linspace({config.x_min}, {config.x_max}, _res_2d)
            _yp = np.linspace({config.y_min}, {config.y_max}, _res_2d)
            _Xg, _Yg = np.meshgrid(_xp, _yp)
            _vmin_2d = None if {config.plot_auto_range} else {config.plot_vmin}
            _vmax_2d = None if {config.plot_auto_range} else {config.plot_vmax}

            fig, axes = plt.subplots(1, _n_snaps, figsize=(5*_n_snaps, 5))
            if _n_snaps == 1: axes = [axes]
            for _ai, _tv in enumerate(_t_snaps):
                _XYT = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _tv)])
                _pred = model.predict(_XYT)[:, _plot_idx].reshape(_res_2d, _res_2d)
                im = axes[_ai].contourf(_Xg, _Yg, _pred, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_2d, vmax=_vmax_2d)
                axes[_ai].set_title(f"t = {{_tv:.3f}}")
                axes[_ai].set_xlabel("x"); axes[_ai].set_ylabel("y")
                if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
            out_name = "{config.output_names}".split(",")[_plot_idx].strip()
            fig.suptitle(f"PINN Solution — {{out_name}}(x,y,t)", fontsize=12)
            plt.tight_layout()
            plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        else:
            # 1D plot
            _plot_type = "{config.plot_type}"

            if _plot_type == "Surface":
                _res = {config.plot_resolution}
                _x_s = np.linspace({config.x_min}, {config.x_max}, _res)
                _t_s = np.linspace({config.t_min}, {config.t_max}, _res)
                _Xs, _Ts = np.meshgrid(_x_s, _t_s)
                _XTs = np.vstack([_Xs.ravel(), _Ts.ravel()]).T
                _u_s = model.predict(_XTs)[:, _plot_idx].reshape(_res, _res)
                _vmin_s = None if {config.plot_auto_range} else {config.plot_vmin}
                _vmax_s = None if {config.plot_auto_range} else {config.plot_vmax}
                print(f"Plot settings: cmap={config.plot_colormap}, levels={config.plot_levels}, dpi={config.plot_dpi}, res={config.plot_resolution}")
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.contourf(_Xs, _Ts, _u_s, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_s, vmax=_vmax_s)
                if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
                ax.set_xlabel("x"); ax.set_ylabel("t")
                ax.set_title(f"PINN Solution — {{_param_name}}={{_pval}}" if _parametric else "PINN Solution")
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

            elif _plot_type.startswith("Line"):
                n_steps_plot = {config.num_timesteps}
                _x_l = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                t_steps_plot = np.linspace({config.t_min}, {config.t_max}, n_steps_plot)
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_plot))
                for i, t_val in enumerate(t_steps_plot):
                    xt = np.column_stack([_x_l, np.full_like(_x_l, t_val)])
                    u_line = model.predict(xt)[:, _plot_idx].flatten()
                    ax.plot(_x_l, u_line, color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{t_val:.3f}}")
                ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
                ax.set_title(f"PINN Solution — {{_param_name}}={{_pval}}" if _parametric else "PINN Solution")
                ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

        import shutil as _shutil
        if _run_loss_path != "/tmp/loss_plot.png":
            _shutil.copy(_run_loss_path, "/tmp/loss_plot.png")
        if _run_solution_path != "/tmp/solution_plot.png":
            _shutil.copy(_run_solution_path, "/tmp/solution_plot.png")

        # ── Inline Error Analysis ─────────────────────────────
        if {config.ea_files}:
            from scipy.interpolate import interp1d as _interp1d
            _ea_files = [(tv, fp) for tv, fp in {config.ea_files}
                         if {config.t_min} - 1e-10 <= tv <= {config.t_max} + 1e-10]
            _ea_dir = _os.path.join(_save_dir if _use_save else "/tmp", "error_analysis")
            _os.makedirs(_ea_dir, exist_ok=True)
            print("\\n=== Running Error Analysis ===")
            print(f"  Filtering to t=[{config.t_min}, {config.t_max}]: {{len(_ea_files)}} files")

            # Load all ground truth files
            _ea_times = []; _ea_x_refs = []; _ea_y_refs = []; _ea_u_refs = []
            for _ea_tv, _ea_fp in _ea_files:
                _ea_d = np.loadtxt(_ea_fp)
                if _ea_d.ndim == 1: _ea_d = _ea_d.reshape(1, -1)
                if _is_2d:
                    # 2D format: x, y, t, u — sort by x then y
                    _ea_idx = np.lexsort((_ea_d[:, 1], _ea_d[:, 0]))
                    _ea_x_refs.append(_ea_d[_ea_idx, 0])
                    _ea_y_refs.append(_ea_d[_ea_idx, 1])
                    _ea_u_refs.append(_ea_d[_ea_idx, 3])
                    _detected_t = float(_ea_d[0, 2])
                    _ea_times.append(_detected_t)
                    print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                else:
                    # 1D format: x, t, u — sort by x
                    _ea_idx = np.argsort(_ea_d[:, 0])
                    _ea_x_refs.append(_ea_d[_ea_idx, 0])
                    _ea_y_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                    _ea_u_refs.append(_ea_d[_ea_idx, 2])
                    _ea_times.append(float(_ea_tv))
                    print(f"  Loaded ground truth t={{_ea_tv:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
            # Sort all loaded data by time value — outside the loop
            _ea_sort_idx = np.argsort(_ea_times)
            _ea_times  = [_ea_times[_i]  for _i in _ea_sort_idx]
            _ea_x_refs = [_ea_x_refs[_i] for _i in _ea_sort_idx]
            _ea_y_refs = [_ea_y_refs[_i] for _i in _ea_sort_idx]
            _ea_u_refs = [_ea_u_refs[_i] for _i in _ea_sort_idx]
            _ea_n_t = len(_ea_times)
            _ea_u_pinns = [None] * _ea_n_t
            _ea_metrics = [None] * _ea_n_t

            if not {config.time_adaptive}:
                # ── Non-adaptive: use single model ───────────────
                for _ei, _ea_tv in enumerate(_ea_times):
                    _ea_xf = _ea_x_refs[_ei]
                    if _is_2d:
                        _ea_yf = _ea_y_refs[_ei]
                        _ea_xt = np.column_stack([_ea_xf, _ea_yf, np.full_like(_ea_xf, _ea_tv)])
                    else:
                        _ea_xt = np.column_stack([_ea_xf, np.full_like(_ea_xf, _ea_tv)])
                    _ea_u_pinns[_ei] = model.predict(_ea_xt)[:, {config.plot_output_idx}].flatten()
                    print(f"  PINN predicted at t={{_ea_tv:.4f}}: {{len(_ea_xf)}} points")
            else:
                # ── Time adaptive: match each GT file to correct step model ──
                # Reconstruct step intervals from saved models
                import glob as _ea_glob, json as _ea_json
                _ta_step_dir = _os.path.join(_save_dir, "time_adaptive_steps")
                _ta_step_dirs = sorted([_sd for _sd in _ea_glob.glob(_os.path.join(_ta_step_dir, "step_*")) if _os.path.isdir(_sd)])
                # Parse t0, t1 from each step directory name
                # Format: step_NNN_tX.XXXX_to_tY.YYYY
                _ta_intervals = []
                for _sd in _ta_step_dirs:
                    _sd_name = _os.path.basename(_sd)
                    try:
                        _parts = _sd_name.split("_")
                        _t0_str = _parts[2].replace("t","")
                        _t1_str = _parts[4].replace("t","")
                        _ta_intervals.append((float(_t0_str), float(_t1_str), _sd))
                    except Exception as _pe:
                        print(f"  Could not parse step dir: {{_sd_name}}: {{_pe}}")

                print(f"  Found {{len(_ta_intervals)}} time-adaptive step models")

                # Load each step model once and predict for all GT files in its interval
                for _si, (_t0_i, _t1_i, _sd_i) in enumerate(_ta_intervals):
                    # Find GT files whose time falls in [t0, t1]
                    # For the last step include t1, for others use t0 <= t < t1
                    # except t0 of first step includes t=t_min
                    _is_last = (_si == len(_ta_intervals) - 1)
                    _matching = []
                    for _ei, _ea_tv in enumerate(_ea_times):
                        if _is_last:
                            _in_range = (_t0_i <= _ea_tv <= _t1_i + 1e-10)
                        else:
                            _in_range = (_t0_i <= _ea_tv < _t1_i - 1e-10) or \
                                        (abs(_ea_tv - _t1_i) < 1e-10)  # boundary goes to this step
                        if _in_range and _ea_u_pinns[_ei] is None:
                            _matching.append(_ei)

                    if not _matching:
                        continue

                    print(f"  Step {{_si+1}} [{{_t0_i:.4f}}→{{_t1_i:.4f}}]: predicting for t = {{[_ea_times[_ei] for _ei in _matching]}}")

                    # Load step model
                    _step_cfg_path = _os.path.join(_sd_i, "step_config.json")
                    try:
                        with open(_step_cfg_path) as _scf:
                            _step_cfg = _ea_json.load(_scf)
                    except Exception:
                        _step_cfg = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}

                    _step_layers = _step_cfg.get("layers", {config.layers})
                    _step_act    = _step_cfg.get("activation", "{config.activation}")
                    _step_loss   = _step_cfg.get("loss_type", "{config.loss_type}")

                    # Build minimal geometry for this step
                    _step_geom = dde.geometry.Interval({config.x_min}, {config.x_max})
                    _step_td   = dde.geometry.TimeDomain(_t0_i, _t1_i)
                    _step_gt   = dde.geometry.GeometryXTime(_step_geom, _step_td)
                    def _step_pde(x, y): return y[:, 0:1] * 0
                    _step_data  = dde.data.TimePDE(_step_gt, _step_pde, [], num_domain=100, num_test=100)
                    _step_net   = dde.nn.FNN(_step_layers, _step_act, "Glorot uniform")
                    _step_model = dde.Model(_step_data, _step_net)

                    # Find best saved model for this step (lbfgs preferred)
                    _step_pt = ""
                    for _pat in ["model_lbfgs-*.pt", "model_lbfgs.pt", "model_adam-*.pt", "model_adam.pt"]:
                        _step_pts = sorted(_ea_glob.glob(_os.path.join(_sd_i, _pat)))
                        if _step_pts:
                            _step_pt = max(_step_pts, key=_os.path.getmtime)
                            break

                    if not _step_pt:
                        print(f"  ⚠️ No model found for step {{_si+1}}, skipping")
                        continue

                    # Compile and restore
                    if "lbfgs" in _os.path.basename(_step_pt):
                        dde.optimizers.set_LBFGS_options(maxiter=1)
                        _step_model.compile("L-BFGS", loss=_step_loss)
                    else:
                        _step_model.compile("adam", lr=0.001, loss=_step_loss)

                    _step_model.restore(_step_pt, verbose=0)
                    print(f"    Restored: {{_os.path.basename(_step_pt)}}")

                    # Predict for each matching GT file
                    for _ei in _matching:
                        _ea_xf = _ea_x_refs[_ei]
                        _ea_tv = _ea_times[_ei]
                        _ea_xt = np.column_stack([_ea_xf, np.full_like(_ea_xf, _ea_tv)])
                        _ea_u_pinns[_ei] = _step_model.predict(_ea_xt)[:, 0].flatten()
                        print(f"    Predicted at t={{_ea_tv:.4f}}: {{len(_ea_xf)}} points")

                # Fill any unmatched with zeros (safety)
                for _ei in range(_ea_n_t):
                    if _ea_u_pinns[_ei] is None:
                        print(f"  ⚠️ No prediction for t={{_ea_times[_ei]:.4f}} — skipping")
                        _ea_u_pinns[_ei] = np.zeros_like(_ea_u_refs[_ei])

            # ── Compute metrics ───────────────────────────────────
            for _ei, _ea_tv in enumerate(_ea_times):
                _up = _ea_u_pinns[_ei]; _uf = _ea_u_refs[_ei]
                _ea_abs = np.abs(_up - _uf)
                _ea_l2  = np.linalg.norm(_up - _uf) / (np.linalg.norm(_uf) + 1e-10)
                _ea_mse = np.mean((_up - _uf)**2)
                _ea_mx  = np.max(_ea_abs)
                _ea_ma  = np.mean(_ea_abs)
                _ea_metrics[_ei] = (_ea_tv, _ea_l2, _ea_mse, _ea_mx, _ea_ma)
                print(f"  t={{_ea_tv:.4f}} — L2={{_ea_l2:.4e}}, MSE={{_ea_mse:.4e}}, Max={{_ea_mx:.4e}}, MeanAbs={{_ea_ma:.4e}}")

            with open(_os.path.join(_ea_dir, "error_metrics.txt"), "w") as _emf:
                _emf.write("t,L2_relative,MSE,Max_error,Mean_abs_error\\n")
                for _ea_tv, _l2, _mse, _mx, _ma in _ea_metrics:
                    _emf.write(f"{{_ea_tv:.6f}},{{_l2:.6e}},{{_mse:.6e}},{{_mx:.6e}},{{_ma:.6e}}\\n")
            print(f"  Metrics saved: {{_os.path.join(_ea_dir, 'error_metrics.txt')}}")

            # ── Line comparison ───────────────────────────────────
            if {config.ea_do_line}:
                _ea_ncols = min(4, _ea_n_t)
                _ea_nrows = (_ea_n_t + _ea_ncols - 1) // _ea_ncols
                fig, axes = plt.subplots(_ea_nrows, _ea_ncols, figsize=(4*_ea_ncols, 3.5*_ea_nrows), squeeze=False)
                fig.suptitle("PINN vs Ground Truth — Line Comparison", fontsize=13, fontweight='bold')
                _ea_ax_flat = axes.flatten()
                for _ei in range(_ea_n_t):
                    ax = _ea_ax_flat[_ei]
                    _xv = _ea_x_refs[_ei]
                    if _is_2d:
                        # For 2D line plot: extract mid-y slice
                        _yv = _ea_y_refs[_ei]
                        _y_mid = ({config.y_min} + {config.y_max}) / 2.0
                        _y_tol = ({config.y_max} - {config.y_min}) / 20.0
                        _mid_mask = np.abs(_yv - _y_mid) < _y_tol
                        if _mid_mask.sum() < 5:
                            _mid_mask = np.abs(_yv - _y_mid) < ({config.y_max} - {config.y_min}) / 5.0
                        _ea_sort = np.argsort(_xv[_mid_mask])
                        _xv_s   = _xv[_mid_mask][_ea_sort]
                        _gt_s   = _ea_u_refs[_ei][_mid_mask][_ea_sort]
                        _pinn_s = _ea_u_pinns[_ei][_mid_mask][_ea_sort]
                    else:
                        _ea_sort = np.argsort(_xv)
                        _xv_s   = _xv[_ea_sort]
                        _gt_s   = _ea_u_refs[_ei][_ea_sort]
                        _pinn_s = _ea_u_pinns[_ei][_ea_sort]
                    _ea_tv, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                    ax.plot(_xv_s, _gt_s,   color='#4dabf7', linewidth=2.0, linestyle='-',  label='Ground Truth')
                    ax.plot(_xv_s, _pinn_s, color='#ff6b6b', linewidth=2.0, linestyle='--', label='PINN')
                    ax.set_title(f"t = {{_ea_tv:.3f}}  |  L2 = {{_l2:.2e}}", fontsize=10)
                    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)"); ax.grid(True, alpha=0.3)
                for _ej in range(_ea_n_t, len(_ea_ax_flat)):
                    _ea_ax_flat[_ej].set_visible(False)
                handles, labels = _ea_ax_flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
                           framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
                plt.tight_layout(rect=[0, 0.06, 1, 1])
                _ea_lp = _os.path.join(_ea_dir, "line_comparison.png")
                plt.savefig(_ea_lp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                print(f"  Line comparison saved: {{_ea_lp}}")

            # ── Surface comparison ────────────────────────────────
            if {config.ea_do_surface}:
                if _is_2d:
                    # 2D: 3 columns (PINN | FEM | Error), one row per time snapshot
                    fig, axes = plt.subplots(_ea_n_t, 3,
                                             figsize=(15, 4*_ea_n_t), squeeze=False)
                    fig.suptitle("PINN vs Ground Truth — 2D Heatmaps", fontsize=13, fontweight='bold')
                    _res_ea = {config.plot_resolution}
                    _xg_ea = np.linspace({config.x_min}, {config.x_max}, _res_ea)
                    _yg_ea = np.linspace({config.y_min}, {config.y_max}, _res_ea)
                    _Xg_ea, _Yg_ea = np.meshgrid(_xg_ea, _yg_ea)
                    from scipy.interpolate import griddata as _gd
                    for _ei, _ea_tv in enumerate(_ea_times):
                        _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                        # PINN prediction on grid
                        _xyt_grid = np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel(), np.full(_Xg_ea.size, _ea_tv)])
                        _u_pinn_grid = model.predict(_xyt_grid)[:, {config.plot_output_idx}].reshape(_res_ea, _res_ea)
                        # FEM interpolated onto same grid
                        _u_fem_grid = _gd(
                            np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei]]),
                            _ea_u_refs[_ei],
                            (_Xg_ea, _Yg_ea), method='linear', fill_value=0.0)
                        _u_err_grid = np.abs(_u_pinn_grid - _u_fem_grid)
                        _vmin_ea = min(_u_pinn_grid.min(), _u_fem_grid.min())
                        _vmax_ea = max(_u_pinn_grid.max(), _u_fem_grid.max())
                        # Column 0: PINN
                        im0 = axes[_ei][0].contourf(_Xg_ea, _Yg_ea, _u_pinn_grid, levels=40,
                                                     cmap='{config.plot_colormap}', vmin=_vmin_ea, vmax=_vmax_ea)
                        axes[_ei][0].set_title(f"PINN  t={{_ea_tv:.3f}}  L2={{_l2:.2e}}", fontsize=10)
                        axes[_ei][0].set_xlabel("x"); axes[_ei][0].set_ylabel("y")
                        fig.colorbar(im0, ax=axes[_ei][0])
                        # Column 1: FEM
                        im1 = axes[_ei][1].contourf(_Xg_ea, _Yg_ea, _u_fem_grid, levels=40,
                                                     cmap='{config.plot_colormap}', vmin=_vmin_ea, vmax=_vmax_ea)
                        axes[_ei][1].set_title(f"Ground Truth  t={{_ea_tv:.3f}}", fontsize=10)
                        axes[_ei][1].set_xlabel("x"); axes[_ei][1].set_ylabel("y")
                        fig.colorbar(im1, ax=axes[_ei][1])
                        # Column 2: Absolute error
                        im2 = axes[_ei][2].contourf(_Xg_ea, _Yg_ea, _u_err_grid, levels={config.plot_levels}, cmap='YlOrRd')
                        axes[_ei][2].set_title(f"|Error|  t={{_ea_tv:.3f}}  Max={{_mx:.2e}}", fontsize=10)
                        axes[_ei][2].set_xlabel("x"); axes[_ei][2].set_ylabel("y")
                        fig.colorbar(im2, ax=axes[_ei][2])
                    plt.tight_layout()
                else:
                    # 1D: standard x vs t surface
                    _ea_x_common = np.linspace({config.x_min}, {config.x_max}, 300)
                    _ea_t_arr = np.array(_ea_times)
                    _ea_U_pinn = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                    _ea_U_fem  = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                    if not {config.time_adaptive}:
                        for _ei, _ea_tv in enumerate(_ea_times):
                            _ea_xt_c = np.column_stack([_ea_x_common, np.full_like(_ea_x_common, _ea_tv)])
                            _ea_U_pinn[_ei, :] = model.predict(_ea_xt_c)[:, {config.plot_output_idx}].flatten()
                            _ea_fi = _interp1d(_ea_x_refs[_ei], _ea_u_refs[_ei], kind='linear', fill_value='extrapolate')
                            _ea_U_fem[_ei, :] = _ea_fi(_ea_x_common)
                    else:
                        for _ei in range(_ea_n_t):
                            _ea_fi_pinn = _interp1d(_ea_x_refs[_ei], _ea_u_pinns[_ei], kind='linear', fill_value='extrapolate')
                            _ea_U_pinn[_ei, :] = _ea_fi_pinn(_ea_x_common)
                            _ea_fi_fem = _interp1d(_ea_x_refs[_ei], _ea_u_refs[_ei], kind='linear', fill_value='extrapolate')
                            _ea_U_fem[_ei, :] = _ea_fi_fem(_ea_x_common)
                    _ea_Xg, _ea_Tg = np.meshgrid(_ea_x_common, _ea_t_arr)
                    _ea_U_err = np.abs(_ea_U_pinn - _ea_U_fem)
                    _ea_vmin = min(_ea_U_pinn.min(), _ea_U_fem.min())
                    _ea_vmax = max(_ea_U_pinn.max(), _ea_U_fem.max())
                    fig, axes_s = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle("PINN vs Ground Truth — Surface Comparison", fontsize=13, fontweight='bold')
                    im0 = axes_s[0].contourf(_ea_Tg, _ea_Xg, _ea_U_pinn, levels={config.plot_levels}, cmap='{config.plot_colormap}', vmin=_ea_vmin, vmax=_ea_vmax)
                    axes_s[0].set_title("PINN  u(x,t)"); axes_s[0].set_xlabel("t"); axes_s[0].set_ylabel("x")
                    fig.colorbar(im0, ax=axes_s[0])
                    im1 = axes_s[1].contourf(_ea_Tg, _ea_Xg, _ea_U_fem, levels={config.plot_levels}, cmap='{config.plot_colormap}', vmin=_ea_vmin, vmax=_ea_vmax)
                    axes_s[1].set_title("Ground Truth  u(x,t)"); axes_s[1].set_xlabel("t"); axes_s[1].set_ylabel("x")
                    fig.colorbar(im1, ax=axes_s[1])
                    im2 = axes_s[2].contourf(_ea_Tg, _ea_Xg, _ea_U_err, levels={config.plot_levels}, cmap='YlOrRd')
                    axes_s[2].set_title("Error  |PINN - Ground Truth|"); axes_s[2].set_xlabel("t"); axes_s[2].set_ylabel("x")
                    fig.colorbar(im2, ax=axes_s[2])
                    plt.tight_layout()
                _ea_sp = _os.path.join(_ea_dir, "surface_comparison.png")
                plt.savefig(_ea_sp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                print(f"  Surface comparison saved: {{_ea_sp}}")
            print("=== Error Analysis Complete ===")

        # ── Export solution data ──────────────────────────────
        if _problem_type != "Inverse":
            _data_dir = _os.path.join(
                _run_dir if (_parametric and _pval is not None)
                else (_sol_dir if _use_save else "/tmp"),
                "solution_data"
            )
            _os.makedirs(_data_dir, exist_ok=True)
            _x_export = np.linspace({config.x_min}, {config.x_max}, {config.export_grid_size})
            _t_export = np.linspace({config.t_min}, {config.t_max}, {config.export_t_steps})

            if _is_2d:
                _y_export = np.linspace({config.y_min}, {config.y_max}, {config.export_grid_size})
                for _t_val in _t_export:
                    _Xe, _Ye = np.meshgrid(_x_export, _y_export)
                    _XYT_exp = np.column_stack([_Xe.ravel(), _Ye.ravel(), np.full(_Xe.size, _t_val)])
                    _u_exp = model.predict(_XYT_exp)
                    _header = "x,y,t," + ",".join(_out_names_list)
                    _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), np.full(_Xe.size, _t_val), _u_exp])
                    _fname = _os.path.join(_data_dir, f"solution_t{{_t_val:.4f}}.txt")
                    np.savetxt(_fname, _out, header=_header, delimiter=",", comments="")
            else:
                for _t_val in _t_export:
                    _xt_exp = np.column_stack([_x_export, np.full_like(_x_export, _t_val)])
                    _u_exp_all = model.predict(_xt_exp)
                    _header_cols = "x,t," + ",".join(_out_names_list)
                    _out = np.column_stack([_x_export, np.full_like(_x_export, _t_val), _u_exp_all])
                    _fname = _os.path.join(_data_dir, f"solution_t{{_t_val:.4f}}.txt")
                    np.savetxt(_fname, _out, header=_header_cols, delimiter=",", comments="")

            dim_str = "2D" if _is_2d else "1D"
            print(f"Solution data saved to: {{_data_dir}} ({{dim_str}}, grid={config.export_grid_size}, t_steps={config.export_t_steps})")

# ── End parametric loop ───────────────────────────────────────

# ── Parametric summary plot ───────────────────────────────────
if _parametric and len(_summary) > 0:
    labels = [str(v) for v, _ in _summary]
    losses = [l for _, l in _summary]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, losses, color="#4dabf7")
    ax.set_yscale("log")
    ax.set_xlabel(_param_name); ax.set_ylabel("Final Loss")
    ax.set_title(f"Parametric Study — Effect of {{_param_name}}")
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, loss, f"{{loss:.2e}}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _summary_path = _os.path.join(_save_dir if _use_save else "/tmp", "parametric_summary.png")
    plt.savefig(_summary_path, dpi=100); plt.close()
    import shutil as _shutil2
    _shutil2.copy(_summary_path, "/tmp/solution_plot.png")
    _shutil2.copy(_summary_path, "/tmp/loss_plot.png")
    print(f"\\n=== Parametric Study Complete ===")
    print(f"Summary saved to: {{_summary_path}}")

# ── Time Adaptive Loop ────────────────────────────────────────
if {config.time_adaptive}:
    print("\\n=== Starting Time-Adaptive Training ===")

    # Clear old step directories to avoid stale models from previous runs
    import shutil as _ta_shutil
    _ta_steps_root = _os.path.join(_save_dir, "time_adaptive_steps")
    if _os.path.isdir(_ta_steps_root):
        _ta_shutil.rmtree(_ta_steps_root)
        print(f"  Cleared old time_adaptive_steps directory")
    _os.makedirs(_ta_steps_root, exist_ok=True)

    grid_size = {config.ta_grid_size}
    if _is_2d:
        _xg_ta = np.linspace({config.x_min}, {config.x_max}, grid_size)
        _yg_ta = np.linspace({config.y_min}, {config.y_max}, grid_size)
        _Xmesh, _Ymesh = np.meshgrid(_xg_ta, _yg_ta)
        x_grid = np.column_stack([_Xmesh.ravel(), _Ymesh.ravel()])
        x = x_grid  # (N*N, 2)
    else:
        x_grid = np.linspace({config.x_min}, {config.x_max}, grid_size)

    all_x = []; all_t = []; all_u = []

    if _is_2d:
        x = x_grid
    else:
        x = x_grid.reshape(-1, 1)
    {_ta_ic_init}
    _lr = {config.learning_rate}
    _prev_step_model_path = ""

    # ── Build flat interval list from step groups ─────────────
    import json as _json_ta_groups
    _ta_groups = _json_ta_groups.loads({repr(config.ta_step_groups) if config.ta_step_groups else repr('[{"t_start":' + str(config.t_min) + ',"t_end":' + str(config.t_max) + ',"steps":' + str(config.ta_num_steps) + '}]')})
    _ta_flat_intervals = []
    for _grp in _ta_groups:
        _grp_dt = (_grp['t_end'] - _grp['t_start']) / _grp['steps']
        for _gi in range(_grp['steps']):
            _t0_g = _grp['t_start'] + _gi * _grp_dt
            _t1_g = _t0_g + _grp_dt
            _ta_flat_intervals.append((_t0_g, _t1_g))
    n_steps = len(_ta_flat_intervals)
    print(f"  Total time steps: {{n_steps}} from {{len(_ta_groups)}} group(s)")

    for step_i, (t0, t1) in enumerate(_ta_flat_intervals):
        print(f"\\n--- Time step {{step_i+1}}/{{n_steps}}: t = {{t0:.4f}} to {{t1:.4f}} ---")

        if _is_2d:
            geom_i = dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])
        else:
            geom_i = dde.geometry.Interval({config.x_min}, {config.x_max})
        time_i     = dde.geometry.TimeDomain(t0, t1)
        geomtime_i = dde.geometry.GeometryXTime(geom_i, time_i)

        _constraints_i = []
        for _oi_ta in range({config.num_outputs}):
            _comp_ta = _oi_ta
            _blt_ta = _bc_left_types[_oi_ta].strip()  if _oi_ta < len(_bc_left_types)  else "Dirichlet"
            _brt_ta = _bc_right_types[_oi_ta].strip() if _oi_ta < len(_bc_right_types) else "Dirichlet"
            _blv_ta = float(_bc_left_values[_oi_ta].strip())  if _oi_ta < len(_bc_left_values)  else 0.0
            _brv_ta = float(_bc_right_values[_oi_ta].strip()) if _oi_ta < len(_bc_right_values) else 0.0

            if _oi_ta < len(_bc_left_active) and _bc_left_active[_oi_ta].strip() == "True":
                def _bl_on_ta(x, on_boundary):
                    return on_boundary and dde.utils.isclose(x[0], {config.x_min})
                if _blt_ta == "Dirichlet":
                    def _mk_dbl_ta(v, comp, on_bd):
                        def _vf(x): return np.full((len(x),1), v)
                        return dde.icbc.DirichletBC(geomtime_i, _vf, on_bd, component=comp)
                    _constraints_i.append(_mk_dbl_ta(_blv_ta, _comp_ta, _bl_on_ta))
                elif _blt_ta == "Neumann":
                    def _mk_nbl_ta(v, comp, on_bd):
                        def _vf(x): return np.full((len(x),1), v)
                        return dde.icbc.NeumannBC(geomtime_i, _vf, on_bd, component=comp)
                    _constraints_i.append(_mk_nbl_ta(_blv_ta, _comp_ta, _bl_on_ta))
                elif _blt_ta == "Periodic":
                    _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 0, _bl_on_ta, derivative_order=0, component=_comp_ta))
                    if _oi_ta < len(_bc_left_deriv_list) and _bc_left_deriv_list[_oi_ta].strip() == "True":
                        _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 0, _bl_on_ta, derivative_order=1, component=_comp_ta))

            if _oi_ta < len(_bc_right_active) and _bc_right_active[_oi_ta].strip() == "True":
                def _br_on_ta(x, on_boundary):
                    return on_boundary and dde.utils.isclose(x[0], {config.x_max})
                if _brt_ta == "Dirichlet":
                    def _mk_dbr_ta(v, comp, on_bd):
                        def _vf(x): return np.full((len(x),1), v)
                        return dde.icbc.DirichletBC(geomtime_i, _vf, on_bd, component=comp)
                    _constraints_i.append(_mk_dbr_ta(_brv_ta, _comp_ta, _br_on_ta))
                elif _brt_ta == "Neumann":
                    def _mk_nbr_ta(v, comp, on_bd):
                        def _vf(x): return np.full((len(x),1), v)
                        return dde.icbc.NeumannBC(geomtime_i, _vf, on_bd, component=comp)
                    _constraints_i.append(_mk_nbr_ta(_brv_ta, _comp_ta, _br_on_ta))
                elif _brt_ta == "Periodic":
                    _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 0, _br_on_ta, derivative_order=0, component=_comp_ta))
                    if _oi_ta < len(_bc_right_deriv_list) and _bc_right_deriv_list[_oi_ta].strip() == "True":
                        _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 0, _br_on_ta, derivative_order=1, component=_comp_ta))

            if _is_2d:
                _bbt_ta = _bc_bottom_types[_oi_ta].strip() if _oi_ta < len(_bc_bottom_types) else "Dirichlet"
                _btt_ta = _bc_top_types[_oi_ta].strip()    if _oi_ta < len(_bc_top_types)    else "Dirichlet"
                _bbv_ta = float(_bc_bottom_values[_oi_ta].strip()) if _oi_ta < len(_bc_bottom_values) else 0.0
                _btv_ta = float(_bc_top_values[_oi_ta].strip())    if _oi_ta < len(_bc_top_values)    else 0.0

                if _oi_ta < len(_bc_bottom_active) and _bc_bottom_active[_oi_ta].strip() == "True":
                    def _bb_on_ta(x, on_boundary):
                        return on_boundary and dde.utils.isclose(x[1], {config.y_min})
                    if _bbt_ta == "Dirichlet":
                        def _mk_dbb_ta(v, comp, on_bd):
                            def _vf(x): return np.full((len(x),1), v)
                            return dde.icbc.DirichletBC(geomtime_i, _vf, on_bd, component=comp)
                        _constraints_i.append(_mk_dbb_ta(_bbv_ta, _comp_ta, _bb_on_ta))
                    elif _bbt_ta == "Periodic":
                        _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 1, _bb_on_ta, derivative_order=0, component=_comp_ta))

                if _oi_ta < len(_bc_top_active) and _bc_top_active[_oi_ta].strip() == "True":
                    def _bt_on_ta(x, on_boundary):
                        return on_boundary and dde.utils.isclose(x[1], {config.y_max})
                    if _btt_ta == "Dirichlet":
                        def _mk_dbt_ta(v, comp, on_bd):
                            def _vf(x): return np.full((len(x),1), v)
                            return dde.icbc.DirichletBC(geomtime_i, _vf, on_bd, component=comp)
                        _constraints_i.append(_mk_dbt_ta(_btv_ta, _comp_ta, _bt_on_ta))
                    elif _btt_ta == "Periodic":
                        _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 1, _bt_on_ta, derivative_order=0, component=_comp_ta))

            if step_i == 0:
                if {config.forward_ic_from_file} and _oi_ta == 0:
                    _xyt_ic_anchor = _ic_xyt  # use the already-loaded IC points
                    _constraints_i.append(dde.icbc.PointSetBC(_ic_xyt, _ic_vals, component=0))
                elif _oi_ta < len(_ic_active_list) and _ic_active_list[_oi_ta].strip() == "True":
                    _ic_expr_ta = _ic_expressions[_oi_ta].strip() if _oi_ta < len(_ic_expressions) else "np.zeros_like(x[:,0])"
                    def _mk_ic_ta(expr, comp):
                        def _ic_fn(x):
                            return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                        return dde.icbc.IC(geomtime_i, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                    _constraints_i.append(_mk_ic_ta(_ic_expr_ta, _comp_ta))
            else:
                if _oi_ta == 0:
                    if _is_2d:
                        _xt_ic_2d = np.column_stack([x_grid, np.full(len(x_grid), t0)])
                        _xyt_ic_anchor = _xt_ic_2d
                        _constraints_i.append(dde.icbc.PointSetBC(_xt_ic_2d, prev_u, component=0))
                    else:
                        xt_ic = np.column_stack([x_grid, np.full_like(x_grid.ravel(), t0)])
                        _xyt_ic_anchor = xt_ic
                        _constraints_i.append(dde.icbc.PointSetBC(xt_ic, prev_u, component=0))

        data_i = dde.data.TimePDE(
            geomtime_i, pde, _constraints_i,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_initial={config.num_initial}, num_test={config.num_test},
            anchors=None if {config.forward_ic_from_file} else (_xyt_ic_anchor if (step_i > 0 and _is_2d) else None)
        )

        net_i   = dde.nn.FNN({config.layers}, "{config.activation}", "Glorot uniform")
        model_i = dde.Model(data_i, net_i)
            
        # ── Transfer learning — warm start from previous step ─
        if {config.ta_transfer_learning} and step_i > 0 and _prev_step_model_path:
            try:
                import torch as _torch
                _ckpt = _torch.load(_prev_step_model_path, map_location='cpu')
                # Extract ONLY weights and biases — nothing else
                _state = _ckpt["model_state_dict"]
                _wb_only = {{
                    k: v for k, v in _state.items()
                    if k.endswith('.weight') or k.endswith('.bias')
                }}
                # Strict=False so geometry mismatch doesn't crash
                # but we verify shapes match before loading
                _cur_state = model_i.net.state_dict()
                _matched = {{}}
                _skipped = []
                for k, v in _wb_only.items():
                    if k in _cur_state and _cur_state[k].shape == v.shape:
                        _matched[k] = v
                    else:
                        _skipped.append(k)
                _cur_state.update(_matched)
                model_i.net.load_state_dict(_cur_state)
                print(f"  ✅ Transfer learning: {{len(_matched)}} weight/bias tensors transferred from step {{step_i}}")
                if _skipped:
                    print(f"  ⚠️  Skipped {{len(_skipped)}} tensors (shape mismatch): {{_skipped}}")
            except Exception as _te:
                print(f"  ⚠️ Transfer learning failed: {{_te}} — training from scratch")

        # IC pre-training for first step only
        if {config.ic_pretrain} and step_i == 0:
            print(f"  === IC Pre-Training: {config.ic_pretrain_iterations} iterations ===")
            # Build IC-only data: no domain/BC points, only IC points
            if _is_2d:
                _ic_geom_pt = dde.geometry.Rectangle([{config.x_min},{config.y_min}],[{config.x_max},{config.y_max}])
            else:
                _ic_geom_pt = dde.geometry.Interval({config.x_min}, {config.x_max})
            _ic_td_pt = dde.geometry.TimeDomain(t0, t1)
            _ic_gt_pt = dde.geometry.GeometryXTime(_ic_geom_pt, _ic_td_pt)
            # Only IC constraint
            _ic_constraints_pt = []
            if {config.forward_ic_from_file}:
                _ic_ta_data = np.loadtxt(r"{config.forward_ic_file}")
                _ic_ta_mask = np.abs(_ic_ta_data[:, 2]) < 1e-10
                _ic_ta_xy   = _ic_ta_data[_ic_ta_mask, :2]
                _ic_ta_xyt  = np.hstack([_ic_ta_xy, np.zeros((_ic_ta_mask.sum(), 1))])
                _ic_ta_vals = _ic_ta_data[_ic_ta_mask, 3:4]
                _ic_constraints_pt.append(dde.icbc.PointSetBC(_ic_ta_xyt, _ic_ta_vals, component=0))
            else:
                for _oi_pt in range({config.num_outputs}):
                    _ic_expr_pt = _ic_expressions[_oi_pt].strip() if _oi_pt < len(_ic_expressions) else "np.zeros_like(x[:,0])"
                    def _mk_ic_pt(expr, comp):
                        def _ic_fn(x):
                            return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                        return dde.icbc.IC(_ic_gt_pt, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                    _ic_constraints_pt.append(_mk_ic_pt(_ic_expr_pt, _oi_pt))
            def _pde_dummy_ta(x, y):
                return [y[:, _oi_d:_oi_d+1] * 0 for _oi_d in range({config.num_outputs})]
            _bc_constraints_pre = [_c for _c in _constraints_i if not isinstance(_c, dde.icbc.PointSetBC)]
            _ic_pre_constraints_ta = _bc_constraints_pre + _ic_constraints_pt
            _data_pt = dde.data.TimePDE(
                _ic_gt_pt, _pde_dummy_ta, _ic_pre_constraints_ta,
                num_domain=0, num_boundary=0,
                num_initial=0, num_test={config.ic_pretrain_num_test},
                train_distribution="{config.point_distribution}",
                anchors=None
            )
            _net_pt = model_i.net
            _model_pt = dde.Model(_data_pt, _net_pt)
            # IC-only weights: PDE=0, BC=0, IC=1000
            _n_bcs_ta = len(_bc_constraints_pre)
            _ic_only_w_ta = [0.0] * {config.num_outputs} + [0.0] * _n_bcs_ta + [1000.0] * len(_ic_constraints_pt)
            _model_pt.compile("{config.ic_pretrain_optimizer}", lr=_lr,
                              loss="MSE", loss_weights=_ic_only_w_ta)
            _ic_pre_save_dir_ta = _os.path.join(r"{config.save_dir}", "ic_pretrain")
            _os.makedirs(_ic_pre_save_dir_ta, exist_ok=True)
            if {config.ic_pretrain_restore} and r"{config.ic_pretrain_restore_path}" and _os.path.exists(r"{config.ic_pretrain_restore_path}"):
                print(f"  Restoring IC pre-train model from: {config.ic_pretrain_restore_path}")
                _ic_ckpt_ta = torch.load(r"{config.ic_pretrain_restore_path}", map_location="cpu")
                _ic_state_ta = _ic_ckpt_ta.get("model_state_dict", _ic_ckpt_ta)
                _model_pt.net.load_state_dict(_ic_state_ta)
                print("  IC pre-train model restored — skipping training.")
            else:
                _ic_lh_ta, _ = _model_pt.train(iterations={config.ic_pretrain_iterations},
                                                display_every=10000, batch_size=None,
                                                model_save_path=_os.path.join(_ic_pre_save_dir_ta, "ic_pretrain_model"))
                print(f"  IC pre-training done. Final loss: {{sum(_ic_lh_ta.loss_train[-1]):.4e}}")
                print(f"  IC pre-train model saved to: {{_ic_pre_save_dir_ta}}")
            print("  === Starting Main Training ===")

        # ── Optimizer Scheduler phases ────────────────────────
        if {config.optimizer_scheduler} and {len(config.scheduler_phases) > 0}:
            import json as _json
            _sched_phases = _json.loads({repr(config.scheduler_phases)})
            for _sp_i, _sp in enumerate(_sched_phases):
                print(f"  === Scheduler Phase {{_sp_i+1}}: {{_sp['optimizer']}} {{_sp['iterations']}} iters ===")
                _sp_weights = [float(w) for w in _sp['weights'].split(',') if w.strip()]
                if _sp['optimizer'] == 'lbfgs':
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter=_sp['iterations'],
                        maxfun=int(_sp['iterations']*1.25), maxls={config.lbfgs_maxls})
                    _lbfgs_float = "{config.lbfgs_float_type}"
                    model_i.compile("L-BFGS", loss=_sp.get('loss', '{config.loss_type}'),
                                    loss_weights=_sp_weights)
                    lh_i, ts_i = model_i.train(display_every=200)
                    if _use_save:
                        _sp_step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                        _os.makedirs(_sp_step_dir, exist_ok=True)
                        _sp_save_path = _os.path.join(_sp_step_dir, f"model_lbfgs-phase{{_sp_i+1}}")
                        model_i.save(_sp_save_path)
                        print(f"  Phase {{_sp_i+2}} L-BFGS model saved: {{_sp_save_path}}.pt")
                else:
                    model_i.compile(_sp['optimizer'], lr=_sp['lr'],
                                    loss=_sp.get('loss', '{config.loss_type}'), loss_weights=_sp_weights)
                    if {config.batch_size} > 0:
                        data_i.batch_size = {config.batch_size}
                    lh_i, ts_i = model_i.train(iterations=_sp['iterations'], display_every=1000)
                    if _use_save:
                        _sp_iters = lh_i.steps[-1] if lh_i.steps else _sp['iterations']
                        _sp_step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                        _os.makedirs(_sp_step_dir, exist_ok=True)
                        _sp_save_path = _os.path.join(_sp_step_dir, f"model_adam-phase{{_sp_i+1}}")
                        model_i.save(_sp_save_path)
                        print(f"  Phase {{_sp_i+2}} Adam model saved: {{_sp_save_path}}.pt")
        elif "{config.optimizer2}" != "none":
            dde.optimizers.set_LBFGS_options(
                    maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                    gtol={config.lbfgs_gtol}, maxiter={config.iterations2},
                    maxfun={config.lbfgs_maxfun}, maxls={config.lbfgs_maxls})
            model_i.compile("L-BFGS", loss="{config.loss_type}",
                            loss_weights=_multi_weights)
            lh_i, ts_i = model_i.train(display_every=200)
            print(f"  L-BFGS phase done. Steps: {{len(lh_i.steps)}}")

            if _use_save:
                _step_dir_lbfgs = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                _os.makedirs(_step_dir_lbfgs, exist_ok=True)
                _step_lbfgs_path = _os.path.join(_step_dir_lbfgs, "model_lbfgs")
                model_i.save(_step_lbfgs_path)
                print(f"Step L-BFGS model saved: {{_step_lbfgs_path}}.pt")

        if _is_2d:
            # 2D: predict on x-y grid at t=t1, store as flat array for PointSetBC next step
            _x_pred = np.linspace({config.x_min}, {config.x_max}, grid_size)
            _y_pred = np.linspace({config.y_min}, {config.y_max}, grid_size)
            _Xp, _Yp = np.meshgrid(_x_pred, _y_pred)
            _xyt_pred = np.column_stack([_Xp.ravel(), _Yp.ravel(), np.full(_Xp.size, t1)])
            prev_u  = model_i.predict(_xyt_pred)[:, 0:1]
            x_grid  = _xyt_pred[:, :2]  # store (x,y) pairs for next step's PointSetBC
        else:
            x_pred  = np.linspace({config.x_min}, {config.x_max}, grid_size)
            t_pred  = np.full_like(x_pred, t1)
            xt_pred = np.column_stack([x_pred, t_pred])
            prev_u  = model_i.predict(xt_pred)[:, 0:1]
            x_grid  = x_pred.reshape(-1, 1)

        if not _is_2d:
            x_plot = np.linspace({config.x_min}, {config.x_max}, 100)
            t_plot = np.linspace(t0, t1, 50)
            Xp, Tp = np.meshgrid(x_plot, t_plot)
            XTp    = np.vstack([Xp.ravel(), Tp.ravel()]).T
            Up     = model_i.predict(XTp)[:, {config.plot_output_idx}].reshape(50, 100)
            all_x.append(Xp); all_t.append(Tp); all_u.append(Up)
        else:
            # 2D: store mid-y slice for combined plot
            x_plot = np.linspace({config.x_min}, {config.x_max}, 100)
            t_plot = np.linspace(t0, t1, 50)
            y_mid  = ({config.y_min} + {config.y_max}) / 2.0
            Xp, Tp = np.meshgrid(x_plot, t_plot)
            XYTp   = np.column_stack([Xp.ravel(), np.full(Xp.size, y_mid), Tp.ravel()])
            Up     = model_i.predict(XYTp)[:, {config.plot_output_idx}].reshape(50, 100)
            all_x.append(Xp); all_t.append(Tp); all_u.append(Up)

        print(f"Step {{step_i+1}} done. Final train loss: {{sum(lh_i.loss_train[-1]):.4e}}")

        # ── Save step plot & models ───────────────────────────
        if _use_save:
            _step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
            _os.makedirs(_step_dir, exist_ok=True)
            _plot_type_step = "{config.plot_type}"
            _x_plot_step = np.linspace({config.x_min}, {config.x_max}, 100)
            _t_plot_step = np.linspace(t0, t1, 50)
            _Xp_step, _Tp_step = np.meshgrid(_x_plot_step, _t_plot_step)
            if _is_2d:
                _y_mid_step = ({config.y_min} + {config.y_max}) / 2.0
                _XTp_step = np.column_stack([_Xp_step.ravel(), np.full(_Xp_step.size, _y_mid_step), _Tp_step.ravel()])
            else:
                _XTp_step = np.vstack([_Xp_step.ravel(), _Tp_step.ravel()]).T
            _Up_step = model_i.predict(_XTp_step)[:, {config.plot_output_idx}].reshape(50, 100)

            if _plot_type_step == "Surface" or _plot_type_step.startswith("📊"):
                _vmin_step = None if {config.plot_auto_range} else {config.plot_vmin}
                _vmax_step = None if {config.plot_auto_range} else {config.plot_vmax}
                _step_fname = _os.path.join(_step_dir, f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}.png")
                if _is_2d:
                    # 2D: x-y heatmaps at t=t0 and t=t1
                    _res_step = {config.plot_resolution}
                    _xg_s = np.linspace({config.x_min}, {config.x_max}, _res_step)
                    _yg_s = np.linspace({config.y_min}, {config.y_max}, _res_step)
                    _Xg_s, _Yg_s = np.meshgrid(_xg_s, _yg_s)
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    for _ai, _tv_s in enumerate([t0, t1]):
                        _xyt_s = np.column_stack([_Xg_s.ravel(), _Yg_s.ravel(), np.full(_Xg_s.size, _tv_s)])
                        _U_s = model_i.predict(_xyt_s)[:, {config.plot_output_idx}].reshape(_res_step, _res_step)
                        im = axes[_ai].contourf(_Xg_s, _Yg_s, _U_s, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_step, vmax=_vmax_step)
                        if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
                        axes[_ai].set_xlabel("x"); axes[_ai].set_ylabel("y")
                        axes[_ai].set_title(f"t = {{_tv_s:.4f}}")
                    fig.suptitle(f"Step {{step_i+1}}: t = {{t0:.4f}} → {{t1:.4f}}", fontsize=11)
                    plt.tight_layout()
                else:
                    _res_step = {config.plot_resolution}
                    _x_s2 = np.linspace({config.x_min}, {config.x_max}, _res_step)
                    _t_s2 = np.linspace(t0, t1, _res_step)
                    _Xs2, _Ts2 = np.meshgrid(_x_s2, _t_s2)
                    _XTs2 = np.vstack([_Xs2.ravel(), _Ts2.ravel()]).T
                    _Us2 = model_i.predict(_XTs2)[:, {config.plot_output_idx}].reshape(_res_step, _res_step)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    im = ax.contourf(_Xs2, _Ts2, _Us2, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_step, vmax=_vmax_step)
                    if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
                    ax.set_xlabel("x"); ax.set_ylabel("t")
                    ax.set_title(f"Step {{step_i+1}}: t = {{t0:.4f}} → {{t1:.4f}}")
                    plt.tight_layout()
                plt.savefig(_step_fname, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _plot_type_step.startswith("Line"):
                n_steps_plot = {config.num_timesteps}
                _x_l2 = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                _t_line = np.linspace(t0, t1, n_steps_plot)
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = plt.cm.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_plot))
                for _ci, _tv in enumerate(_t_line):
                    _xt_line = np.column_stack([_x_l2, np.full_like(_x_l2, _tv)])
                    _u_line = model_i.predict(_xt_line)[:, {config.plot_output_idx}].flatten()
                    ax.plot(_x_l2, _u_line, color=colors[_ci], linewidth={config.plot_linewidth}, label=f"t={{_tv:.3f}}")
                ax.set_xlabel("x"); ax.set_ylabel("u")
                ax.set_title(f"Step {{step_i+1}}: t = {{t0:.4f}} → {{t1:.4f}}")
                ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.2)
                plt.tight_layout()
                _step_fname = _os.path.join(_step_dir, f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}.png")
                plt.savefig(_step_fname, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            print(f"Step plot saved: {{_step_fname}}")

            # ── Final model already saved per-phase, just track for transfer learning ───
            _last_phase_opt = "{config.optimizer}"
            if {config.optimizer_scheduler} and {len(config.scheduler_phases) > 0}:
                import json as _json_lp
                _lp_phases = _json_lp.loads({repr(config.scheduler_phases)})
                if _lp_phases:
                    _last_phase_opt = _lp_phases[-1]["optimizer"]

            # Track for transfer learning — prefer lbfgs if chosen, else adam
            _prev_step_model_path = ""
            import glob as _tl_glob
            if "{config.ta_transfer_optimizer}" == "lbfgs":
                # Look for any lbfgs .pt file (may have iteration number suffix)
                _lbfgs_pts = sorted(_tl_glob.glob(_os.path.join(_step_dir, "model_lbfgs*.pt")))
                if _lbfgs_pts:
                    _prev_step_model_path = max(_lbfgs_pts, key=_os.path.getmtime)
                    print(f"  Transfer learning source (lbfgs): {{_os.path.basename(_prev_step_model_path)}}")
            if not _prev_step_model_path:
                # Fall back to adam
                _adam_pts = sorted(_tl_glob.glob(_os.path.join(_step_dir, "model_adam*.pt")))
                if _adam_pts:
                    _prev_step_model_path = max(_adam_pts, key=_os.path.getmtime)
                    print(f"  Transfer learning source (adam): {{_os.path.basename(_prev_step_model_path)}}")

            # ── Save step config JSON ─────────────────────────
            _step_cfg = {{
                "step": step_i + 1,
                "t0": t0, "t1": t1,
                "layers": {config.layers},
                "activation": "{config.activation}",
                "x_min": {config.x_min}, "x_max": {config.x_max},
                "t_min": t0, "t_max": t1,
                "problem_dim": "{config.problem_dim}",
                "loss_type": "{config.loss_type}",
                "optimizer": "{config.optimizer}",
                "optimizer2": "{config.optimizer2}",
                "output_names": "{config.output_names}",
                "num_outputs": {config.num_outputs},
            }}
            with open(_os.path.join(_step_dir, "step_config.json"), "w") as _scf:
                _json.dump(_step_cfg, _scf, indent=2)
            print(f"Step config saved: {{_os.path.join(_step_dir, 'step_config.json')}}")

    X_full = np.vstack(all_x); T_full = np.vstack(all_t); U_full = np.vstack(all_u)
    print("\\n=== Time-Adaptive Training Complete ===")

    _ta_solution_path = _os.path.join(_sol_dir, "solution_plot.png")
    _ta_loss_path     = _os.path.join(_sol_dir, "loss_plot.png")

    _plot_type_ta = "{config.plot_type}"
    if _plot_type_ta == "Surface":
        _vmin_ta = None if {config.plot_auto_range} else {config.plot_vmin}
        _vmax_ta = None if {config.plot_auto_range} else {config.plot_vmax}
        if _is_2d:
            # 2D: x-y heatmaps at n snapshots using the last step model
            _n_snaps_ta = {config.plot_n_2d_snapshots}
            _t_snaps_ta = np.linspace({config.t_min}, {config.t_max}, _n_snaps_ta)
            _res_ta = {config.plot_resolution}
            _xp_ta = np.linspace({config.x_min}, {config.x_max}, _res_ta)
            _yp_ta = np.linspace({config.y_min}, {config.y_max}, _res_ta)
            _Xg_ta, _Yg_ta = np.meshgrid(_xp_ta, _yp_ta)
            # Collect all step models to predict at each snapshot time
            import glob as _ta_glob_sol, json as _ta_json_sol
            _ta_sol_dirs = sorted([_sd for _sd in _ta_glob_sol.glob(_os.path.join(_ta_steps_root, "step_*")) if _os.path.isdir(_sd)])
            _ta_sol_intervals = []
            for _sd in _ta_sol_dirs:
                try:
                    _p = _os.path.basename(_sd).split("_")
                    _ta_sol_intervals.append((float(_p[2].replace("t","")), float(_p[4].replace("t","")), _sd))
                except Exception: pass
            fig, axes = plt.subplots(1, _n_snaps_ta, figsize=(5*_n_snaps_ta, 5))
            if _n_snaps_ta == 1: axes = [axes]
            for _ai, _tv_ta in enumerate(_t_snaps_ta):
                # Find which step model covers this time
                _sd_for_t = _ta_sol_dirs[-1] if _ta_sol_dirs else ""
                for _t0s, _t1s, _sds in _ta_sol_intervals:
                    if _t0s <= _tv_ta <= _t1s + 1e-10:
                        _sd_for_t = _sds; break
                # Load step model
                _spt_ta = ""
                for _pat_ta in ["model_lbfgs-*.pt","model_lbfgs.pt","model_adam-*.pt","model_adam.pt"]:
                    _pts_ta = sorted(_ta_glob_sol.glob(_os.path.join(_sd_for_t, _pat_ta)))
                    if _pts_ta: _spt_ta = max(_pts_ta, key=_os.path.getmtime); break
                try:
                    with open(_os.path.join(_sd_for_t, "step_config.json")) as _scf_ta:
                        _sc_ta = _ta_json_sol.load(_scf_ta)
                except: _sc_ta = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}
                _sn_ta = dde.nn.FNN(_sc_ta.get("layers",{config.layers}), _sc_ta.get("activation","{config.activation}"), "Glorot uniform")
                _sg_ta = dde.geometry.Rectangle([{config.x_min},{config.y_min}],[{config.x_max},{config.y_max}])
                _st_ta = dde.geometry.TimeDomain(_sc_ta.get("t_min",0), _sc_ta.get("t_max",1))
                _sgt_ta = dde.geometry.GeometryXTime(_sg_ta, _st_ta)
                _sd_ta = dde.data.TimePDE(_sgt_ta, lambda x,y: y[:,0:1]*0, [], num_domain=100, num_test=100)
                _sm_ta = dde.Model(_sd_ta, _sn_ta)
                if _spt_ta:
                    if "lbfgs" in _os.path.basename(_spt_ta):
                        dde.optimizers.set_LBFGS_options(maxiter=1)
                        _sm_ta.compile("L-BFGS", loss=_sc_ta.get("loss_type","{config.loss_type}"))
                    else:
                        _sm_ta.compile("adam", lr=0.001, loss=_sc_ta.get("loss_type","{config.loss_type}"))
                    _sm_ta.restore(_spt_ta, verbose=0)
                    _xyt_ta = np.column_stack([_Xg_ta.ravel(), _Yg_ta.ravel(), np.full(_Xg_ta.size, _tv_ta)])
                    _pred_ta = _sm_ta.predict(_xyt_ta)[:, {config.plot_output_idx}].reshape(_res_ta, _res_ta)
                else:
                    _pred_ta = np.zeros((_res_ta, _res_ta))
                im = axes[_ai].contourf(_Xg_ta, _Yg_ta, _pred_ta, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_ta, vmax=_vmax_ta)
                axes[_ai].set_title(f"t = {{_tv_ta:.3f}}")
                axes[_ai].set_xlabel("x"); axes[_ai].set_ylabel("y")
                if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
            fig.suptitle("Time-Adaptive PINN Solution", fontsize=12)
            plt.tight_layout()
            plt.savefig(_ta_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.contourf(X_full, T_full, U_full, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_ta, vmax=_vmax_ta)
            if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
            ax.set_xlabel("x"); ax.set_ylabel("t")
            ax.set_title("Time-Adaptive PINN Solution")
            plt.tight_layout(); plt.savefig(_ta_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
    elif _plot_type_ta.startswith("Line"):
        n_ts   = {config.num_timesteps}
        t_vals = np.linspace(_ta_flat_intervals[0][0], _ta_flat_intervals[-1][1], n_ts)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_ts))
        for ci, tv in enumerate(t_vals):
            idx = np.argmin(np.abs(T_full[:,0] - tv))
            ax.plot(X_full[idx,:], U_full[idx,:], color=colors[ci], linewidth={config.plot_linewidth}, label=f"t={{tv:.3f}}")
        ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
        ax.set_title("Time-Adaptive PINN — Line Plot")
        ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
        plt.tight_layout(); plt.savefig(_ta_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

    train_loss_ta = lh_i.loss_train; test_loss_ta = lh_i.loss_test
    steps_ta = list(range(len(train_loss_ta)))
    plt.figure(figsize=(6, 4))
    plt.semilogy(steps_ta, [sum(l) for l in train_loss_ta], label="Train", color="#4dabf7")
    plt.semilogy(steps_ta, [sum(l) for l in test_loss_ta],  label="Test",  color="#ff8787", linestyle="--")
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.title("Loss — Last Time Sub-domain")
    plt.legend(); plt.tight_layout()
    plt.savefig(_ta_loss_path, dpi=100); plt.close()

    # ── Time Adaptive Error Analysis ──────────────────────────
    if {config.ea_files}:
        from scipy.interpolate import interp1d as _interp1d
        import glob as _ea_glob, json as _ea_json
        _ea_files = [(tv, fp) for tv, fp in {config.ea_files}
                     if {config.t_min} - 1e-10 <= tv <= {config.t_max} + 1e-10]
        _ea_dir = _os.path.join(_save_dir if _use_save else "/tmp", "error_analysis")
        _os.makedirs(_ea_dir, exist_ok=True)
        print(f"  Filtering to t=[{config.t_min}, {config.t_max}]: {{len(_ea_files)}} files")
        print("\\n=== Running Time-Adaptive Error Analysis ===")

        # Load all ground truth files
        _ea_times = []; _ea_x_refs = []; _ea_y_refs = []; _ea_u_refs = []
        for _ea_tv, _ea_fp in _ea_files:
            _ea_d = np.loadtxt(_ea_fp)
            if _ea_d.ndim == 1: _ea_d = _ea_d.reshape(1, -1)
            if _is_2d:
                _ea_idx = np.lexsort((_ea_d[:, 1], _ea_d[:, 0]))
                _ea_x_refs.append(_ea_d[_ea_idx, 0])
                _ea_y_refs.append(_ea_d[_ea_idx, 1])
                _ea_u_refs.append(_ea_d[_ea_idx, 3])
                _detected_t = float(_ea_d[0, 2])
                _ea_times.append(_detected_t)
                print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
            else:
                _ea_idx = np.argsort(_ea_d[:, 0])
                _ea_x_refs.append(_ea_d[_ea_idx, 0])
                _ea_y_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                _ea_u_refs.append(_ea_d[_ea_idx, 2])
                _ea_times.append(float(_ea_tv))
                print(f"  Loaded ground truth t={{_ea_tv:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
        # Sort by time
        _ea_sort_idx = np.argsort(_ea_times)
        _ea_times  = [_ea_times[_i]  for _i in _ea_sort_idx]
        _ea_x_refs = [_ea_x_refs[_i] for _i in _ea_sort_idx]
        _ea_y_refs = [_ea_y_refs[_i] for _i in _ea_sort_idx]
        _ea_u_refs = [_ea_u_refs[_i] for _i in _ea_sort_idx]
        _ea_n_t = len(_ea_times)
        _ea_u_pinns = [None] * _ea_n_t

        # Find all step directories
        _ta_step_dir = _os.path.join(_save_dir, "time_adaptive_steps")
        _ta_step_dirs = sorted([_sd for _sd in _ea_glob.glob(_os.path.join(_ta_step_dir, "step_*")) if _os.path.isdir(_sd)])
        _ta_intervals = []
        for _sd in _ta_step_dirs:
            _sd_name = _os.path.basename(_sd)
            try:
                _parts = _sd_name.split("_")
                _t0_str = _parts[2].replace("t","")
                _t1_str = _parts[4].replace("t","")
                _ta_intervals.append((float(_t0_str), float(_t1_str), _sd))
            except Exception as _pe:
                print(f"  Could not parse step dir: {{_sd_name}}: {{_pe}}")
        print(f"  Found {{len(_ta_intervals)}} time-adaptive step models")

        for _si, (_t0_i, _t1_i, _sd_i) in enumerate(_ta_intervals):
            _is_last = (_si == len(_ta_intervals) - 1)
            _matching = []
            for _ei, _ea_tv in enumerate(_ea_times):
                if _is_last:
                    _in_range = (_t0_i <= _ea_tv <= _t1_i + 1e-10)
                else:
                    _in_range = (_t0_i <= _ea_tv < _t1_i - 1e-10) or \
                                (abs(_ea_tv - _t1_i) < 1e-10)
                if _in_range and _ea_u_pinns[_ei] is None:
                    _matching.append(_ei)

            if not _matching:
                continue

            print(f"  Step {{_si+1}} [{{_t0_i:.4f}}→{{_t1_i:.4f}}]: t = {{[_ea_times[_ei] for _ei in _matching]}}")

            _step_cfg_path = _os.path.join(_sd_i, "step_config.json")
            try:
                with open(_step_cfg_path) as _scf:
                    _step_cfg = _ea_json.load(_scf)
            except Exception:
                _step_cfg = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}

            _step_layers = _step_cfg.get("layers", {config.layers})
            _step_act    = _step_cfg.get("activation", "{config.activation}")
            _step_loss   = _step_cfg.get("loss_type", "{config.loss_type}")

            _step_geom  = dde.geometry.Interval({config.x_min}, {config.x_max})
            _step_td    = dde.geometry.TimeDomain(_t0_i, _t1_i)
            _step_gt    = dde.geometry.GeometryXTime(_step_geom, _step_td)
            def _step_pde(x, y): return y[:, 0:1] * 0
            _step_data  = dde.data.TimePDE(_step_gt, _step_pde, [], num_domain=100, num_test=100)
            _step_net   = dde.nn.FNN(_step_layers, _step_act, "Glorot uniform")
            _step_model = dde.Model(_step_data, _step_net)

            _step_pt = ""
            for _pat in ["model_lbfgs-*.pt", "model_lbfgs.pt", "model_adam-*.pt", "model_adam.pt"]:
                _step_pts = sorted(_ea_glob.glob(_os.path.join(_sd_i, _pat)))
                if _step_pts:
                    _step_pt = max(_step_pts, key=_os.path.getmtime)
                    break

            if not _step_pt:
                print(f"  ⚠️ No model found for step {{_si+1}}, skipping")
                continue

            if "lbfgs" in _os.path.basename(_step_pt):
                dde.optimizers.set_LBFGS_options(maxiter=1)
                _step_model.compile("L-BFGS", loss=_step_loss)
            else:
                _step_model.compile("adam", lr=0.001, loss=_step_loss)

            _step_model.restore(_step_pt, verbose=0)
            print(f"    Restored: {{_os.path.basename(_step_pt)}}")

            for _ei in _matching:
                _ea_xf = _ea_x_refs[_ei]
                _ea_tv = _ea_times[_ei]
                if _is_2d:
                    _ea_yf = _ea_y_refs[_ei]
                    _ea_xt = np.column_stack([_ea_xf, _ea_yf, np.full_like(_ea_xf, _ea_tv)])
                else:
                    _ea_xt = np.column_stack([_ea_xf, np.full_like(_ea_xf, _ea_tv)])
                _ea_u_pinns[_ei] = _step_model.predict(_ea_xt)[:, 0].flatten()
                print(f"    Predicted at t={{_ea_tv:.4f}}: {{len(_ea_xf)}} points")

        for _ei in range(_ea_n_t):
            if _ea_u_pinns[_ei] is None:
                print(f"  ⚠️ No prediction for t={{_ea_times[_ei]:.4f}} — zero fill")
                _ea_u_pinns[_ei] = np.zeros_like(_ea_u_refs[_ei])

        # Metrics
        _ea_metrics = []
        for _ei, _ea_tv in enumerate(_ea_times):
            _up = _ea_u_pinns[_ei]; _uf = _ea_u_refs[_ei]
            _ea_abs = np.abs(_up - _uf)
            _ea_l2  = np.linalg.norm(_up - _uf) / (np.linalg.norm(_uf) + 1e-10)
            _ea_mse = np.mean((_up - _uf)**2)
            _ea_mx  = np.max(_ea_abs)
            _ea_ma  = np.mean(_ea_abs)
            _ea_metrics.append((_ea_tv, _ea_l2, _ea_mse, _ea_mx, _ea_ma))
            print(f"  t={{_ea_tv:.4f}} — L2={{_ea_l2:.4e}}, MSE={{_ea_mse:.4e}}, Max={{_ea_mx:.4e}}")

        with open(_os.path.join(_ea_dir, "error_metrics.txt"), "w") as _emf:
            _emf.write("t,L2_relative,MSE,Max_error,Mean_abs_error\\n")
            for _ea_tv, _l2, _mse, _mx, _ma in _ea_metrics:
                _emf.write(f"{{_ea_tv:.6f}},{{_l2:.6e}},{{_mse:.6e}},{{_mx:.6e}},{{_ma:.6e}}\\n")
        print(f"  Metrics saved: {{_os.path.join(_ea_dir, 'error_metrics.txt')}}")

        # Line comparison
        if {config.ea_do_line}:
            _ea_ncols = min(4, _ea_n_t)
            _ea_nrows = (_ea_n_t + _ea_ncols - 1) // _ea_ncols
            fig, axes = plt.subplots(_ea_nrows, _ea_ncols, figsize=(4*_ea_ncols, 3.5*_ea_nrows), squeeze=False)
            fig.suptitle("PINN vs Ground Truth — Line Comparison", fontsize=13, fontweight='bold')
            _ea_ax_flat = axes.flatten()
            for _ei in range(_ea_n_t):
                ax = _ea_ax_flat[_ei]
                _xv = _ea_x_refs[_ei]
                _ea_sort = np.argsort(_xv)
                _xv_s = _xv[_ea_sort]
                _gt_s = _ea_u_refs[_ei][_ea_sort]
                _pinn_s = _ea_u_pinns[_ei][_ea_sort]
                _ea_tv, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                ax.plot(_xv_s, _gt_s,   color='#4dabf7', linewidth=2.0, linestyle='-',  label='Ground Truth')
                ax.plot(_xv_s, _pinn_s, color='#ff6b6b', linewidth=2.0, linestyle='--', label='PINN')
                ax.set_title(f"t = {{_ea_tv:.3f}}  |  L2 = {{_l2:.2e}}", fontsize=10)
                ax.set_xlabel("x"); ax.set_ylabel("u(x,t)"); ax.grid(True, alpha=0.3)
            for _ej in range(_ea_n_t, len(_ea_ax_flat)):
                _ea_ax_flat[_ej].set_visible(False)
            handles, labels = _ea_ax_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
                       framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
            plt.tight_layout(rect=[0, 0.06, 1, 1])
            _ea_lp = _os.path.join(_ea_dir, "line_comparison.png")
            plt.savefig(_ea_lp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            print(f"  Line comparison saved: {{_ea_lp}}")

        # Surface comparison
        if {config.ea_do_surface}:
            if _is_2d:
                # 2D: PINN | FEM | Error heatmaps, one row per time snapshot
                from scipy.interpolate import griddata as _gd
                _res_ea = {config.plot_resolution}
                _xg_ea = np.linspace({config.x_min}, {config.x_max}, _res_ea)
                _yg_ea = np.linspace({config.y_min}, {config.y_max}, _res_ea)
                _Xg_ea, _Yg_ea = np.meshgrid(_xg_ea, _yg_ea)
                fig, axes = plt.subplots(_ea_n_t, 3, figsize=(15, 4*_ea_n_t), squeeze=False)
                fig.suptitle("PINN vs Ground Truth — 2D Heatmaps", fontsize=13, fontweight='bold')
                # Need step models to predict on grid — collect from step dirs
                import glob as _ea_glob2, json as _ea_json2
                _ta_step_dir2 = _os.path.join(_save_dir, "time_adaptive_steps")
                _ta_step_dirs2 = sorted([_sd for _sd in _ea_glob2.glob(_os.path.join(_ta_step_dir2, "step_*")) if _os.path.isdir(_sd)])
                _ta_intervals2 = []
                for _sd in _ta_step_dirs2:
                    try:
                        _parts = _os.path.basename(_sd).split("_")
                        _ta_intervals2.append((float(_parts[2].replace("t","")), float(_parts[4].replace("t","")), _sd))
                    except Exception: pass
                # Map each time to its step model
                _step_model_map = {{}}
                for _si2, (_t0_i2, _t1_i2, _sd_i2) in enumerate(_ta_intervals2):
                    for _ei in range(_ea_n_t):
                        _tv = _ea_times[_ei]
                        _is_last2 = (_si2 == len(_ta_intervals2) - 1)
                        if _is_last2:
                            _in = (_t0_i2 <= _tv <= _t1_i2 + 1e-10)
                        else:
                            _in = (_t0_i2 <= _tv < _t1_i2 - 1e-10) or (abs(_tv - _t1_i2) < 1e-10)
                        if _in:
                            _step_model_map[_ei] = _sd_i2
                for _ei, _ea_tv in enumerate(_ea_times):
                    _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                    # PINN on grid using step model
                    _sd_for_ei = _step_model_map.get(_ei, "")
                    _xyt_grid = np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel(), np.full(_Xg_ea.size, _ea_tv)])
                    if _sd_for_ei:
                        try:
                            with open(_os.path.join(_sd_for_ei, "step_config.json")) as _scf2: _sc2 = _ea_json2.load(_scf2)
                        except: _sc2 = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}
                        _sn2 = dde.nn.FNN(_sc2.get("layers",{config.layers}), _sc2.get("activation","{config.activation}"), "Glorot uniform")
                        _sg2 = dde.geometry.Rectangle([{config.x_min},{config.y_min}],[{config.x_max},{config.y_max}])
                        _st2 = dde.geometry.TimeDomain(_sc2.get("t_min",0), _sc2.get("t_max",1))
                        _sgt2 = dde.geometry.GeometryXTime(_sg2, _st2)
                        _sd2 = dde.data.TimePDE(_sgt2, lambda x,y: y[:,0:1]*0, [], num_domain=100, num_test=100)
                        _sm2 = dde.Model(_sd2, _sn2)
                        _spt2 = ""
                        for _pat2 in ["model_lbfgs-*.pt","model_lbfgs.pt","model_adam-*.pt","model_adam.pt"]:
                            _pts2 = sorted(_ea_glob2.glob(_os.path.join(_sd_for_ei, _pat2)))
                            if _pts2: _spt2 = max(_pts2, key=_os.path.getmtime); break
                        if _spt2:
                            if "lbfgs" in _os.path.basename(_spt2):
                                dde.optimizers.set_LBFGS_options(maxiter=1)
                                _sm2.compile("L-BFGS", loss=_sc2.get("loss_type","{config.loss_type}"))
                            else:
                                _sm2.compile("adam", lr=0.001, loss=_sc2.get("loss_type","{config.loss_type}"))
                            _sm2.restore(_spt2, verbose=0)
                            _u_pinn_grid = _sm2.predict(_xyt_grid)[:, 0].reshape(_res_ea, _res_ea)
                        else:
                            _u_pinn_grid = np.zeros((_res_ea, _res_ea))
                    else:
                        _u_pinn_grid = np.zeros((_res_ea, _res_ea))
                    _u_fem_grid = _gd(np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei]]),
                                      _ea_u_refs[_ei], (_Xg_ea, _Yg_ea), method='linear', fill_value=0.0)
                    _u_err_grid = np.abs(_u_pinn_grid - _u_fem_grid)
                    _vmin_ea = min(_u_pinn_grid.min(), _u_fem_grid.min())
                    _vmax_ea = max(_u_pinn_grid.max(), _u_fem_grid.max())
                    im0 = axes[_ei][0].contourf(_Xg_ea, _Yg_ea, _u_pinn_grid, levels={config.plot_levels}, cmap='{config.plot_colormap}', vmin=_vmin_ea, vmax=_vmax_ea)
                    axes[_ei][0].set_title(f"PINN  t={{_ea_tv:.3f}}  L2={{_l2:.2e}}", fontsize=10)
                    axes[_ei][0].set_xlabel("x"); axes[_ei][0].set_ylabel("y")
                    fig.colorbar(im0, ax=axes[_ei][0])
                    im1 = axes[_ei][1].contourf(_Xg_ea, _Yg_ea, _u_fem_grid, levels={config.plot_levels}, cmap='{config.plot_colormap}', vmin=_vmin_ea, vmax=_vmax_ea)
                    axes[_ei][1].set_title(f"Ground Truth  t={{_ea_tv:.3f}}", fontsize=10)
                    axes[_ei][1].set_xlabel("x"); axes[_ei][1].set_ylabel("y")
                    fig.colorbar(im1, ax=axes[_ei][1])
                    im2 = axes[_ei][2].contourf(_Xg_ea, _Yg_ea, _u_err_grid, levels={config.plot_levels}, cmap='YlOrRd')
                    axes[_ei][2].set_title(f"|Error|  t={{_ea_tv:.3f}}  Max={{_mx:.2e}}", fontsize=10)
                    axes[_ei][2].set_xlabel("x"); axes[_ei][2].set_ylabel("y")
                    fig.colorbar(im2, ax=axes[_ei][2])
                plt.tight_layout()
            else:
                _ea_x_common = np.linspace({config.x_min}, {config.x_max}, 300)
                _ea_t_arr = np.array(_ea_times)
                _ea_U_pinn = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                _ea_U_fem  = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                for _ei in range(_ea_n_t):
                    _ea_fi_p = _interp1d(_ea_x_refs[_ei], _ea_u_pinns[_ei], kind='linear', fill_value='extrapolate')
                    _ea_U_pinn[_ei, :] = _ea_fi_p(_ea_x_common)
                    _ea_fi_f = _interp1d(_ea_x_refs[_ei], _ea_u_refs[_ei], kind='linear', fill_value='extrapolate')
                    _ea_U_fem[_ei, :] = _ea_fi_f(_ea_x_common)
                _ea_Xg, _ea_Tg = np.meshgrid(_ea_x_common, _ea_t_arr)
                _ea_U_err = np.abs(_ea_U_pinn - _ea_U_fem)
                _ea_vmin = min(_ea_U_pinn.min(), _ea_U_fem.min())
                _ea_vmax = max(_ea_U_pinn.max(), _ea_U_fem.max())
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle("PINN vs Ground Truth — Surface Comparison", fontsize=13, fontweight='bold')
                im0 = axes[0].contourf(_ea_Tg, _ea_Xg, _ea_U_pinn, levels={config.plot_levels}, cmap='{config.plot_colormap}', vmin=_ea_vmin, vmax=_ea_vmax)
                axes[0].set_title("PINN  u(x,t)"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
                fig.colorbar(im0, ax=axes[0])
                im1 = axes[1].contourf(_ea_Tg, _ea_Xg, _ea_U_fem, levels={config.plot_levels}, cmap='{config.plot_colormap}', vmin=_ea_vmin, vmax=_ea_vmax)
                axes[1].set_title("Ground Truth  u(x,t)"); axes[1].set_xlabel("t"); axes[1].set_ylabel("x")
                fig.colorbar(im1, ax=axes[1])
                im2 = axes[2].contourf(_ea_Tg, _ea_Xg, _ea_U_err, levels={config.plot_levels}, cmap='YlOrRd')
                axes[2].set_title("Error  |PINN - Ground Truth|"); axes[2].set_xlabel("t"); axes[2].set_ylabel("x")
                fig.colorbar(im2, ax=axes[2])
                plt.tight_layout()
            _ea_sp = _os.path.join(_ea_dir, "surface_comparison.png")
            plt.savefig(_ea_sp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            print(f"  Surface comparison saved: {{_ea_sp}}")

        print("=== Time-Adaptive Error Analysis Complete ===")

print("DONE")
"""
    return script

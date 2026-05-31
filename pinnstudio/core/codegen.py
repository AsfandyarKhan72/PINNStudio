import os
os.environ["DDE_BACKEND"] = "pytorch"

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

    script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*cuBLAS.*")

# ── Force GPU initialization ──────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Warm up CUDA context with a dummy forward+backward pass
    _dummy = torch.zeros(10, 10, requires_grad=True, device='cuda')
    _loss = (_dummy ** 2).sum()
    _loss.backward()
    torch.cuda.synchronize()
    del _dummy, _loss
    print(f"✅ GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"   Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")
    print(f"   TF32 and cuDNN benchmark enabled")
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
    with open(_os.path.join(_save_dir, "model_config.json"), "w") as _mcf:
        _json.dump(_model_config, _mcf, indent=2)
    print(f"Model config saved to: {{_os.path.join(_save_dir, 'model_config.json')}}")
_loss_path     = _os.path.join(_save_dir, "loss_plot.png")     if _use_save else "/tmp/loss_plot.png"
_solution_path = _os.path.join(_save_dir, "solution_plot.png") if _use_save else "/tmp/solution_plot.png"
_log_path      = _os.path.join(_save_dir, "training_log.txt")  if _use_save else None

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
            self._step = 0
            self._offset = 0
        def set_offset(self, offset):
            self._offset = offset
            self._step = 0
        def on_batch_end(self):
            self._step += 1
            actual_iter = self._offset + self._step
            if actual_iter % self.period == 0:
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                print(f"  [{config.inverse_param_name}] Iter {{actual_iter}}: {{val:.6f}}", flush=True)

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
            self._step = 0
            self._offset = 0
        def set_offset(self, offset):
            self._offset = offset
            self._step = 0
        def on_batch_end(self):
            if self.period == 0: return
            self._step += 1
            actual_iter = self._offset + self._step
            if actual_iter % self.period == 0:
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                with open(self.path, "a") as _f:
                    _f.write(f"{{actual_iter}},{{val:.8f}}\\n")

    _save_cb = _SaveParamCallback(
        {config.inverse_param_name},
        "{config.inverse_param_name}",
        _param_save_period,
        _param_save_path if _param_save_path else "/tmp/param_save.txt"
    )


# ── PDE definition ──────────────────────────────────────────
def pde(x, y):
    _n_out = {config.num_outputs}
    _out_names = [n.strip() for n in "{config.output_names}".split(",")]
    _is_2d_pde = "{config.problem_dim}" == "2D"

    _dvars = {{}}
    for _oi, _oname in enumerate(_out_names):
        _dvars[_oname] = y[:, _oi:_oi+1]
        if _is_2d_pde:
            # 2D: inputs are (x, y, t) → j=0,1,2
            _dvars[f"d{{_oname}}_x"]  = dde.grad.jacobian(y, x, i=_oi, j=0)
            _dvars[f"d{{_oname}}_y"]  = dde.grad.jacobian(y, x, i=_oi, j=1)
            _dvars[f"d{{_oname}}_t"]  = dde.grad.jacobian(y, x, i=_oi, j=2)
            _dvars[f"d{{_oname}}_xx"] = dde.grad.hessian(y, x, component=_oi, i=0, j=0)
            _dvars[f"d{{_oname}}_yy"] = dde.grad.hessian(y, x, component=_oi, i=1, j=1)
            _dvars[f"d{{_oname}}_xy"] = dde.grad.hessian(y, x, component=_oi, i=0, j=1)
            _dvars[f"d{{_oname}}_tt"] = dde.grad.hessian(y, x, component=_oi, i=2, j=2)
            _dvars[f"d{{_oname}}_xt"] = dde.grad.hessian(y, x, component=_oi, i=0, j=2)
            _dvars[f"d{{_oname}}_yt"] = dde.grad.hessian(y, x, component=_oi, i=1, j=2)
            _dvars[f"d{{_oname}}_xxxx"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            _dvars[f"d{{_oname}}_yyyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=1, j=1)
            _dvars[f"d{{_oname}}_xxyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=1, j=1)
            _dvars[f"d{{_oname}}_xxtt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=2, j=2)
            _dvars[f"d{{_oname}}_yytt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=2, j=2)
        else:
            # 1D: inputs are (x, t) → j=0,1
            _dvars[f"d{{_oname}}_x"]  = dde.grad.jacobian(y, x, i=_oi, j=0)
            _dvars[f"d{{_oname}}_t"]  = dde.grad.jacobian(y, x, i=_oi, j=1)
            _dvars[f"d{{_oname}}_xx"] = dde.grad.hessian(y, x, component=_oi, i=0, j=0)
            _dvars[f"d{{_oname}}_tt"] = dde.grad.hessian(y, x, component=_oi, i=1, j=1)
            _dvars[f"d{{_oname}}_xt"] = dde.grad.hessian(y, x, component=_oi, i=0, j=1)
            _dvars[f"d{{_oname}}_xxxx"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            _dvars[f"d{{_oname}}_xxtt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=1, j=1)
            _dvars[f"d{{_oname}}_tttt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_tt"], x, i=1, j=1)

    _eval_ns = {{**globals(), **_dvars}}
    _eval_ns["dde"] = dde
    _eval_ns["np"]  = np
    _eval_ns["x"]   = x
    _eval_ns["y"]   = y

    if _problem_type == "Inverse":
        _eval_ns["{config.inverse_param_name}"] = {config.inverse_param_name}

    if _n_out == 1:
        if _is_2d_pde:
            u     = _dvars.get("u",     y[:, 0:1])
            du_x  = _dvars.get("du_x",  dde.grad.jacobian(y, x, i=0, j=0))
            du_y  = _dvars.get("du_y",  dde.grad.jacobian(y, x, i=0, j=1))
            du_t  = _dvars.get("du_t",  dde.grad.jacobian(y, x, i=0, j=2))
            du_xx = _dvars.get("du_xx", dde.grad.hessian(y, x, component=0, i=0, j=0))
            du_yy = _dvars.get("du_yy", dde.grad.hessian(y, x, component=0, i=1, j=1))
            du_xy = _dvars.get("du_xy", dde.grad.hessian(y, x, component=0, i=0, j=1))
            du_tt = _dvars.get("du_tt", dde.grad.hessian(y, x, component=0, i=2, j=2))
            _eval_ns.update({{
                "u": u, "du_x": du_x, "du_y": du_y, "du_t": du_t,
                "du_xx": du_xx, "du_yy": du_yy, "du_xy": du_xy, "du_tt": du_tt
            }})
        else:
            u     = _dvars.get("u",     y[:, 0:1])
            du_x  = _dvars.get("du_x",  dde.grad.jacobian(y, x, i=0, j=0))
            du_t  = _dvars.get("du_t",  dde.grad.jacobian(y, x, i=0, j=1))
            du_xx = _dvars.get("du_xx", dde.grad.hessian(y, x, component=0, i=0, j=0))
            du_tt = _dvars.get("du_tt", dde.grad.hessian(y, x, component=0, i=1, j=1))
            du_xt = _dvars.get("du_xt", dde.grad.hessian(y, x, component=0, i=0, j=1))
            _eval_ns.update({{
                "u": u, "du_x": du_x, "du_t": du_t,
                "du_xx": du_xx, "du_tt": du_tt, "du_xt": du_xt
            }})
        return eval("{pde_expr_single}", _eval_ns)
    else:
        _pde_exprs = "{config_pde_expressions}".split("|")
        return [eval(_expr.strip(), _eval_ns) for _expr in _pde_exprs]

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
        if _oi < len(_ic_active_list) and _ic_active_list[_oi].strip() == "True":
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
    if _is_2d:
        if _oi_w < len(_bc_ba) and _bc_ba[_oi_w].strip() == "True":
            _bcb_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _bcb_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0
        _btt_check = "{config.bc_top_types}".split(",")
        _bbt_check = "{config.bc_bottom_types}".split(",")
        _btt_is_periodic = _oi_w < len(_btt_check) and _btt_check[_oi_w].strip() == "Periodic"
        _bbt_is_periodic = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"
        if _oi_w < len(_bc_ta) and _bc_ta[_oi_w].strip() == "True" and not _btt_is_periodic and not _bbt_is_periodic:
            _bct_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _bct_w.append(None)  # do NOT advance _wi — GUI sends no value for this slot
    else:
        _bcb_w.append(None); _bct_w.append(None)
    if _oi_w < len(_ic_a) and _ic_a[_oi_w].strip() == "True":
        _ic_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
    else:
        _ic_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0

_multi_weights = []
for _oi_w in range(_n_out_w):
    _multi_weights.append(_pde_w[_oi_w])
for _oi_w in range(_n_out_w):
    if _bcl_w[_oi_w] is not None: _multi_weights.append(_bcl_w[_oi_w])
    if _bcr_w[_oi_w] is not None: _multi_weights.append(_bcr_w[_oi_w])
    if _bcb_w[_oi_w] is not None: _multi_weights.append(_bcb_w[_oi_w])
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
        train_distribution="{config.point_distribution}"
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

    _iters = int(_pval) if (_parametric and _param_name == "phase1_iterations" and _pval is not None) else {config.iterations}

    if not {config.time_adaptive}:
        if _problem_type == "Inverse":
            with open("/tmp/param_history.txt", "w") as _f:
                pass
            _var_cb = dde.callbacks.VariableValue(
                [{config.inverse_param_name}], period=1000, filename="/tmp/param_history.txt",
                precision=6
            )
            loss_history, train_state = model.train(iterations=_iters, display_every=1000, callbacks=[_var_cb, _print_cb, _save_cb])
        else:
            loss_history, train_state = model.train(iterations=_iters, display_every=1000)
            if _use_save:
                _adam_model_path = _os.path.join(_save_dir, "model_adam")
                model.save(_adam_model_path)
                _adam_cfg_path = _os.path.join(_save_dir, f"model_adam-{{_iters}}.json")
                with open(_adam_cfg_path, "w") as _acf:
                    _json.dump(_model_config, _acf, indent=2)
                print(f"Adam config saved to: {{_adam_cfg_path}}")

        # Phase 2
        if "{config.optimizer2}" != "none":
            _phase2_weights = _multi_weights
            if "{config.optimizer2}" == "lbfgs":
                if {config.lbfgs_use_default}:
                    dde.optimizers.set_LBFGS_options(maxiter={config.iterations2})
                else:
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter={config.lbfgs_maxiter},
                        maxfun={config.lbfgs_maxfun}, maxls={config.lbfgs_maxls}
                    )
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
                    if _use_save:
                        _lbfgs_model_path = _os.path.join(_save_dir, "model_lbfgs")
                        model.save(_lbfgs_model_path)
                        _lbfgs_iter = loss_history.steps[-1] if loss_history.steps else _iters
                        _lbfgs_cfg_path = _os.path.join(_save_dir, f"model_lbfgs-{{_lbfgs_iter}}.json")
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
                if {config.lbfgs_use_default}:
                    dde.optimizers.set_LBFGS_options(maxiter={config.rar_lbfgs_iters})
                else:
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter={config.lbfgs_maxiter},
                        maxfun={config.lbfgs_maxfun}, maxls={config.lbfgs_maxls}
                    )
                model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_multi_weights)
                loss_history, train_state = model.train(display_every=200)
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
            # 2D: heatmap at multiple time snapshots
            _n_snaps = min(4, {config.num_timesteps})
            _t_snaps = np.linspace({config.t_min}, {config.t_max}, _n_snaps)
            _xp = np.linspace({config.x_min}, {config.x_max}, 80)
            _yp = np.linspace({config.y_min}, {config.y_max}, 80)
            _Xg, _Yg = np.meshgrid(_xp, _yp)

            fig, axes = plt.subplots(1, _n_snaps, figsize=(4*_n_snaps, 4))
            if _n_snaps == 1:
                axes = [axes]
            for _ai, _tv in enumerate(t_snaps if False else _t_snaps):
                _XYT = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _tv)])
                _pred = model.predict(_XYT)[:, _plot_idx].reshape(80, 80)
                im = axes[_ai].contourf(_Xg, _Yg, _pred, levels=40, cmap="RdBu_r")
                axes[_ai].set_title(f"t = {{_tv:.3f}}")
                axes[_ai].set_xlabel("x"); axes[_ai].set_ylabel("y")
                fig.colorbar(im, ax=axes[_ai])
            out_name = "{config.output_names}".split(",")[_plot_idx].strip()
            fig.suptitle(f"PINN Solution — {{out_name}}(x,y,t)", fontsize=12)
            plt.tight_layout()
            plt.savefig(_run_solution_path, dpi=100); plt.close()
        else:
            # 1D plot
            x_vals      = np.linspace({config.x_min}, {config.x_max}, 100)
            t_vals_plot = np.linspace({config.t_min}, {config.t_max}, 100)
            X, T = np.meshgrid(x_vals, t_vals_plot)
            XT   = np.vstack([X.ravel(), T.ravel()]).T
            _pred_all = model.predict(XT)
            u_pred = _pred_all[:, _plot_idx].reshape(100, 100)

            _plot_type = "{config.plot_type}"
            if _plot_type == "Surface":
                plt.figure(figsize=(7, 5))
                plt.contourf(X, T, u_pred, levels=50, cmap="RdBu_r")
                plt.colorbar()
                plt.xlabel("x"); plt.ylabel("t")
                plt.title(f"PINN Solution — {{_param_name}}={{_pval}}" if _parametric else "PINN Solution")
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi=100); plt.close()
            elif _plot_type.startswith("Line"):
                n_steps_plot = {config.num_timesteps}
                t_steps_plot = np.linspace({config.t_min}, {config.t_max}, n_steps_plot)
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.viridis(np.linspace(0, 1, n_steps_plot))
                for i, t_val in enumerate(t_steps_plot):
                    xt = np.column_stack([x_vals, np.full_like(x_vals, t_val)])
                    u_line = model.predict(xt)[:, _plot_idx].flatten()
                    ax.plot(x_vals, u_line, color=colors[i], label=f"t = {{t_val:.3f}}")
                ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
                ax.set_title(f"PINN Solution — {{_param_name}}={{_pval}}" if _parametric else "PINN Solution")
                ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi=100); plt.close()

        import shutil as _shutil
        if _run_loss_path != "/tmp/loss_plot.png":
            _shutil.copy(_run_loss_path, "/tmp/loss_plot.png")
        if _run_solution_path != "/tmp/solution_plot.png":
            _shutil.copy(_run_solution_path, "/tmp/solution_plot.png")

        # ── Export solution data ──────────────────────────────
        if _problem_type != "Inverse":
            _data_dir = _os.path.join(
                _run_dir if (_parametric and _pval is not None)
                else (_save_dir if _use_save else "/tmp"),
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

    n_steps   = {config.ta_num_steps}
    grid_size = {config.ta_grid_size}
    t_start   = {config.t_min}
    t_end     = {config.t_max}
    dt        = (t_end - t_start) / n_steps
    x_grid    = np.linspace({config.x_min}, {config.x_max}, grid_size)

    all_x = []; all_t = []; all_u = []

    x      = x_grid.reshape(-1, 1)
    prev_u = np.reshape({ta_ic_expr}, (-1, 1))

    for step_i in range(n_steps):
        t0 = t_start + step_i * dt
        t1 = t0 + dt
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
                if _oi_ta < len(_ic_active_list) and _ic_active_list[_oi_ta].strip() == "True":
                    _ic_expr_ta = _ic_expressions[_oi_ta].strip() if _oi_ta < len(_ic_expressions) else "np.zeros_like(x[:,0])"
                    def _mk_ic_ta(expr, comp):
                        def _ic_fn(x):
                            return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                        return dde.icbc.IC(geomtime_i, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                    _constraints_i.append(_mk_ic_ta(_ic_expr_ta, _comp_ta))
            else:
                if _oi_ta == 0:
                    xt_ic = np.column_stack([x_grid, np.full_like(x_grid, t0)])
                    _constraints_i.append(dde.icbc.PointSetBC(xt_ic, prev_u, component=0))

        data_i = dde.data.TimePDE(
            geomtime_i, pde, _constraints_i,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_initial={config.num_initial}, num_test={config.num_test}
        )

        net_i   = dde.nn.FNN({config.layers}, "{config.activation}", "Glorot uniform")
        model_i = dde.Model(data_i, net_i)
        model_i.compile("{config.optimizer}", lr=_lr, loss="{config.loss_type}", loss_weights=_multi_weights)
        lh_i, ts_i = model_i.train(iterations={config.iterations}, display_every=1000)

        if "{config.optimizer2}" != "none":
            if {config.lbfgs_use_default}:
                dde.optimizers.set_LBFGS_options(maxiter={config.iterations2})
            else:
                dde.optimizers.set_LBFGS_options(
                    maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                    gtol={config.lbfgs_gtol}, maxiter={config.lbfgs_maxiter},
                    maxfun={config.lbfgs_maxfun}, maxls={config.lbfgs_maxls}
                )
            model_i.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_multi_weights)
            lh_i, ts_i = model_i.train(display_every=200)

        x_pred  = np.linspace({config.x_min}, {config.x_max}, grid_size)
        t_pred  = np.full_like(x_pred, t1)
        xt_pred = np.column_stack([x_pred, t_pred])
        prev_u  = model_i.predict(xt_pred)[:, 0:1]

        x_plot = np.linspace({config.x_min}, {config.x_max}, 100)
        t_plot = np.linspace(t0, t1, 50)
        Xp, Tp = np.meshgrid(x_plot, t_plot)
        XTp    = np.vstack([Xp.ravel(), Tp.ravel()]).T
        Up     = model_i.predict(XTp)[:, {config.plot_output_idx}].reshape(50, 100)
        all_x.append(Xp); all_t.append(Tp); all_u.append(Up)

        print(f"Step {{step_i+1}} done. Final train loss: {{sum(lh_i.loss_train[-1]):.4e}}")

        # ── Save step plot ────────────────────────────────────
        if _use_save:
            _step_dir = _os.path.join(_save_dir, "time_adaptive_steps")
            _os.makedirs(_step_dir, exist_ok=True)
            _plot_type_step = "{config.plot_type}"
            _x_plot_step = np.linspace({config.x_min}, {config.x_max}, 100)
            _t_plot_step = np.linspace(t0, t1, 50)
            _Xp_step, _Tp_step = np.meshgrid(_x_plot_step, _t_plot_step)
            _XTp_step = np.vstack([_Xp_step.ravel(), _Tp_step.ravel()]).T
            _Up_step = model_i.predict(_XTp_step)[:, {config.plot_output_idx}].reshape(50, 100)

            if _plot_type_step == "Surface" or _plot_type_step.startswith("📊"):
                plt.figure(figsize=(7, 4))
                plt.contourf(_Xp_step, _Tp_step, _Up_step, levels=50, cmap="RdBu_r")
                plt.colorbar(); plt.xlabel("x"); plt.ylabel("t")
                plt.title(f"Step {{step_i+1}}: t = {{t0:.4f}} → {{t1:.4f}}")
                plt.tight_layout()
                _step_fname = _os.path.join(_step_dir, f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}.png")
                plt.savefig(_step_fname, dpi=100); plt.close()
            elif _plot_type_step.startswith("Line"):
                n_steps_plot = {config.num_timesteps}
                _t_line = np.linspace(t0, t1, n_steps_plot)
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = plt.cm.viridis(np.linspace(0, 1, n_steps_plot))
                for _ci, _tv in enumerate(_t_line):
                    _xt_line = np.column_stack([_x_plot_step, np.full_like(_x_plot_step, _tv)])
                    _u_line = model_i.predict(_xt_line)[:, {config.plot_output_idx}].flatten()
                    ax.plot(_x_plot_step, _u_line, color=colors[_ci], label=f"t={{_tv:.3f}}")
                ax.set_xlabel("x"); ax.set_ylabel("u")
                ax.set_title(f"Step {{step_i+1}}: t = {{t0:.4f}} → {{t1:.4f}}")
                ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.2)
                plt.tight_layout()
                _step_fname = _os.path.join(_step_dir, f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}.png")
                plt.savefig(_step_fname, dpi=100); plt.close()
            print(f"Step plot saved: {{_step_fname}}")

    X_full = np.vstack(all_x); T_full = np.vstack(all_t); U_full = np.vstack(all_u)
    print("\\n=== Time-Adaptive Training Complete ===")

    _ta_solution_path = _os.path.join(_save_dir, "solution_plot.png") if _use_save else "/tmp/solution_plot.png"
    _ta_loss_path     = _os.path.join(_save_dir, "loss_plot.png")     if _use_save else "/tmp/loss_plot.png"

    _plot_type_ta = "{config.plot_type}"
    if _plot_type_ta == "Surface":
        plt.figure(figsize=(7, 5))
        plt.contourf(X_full, T_full, U_full, levels=50, cmap="RdBu_r")
        plt.colorbar(); plt.xlabel("x"); plt.ylabel("t")
        plt.title("Time-Adaptive PINN Solution")
        plt.tight_layout(); plt.savefig(_ta_solution_path, dpi=100); plt.close()
    elif _plot_type_ta.startswith("Line"):
        n_ts   = {config.num_timesteps}
        t_vals = np.linspace(t_start, t_end, n_ts)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors  = plt.cm.viridis(np.linspace(0, 1, n_ts))
        for ci, tv in enumerate(t_vals):
            idx = np.argmin(np.abs(T_full[:,0] - tv))
            ax.plot(X_full[idx,:], U_full[idx,:], color=colors[ci], label=f"t={{tv:.3f}}")
        ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
        ax.set_title("Time-Adaptive PINN — Line Plot")
        ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
        plt.tight_layout(); plt.savefig(_ta_solution_path, dpi=100); plt.close()

    train_loss_ta = lh_i.loss_train; test_loss_ta = lh_i.loss_test
    steps_ta = list(range(len(train_loss_ta)))
    plt.figure(figsize=(6, 4))
    plt.semilogy(steps_ta, [sum(l) for l in train_loss_ta], label="Train", color="#4dabf7")
    plt.semilogy(steps_ta, [sum(l) for l in test_loss_ta],  label="Test",  color="#ff8787", linestyle="--")
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.title("Loss — Last Time Sub-domain")
    plt.legend(); plt.tight_layout()
    plt.savefig(_ta_loss_path, dpi=100); plt.close()

print("DONE")
"""
    return script
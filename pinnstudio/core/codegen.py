import os
os.environ["DDE_BACKEND"] = "pytorch"
# ── CUDA performance environment variables ────────────────────
# (No TORCH_CUDA_ARCH_LIST here -- that used to be pinned to a specific
# GPU model (RTX 4090 / sm_89) on the original dev machine, which only
# matters if something JIT-compiles a CUDA extension from source, but
# would silently restrict that compile to sm_89 on anyone else's card.
# Leaving it unset lets PyTorch/DeepXDE target whatever GPU is actually
# present, same as it already does for everyone via the ordinary prebuilt
# CUDA wheels install.sh/select_torch_index.py picks for their driver.)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"        # async CUDA launches
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

def _simplify_expr(expr, is_2d=False, is_3d=False):
    """Convert user-friendly math syntax to numpy syntax."""
    import re
    e = expr.strip()
    # Replace math functions with numpy equivalents
    for fn in ["sin","cos","tan","sinh","cosh","tanh","arcsin","arccos","arctan",
               "exp","log","log10","sqrt","abs","ceil","floor"]:
        e = re.sub(rf'\b{fn}\(', f'np.{fn}(', e)
    # Replace the pi constant. (There used to be a similar line here for a
    # bare Euler's-number "e" constant, but it was written as \bexp\b
    # instead of \be\b -- since \b is a boundary between a word character
    # and a non-word character, \bexp\b matches "exp" immediately before
    # "(" too (a word/non-word boundary either way), so it silently
    # clobbered every use of the exp() function into "np.e(", right after
    # the loop above had already correctly turned it into "np.exp(" --
    # e.g. "exp(-20*x)" -> "np.exp(-20*x)" -> "np.np.e(-20*x)". No shipped
    # template used exp() before this patch's Diffusion-Reaction and 2D
    # Burgers examples, so it went unnoticed. Removed rather than fixed to
    # \be\b, matching _simplify_pde_expr in main_window.py (used for the
    # same user-facing "type math like you'd write it" convention
    # elsewhere), which never had a bare-e substitution to begin with.)
    e = re.sub(r'\bpi\b', 'np.pi', e)
    # Replace x, y, z variables — must be done carefully to avoid replacing
    # inside words
    if is_3d:
        e = re.sub(r'\bz\b', '__Z__', e)
        e = re.sub(r'\by\b', '__Y__', e)
        e = re.sub(r'\bx\b', 'x[:, 0]', e)
        e = e.replace('__Y__', 'x[:, 1]')
        e = e.replace('__Z__', 'x[:, 2]')
    elif is_2d:
        e = re.sub(r'\by\b', '__Y__', e)
        e = re.sub(r'\bx\b', 'x[:, 0]', e)
        e = e.replace('__Y__', 'x[:, 1]')
    else:
        e = re.sub(r'\bx\b', 'x[:, 0]', e)
    return e

def _parse_vertex_list(text):
    """'x1,y1;x2,y2;...' -> [[x1,y1], [x2,y2], ...]. Skips malformed pairs --
    mirrors MainWindow._parse_vertices() exactly, so a Triangle/Polygon
    trains on the same vertices the domain preview already draws."""
    verts = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            continue
        try:
            verts.append([float(parts[0].strip()), float(parts[1].strip())])
        except ValueError:
            continue
    return verts

_CUSTOM_GEOM_OP_CTORS = {"union": "CSGUnion", "subtract": "CSGDifference", "intersect": "CSGIntersection"}
_CUSTOM_GEOM_BUGGY_LEAF_TYPES = ("Disk", "Ellipse", "Triangle", "Polygon", "Sphere")


def _custom_geom_leaf_literal(shape_type, params):
    """One Custom-geometry leaf shape -> its dde.geometry constructor call,
    as a literal string, from the params dict the GUI's
    _custom_geom_row_to_dict() produces (see config.py's
    geom_custom_shapes_json docs). Mirrors the per-type literal building
    the other geometry types already do above (_triangle_vertices_literal
    etc.) and in _clean_geom_line() below -- falls back to a sane default
    for a missing/malformed param rather than raising, since codegen must
    never crash on a stale or hand-edited config."""
    params = params or {}
    if shape_type == "Disk":
        cx = params.get("cx", 0.5); cy = params.get("cy", 0.5); r = params.get("r", 0.5)
        return f"dde.geometry.Disk([{cx}, {cy}], {r})"
    if shape_type == "Ellipse":
        cx = params.get("cx", 0.5); cy = params.get("cy", 0.5)
        a = params.get("a", 0.5); b = params.get("b", 0.3); angle = params.get("angle", 0.0)
        return f"dde.geometry.Ellipse([{cx}, {cy}], {a}, {b}, {angle})"
    if shape_type == "Triangle":
        verts = _parse_vertex_list(params.get("vertices_text", ""))
        if len(verts) != 3:
            verts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        return f"dde.geometry.Triangle({verts[0]}, {verts[1]}, {verts[2]})"
    if shape_type == "Polygon":
        verts = _parse_vertex_list(params.get("vertices_text", ""))
        if len(verts) < 3:
            verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        return f"dde.geometry.Polygon({verts})"
    if shape_type == "Cuboid":
        x_min = params.get("x_min", 0.0); x_max = params.get("x_max", 1.0)
        y_min = params.get("y_min", 0.0); y_max = params.get("y_max", 1.0)
        z_min = params.get("z_min", 0.0); z_max = params.get("z_max", 1.0)
        if x_min >= x_max:
            x_min, x_max = 0.0, 1.0
        if y_min >= y_max:
            y_min, y_max = 0.0, 1.0
        if z_min >= z_max:
            z_min, z_max = 0.0, 1.0
        return f"dde.geometry.Cuboid([{x_min}, {y_min}, {z_min}], [{x_max}, {y_max}, {z_max}])"
    if shape_type == "Sphere":
        cx = params.get("cx", 0.5); cy = params.get("cy", 0.5); cz = params.get("cz", 0.5)
        r = params.get("r", 0.5)
        return f"dde.geometry.Sphere([{cx}, {cy}, {cz}], {r})"
    # Rectangle, and any unrecognized/missing type -- fall back to a unit
    # square Rectangle rather than crashing codegen over a stale entry.
    x_min = params.get("x_min", 0.0); x_max = params.get("x_max", 1.0)
    y_min = params.get("y_min", 0.0); y_max = params.get("y_max", 1.0)
    if x_min >= x_max:
        x_min, x_max = 0.0, 1.0
    if y_min >= y_max:
        y_min, y_max = 0.0, 1.0
    return f"dde.geometry.Rectangle([{x_min}, {y_min}], [{x_max}, {y_max}])"


def _build_custom_geom_code(config):
    """Custom geometry: config.geom_custom_shapes_json -> (a dde.geometry
    constructor call chaining CSGUnion/CSGDifference/CSGIntersection in
    build order, wrap_needed). Shared by generate_script()'s _build_geom()
    template and generate_clean_script()'s _clean_geom_line() below --
    the same two call sites every other geometry type already has its own
    literal-building logic at (see _triangle_vertices_literal above vs
    tri_verts in _clean_geom_line). Falls back to a unit-square Rectangle
    (2D) or unit cube Cuboid (3D, per config.problem_dim) if the shape
    list is empty or unparseable, matching every other shape's
    fallback-to-sane-default philosophy in this file.

    wrap_needed mirrors _DTypeSafeGeom's existing single-shape rule
    (Disk/Ellipse/Triangle/Polygon/Sphere get wrapped, Rectangle/Cuboid
    don't): a CSG combination of 2+ shapes never needs it -- verified
    empirically against the installed DeepXDE version, in both 2D and 3D,
    CSGUnion/CSGDifference/CSGIntersection always normalize their sampled
    points to the network's own configured float dtype regardless of
    which leaf types feed into them, even when every leaf is individually
    "buggy" -- so wrapping is only applied for the degenerate single-shape
    Custom case, exactly like a plain (non-Custom) Disk/Ellipse/Triangle/
    Polygon/Sphere already gets."""
    import json as _json_cg
    try:
        entries = _json_cg.loads(config.geom_custom_shapes_json or "[]")
    except (ValueError, TypeError):
        entries = []
    if not isinstance(entries, list):
        entries = []
    entries = [e for e in entries if isinstance(e, dict)]
    if not entries:
        if getattr(config, "problem_dim", "2D") == "3D":
            entries = [{"type": "Cuboid", "params": {
                "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}}]
        else:
            entries = [{"type": "Rectangle", "params": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}}]

    code = None
    for entry in entries:
        leaf = _custom_geom_leaf_literal(entry.get("type"), entry.get("params"))
        if code is None:
            code = leaf
        else:
            ctor = _CUSTOM_GEOM_OP_CTORS.get(entry.get("op") or "union", "CSGUnion")
            code = f"dde.geometry.{ctor}({code}, {leaf})"

    wrap_needed = len(entries) == 1 and entries[0].get("type") in _CUSTOM_GEOM_BUGGY_LEAF_TYPES
    return code, wrap_needed


def _simplify_pde_expr(expr):
    """Convert user-friendly math in PDE — only functions and constants, not variables."""
    import re
    e = expr.strip()
    for fn in ["sin","cos","tan","sinh","cosh","tanh","arcsin","arccos","arctan",
               "exp","log","log10","sqrt","abs","ceil","floor"]:
        e = re.sub(rf'(?<![a-zA-Z_]){fn}\(', f'np.{fn}(', e)
    e = re.sub(r'(?<![a-zA-Z_])pi(?![a-zA-Z_])', 'np.pi', e)
    return e

def _sanitize_python_identifier(name, fallback):
    """Coerce a free-typed string into a legal Python identifier. Inverse
    trainable-variable names come straight from a QLineEdit with no
    character restriction, but get spliced directly into generated code
    as a bare variable name (`{name} = dde.Variable(...)`), not inside a
    string literal -- so the repr()-escaping used for other free-text
    fields elsewhere in this file doesn't apply here; a name like
    "diffusion coefficient" would otherwise produce a SyntaxError in the
    generated script. An already-valid identifier (which covers every
    built-in template and any already-well-formed custom name) is
    returned completely unchanged -- this only rewrites the ones that
    would otherwise break."""
    import re as _re_id
    import keyword as _keyword_id
    _name = (name or "").strip()
    if _name.isidentifier() and not _keyword_id.iskeyword(_name):
        return _name
    _clean = _re_id.sub(r'[^A-Za-z0-9_]', '_', _name).strip('_')
    if not _clean:
        _clean = fallback
    elif _clean[0].isdigit():
        _clean = "_" + _clean
    if _keyword_id.iskeyword(_clean):
        _clean = _clean + "_var"
    return _clean


def _parse_inverse_variables(config):
    """Parse config.inverse_variables_json into an ordered list of
    (name, init, true) triples, one per trainable variable (variable 1
    first). `true` is the known ground-truth value for a built-in template
    whose PDE constant was substituted for this variable (see
    main_window.py's INVERSE_AUTO_VARS) -- None when not known (a manually
    added variable, or a config saved before this field existed), in which
    case downstream code (the parameter-convergence plot) simply skips
    drawing a true-value reference line for that variable. Falls back to a
    single entry built from the legacy inverse_param_name/inverse_param_init
    fields when the JSON is empty or unparseable -- covers configs saved
    before this feature existed, and stays perfectly single-variable-
    compatible: one variable named inverse_param_name, true=None, in that
    case."""
    import json
    raw = getattr(config, "inverse_variables_json", "") or ""
    variables = []
    if raw:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = []
        for i, v in enumerate(parsed or []):
            name = str((v or {}).get("name") or f"trainable_variable_{i + 1}").strip()
            if not name:
                name = f"trainable_variable_{i + 1}"
            name = _sanitize_python_identifier(name, f"trainable_variable_{i + 1}")
            try:
                init = float((v or {}).get("init", 1.0))
            except (TypeError, ValueError):
                init = 1.0
            _true_raw = (v or {}).get("true", None)
            try:
                true = float(_true_raw) if _true_raw is not None else None
            except (TypeError, ValueError):
                true = None
            variables.append((name, init, true))
    if not variables:
        variables.append((
            _sanitize_python_identifier(config.inverse_param_name, "trainable_variable_1"),
            config.inverse_param_init, None))

    # Two different user-typed names can sanitize to the same identifier
    # (e.g. "D 1" and "D_1" both become "D_1") -- de-duplicate so the
    # generated script never defines the same variable name twice.
    _seen = {}
    _deduped = []
    for name, init, true in variables:
        if name in _seen:
            _seen[name] += 1
            name = f"{name}_{_seen[name]}"
        else:
            _seen[name] = 0
        _deduped.append((name, init, true))
    return _deduped

def _parse_inverse_obs_files(config):
    """Parse config.inverse_obs_files_json into an ordered list of
    {"path", "output_idx", "weight", "custom_expr"} dicts, one per
    measured-data file (file 1 first). "custom_expr" is non-empty only
    when the measured data corresponds to a *derived* field the raw
    network outputs don't have as a single column -- e.g. 1D
    Schrodinger's data is the magnitude |h| = sqrt(u**2+v**2), not
    output u or v alone. When set, the generated script matches this
    file against that expression (evaluated against ALL of this
    problem's outputs by name, via dde.icbc.PointSetOperatorBC) instead
    of a single output column (dde.icbc.PointSetBC(component=output_idx))
    -- mirrors the existing Results-panel "Custom..." plot-field option
    (see main_window.py's plot_custom_expr / codegen.py's
    _extract_plot_field), just for the *measured-data* side of an
    Inverse problem instead of the plotted solution. Falls back to a
    single entry built from the legacy inverse_data_file/
    inverse_obs_output_idx/loss_weight_obs fields when the JSON is empty
    or unparseable -- covers configs saved before this feature existed,
    and stays perfectly single-file-compatible: one observation file/
    constraint/weight, no custom expression, in that case, exactly like
    before. Unlike trainable variables, every entry here is *data*, not
    a Python identifier, so it needs no codegen-time unrolling -- the
    whole list is embedded as one repr() literal and looped over at
    runtime."""
    import json
    raw = getattr(config, "inverse_obs_files_json", "") or ""
    files = []
    if raw:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = []
        for f in (parsed or []):
            f = f or {}
            path = str(f.get("path") or "").strip()
            try:
                output_idx = int(f.get("output_idx", 0))
            except (TypeError, ValueError):
                output_idx = 0
            try:
                weight = float(f.get("weight", 100.0))
            except (TypeError, ValueError):
                weight = 100.0
            custom_expr = str(f.get("custom_expr") or "").strip()
            if path:
                files.append({"path": path, "output_idx": output_idx,
                               "weight": weight, "custom_expr": custom_expr})
    if not files:
        files.append({
            "path": config.inverse_data_file or "",
            "output_idx": getattr(config, "inverse_obs_output_idx", 0),
            "weight": config.loss_weight_obs,
            "custom_expr": "",
        })
    return files


def _build_train_cbs_code(config, var_name="_train_cbs", indent=4):
    """Builds the literal Python source (already indented at `indent`
    spaces, matching whatever base indent level the generated script has
    at the call site this is embedded into -- 4 for the Standard path, 8
    for the Time-Adaptive per-step loop) that constructs the shared list
    of opt-in DeepXDE training callbacks (EarlyStopping, PDEPointResampler,
    ModelCheckpoint, Timer) from config.cb_*. All four are off by default,
    so an unconfigured run gets exactly `{var_name} = []`, appended into
    every mainline .train() call's existing callbacks list -- a no-op,
    byte-for-byte reproducing every pre-existing generated script. NOT
    threaded into IC pre-training or the RAR refinement sub-loop (see the
    call sites) -- those are short, purpose-built inner loops where early-
    stopping/point-resampling/checkpointing would fight their intent
    rather than help it. EarlyStopping's start_from_epoch needs
    deepxde>=1.12.0 (older than this app's minimum pinned 1.10.0) -- only
    attempted when the user set a nonzero value, and falls back gracefully
    with a printed note rather than crashing if the installed deepxde is
    too old for it."""
    pad = " " * indent
    pad2 = " " * (indent + 4)
    lines = [f"{pad}{var_name} = []"]
    if config.cb_early_stopping:
        baseline_src = "None"
        raw_baseline = str(config.cb_early_stopping_baseline or "").strip()
        if raw_baseline:
            try:
                baseline_src = repr(float(raw_baseline))
            except ValueError:
                baseline_src = "None"
        es_kwargs = (
            f"min_delta={config.cb_early_stopping_min_delta}, "
            f"patience={int(config.cb_early_stopping_patience)}, "
            f"baseline={baseline_src}, "
            f"monitor={config.cb_early_stopping_monitor!r}"
        )
        if config.cb_early_stopping_start_from > 0:
            lines.append(f"{pad}try:")
            lines.append(
                f"{pad2}{var_name}.append(dde.callbacks.EarlyStopping("
                f"{es_kwargs}, start_from_epoch={int(config.cb_early_stopping_start_from)}))"
            )
            lines.append(f"{pad}except TypeError:")
            lines.append(
                f"{pad2}print(\"Note: this deepxde version doesn't support "
                "EarlyStopping's start_from_epoch (added in 1.12.0) -- ignoring it.\")"
            )
            lines.append(f"{pad2}{var_name}.append(dde.callbacks.EarlyStopping({es_kwargs}))")
        else:
            lines.append(f"{pad}{var_name}.append(dde.callbacks.EarlyStopping({es_kwargs}))")
    if config.cb_point_resampler:
        lines.append(
            f"{pad}{var_name}.append(dde.callbacks.PDEPointResampler("
            f"period={int(config.cb_point_resampler_period)}, "
            f"pde_points={bool(config.cb_point_resampler_pde_points)}, "
            f"bc_points={bool(config.cb_point_resampler_bc_points)}))"
        )
    if config.cb_model_checkpoint:
        lines.append(
            f'{pad}{var_name}.append(dde.callbacks.ModelCheckpoint('
            f'_os.path.join(_sol_dir, "checkpoint_best"), verbose=1, '
            f"save_better_only={bool(config.cb_checkpoint_save_better_only)}, "
            f"period={int(config.cb_checkpoint_period)}, "
            f"monitor={config.cb_checkpoint_monitor!r}))"
        )
    if config.cb_timer:
        lines.append(f"{pad}{var_name}.append(dde.callbacks.Timer(available_time={config.cb_timer_minutes}))")
    return "\n".join(lines)


def _net_construction_helper_code(network_type):
    """Builds the literal Python source for a `_make_net(_flat_layers,
    _activation, _kernel_init, _weight_decay=0.0)` helper, embedded once
    near the top of every generated/restore script and called at every
    network-construction site in place of a bare `dde.nn.FNN(...)`. This is
    the SINGLE place that decides network_type (FNN vs DeepXDE's own
    built-in PFNN class) -- shared verbatim (via this same function) by
    generate_script(), generate_clean_script(), AND every Model Restore /
    Error Analysis restore-script builder in main_window.py, so a training
    script and the script that later restores its checkpoint can never
    independently drift apart on how the network is shaped. That parity is
    the critical correctness requirement here: DeepXDE's model.restore()
    loads a raw state_dict by key/shape, so a restore-side net built with
    even a slightly different layer shape either crashes outright or
    silently loads nothing useful.

    layers[0] (raw input dim) and layers[-1] (num_outputs) always come from
    the caller's flat `_flat_layers` list unchanged -- only the hidden-layer
    shape (nested per-branch lists for PFNN) is computed here, purely
    locally, rather than ever being written back into a stored layers list
    -- deliberately avoiding a second place that needs the same shape
    derivation (see config.py's network_type field comment).

    PFNN has no `regularization` kwarg in DeepXDE (confirmed against its
    PyTorch backend) -- weight_decay is silently dropped for PFNN here
    rather than raising, since weight decay is a secondary training knob,
    not something that should block switching architectures."""
    return f'''# ── Network architecture: FNN or DeepXDE's built-in PFNN ──
_network_type = {network_type!r}

def _make_net(_flat_layers, _activation, _kernel_init, _weight_decay=0.0):
    _flat_layers = list(_flat_layers)
    _hidden = list(_flat_layers[1:-1])
    _n_out = _flat_layers[-1]
    if _network_type == "PFNN":
        _eff_layers = [_flat_layers[0]] + [[_w] * _n_out for _w in _hidden] + [_n_out]
        _net = dde.nn.PFNN(_eff_layers, _activation, _kernel_init)
    else:
        if _weight_decay and _weight_decay > 0:
            _net = dde.nn.FNN(_flat_layers, _activation, _kernel_init, regularization=("l2", _weight_decay))
        else:
            _net = dde.nn.FNN(_flat_layers, _activation, _kernel_init)
    return _net
'''


def _loss_plot_runtime_code(config):
    """Builds the literal Python source for a module-level `_make_loss_plot(
    train_rows, test_rows, steps, save_path, title, xlabel="Iteration",
    extra_fn=None)` helper, embedded once near the top of the generated
    script (alongside `_net_helper_code`/`_tm_runtime_code`, same pattern)
    and called at every loss-plotting site in generate_script() (the
    Standard/RAR path and the Time-Adaptive path) and
    generate_clean_script() -- honoring the Round 28 Loss Plot Settings
    (config.loss_plot_mode/loss_plot_log_y/loss_plot_linewidth).

    Defining this as a reusable FUNCTION at module level, rather than a
    block of statements spliced in at each (differently-indented) call
    site, sidesteps the indentation problem a multi-line splice would
    hit inside, e.g., the Standard path's `if not {{config.time_adaptive}}:`
    block -- every call site only needs ONE correctly-indented statement
    calling this function, exactly like `_make_net(...)` below it.

    `train_rows`/`test_rows` must be DeepXDE's own loss_history.loss_train/
    loss_test shape: one row per recorded step, each row a list of every
    individual loss TERM's value that step, PDE terms first -- one per
    output, in output order -- followed by every BC/IC term in whatever
    order codegen's own data.add_boundary_condition/IC-constraint-building
    added them.

    loss_plot_mode="train_test" (default) reproduces the exact plot every
    loss figure in the app already drew before this feature existed --
    summed Train + Test, semilogy -- so this is a verified no-op for any
    config that doesn't touch the new setting. "individual" additionally
    needs per-term labels: DeepXDE provides no per-term names at all, and
    reconstructing codegen's own full BC/IC term order statically (outside
    the generated script, which builds `_constraints`/`_constraints_i`
    from runtime-parsed custom_bc_json) would risk a label drifting out of
    sync with the real term it's attached to -- a mislabeled line is worse
    than an unlabeled one. So only the leading `config.num_outputs` PDE
    terms (a static, always-true fact -- see the matching comment in
    _rar_save_round_diagnostics/_ea_extract's own loss-term-adjacent code)
    get a real label ("PDE (u)", from config.output_names); every
    following term is labeled generically ("Constraint 1", "Constraint
    2", ...) by its position past the PDE terms. DeepXDE never tracks Test
    loss per-term (only the single summed total loss_history.loss_test
    itself), so "individual"/"all" only ever add TRAIN per-term lines.

    `extra_fn`, if given, is called with the Axes object AFTER the loss
    lines are drawn but BEFORE the figure is finalized/saved -- the one
    call site with its own extra (the Time-Adaptive accumulated loss
    plot's per-sub-domain-boundary `axvline` markers) passes a small
    closure here instead of this shared helper needing to know about
    Time-Adaptive sub-domain boundaries at all."""
    _pde_names = [n.strip() or f"u{i+1}" for i, n in enumerate((config.output_names or "u").split(","))]
    return f'''_LOSS_MODE = {config.loss_plot_mode!r}
_LOSS_LOG_Y = {config.loss_plot_log_y!r}
_LOSS_LW = {config.loss_plot_linewidth!r}
_LOSS_PDE_NAMES = {_pde_names!r}

def _loss_term_label(li):
    if li < len(_LOSS_PDE_NAMES):
        return f"PDE ({{_LOSS_PDE_NAMES[li]}})"
    return f"Constraint {{li - len(_LOSS_PDE_NAMES) + 1}}"

def _make_loss_plot(train_rows, test_rows, steps, save_path, title, xlabel="Iteration", extra_fn=None):
    train_rows = list(train_rows); steps = list(steps)
    total_train = [sum(r) for r in train_rows]
    total_test = [sum(r) for r in test_rows] if test_rows else []

    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
    plotfn = ax.semilogy if _LOSS_LOG_Y else ax.plot
    if _LOSS_MODE in ("train_test", "all"):
        plotfn(steps, total_train, label="Train loss", color="#4dabf7", linewidth=_LOSS_LW)
        if total_test:
            plotfn(steps, total_test, label="Test loss", color="#ff8787", linestyle="--", linewidth=_LOSS_LW)
    if _LOSS_MODE in ("individual", "all") and train_rows:
        n_terms = len(train_rows[0])
        term_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_terms, 1)))
        for li in range(n_terms):
            plotfn(steps, [r[li] if li < len(r) else float("nan") for r in train_rows],
                   label=_loss_term_label(li), color=term_colors[li],
                   linewidth=max(_LOSS_LW - 0.5, 0.5), alpha=0.85)
    if extra_fn is not None:
        extra_fn(ax)
    ax.set_xlabel(xlabel); ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize=8 if _LOSS_MODE != "train_test" else 10)
    fig.tight_layout()
    fig.savefig(save_path, dpi={config.plot_dpi}); plt.close(fig)
'''


def _training_monitor_runtime_code():
    """Builds the literal Python source for the Training Monitors feature's
    shared, static derivative-building/expression-evaluation helpers --
    embedded once near the top of every generated script, exactly like
    _net_construction_helper_code() above.

    Deliberately NOT shared with _pde_standard()'s own derivative builder
    (the nested `_hess`/`_dvars` logic inside generate_script()'s PDE
    definition) even though the math is identical -- this is an
    intentionally separate, self-contained copy so a diagnostic-only
    feature can never perturb the already-tested PDE-residual codegen
    path. The duplication is the deliberate tradeoff (see Round 15's
    network-architecture lesson about avoiding a second place that needs
    the same derivation logic -- accepted here specifically to keep this
    feature's blast radius at zero for the PDE itself).

    `_tm_build_dvars(x, y, n_out, out_names, is_steady, dim, expr_check)`
    builds only the derivative entries actually referenced as substrings
    of `expr_check` (a Training Monitor's own expression text), using the
    same d{name}_x / d{name}_xx / d{name}_xy / d{name}_xxxx naming the
    Custom PDE expression box already uses -- so a user who already knows
    that syntax from writing PDEs in this app can use it here unchanged.
    `dim` is "1D"/"2D"/"3D"; higher-order terms referencing a "t" column
    that doesn't exist for a steady problem are only safe to write if the
    monitor's own `is_steady` is False, exactly mirroring the same
    unenforced assumption _pde_standard()'s PDE box already relies on
    (there's no "t" in a steady problem's vocabulary to begin with).

    `_tm_eval_terms(expr_list, x, y, dvars, out_names)` splits a monitor's
    comma-separated expression list, evaluates each term against the
    built `dvars` (plus x/y/dde/torch/np), and torch.cat's the results
    into one (batch, k) tensor -- the "beyond a single derivative pair"
    capability: one dde.callbacks.OperatorPredictor can then log several
    related quantities (e.g. "u, du_x, du_xx") together, in one file,
    against the same points, every call."""
    return '''# ── Training Monitors: shared derivative-building/eval helpers ──
# (see config.py's training_monitors_enabled/training_monitors docstring)
def _tm_hess(ys, xs, i, j, _oi, _n_out):
    if _n_out == 1:
        return dde.grad.hessian(ys, xs, i=i, j=j)
    return dde.grad.hessian(ys, xs, component=_oi, i=i, j=j)

def _tm_build_dvars(x, y, n_out, out_names, is_steady, dim, expr_check):
    dvars = {}
    for oi, oname in enumerate(out_names):
        dvars[oname] = y[:, oi:oi + 1]
        if dim == "3D":
            dvars[f"d{oname}_x"] = dde.grad.jacobian(y, x, i=oi, j=0)
            dvars[f"d{oname}_y"] = dde.grad.jacobian(y, x, i=oi, j=1)
            dvars[f"d{oname}_z"] = dde.grad.jacobian(y, x, i=oi, j=2)
            if not is_steady:
                dvars[f"d{oname}_t"] = dde.grad.jacobian(y, x, i=oi, j=3)
            if f"d{oname}_xx" in expr_check:
                dvars[f"d{oname}_xx"] = _tm_hess(y, x, 0, 0, oi, n_out)
            if f"d{oname}_yy" in expr_check:
                dvars[f"d{oname}_yy"] = _tm_hess(y, x, 1, 1, oi, n_out)
            if f"d{oname}_zz" in expr_check:
                dvars[f"d{oname}_zz"] = _tm_hess(y, x, 2, 2, oi, n_out)
            if f"d{oname}_tt" in expr_check:
                dvars[f"d{oname}_tt"] = _tm_hess(y, x, 3, 3, oi, n_out)
            if f"d{oname}_xy" in expr_check:
                dvars[f"d{oname}_xy"] = _tm_hess(y, x, 0, 1, oi, n_out)
            if f"d{oname}_xz" in expr_check:
                dvars[f"d{oname}_xz"] = _tm_hess(y, x, 0, 2, oi, n_out)
            if f"d{oname}_yz" in expr_check:
                dvars[f"d{oname}_yz"] = _tm_hess(y, x, 1, 2, oi, n_out)
            if f"d{oname}_xt" in expr_check:
                dvars[f"d{oname}_xt"] = _tm_hess(y, x, 0, 3, oi, n_out)
            if f"d{oname}_yt" in expr_check:
                dvars[f"d{oname}_yt"] = _tm_hess(y, x, 1, 3, oi, n_out)
            if f"d{oname}_zt" in expr_check:
                dvars[f"d{oname}_zt"] = _tm_hess(y, x, 2, 3, oi, n_out)
            if f"d{oname}_xxxx" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxxx"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=0, j=0)
            if f"d{oname}_yyyy" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yyyy"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=1, j=1)
            if f"d{oname}_zzzz" in expr_check and f"d{oname}_zz" in dvars:
                dvars[f"d{oname}_zzzz"] = dde.grad.hessian(dvars[f"d{oname}_zz"], x, i=2, j=2)
            if f"d{oname}_xxyy" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxyy"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=1, j=1)
            if f"d{oname}_xxzz" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxzz"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=2, j=2)
            if f"d{oname}_yyzz" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yyzz"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=2, j=2)
            if f"d{oname}_xxtt" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxtt"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=3, j=3)
            if f"d{oname}_yytt" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yytt"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=3, j=3)
            if f"d{oname}_zztt" in expr_check and f"d{oname}_zz" in dvars:
                dvars[f"d{oname}_zztt"] = dde.grad.hessian(dvars[f"d{oname}_zz"], x, i=3, j=3)
        elif dim == "2D":
            dvars[f"d{oname}_x"] = dde.grad.jacobian(y, x, i=oi, j=0)
            dvars[f"d{oname}_y"] = dde.grad.jacobian(y, x, i=oi, j=1)
            if not is_steady:
                dvars[f"d{oname}_t"] = dde.grad.jacobian(y, x, i=oi, j=2)
            if f"d{oname}_xx" in expr_check:
                dvars[f"d{oname}_xx"] = _tm_hess(y, x, 0, 0, oi, n_out)
            if f"d{oname}_yy" in expr_check:
                dvars[f"d{oname}_yy"] = _tm_hess(y, x, 1, 1, oi, n_out)
            if f"d{oname}_xy" in expr_check:
                dvars[f"d{oname}_xy"] = _tm_hess(y, x, 0, 1, oi, n_out)
            if f"d{oname}_tt" in expr_check:
                dvars[f"d{oname}_tt"] = _tm_hess(y, x, 2, 2, oi, n_out)
            if f"d{oname}_xt" in expr_check:
                dvars[f"d{oname}_xt"] = _tm_hess(y, x, 0, 2, oi, n_out)
            if f"d{oname}_yt" in expr_check:
                dvars[f"d{oname}_yt"] = _tm_hess(y, x, 1, 2, oi, n_out)
            if f"d{oname}_xxxx" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxxx"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=0, j=0)
            if f"d{oname}_yyyy" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yyyy"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=1, j=1)
            if f"d{oname}_xxyy" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxyy"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=1, j=1)
            if f"d{oname}_xxtt" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxtt"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=2, j=2)
            if f"d{oname}_yytt" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yytt"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=2, j=2)
        else:
            dvars[f"d{oname}_x"] = dde.grad.jacobian(y, x, i=oi, j=0)
            if not is_steady:
                dvars[f"d{oname}_t"] = dde.grad.jacobian(y, x, i=oi, j=1)
            if f"d{oname}_xx" in expr_check or f"d{oname}_xxx" in expr_check or f"d{oname}_xxxx" in expr_check or f"d{oname}_xxtt" in expr_check:
                dvars[f"d{oname}_xx"] = _tm_hess(y, x, 0, 0, oi, n_out)
            if f"d{oname}_tt" in expr_check or f"d{oname}_tttt" in expr_check or f"d{oname}_xxtt" in expr_check:
                dvars[f"d{oname}_tt"] = _tm_hess(y, x, 1, 1, oi, n_out)
            if f"d{oname}_xt" in expr_check:
                dvars[f"d{oname}_xt"] = _tm_hess(y, x, 0, 1, oi, n_out)
            if f"d{oname}_xxx" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxx"] = dde.grad.jacobian(dvars[f"d{oname}_xx"], x, i=0, j=0)
            if f"d{oname}_xxxx" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxxx"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=0, j=0)
            if f"d{oname}_xxtt" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxtt"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=1, j=1)
            if f"d{oname}_tttt" in expr_check and f"d{oname}_tt" in dvars:
                dvars[f"d{oname}_tttt"] = dde.grad.hessian(dvars[f"d{oname}_tt"], x, i=1, j=1)
    return dvars

def _tm_eval_terms(expr_list, x, y, dvars, out_names):
    _ns = dict(dvars)
    _ns["x"] = x
    _ns["y"] = y
    _ns["dde"] = dde
    _ns["torch"] = torch
    _ns["np"] = np
    _terms = [eval(_e.strip(), _ns) for _e in expr_list.split(",") if _e.strip()]
    return torch.cat(_terms, dim=1)
'''


def _build_training_monitors_code(config, cbs_var_name="_train_cbs", sol_dir_var="_sol_dir", indent=4):
    """Builds the literal source that constructs one dde.callbacks.
    OperatorPredictor per configured Training Monitor (see config.py's
    training_monitors_enabled/training_monitors) and appends each to the
    shared `{cbs_var_name}` list that _build_train_cbs_code() builds --
    so monitors ride along in every Standard-path .train() call exactly
    like EarlyStopping/PDEPointResampler/ModelCheckpoint/Timer already do,
    with zero per-call-site changes needed. NOT wired into the Time-
    Adaptive per-step loop (`_train_cbs_ta`): each TA step trains an
    independent model from scratch, so a monitor's log file would be
    truncated and restarted every single step rather than accumulating
    one continuous trace -- out of scope for this round, not silently
    wrong.

    Each monitor also gets a small `<name>.meta.json` sidecar (point
    coordinates + expression labels + period), written directly by this
    generated code via plain json.dump, separate from the .txt log file
    OperatorPredictor itself writes -- kept as two separate files
    specifically because OperatorPredictor's own __init__ opens its log
    file in "w" mode (truncating), so anything this code wrote into that
    same file first would simply be wiped the moment the callback is
    constructed. The Results panel reads both: the sidecar to know how
    many points/expressions there are and what to label them, the log to
    get the actual (iteration, value...) time series.

    Returns "" (a true no-op) when Training Monitors is off or no
    monitors are configured, same convention _build_train_cbs_code() uses."""
    if not config.training_monitors_enabled:
        return ""
    import json as _json_tm
    try:
        _monitors = _json_tm.loads(config.training_monitors or "[]")
    except (ValueError, TypeError):
        _monitors = []
    if not isinstance(_monitors, list) or not _monitors:
        return ""

    pad = " " * indent
    _out_names_list = [n.strip() for n in (config.output_names or "").split(",") if n.strip()]
    _dim = config.problem_dim
    _is_steady = bool(config.steady_state)
    _n_out = int(config.num_outputs)

    lines = []
    for _mi, _mon in enumerate(_monitors):
        if not isinstance(_mon, dict):
            continue
        _name = str(_mon.get("name") or f"monitor{_mi}").strip() or f"monitor{_mi}"
        _safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in _name)
        _points = _mon.get("points") or []
        _expr = str(_mon.get("expr") or "").strip()
        _period = int(_mon.get("period") or 1000)
        if not _points or not _expr:
            continue
        _expr_labels = [e.strip() for e in _expr.split(",") if e.strip()]

        lines.append(f"{pad}_tm_points_{_mi} = {_points!r}")
        lines.append(f"{pad}_tm_log_path_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}.txt')")
        lines.append(f"{pad}_tm_meta_path_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}.meta.json')")
        lines.append(f"{pad}with open(_tm_meta_path_{_mi}, 'w') as _tm_mf_{_mi}:")
        lines.append(
            f"{pad}    _json_tm_write.dump({{'name': {_name!r}, 'points': _tm_points_{_mi}, "
            f"'expressions': {_expr_labels!r}, 'period': {_period}}}, _tm_mf_{_mi})"
        )
        lines.append(
            f"{pad}def _tm_op_{_mi}(x, y, _expr={_expr!r}, _out_names={_out_names_list!r}, "
            f"_n_out={_n_out}, _is_steady={_is_steady!r}, _dim={_dim!r}):"
        )
        lines.append(f"{pad}    _dvars = _tm_build_dvars(x, y, _n_out, _out_names, _is_steady, _dim, _expr)")
        lines.append(f"{pad}    return _tm_eval_terms(_expr, x, y, _dvars, _out_names)")
        lines.append(
            f"{pad}{cbs_var_name}.append(dde.callbacks.OperatorPredictor(_tm_points_{_mi}, _tm_op_{_mi}, "
            f"period={_period}, filename=_tm_log_path_{_mi}, precision=8))"
        )
    if not lines:
        return ""
    return (f"{pad}import json as _json_tm_write\n") + "\n".join(lines)


def _build_training_monitors_plot_code(config, sol_dir_var="_sol_dir", indent=0):
    """Builds the literal source (placed near the very end of the
    generated script, just before the final "DONE" print) that reads
    each Training Monitor's own .txt log + .meta.json sidecar back in
    (the exact same files _build_training_monitors_code() wrote the
    OperatorPredictor callbacks to write) and renders one line plot per
    monitor via matplotlib (already a hard dependency of this app,
    imported as `plt` everywhere else in this script), saved as
    `monitor_<name>_plot.png` next to the log itself -- so the Results
    panel can display it exactly the way it already displays
    loss_plot.png/solution_plot.png (see main_window.py's
    _display_run_result_plots()), with zero new rendering logic needed
    on the GUI side.

    dde.callbacks.OperatorPredictor logs each line as
    "<iteration> [v0, v1, ...]" (space before the bracket, DeepXDE's own
    utils.list_to_str() format) with the flattened values in point-major,
    expression-minor order -- matching exactly how _tm_eval_terms()
    builds its (n_points, n_expr) tensor via torch.cat(..., dim=1) before
    flattening, confirmed by a real training run's logged output during
    this feature's own development, not just inferred from the source.

    Wrapped in its own try/except per monitor -- a plotting failure
    (e.g. an empty log because training was stopped at iteration 0)
    prints a warning and moves on rather than crashing a run that
    otherwise trained and saved successfully."""
    if not config.training_monitors_enabled:
        return ""
    import json as _json_tm2
    try:
        _monitors = _json_tm2.loads(config.training_monitors or "[]")
    except (ValueError, TypeError):
        _monitors = []
    if not isinstance(_monitors, list) or not _monitors:
        return ""

    pad = " " * indent
    lines = [f"{pad}import json as _json_tm_read"]
    _any = False
    for _mi, _mon in enumerate(_monitors):
        if not isinstance(_mon, dict):
            continue
        _name = str(_mon.get("name") or f"monitor{_mi}").strip() or f"monitor{_mi}"
        _safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in _name)
        if not (_mon.get("points") and _mon.get("expr")):
            continue
        _any = True
        lines.append(f"{pad}_tm_plot_log_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}.txt')")
        lines.append(f"{pad}_tm_plot_meta_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}.meta.json')")
        lines.append(f"{pad}if _os.path.exists(_tm_plot_log_{_mi}) and _os.path.exists(_tm_plot_meta_{_mi}):")
        lines.append(f"{pad}    try:")
        lines.append(f"{pad}        with open(_tm_plot_meta_{_mi}) as _tm_mf_{_mi}:")
        lines.append(f"{pad}            _tm_meta_{_mi} = _json_tm_read.load(_tm_mf_{_mi})")
        lines.append(f"{pad}        _tm_npts_{_mi} = len(_tm_meta_{_mi}['points'])")
        lines.append(f"{pad}        _tm_nexpr_{_mi} = len(_tm_meta_{_mi}['expressions'])")
        lines.append(f"{pad}        _tm_iters_{_mi} = []")
        lines.append(f"{pad}        _tm_rows_{_mi} = []")
        lines.append(f"{pad}        with open(_tm_plot_log_{_mi}) as _tm_lf_{_mi}:")
        lines.append(f"{pad}            for _tm_line_{_mi} in _tm_lf_{_mi}:")
        lines.append(f"{pad}                _tm_line_{_mi} = _tm_line_{_mi}.strip()")
        lines.append(f"{pad}                if not _tm_line_{_mi}:")
        lines.append(f"{pad}                    continue")
        lines.append(f"{pad}                _tm_it_s_{_mi}, _tm_rest_{_mi} = _tm_line_{_mi}.split(' ', 1)")
        lines.append(f"{pad}                _tm_rows_{_mi}.append([float(_v) for _v in _tm_rest_{_mi}.strip('[]').split(',')])")
        lines.append(f"{pad}                _tm_iters_{_mi}.append(int(_tm_it_s_{_mi}))")
        lines.append(f"{pad}        if _tm_iters_{_mi} and _tm_npts_{_mi} > 0 and _tm_nexpr_{_mi} > 0:")
        lines.append(f"{pad}            _tm_arr_{_mi} = np.array(_tm_rows_{_mi}).reshape(-1, _tm_npts_{_mi}, _tm_nexpr_{_mi})")
        lines.append(f"{pad}            _tm_fig_{_mi}, _tm_ax_{_mi} = plt.subplots(figsize=(7, 5))")
        lines.append(f"{pad}            for _tm_pi_{_mi} in range(_tm_npts_{_mi}):")
        lines.append(f"{pad}                for _tm_ei_{_mi} in range(_tm_nexpr_{_mi}):")
        lines.append(
            f"{pad}                    _tm_ax_{_mi}.plot(_tm_iters_{_mi}, "
            f"_tm_arr_{_mi}[:, _tm_pi_{_mi}, _tm_ei_{_mi}], marker='o', markersize=3, "
            f"label=f\"{{_tm_meta_{_mi}['expressions'][_tm_ei_{_mi}]}} @ pt{{_tm_pi_{_mi}}}\")"
        )
        lines.append(f"{pad}            _tm_ax_{_mi}.set_xlabel('Iteration'); _tm_ax_{_mi}.set_ylabel('Value')")
        lines.append(f"{pad}            _tm_ax_{_mi}.set_title({('Training Monitor: ' + _name)!r})")
        lines.append(f"{pad}            _tm_ax_{_mi}.legend(fontsize=7, loc='best'); _tm_ax_{_mi}.grid(alpha=0.3)")
        lines.append(f"{pad}            _tm_plot_path_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}_plot.png')")
        lines.append(f"{pad}            plt.tight_layout(); plt.savefig(_tm_plot_path_{_mi}, dpi=150); plt.close(_tm_fig_{_mi})")
        lines.append(f"{pad}            print(f'  Training monitor plot saved: {{_tm_plot_path_{_mi}}}')")
        lines.append(f"{pad}    except Exception as _tm_plot_err_{_mi}:")
        lines.append(f'{pad}        print(f"  Warning: failed to plot training monitor {_name!r}: {{_tm_plot_err_{_mi}}}")')
    if not _any:
        return ""
    return "\n".join(lines)


def generate_script(config):
    is_2d = config.problem_dim == "2D"
    is_3d = config.problem_dim == "3D"
    # The "solution" output is a static PNG for every plot type except the
    # two GIF animations, where it's an actual animated .gif file instead
    # -- known now (at generation time) from the selected plot type, so the
    # right extension can be baked into every solution-path literal below
    # rather than guessed at runtime.
    # Steady-state problems have no time axis, so a GIF animation (which is
    # inherently a walk over time) never applies -- always .png there, even
    # if a stale config still has one of the GIF plot types selected.
    _sol_ext = "gif" if (not config.steady_state and config.plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)")) else "png"
    # Convert user-friendly IC expressions
    ic_exprs_raw = config.ic_expressions.split("|")
    ic_exprs_converted = [_simplify_expr(e, is_2d, is_3d) for e in ic_exprs_raw]
    config_ic_expressions = "|".join(ic_exprs_converted)
    ta_ic_expr = _simplify_expr(config.ic_expression, is_2d, is_3d)

    # Convert user-friendly PDE expressions
    pde_exprs_raw = config.pde_expressions.split("|")
    pde_exprs_converted = [_simplify_pde_expr(e) for e in pde_exprs_raw]
    config_pde_expressions = "|".join(pde_exprs_converted)

    pde_expr_single = _simplify_pde_expr(config.pde_expression)

    # Every 2D/3D line-type plot (Line (time steps), Line Animation (GIF),
    # the Time-Adaptive per-step preview, and the inline Error Analysis
    # line comparison) shows u vs x only, fixing y (and z, in 3D) at one
    # value -- previously each site hardcoded the domain midpoint
    # separately; now every site reads the same user-configurable value,
    # computed once here. _auto (the default) reproduces the previous
    # hardcoded behavior exactly.
    _line_slice_y = (config.y_min + config.y_max) / 2.0 if config.line_slice_y_auto else config.line_slice_y
    _line_slice_z = (config.z_min + config.z_max) / 2.0 if config.line_slice_z_auto else config.line_slice_z

    # The optional custom Results-panel plot expression (e.g. Schrodinger's
    # "sqrt(u**2+v**2)") is passed through UNCONVERTED, deliberately NOT
    # run through _simplify_pde_expr() the way PDE expressions are above.
    # PDE expressions are eval'd in the generated script's own MODULE
    # namespace (where `import numpy as np` already ran), so prefixing
    # their function calls with "np." is correct there. This expression is
    # instead eval'd by _plot_custom_op/_extract_plot_field below inside a
    # small, ISOLATED namespace dict (_pf_ns) built from _PLOT_TORCH_MATH_NS
    # -- which already maps every one of the same bare names
    # (sin/cos/.../sqrt/pi, ...) straight to their torch equivalent, so a
    # raw "sqrt(...)" resolves correctly there. Running _simplify_pde_expr
    # on it rewrites "sqrt(" to "np.sqrt(" instead, which that namespace
    # has no "np" binding for -- a real bug, caught via an actual exec-level
    # run of the 1D Schrodinger template's GIF export ("NameError: name
    # 'np' is not defined"), since _plot_custom_op's eval() only sees
    # whatever's in _pf_ns, not the module's real globals. Both
    # generate_clean_script()'s own _plot_custom_expr_val and every Restore
    # & Visualize script builder's _custom_expr_val already pass this value
    # through raw/unconverted -- this just brings generate_script() in
    # line with them.
    _plot_custom_expr_raw = (getattr(config, "plot_custom_expr", "") or "").strip()
    plot_custom_expr_converted = _plot_custom_expr_raw
    plot_custom_label_resolved = (getattr(config, "plot_custom_label", "") or "").strip() or _plot_custom_expr_raw

    # Optional title/axis-label overrides for the solution plot -- same
    # blank-means-default convention, and same reasoning, as Restore &
    # Visualize's own title_override/xlabel_override/ylabel_override in
    # _build_restore_script. Baked in as a plain string here (possibly
    # empty) and combined with each default title/label expression via
    # `_plot_title_override or <default>` INSIDE the generated script
    # itself (see the "Plot solution" dispatch below) -- unlike the
    # restore builders, out_name there is only computed at the generated
    # script's own runtime, not at codegen time, so the override can't be
    # resolved here already.
    plot_title_override = (getattr(config, "plot_title_override", "") or "").strip()
    plot_xlabel_override = (getattr(config, "plot_xlabel_override", "") or "").strip()
    plot_ylabel_override = (getattr(config, "plot_ylabel_override", "") or "").strip()

    if config.forward_ic_from_file:
        _ta_ic_init = f"""_ic_ta_xt, _ic_ta_vals = _load_ic_from_file({repr(config.forward_ic_file)})
    prev_u = _ic_ta_vals"""
    else:
        _ta_ic_init = f"prev_u = np.reshape({ta_ic_expr}, (-1, 1))"
    _fecr_pde_block = ""

    # Boundary Conditions panel entries -- embedded via repr() (not a raw
    # f-string substitution) because the JSON text contains double quotes
    # and whatever punctuation the user typed into location/value fields;
    # repr() guarantees a syntactically valid, safely-escaped Python string
    # literal in the generated script no matter what's inside it.
    _custom_bc_json_literal = repr(config.custom_bc_json or "[]")
    # Whether this config actually has Boundary Conditions panel data at
    # all, checked BEFORE the "or '[]'" fallback above collapses the
    # distinction: a config built by the current GUI always has a real
    # (possibly empty-array) custom_bc_json string, since the panel is
    # always serialized -- only a config saved before this panel existed
    # would have this field genuinely blank. Used so an intentionally
    # emptied panel (any geometry) means "zero custom BCs", never a
    # silent fall-back to the legacy per-side scheme.
    _bc_panel_data_present_literal = repr(bool(config.custom_bc_json))

    # Geometry-type dispatch (Phase 2): the shape's own parameters, baked in
    # as literals the same way _custom_bc_json_literal is above. Triangle
    # needs exactly 3 vertices and Polygon at least 3 -- fall back to the
    # same defaults PINNConfig itself uses if parsing comes up short (a
    # malformed/edited-away vertex string shouldn't crash codegen).
    _geometry_type_literal = repr(config.geometry_type or "Rectangle")
    _triangle_verts_parsed = _parse_vertex_list(config.geom_triangle_vertices)
    if len(_triangle_verts_parsed) != 3:
        _triangle_verts_parsed = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    _polygon_verts_parsed = _parse_vertex_list(config.geom_polygon_vertices)
    if len(_polygon_verts_parsed) < 3:
        _polygon_verts_parsed = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    _triangle_vertices_literal = repr(_triangle_verts_parsed)
    _polygon_vertices_literal = repr(_polygon_verts_parsed)
    # Custom geometry: the CSG-chained constructor call is built once here
    # (at codegen time, not inside the generated script) and baked in as
    # a literal, same as every other shape's parameters above -- see
    # _build_custom_geom_code()'s docstring for the wrap_needed rule.
    _custom_geom_inner_code, _custom_geom_wrap_needed = _build_custom_geom_code(config)
    _custom_geom_final_literal = (
        f"_DTypeSafeGeom({_custom_geom_inner_code})" if _custom_geom_wrap_needed
        else _custom_geom_inner_code
    )

    # Inverse: one or more trainable variables (generalized from the single
    # trainable_variable this used to be limited to). Only the variable
    # *definitions* need to be unrolled per-variable here at generation
    # time -- every downstream use (external_trainable_variables,
    # VariableValue, print/save callbacks, the eval() namespace, the
    # convergence plot) operates generically over the _inv_vars /
    # _inv_var_names lists built right after these definitions run, so
    # none of that has to be unrolled.
    _inv_vars_parsed = _parse_inverse_variables(config)
    _inv_var_def_lines = []
    for _iv_name, _iv_init, _iv_true in _inv_vars_parsed:
        _inv_var_def_lines.append(f"    {_iv_name} = dde.Variable({_iv_init})")
        _inv_var_def_lines.append(
            f'    print(f"Inverse PINN: inferring {_iv_name}, init = {_iv_init}")')
    _inv_var_defs_code = "\n".join(_inv_var_def_lines)
    _inv_var_list_literal = "[" + ", ".join(n for n, _i, _t in _inv_vars_parsed) + "]"
    # Recorded into model_config.json below so a later Model Restore knows
    # this run was Inverse and exactly which trainable variables it had --
    # restoring an Inverse checkpoint needs to recompile with the same
    # number of external_trainable_variables the optimizer was originally
    # given (init's own value doesn't matter for that -- only the count,
    # since neither the .pt file nor model.restore() ever stores/recovers a
    # trainable variable's actual value, only the network weights). "true"
    # is recorded too (None when not known) purely so a later Restore's
    # Parameter Convergence Plot/Animation can auto-fill the same true-value
    # dashed reference line the training-time plot already draws (see
    # main_window.py's _on_browse_restore_model auto-detect and
    # _build_restore_param_script) -- it plays no part in the restore
    # itself. Empty list for Forward configs, where none of this is
    # relevant.
    _mc_inv_vars_literal = (
        repr([{"name": n, "init": i, "true": t} for n, i, t in _inv_vars_parsed])
        if config.problem_type == "Inverse" else "[]"
    )
    _inv_var_names_literal = repr([n for n, _i, _t in _inv_vars_parsed])
    # One entry per trainable variable, the known ground-truth value for a
    # built-in template's auto-substituted PDE constant (see
    # main_window.py's INVERSE_AUTO_VARS), or None for a manually added
    # variable/legacy config with no known true value -- used only by the
    # parameter-convergence plot below to optionally draw a dashed
    # reference line at the true value, alongside the existing dashed
    # line at the run's own final inferred value.
    _inv_var_true_literal = repr([t for _n, _i, t in _inv_vars_parsed])

    # Inverse: one or more measured-data files, each with its own output
    # column and its own loss weight (a separate observation loss term
    # per file). This is pure data (path/index/weight), not a Python
    # identifier, so -- unlike the variables above -- it needs no
    # per-entry unrolling: the whole parsed list is just one repr()
    # literal, looped over at runtime.
    _obs_files_parsed = _parse_inverse_obs_files(config)
    _obs_files_literal = repr(_obs_files_parsed)

    # Whether this run uses the NNCG optimizer anywhere (a Training Phase,
    # or the legacy phase-2 field for pre-scheduler configs) -- NNCG was
    # added in deepxde 1.13.0, but this app's minimum pinned version is
    # 1.10.0, so a version-guard block is only emitted into the generated
    # script when actually needed, printing a clear upgrade message instead
    # of letting an old deepxde installation hit a bare NotImplementedError
    # deep inside training. (The GUI itself also checks this at Solve time,
    # in _validate_optimizer_settings() -- this is the belt-and-suspenders
    # copy for a script saved and re-run standalone, possibly on a
    # different machine/environment than the one that generated it.)
    import json as _json_nncg
    try:
        _sched_phases_for_nncg = _json_nncg.loads(config.scheduler_phases) if config.scheduler_phases else []
    except (ValueError, TypeError):
        _sched_phases_for_nncg = []
    _uses_nncg = (
        any(p.get("optimizer") == "nncg" for p in _sched_phases_for_nncg)
        or config.optimizer2 == "nncg"
    )
    _nncg_guard_code = ""
    if _uses_nncg:
        _nncg_guard_code = '''
_dxde_ver_parts = dde.__version__.split(".")[:3]
try:
    _dxde_ver = tuple(int(_p) for _p in _dxde_ver_parts)
except ValueError:
    _dxde_ver = None
if _dxde_ver is not None and _dxde_ver < (1, 13, 0):
    print(f"ERROR: The NNCG optimizer requires deepxde>=1.13.0 "
          f"(you have {dde.__version__} installed). "
          f"Please upgrade: pip install --upgrade deepxde")
    raise SystemExit(1)
'''

    # Reproducibility: seed NumPy/PyTorch/random together via DeepXDE's own
    # helper, right before anything random happens (point sampling, network
    # weight init). Off entirely when use_random_seed is False -- emits
    # nothing, byte-for-byte the same as before this feature existed.
    # Deliberately NOT touching torch.backends.cudnn.deterministic below --
    # that would buy bit-for-bit GPU reproducibility too, at a real training-
    # speed cost that isn't worth it here.
    _seed_code = ""
    if config.use_random_seed:
        _seed_code = f'''dde.config.set_random_seed({config.random_seed})
'''

    # Weight decay (L2 regularization). 0.0 (the default) reproduces every
    # pre-existing generated script byte-for-byte in this section (an empty
    # regularization arg was never emitted before this feature existed).
    # Superseded by _make_net()'s own _weight_decay parameter (see
    # _net_construction_helper_code below), which additionally knows to
    # drop this entirely for PFNN (no `regularization` kwarg on DeepXDE's
    # PFNN) -- config.weight_decay is now passed straight through to
    # _make_net as a plain float at each call site instead of this being
    # pre-built into an arg-string splice.

    # Network architecture: FNN or DeepXDE's built-in PFNN class. See
    # _net_construction_helper_code()'s own docstring for the full design
    # -- this is the literal `_make_net(...)` helper's source, embedded
    # once near the top of the generated script below and called at every
    # network-construction site in place of a bare dde.nn.FNN(...).
    _net_helper_code = _net_construction_helper_code(config.network_type)

    # Training Monitors' shared, static derivative-building/expression-
    # evaluation helpers -- see _training_monitor_runtime_code()'s own
    # docstring. Embedded once near the top of the generated script
    # (module level, alongside _net_helper_code) regardless of whether
    # any monitors are actually configured -- cheap, and keeps this a
    # true no-op wiring point like _train_cbs_code below.
    _tm_runtime_code = _training_monitor_runtime_code()

    # Loss Plot Settings (Round 28) -- see _loss_plot_runtime_code()'s own
    # docstring. Embeds a single reusable `_make_loss_plot(...)` helper
    # once at module level (same pattern as _net_helper_code/_make_net);
    # every loss-plotting call site (Standard/RAR path, Time-Adaptive
    # path, and generate_clean_script()'s own two analogous sites) then
    # just calls it with its own data/title/extra_fn, no further splicing.
    _loss_helper_code = _loss_plot_runtime_code(config)

    # Opt-in training callbacks (EarlyStopping/PDEPointResampler/
    # ModelCheckpoint/Timer/Training Monitors) -- one block for the
    # Standard path (built once, reused across every scheduler phase's
    # .train() call so state like Timer's running clock and
    # EarlyStopping's patience counter carry correctly across phases
    # within one run), one for the Time-Adaptive per-step loop (rebuilt
    # fresh each step, since each step is its own independent training
    # problem -- Training Monitors are deliberately NOT appended to this
    # one, see _build_training_monitors_code()'s docstring for why).
    _tm_cbs_code = _build_training_monitors_code(config, "_train_cbs", "_sol_dir", indent=4)
    _train_cbs_code = _build_train_cbs_code(config, "_train_cbs", indent=4) + ("\n" + _tm_cbs_code if _tm_cbs_code else "")
    _train_cbs_code_ta = _build_train_cbs_code(config, "_train_cbs_ta", indent=8)
    # Rendered at the very end of the script (see the {_tm_plot_code}
    # splice near the final "DONE" print below) -- reads each monitor's
    # own .txt/.meta.json back in and saves a line plot PNG, the same way
    # loss_plot.png/solution_plot.png already get saved, so the Results
    # panel can display it with no new rendering logic of its own.
    _tm_plot_code = _build_training_monitors_plot_code(config, "_sol_dir", indent=0)

    # IC pre-training needs at least 1 point sampled at the initial-time
    # slice, or DeepXDE's data construction fails outright (see the IC
    # pre-training blocks below for why). The GUI's own spinner is now
    # range-limited to >=1, but a config saved before that existed (or
    # hand-edited) could still carry 0 -- clamp here too as a second line
    # of defense. The field's own default (1000) is unaffected either way.
    _ic_pretrain_num_initial_safe = max(1, config.ic_pretrain_num_initial)

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

# ── Plot Settings: figure-size standardization ─────────────────
# Applied to every single-panel results plot below (Loss, Line, Surface,
# both Animation GIF types, Parameter Convergence, and their
# Time-Adaptive equivalents) via figsize=_plot_figsize(default_w,
# default_h) in place of a bare literal tuple -- "Default" mode (below)
# returns (default_w, default_h) unchanged, so every one of those call
# sites renders byte-identical output to before this existed unless the
# user has actually changed Plot Settings' Figure size option. NOT
# applied to the multi-column Error Analysis comparison grids (sized by
# however many files/columns are being compared) or the domain-sampling
# preview plots (sized to the problem's own geometry).
def _plot_figsize(_default_w, _default_h):
    _fs_mode = "{config.plot_figsize_mode}"
    if _fs_mode == "Square":
        _fs_s = max(_default_w, _default_h)
        return (_fs_s, _fs_s)
    elif _fs_mode == "Wide":
        return (_default_h * 1.8, _default_h)
    elif _fs_mode == "Custom":
        return ({config.plot_figsize_w}, {config.plot_figsize_h})
    return (_default_w, _default_h)
{_nncg_guard_code}

# ── Force GPU initialization ──────────────────────────────────
_effective_float = "{config.float_type}"
if "{config.lbfgs_float_type}" == "float64" and _effective_float == "float32":
    _effective_float = "float64"
    print("  [Info] Using float64 globally (required for L-BFGS float64 mode)")
dde.config.set_default_float(_effective_float)
{_seed_code}_gpu_device_index = {config.gpu_device_index}
_gpu_memory_fraction = {config.gpu_memory_fraction}
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(_gpu_device_index)

    # ── Maximize GPU memory usage ─────────────────────────────
    # Reserve up to gpu_memory_fraction of the selected device's memory
    # upfront (defaults: device 0, 95% -- unchanged from prior behavior;
    # both are configurable from the GUI's Hardware settings).
    torch.cuda.empty_cache()
    total_mem = torch.cuda.get_device_properties(_gpu_device_index).total_memory
    torch.cuda.set_per_process_memory_fraction(_gpu_memory_fraction, device=_gpu_device_index)

    # ── Performance settings ──────────────────────────────────
    _is_f64 = "{config.float_type}" == "float64"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    torch.backends.cudnn.benchmark        = False
    torch.backends.cudnn.deterministic    = False

    # ── Warm up CUDA ──────────────────────────────────────────
    _dummy = torch.zeros(10, 10, requires_grad=True, device=f'cuda:{{_gpu_device_index}}')
    _loss = (_dummy ** 2).sum()
    _loss.backward()
    torch.cuda.synchronize()
    del _dummy, _loss
    torch.cuda.empty_cache()

    _free, _total = torch.cuda.mem_get_info(_gpu_device_index)
    print(f"✅ GPU: {{torch.cuda.get_device_name(_gpu_device_index)}}")
    print(f"   Total memory: {{_total / 1e9:.1f}} GB")
    print(f"   Available:    {{_free / 1e9:.1f}} GB")
    print(f"   TF32:         False")
    print(f"   Float type:   {config.float_type}")
    print(f"   cuDNN bench:  False")
else:
    print("⚠️ No GPU found — running on CPU")

# ── Save directory setup ─────────────────────────────────────
import os as _os
_save_dir = {repr(config.save_dir)}.strip()
_use_save = bool(_save_dir)
if _use_save:
    _os.makedirs(_save_dir, exist_ok=True)
    import json as _json
    _model_config = {{
        "layers": {config.layers},
        "activation": {repr(config.activation)},
        # Network architecture -- read back by every Model Restore / Error
        # Analysis restore-script builder in main_window.py so a restored
        # net is reconstructed with the exact same shape it was trained
        # with (see _net_construction_helper_code() in this file). Absent
        # from a config saved before this feature existed; cfg.get(...) on
        # the restore side defaults to "FNN", i.e. the old behavior.
        "network_type": {repr(config.network_type)},
        "num_outputs": {config.num_outputs},
        "output_names": {repr(config.output_names)},
        "x_min": {config.x_min}, "x_max": {config.x_max},
        "y_min": {config.y_min}, "y_max": {config.y_max},
        "z_min": {config.z_min}, "z_max": {config.z_max},
        "t_min": {config.t_min}, "t_max": {config.t_max},
        # 2D/3D Line-type plots' y/z slice -- read back by Restore &
        # Visualize's own "Line (time steps)"/"Animation Line (GIF)"
        # builders so a restored model's line plot uses the same slice
        # the run was trained/plotted with, not a different one chosen
        # later in the Restore panel. Absent from a config saved before
        # this feature existed; cfg.get(...) on the restore side defaults
        # to "auto" (domain midpoint), i.e. the old hardcoded behavior.
        "line_slice_y_auto": {config.line_slice_y_auto}, "line_slice_y": {config.line_slice_y},
        "line_slice_z_auto": {config.line_slice_z_auto}, "line_slice_z": {config.line_slice_z},
        "problem_dim": {repr(config.problem_dim)},
        # Restoring a model has no other way to tell a steady-state
        # checkpoint from a transient one -- _build_restore_script()/
        # _build_restore_ea_script() in main_window.py both read this key
        # (defaulting to False, i.e. transient, when absent) to decide
        # whether model.predict(...)'s input arrays need a time column.
        # Missing here before this fix -- so EVERY restored steady-state
        # model was silently treated as transient, appending a time column
        # the restored network's first layer was never sized for, crashing
        # with "mat1 and mat2 shapes cannot be multiplied" the moment you
        # tried to plot or run Error Analysis against it.
        "steady_state": {config.steady_state},
        # Geometry definition -- without these, a restored model can only
        # ever be rebuilt against its rectangular/cuboid bounding box (see
        # _build_restore_script() in main_window.py), so a non-box geometry
        # (Disk/Ellipse/Triangle/Polygon/Sphere/Custom CSG combos) gets
        # predicted and plotted over the WRONG domain on restore -- the
        # actual shape, e.g. a triangular cavity, never gets reconstructed,
        # and nothing here lets the restored prediction be masked back down
        # to it either. Missing here before this fix.
        "geometry_type": {repr(config.geometry_type)},
        "geom_center_x": {config.geom_center_x}, "geom_center_y": {config.geom_center_y},
        "geom_center_z": {config.geom_center_z}, "geom_radius": {config.geom_radius},
        "geom_semi_major": {config.geom_semi_major}, "geom_semi_minor": {config.geom_semi_minor},
        "geom_angle": {config.geom_angle},
        "geom_triangle_vertices": {repr(config.geom_triangle_vertices)},
        "geom_polygon_vertices": {repr(config.geom_polygon_vertices)},
        "geom_custom_shapes_json": {repr(config.geom_custom_shapes_json)},
        "pde_expressions": {repr(config.pde_expressions)},
        "optimizer": {repr(config.optimizer)},
        "optimizer2": {repr(config.optimizer2)},
        "loss_type": {repr(config.loss_type)},
        "problem_type": {config.problem_type!r},
        "inverse_variables": {_mc_inv_vars_literal},
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
_solution_path = _os.path.join(_sol_dir, "solution_plot.{_sol_ext}")
_log_path      = _os.path.join(_sol_dir, "training_log.txt") if _use_save else None

# ── Problem dimension ─────────────────────────────────────────
_is_2d = "{config.problem_dim}" == "2D"
_is_3d = "{config.problem_dim}" == "3D"
# Steady-state (time-independent) problem, e.g. a Poisson equation -- no
# time axis on the network input, no GeometryXTime, no Initial Condition,
# a plain dde.data.PDE instead of dde.data.TimePDE. See _build_geom() and
# the "Data" section below for where this actually changes construction.
_is_steady = {str(config.steady_state)}

# Load an Initial Condition from a plain, header-less data file and
# return (xt, vals) ready for dde.icbc.PointSetBC -- xt at the domain's
# START time ({config.t_min}), since that's where an IC is defined by
# default. Column layout matches the problem's dimension, one
# coordinate column per spatial axis, then time, then value:
#   1D: x, t, c        2D: x, y, t, c        3D: x, y, z, t, c
# No header row. Rows are filtered to whichever ones already sit at
# t == {config.t_min} (small tolerance for floating-point data) --
# a file covering the whole time range works fine, only its t=t_min
# slice is used as the IC.
def _load_ic_from_file(_path):
    _raw = np.loadtxt(_path)
    _n_coord = 3 if _is_3d else (2 if _is_2d else 1)
    _t_col = _n_coord
    _v_col = _n_coord + 1
    _t_start = {config.t_min}
    _mask = np.abs(_raw[:, _t_col] - _t_start) < 1e-8
    _n_matched = int(_mask.sum())
    if _n_matched == 0:
        raise ValueError(
            f"IC file {{_path}} has no rows at t = {{_t_start}} (the domain's "
            f"start time) -- expected a header-less file with columns "
            f"{{'x, t, c' if not (_is_2d or _is_3d) else ('x, y, t, c' if _is_2d else 'x, y, z, t, c')}}, "
            f"with at least one row at t = {{_t_start}}."
        )
    _coords = _raw[_mask, :_n_coord]
    _t0 = np.full((_n_matched, 1), _t_start)
    _xt = np.hstack([_coords, _t0])
    _vals = _raw[_mask, _v_col:_v_col + 1]
    return _xt, _vals

# Optional input/output transform, applied to every net this script
# builds (the one that actually trains, and every per-step net rebuilt
# later purely to restore a checkpoint for plotting/error-analysis --
# restoring only loads weights, not a transform, since apply_feature_
# transform/apply_output_transform just attach a plain Python closure to
# the net instance rather than something saved in a .pt checkpoint, so a
# reloaded net needs the same transform re-applied to predict correctly).
# Empty list means "off"; scale/shift lists line up column-for-column
# with the net's raw input (x, [y], [z], t) or raw output (one entry per
# output component).
_in_scale  = {config.input_transform_scale if config.input_transform_enabled else []}
_in_shift  = {config.input_transform_shift if config.input_transform_enabled else []}
_out_scale = {config.output_transform_scale if config.output_transform_enabled else []}
_out_shift = {config.output_transform_shift if config.output_transform_enabled else []}

{_net_helper_code}

{_tm_runtime_code}

{_loss_helper_code}

def _apply_net_transforms(_net):
    if _in_scale:
        def _input_transform(x):
            _sc = torch.tensor(_in_scale, dtype=x.dtype, device=x.device)
            _sh = torch.tensor(_in_shift, dtype=x.dtype, device=x.device)
            return x * _sc + _sh
        _net.apply_feature_transform(_input_transform)
    if _out_scale:
        def _output_transform(x, y):
            _sc = torch.tensor(_out_scale, dtype=y.dtype, device=y.device)
            _sh = torch.tensor(_out_shift, dtype=y.dtype, device=y.device)
            return y * _sc + _sh
        _net.apply_output_transform(_output_transform)
    return _net

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

{_inv_var_defs_code}
    _inv_vars = {_inv_var_list_literal}
    _inv_var_names = {_inv_var_names_literal}
    _inv_var_trues = {_inv_var_true_literal}

    def _load_data(fpath):
        try:
            arr = np.loadtxt(fpath, delimiter=",", skiprows=1)
        except Exception:
            try:
                arr = np.loadtxt(fpath, skiprows=1)
            except Exception:
                arr = np.loadtxt(fpath, delimiter=",")
        return arr

    # Number of leading coordinate columns in a measured-data file (before
    # its value column): x/y/z plus a time column, EXCEPT for a steady-state
    # problem, which has no time axis at all -- same distinction
    # _n_coord_cols_bc already makes for BC-panel points below. Getting this
    # wrong for a steady 2D/3D template (assuming a "t" column that doesn't
    # exist) would silently read the value column as coordinates and vice
    # versa instead of raising, so this is computed generically rather than
    # copying the old is_3d/is_2d-only ladder that never checked _is_steady.
    _n_obs_coord_cols = ((3 if _is_3d else (2 if _is_2d else 1)) if _is_steady
                         else (4 if _is_3d else (3 if _is_2d else 2)))

    def _make_obs_func(_expr, _onames):
        # Builds a dde.icbc.PointSetOperatorBC-compatible func(inputs,
        # outputs, X) for a measured-data file whose value is a DERIVED
        # field (e.g. 1D Schrodinger's |h| = sqrt(u**2+v**2)) rather than
        # a single raw output column -- same expression syntax as the
        # Results panel's "Custom..." plot field (_extract_plot_field
        # below), but evaluated on live torch tensors during training
        # instead of a numpy array after predict(), so it uses real torch
        # ops (gradient-safe) rather than np.* (which errors or silently
        # detaches on a tensor that requires grad).
        _code_obs = compile(_expr, "<obs_custom_field>", "eval")
        def _f(inputs, outputs, X):
            _ns_obs = {{
                "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
                "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
                "arcsin": torch.arcsin, "arccos": torch.arccos, "arctan": torch.arctan,
                "exp": torch.exp, "log": torch.log, "log10": torch.log10,
                # +1e-12 baked into sqrt itself here (not just relying on a
                # default custom_expr string like "sqrt(u**2+v**2+1e-12)"
                # already spelling it out, e.g. 1D Schrodinger's |h|
                # default) -- unlike a plotting-only sqrt evaluation, this
                # namespace is what DeepXDE actually differentiates through
                # during training via PointSetOperatorBC, and
                # d/du sqrt(u**2+v**2) = u/sqrt(u**2+v**2) is genuinely
                # singular at u=v=0, which a freshly-initialized network can
                # (and does) land on exactly at some collocation points
                # early in training, producing an inf/nan gradient that
                # poisons every other loss term. Guarding the function
                # itself protects ANY custom expression using sqrt(...),
                # including ones a user hand-types without adding their own
                # epsilon, not just whichever default happens to spell it
                # out already. The shift is far below any physically
                # meaningful value for a sum-of-squares argument, so it
                # doesn't change what's being fit.
                "sqrt": lambda _v: torch.sqrt(_v + 1e-12), "abs": torch.abs, "ceil": torch.ceil,
                "floor": torch.floor, "pi": np.pi, "torch": torch,
            }}
            for _oi_ob, _on_ob in enumerate(_onames):
                _ns_obs[_on_ob.strip()] = outputs[:, _oi_ob:_oi_ob + 1]
            _r_ob = eval(_code_obs, _ns_obs)
            return _r_ob if _r_ob.dim() == 2 else _r_ob.reshape(-1, 1)
        return _f

    _obs_output_names_list = {repr(config.output_names)}.split(",")
    _obs_files = {_obs_files_literal}
    _obs_entries = []  # list of (xt, u, output_idx, weight, custom_expr), one per measured-data file
    for _of in _obs_files:
        _of_data = _load_data(_of["path"])
        _of_xt = _of_data[:, 0:_n_obs_coord_cols]
        _of_u  = _of_data[:, _n_obs_coord_cols:_n_obs_coord_cols + 1]
        _obs_entries.append((_of_xt, _of_u, _of.get("output_idx", 0),
                              _of.get("weight", 100.0), _of.get("custom_expr", "")))
        print(f"Loaded {{len(_of_xt)}} observation points from {{_of['path']}} (output {{_of.get('output_idx', 0)}}, weight {{_of.get('weight', 100.0)}})")
    # Kept as aliases to the first measured-data file's points, in case
    # anything downstream still expects the old single-file names.
    _obs_xt = _obs_entries[0][0] if _obs_entries else None
    _obs_u  = _obs_entries[0][1] if _obs_entries else None

    if "{config.inverse_ic_type}" == "File (x, t, u)":
        _ic_data = _load_data({repr(config.inverse_ic_file)})
        if _is_3d:
            _ic_xt = _ic_data[:, 0:4]  # x, y, z, t
            _ic_u  = _ic_data[:, 4:5]  # u
        elif _is_2d:
            _ic_xt = _ic_data[:, 0:3]  # x, y, t
            _ic_u  = _ic_data[:, 3:4]  # u
        else:
            _ic_xt = _ic_data[:, 0:2]  # x, t
            _ic_u  = _ic_data[:, 2:3]  # u
        print(f"Loaded {{len(_ic_xt)}} IC points from file")

    def _split_param_history_line(line):
        # Robustly split one dde.callbacks.VariableValue output line,
        # "<iter> [<v1>, <v2>, ...]", into (iter:int, raw_values_text:str)
        # without reparsing/reformatting the numeric values -- so the
        # precision the callback wrote is preserved exactly when this is
        # used to merge/append history files. Returns None if the line
        # doesn't look like a VariableValue line at all. Replaces the old
        # `.replace("[","").replace("]","").split()` approach, which broke
        # as soon as more than one comma-separated value appeared inside
        # the brackets (whitespace-split leaves a trailing comma on each
        # value, which float() rejects).
        line = line.strip()
        if not line or "[" not in line or "]" not in line:
            return None
        try:
            i0 = line.index("[")
            i1 = line.rindex("]")
            return int(float(line[:i0].strip())), line[i0 + 1:i1]
        except (ValueError, IndexError):
            return None

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
                print(f"  [{{self.name}}] Iter {{cur}}: {{val:.6f}}", flush=True)

    _print_cbs = [_PrintParamCallback(_iv, _in, period=1000) for _iv, _in in zip(_inv_vars, _inv_var_names)]

    # Parameter saving to text file -- one convergence file per variable,
    # all sharing the same save-period setting from the panel.
    _param_save_opt = "{config.inv_param_save}"
    _param_save_period = 100 if _param_save_opt == "Every 100 iters" else 1000 if _param_save_opt == "Every 1000 iters" else 0
    _param_save_paths = [
        _os.path.join(_save_dir if _use_save else "/tmp", f"{{_in}}_convergence.txt")
        for _in in _inv_var_names
    ] if _param_save_period > 0 else [None] * len(_inv_var_names)

    if _param_save_period > 0:
        for _in, _ip in zip(_inv_var_names, _param_save_paths):
            with open(_ip, "w") as _psf:
                _psf.write(f"iteration,{{_in}}\\n")
        print(f"Parameter convergence will be saved to: {{_param_save_paths}}")

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

    _save_cbs = [
        _SaveParamCallback(_iv, _in, _param_save_period, _ip if _ip else f"/tmp/{{_in}}_param_save.txt")
        for _iv, _in, _ip in zip(_inv_vars, _inv_var_names, _param_save_paths)
    ]


{_fecr_pde_block}
_IS_FECR = False
# ── PDE definition (standard) ───────────────────────────────
def _pde_standard(x, y):
    _n_out = {config.num_outputs}
    _out_names = [n.strip() for n in {repr(config.output_names)}.split(",")]
    _is_2d_pde = "{config.problem_dim}" == "2D"
    _is_3d_pde = "{config.problem_dim}" == "3D"

    def _hess(ys, xs, i, j):
        # DeepXDE 1.10.0 (the floor version pinned in setup.py) raises
        # "Do not use component for 1D y." if `component` is passed at all
        # for a single-output model, while multi-output models require it.
        # Newer deepxde releases accept it either way, so omitting `component`
        # only for the single-output case is safe across the whole supported range.
        if _n_out == 1:
            return dde.grad.hessian(ys, xs, i=i, j=j)
        return dde.grad.hessian(ys, xs, component=_oi, i=i, j=j)

    _dvars = {{}}
    for _oi, _oname in enumerate(_out_names):
        _dvars[_oname] = y[:, _oi:_oi+1]
        if _is_3d_pde:
            # 3D: inputs are (x, y, z, t) → j=0,1,2,3 -- or, for a steady
            # (time-independent) problem, just (x, y, z) → j=0,1,2, with no
            # time column at all, so the unconditional "_t" jacobian below
            # (which would otherwise index a column that doesn't exist) is
            # skipped entirely rather than computed and left unused.
            # Always compute first-order derivatives
            _dvars[f"d{{_oname}}_x"] = dde.grad.jacobian(y, x, i=_oi, j=0)
            _dvars[f"d{{_oname}}_y"] = dde.grad.jacobian(y, x, i=_oi, j=1)
            _dvars[f"d{{_oname}}_z"] = dde.grad.jacobian(y, x, i=_oi, j=2)
            if not _is_steady:
                _dvars[f"d{{_oname}}_t"] = dde.grad.jacobian(y, x, i=_oi, j=3)
            # Only compute second-order derivatives if used in PDE (pure and
            # mixed). 4th-order pure/mixed terms (xxxx/yyyy/zzzz/xxyy/xxzz/
            # yyzz/xxtt/yytt/zztt) are also supported, mirroring the 2D/1D
            # cases below, computed as the hessian of the matching 2nd-order
            # term (which the substring check above already guarantees gets
            # built, since e.g. "xx" is a substring of "xxxx"/"xxyy"/"xxzz"/
            # "xxtt").
            _pde_str_check = {repr(config_pde_expressions)}
            _oname_check = _oname
            if f"d{{_oname_check}}_xx" in _pde_str_check:
                _dvars[f"d{{_oname}}_xx"] = _hess(y, x, 0, 0)
            if f"d{{_oname_check}}_yy" in _pde_str_check:
                _dvars[f"d{{_oname}}_yy"] = _hess(y, x, 1, 1)
            if f"d{{_oname_check}}_zz" in _pde_str_check:
                _dvars[f"d{{_oname}}_zz"] = _hess(y, x, 2, 2)
            if f"d{{_oname_check}}_tt" in _pde_str_check:
                _dvars[f"d{{_oname}}_tt"] = _hess(y, x, 3, 3)
            if f"d{{_oname_check}}_xy" in _pde_str_check:
                _dvars[f"d{{_oname}}_xy"] = _hess(y, x, 0, 1)
            if f"d{{_oname_check}}_xz" in _pde_str_check:
                _dvars[f"d{{_oname}}_xz"] = _hess(y, x, 0, 2)
            if f"d{{_oname_check}}_yz" in _pde_str_check:
                _dvars[f"d{{_oname}}_yz"] = _hess(y, x, 1, 2)
            if f"d{{_oname_check}}_xt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xt"] = _hess(y, x, 0, 3)
            if f"d{{_oname_check}}_yt" in _pde_str_check:
                _dvars[f"d{{_oname}}_yt"] = _hess(y, x, 1, 3)
            if f"d{{_oname_check}}_zt" in _pde_str_check:
                _dvars[f"d{{_oname}}_zt"] = _hess(y, x, 2, 3)
            if f"d{{_oname_check}}_xxxx" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxxx"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            if f"d{{_oname_check}}_yyyy" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yyyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=1, j=1)
            if f"d{{_oname_check}}_zzzz" in _pde_str_check and f"d{{_oname}}_zz" in _dvars:
                _dvars[f"d{{_oname}}_zzzz"] = dde.grad.hessian(_dvars[f"d{{_oname}}_zz"], x, i=2, j=2)
            if f"d{{_oname_check}}_xxyy" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=1, j=1)
            if f"d{{_oname_check}}_xxzz" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxzz"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=2, j=2)
            if f"d{{_oname_check}}_yyzz" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yyzz"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=2, j=2)
            if f"d{{_oname_check}}_xxtt" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxtt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=3, j=3)
            if f"d{{_oname_check}}_yytt" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yytt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=3, j=3)
            if f"d{{_oname_check}}_zztt" in _pde_str_check and f"d{{_oname}}_zz" in _dvars:
                _dvars[f"d{{_oname}}_zztt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_zz"], x, i=3, j=3)
        elif _is_2d_pde:
            # 2D: inputs are (x, y, t) → j=0,1,2 -- or, for a steady
            # (time-independent) problem, just (x, y) → j=0,1, so the
            # unconditional "_t" jacobian below (which would otherwise
            # index a column that doesn't exist) is skipped entirely.
            # Always compute first-order derivatives
            _dvars[f"d{{_oname}}_x"] = dde.grad.jacobian(y, x, i=_oi, j=0)
            _dvars[f"d{{_oname}}_y"] = dde.grad.jacobian(y, x, i=_oi, j=1)
            if not _is_steady:
                _dvars[f"d{{_oname}}_t"] = dde.grad.jacobian(y, x, i=_oi, j=2)
            # Only compute higher-order derivatives if used in PDE
            _pde_str_check = {repr(config_pde_expressions)}
            _oname_check = _oname
            if f"d{{_oname_check}}_xx" in _pde_str_check or f"d{{_oname_check}}_xxxx" in _pde_str_check or f"d{{_oname_check}}_xxyy" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xx"] = _hess(y, x, 0, 0)
            if f"d{{_oname_check}}_yy" in _pde_str_check or f"d{{_oname_check}}_yyyy" in _pde_str_check or f"d{{_oname_check}}_xxyy" in _pde_str_check or f"d{{_oname_check}}_yytt" in _pde_str_check:
                _dvars[f"d{{_oname}}_yy"] = _hess(y, x, 1, 1)
            if f"d{{_oname_check}}_xy" in _pde_str_check:
                _dvars[f"d{{_oname}}_xy"] = _hess(y, x, 0, 1)
            if f"d{{_oname_check}}_tt" in _pde_str_check:
                _dvars[f"d{{_oname}}_tt"] = _hess(y, x, 2, 2)
            if f"d{{_oname_check}}_xt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xt"] = _hess(y, x, 0, 2)
            if f"d{{_oname_check}}_yt" in _pde_str_check:
                _dvars[f"d{{_oname}}_yt"] = _hess(y, x, 1, 2)
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
            # 1D: inputs are (x, t) → j=0,1 -- or, for a steady
            # (time-independent) problem, just (x) → j=0, so the
            # unconditional "_t" jacobian below (which would otherwise
            # index a column that doesn't exist) is skipped entirely.
            _dvars[f"d{{_oname}}_x"] = dde.grad.jacobian(y, x, i=_oi, j=0)
            if not _is_steady:
                _dvars[f"d{{_oname}}_t"] = dde.grad.jacobian(y, x, i=_oi, j=1)
            _pde_str_check = {repr(config_pde_expressions)}
            _oname_check = _oname
            if f"d{{_oname_check}}_xx" in _pde_str_check or f"d{{_oname_check}}_xxx" in _pde_str_check or f"d{{_oname_check}}_xxxx" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xx"] = _hess(y, x, 0, 0)
            if f"d{{_oname_check}}_tt" in _pde_str_check or f"d{{_oname_check}}_tttt" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_tt"] = _hess(y, x, 1, 1)
            if f"d{{_oname_check}}_xt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xt"] = _hess(y, x, 0, 1)
            # Third-order pure x-derivative (e.g. for the KdV equation's
            # u_xxx dispersive term): one more jacobian pass on top of the
            # already-computed u_xx, rather than a hessian (which would
            # give the 4th-order xxxx term instead -- see below).
            if f"d{{_oname_check}}_xxx" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxx"] = dde.grad.jacobian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
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
        for _iv_name, _iv_var in zip(_inv_var_names, _inv_vars):
            _eval_ns[_iv_name] = _iv_var

    if _n_out == 1:
        return eval({repr(pde_expr_single)}, _eval_ns)
    else:
        _pde_exprs = {repr(config_pde_expressions)}.split("|")
        return [eval(_expr.strip(), _eval_ns) for _expr in _pde_exprs]
if not _IS_FECR:
    pde = _pde_standard

# ── Geometry & Time domain ──────────────────────────────────
# _build_geom() is the one place that turns the Geometry Type selector
# (Rectangle/Disk/Ellipse/Triangle/Polygon in 2D; Cuboid/Sphere in 3D;
# Interval for 1D) into an actual DeepXDE geometry object -- every other
# geometry-construction site in this script (IC pre-training, Time-Adaptive
# per-step) calls this same helper instead of re-deriving the shape, so
# there's exactly one place that needs to know each shape's constructor.
_geom_type = {_geometry_type_literal}
# The legacy per-side BC scheme below (bc_left/right/bottom/top) only ever
# meant "x = x_min" / "x = x_max" / etc, so it only makes sense for a box
# shape -- it's kept purely for configs saved before the Boundary
# Conditions panel existed, and every one of those was necessarily a box
# shape (Disk/Ellipse/Triangle/Polygon/Sphere didn't exist as options back
# then). So for a non-box shape, an empty panel should just mean "no
# custom BCs" (PDE + IC only) rather than silently reinterpreting hidden,
# always-on left/right/bottom/top widgets that don't correspond to
# anything the user can see or edit for that shape.
_box_shape_geom = _geom_type in ("Interval", "Rectangle", "Cuboid")

# Disk/Ellipse/Triangle/Polygon/Sphere -- in this installed DeepXDE version,
# some of these (observed on Ellipse.random_points, Triangle.random_points,
# Polygon.random_boundary_points) don't consistently respect
# dde.config.set_default_float(): they return float64 point arrays even
# when the network itself is float32, which crashes training the moment
# DeepXDE evaluates those points through the net (dtype mismatch error).
# Rather than force the whole run to float64 (worse memory/speed) just for
# these shapes, wrap them so every sampled point array is cast to whatever
# float type the run is actually using -- everything else (on_boundary,
# inside, bbox, dim, ...) passes straight through to the real geometry via
# __getattr__, same adapter pattern as _Interface2DGeomAdapter below.
class _DTypeSafeGeom:
    def __init__(self, geom):
        self._geom = geom
    def __getattr__(self, name):
        return getattr(self._geom, name)
    def random_points(self, n, random="pseudo"):
        return self._geom.random_points(n, random=random).astype(dde.config.real(np))
    def uniform_points(self, n, boundary=True):
        return self._geom.uniform_points(n, boundary=boundary).astype(dde.config.real(np))
    def random_boundary_points(self, n, random="pseudo"):
        return self._geom.random_boundary_points(n, random=random).astype(dde.config.real(np))
    def uniform_boundary_points(self, n):
        return self._geom.uniform_boundary_points(n).astype(dde.config.real(np))

def _build_geom():
    if _geom_type == "Disk":
        return _DTypeSafeGeom(dde.geometry.Disk([{config.geom_center_x}, {config.geom_center_y}], {config.geom_radius}))
    elif _geom_type == "Ellipse":
        return _DTypeSafeGeom(dde.geometry.Ellipse([{config.geom_center_x}, {config.geom_center_y}],
                                     {config.geom_semi_major}, {config.geom_semi_minor}, {config.geom_angle}))
    elif _geom_type == "Triangle":
        _tv = {_triangle_vertices_literal}
        return _DTypeSafeGeom(dde.geometry.Triangle(_tv[0], _tv[1], _tv[2]))
    elif _geom_type == "Polygon":
        return _DTypeSafeGeom(dde.geometry.Polygon({_polygon_vertices_literal}))
    elif _geom_type == "Sphere":
        return _DTypeSafeGeom(dde.geometry.Sphere([{config.geom_center_x}, {config.geom_center_y}, {config.geom_center_z}],
                                    {config.geom_radius}))
    elif _geom_type == "Custom":
        return {_custom_geom_final_literal}
    elif _is_2d:
        return dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])
    elif _is_3d:
        return dde.geometry.Cuboid([{config.x_min}, {config.y_min}, {config.z_min}],
                                    [{config.x_max}, {config.y_max}, {config.z_max}])
    else:
        return dde.geometry.Interval({config.x_min}, {config.x_max})

geom = _build_geom()
if _is_steady:
    # No time axis at all -- every downstream BC constructor (DirichletBC,
    # NeumannBC, PeriodicBC, OperatorBC, ...) only ever needs a plain
    # Geometry (on_boundary/random_points/bbox/...), not specifically a
    # GeometryXTime, so aliasing geomtime straight to geom lets every BC-
    # construction call site below stay completely unchanged. Only the IC
    # section (which needs GeometryXTime's on_initial) and the "Data"
    # section (TimePDE vs plain PDE) below actually branch on _is_steady.
    geomtime = geom
else:
    timedomain = dde.geometry.TimeDomain({config.t_min}, {config.t_max})
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

# Bounding box of the ACTUAL shape (every DeepXDE geometry exposes this),
# used for plot/export grids so a Disk/Ellipse/Triangle/Polygon/Sphere gets
# plotted over its own true extent instead of the plain x_min/x_max/y_min/
# y_max fields, which only mean something for the box shapes (those fields
# are hidden in the UI for every other shape and can hold stale values).
_geom_bbox_arr = np.asarray(geom.bbox, dtype=float)
_plot_x_min, _plot_x_max = float(_geom_bbox_arr[0][0]), float(_geom_bbox_arr[1][0])
_plot_y_min = float(_geom_bbox_arr[0][1]) if _geom_bbox_arr.shape[1] > 1 else 0.0
_plot_y_max = float(_geom_bbox_arr[1][1]) if _geom_bbox_arr.shape[1] > 1 else 1.0
_plot_z_min = float(_geom_bbox_arr[0][2]) if _geom_bbox_arr.shape[1] > 2 else {config.z_min}
_plot_z_max = float(_geom_bbox_arr[1][2]) if _geom_bbox_arr.shape[1] > 2 else {config.z_max}

# ── Boundary & Initial conditions ────────────────────────────
_out_names_list  = {repr(config.output_names)}.split(",")
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

# ── Boundary conditions ──────────────────────────────────────
# The Boundary Conditions panel (custom_bc_json) is now the single
# source of truth for BCs -- whatever the panel shows (hand-built,
# or pre-filled-but-editable from a template) is exactly what
# trains. The legacy per-side fields below are kept only as a
# fallback for configs saved before this panel existed.
import json as _bc_json_mod
_custom_bc_entries = _bc_json_mod.loads({_custom_bc_json_literal}) if {_custom_bc_json_literal} else []
_bc_panel_data_present = {_bc_panel_data_present_literal}
# One fewer coordinate column for a steady-state problem -- no time axis
# at all, so a BC point's last column is whatever spatial axis is last
# (z for 3D, y for 2D, x for 1D), not time.
_n_coord_cols_bc = (3 if _is_3d else (2 if _is_2d else 1)) if _is_steady else (4 if _is_3d else (3 if _is_2d else 2))


# Named math functions/constants available directly (unprefixed) in every
# Boundary Conditions panel expression field ("Where:"/"Value:"), matching
# the fields' own placeholder examples (e.g. "sin(pi*y)") -- unlike the
# PDE/IC expression boxes, these are NOT run through a "np." text-prefixing
# pass first (see _simplify_expr above), so without this dict "sin(pi*y)"
# or "exp(...)" would raise a plain NameError the first time anyone typed
# one, since eval()'s namespace otherwise only has x/y/z/t/np in it.
_BC_MATH_NS = {{
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "exp": np.exp, "log": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "ceil": np.ceil, "floor": np.floor,
    "pi": np.pi,
}}

def _bc_loc_fn(_expr):
    _src = _expr if _expr.strip() else "True"
    _code = compile(_src, "<bc_location>", "eval")
    def _f(_X, _on):
        if not _on:
            return False
        _ns = {{**_BC_MATH_NS, "x": _X[0], "np": np}}
        if not _is_steady:
            _ns["t"] = _X[_n_coord_cols_bc - 1]
        if _is_2d or _is_3d:
            _ns["y"] = _X[1]
        if _is_3d:
            _ns["z"] = _X[2]
        return bool(eval(_code, _ns))
    return _f

def _bc_val_fn(_expr):
    _src = _expr if _expr.strip() else "0"
    _code = compile(_src, "<bc_value>", "eval")
    def _f(_Xb):
        _ns = {{**_BC_MATH_NS, "x": _Xb[:, 0], "np": np}}
        if not _is_steady:
            _ns["t"] = _Xb[:, _n_coord_cols_bc - 1]
        if _is_2d or _is_3d:
            _ns["y"] = _Xb[:, 1]
        if _is_3d:
            _ns["z"] = _Xb[:, 2]
        _r = np.asarray(eval(_code, _ns), dtype=float)
        return np.full((len(_Xb), 1), float(_r)) if _r.ndim == 0 else _r.reshape(-1, 1)
    return _f

def _bc_robin_fn(_expr, _comp):
    # Robin's func(X, y) is NOT wrapped by DeepXDE the way Dirichlet/Neumann
    # are (no return_tensor()) -- error() subtracts our return value directly
    # from a live torch tensor (normal_derivative(...)), so whatever this
    # returns has to already be a torch tensor on the same device/dtype as
    # _Yb, whether the expression references u (already a tensor) or is a
    # plain numpy/constant expression in x, y, z.
    _src = _expr if _expr.strip() else "0"
    _code = compile(_src, "<bc_robin_value>", "eval")
    def _f(_Xb, _Yb):
        _ns = {{**_BC_MATH_NS, "x": _Xb[:, 0], "np": np, "torch": torch, "u": _Yb[:, _comp:_comp + 1]}}
        if _is_2d or _is_3d:
            _ns["y"] = _Xb[:, 1]
        if _is_3d:
            _ns["z"] = _Xb[:, 2]
        _r = eval(_code, _ns)
        if isinstance(_r, torch.Tensor):
            return _r if _r.dim() == 2 else _r.reshape(-1, 1)
        _arr = np.asarray(_r, dtype=float)
        if _arr.ndim == 0:
            return _Yb.new_full((len(_Xb), 1), float(_arr))
        return _Yb.new_tensor(_arr.reshape(-1, 1))
    return _f

def _bc_operator_fn(_expr):
    _src = _expr if _expr.strip() else "0"
    _code = compile(_src, "<bc_operator>", "eval")
    def _f(_inputs, _outputs, _X):
        return eval(_code, {{"inputs": _inputs, "outputs": _outputs, "X": _X, "np": np, "torch": torch}})
    return _f

def _bc_load_points_file(_path, _bc_label):
    if not _path or not _os.path.isfile(_path):
        raise RuntimeError(f"Boundary Conditions panel: {{_bc_label}} needs a points file, but "
                            f"'{{_path}}' was not found. Browse for a valid file in that BC's row.")
    _data = np.loadtxt(_path, delimiter=",") if _path.lower().endswith(".csv") else np.loadtxt(_path)
    if _data.ndim == 1:
        _data = _data.reshape(1, -1)
    return _data[:, :_n_coord_cols_bc], _data[:, _n_coord_cols_bc:_n_coord_cols_bc + 1]

_axis_idx_map = {{"x": 0, "y": 1, "z": 2}}

class _Interface2DGeomAdapter:
    # Interface2DBC's error() dots the network output (one value per
    # spatial dimension -- e.g. 2 for a 2D problem) against
    # geom.boundary_normal(X); GeometryXTime.boundary_normal() appends an
    # extra always-zero "time normal" column, which breaks that dot
    # product (2-vector against a 3-vector). on_boundary() is unaffected
    # (GeometryXTime already strips the time column itself), so this
    # adapter only needs to trim boundary_normal's result back down to
    # the spatial dimensions Interface2DBC actually expects.
    def __init__(self, gt):
        self._gt = gt
    def on_boundary(self, x):
        return self._gt.on_boundary(x)
    def boundary_normal(self, x):
        return self._gt.boundary_normal(x)[:, :-1]

if _custom_bc_entries or _bc_panel_data_present:
    for _bi, _be in enumerate(_custom_bc_entries):
        _btype = _be.get('type', 'dirichlet')
        _bcomp = int(_be.get('component', 0) or 0)
        _bloc  = _be.get('location', '')
        _bval  = _be.get('value', '0')
        _baxis = _be.get('axis', 'x')
        _bderiv = int(_be.get('deriv_order', 0) or 0)
        _bpts_file = _be.get('points_file', '')
        _bloc2 = _be.get('location2', '')
        _bdir  = _be.get('direction', 'normal')
        _blabel = f"BC {{_bi + 1}} ({{_btype}})"

        if _btype == 'neumann':
            _constraints.append(dde.icbc.NeumannBC(geomtime, _bc_val_fn(_bval), _bc_loc_fn(_bloc), component=_bcomp))
        elif _btype == 'robin':
            _constraints.append(dde.icbc.RobinBC(geomtime, _bc_robin_fn(_bval, _bcomp), _bc_loc_fn(_bloc), component=_bcomp))
        elif _btype == 'periodic':
            _constraints.append(dde.icbc.PeriodicBC(geomtime, _axis_idx_map.get(_baxis, 0), _bc_loc_fn(_bloc),
                                                      derivative_order=_bderiv, component=_bcomp))
        elif _btype == 'pointset':
            _pts, _pvals = _bc_load_points_file(_bpts_file, _blabel)
            _constraints.append(dde.icbc.PointSetBC(_pts, _pvals, component=_bcomp))
        elif _btype == 'pointset_operator':
            _pts, _pvals = _bc_load_points_file(_bpts_file, _blabel)
            _constraints.append(dde.icbc.PointSetOperatorBC(_pts, _pvals, _bc_operator_fn(_bval)))
        elif _btype == 'operator':
            _constraints.append(dde.icbc.OperatorBC(geomtime, _bc_operator_fn(_bval), _bc_loc_fn(_bloc)))
        elif _btype == 'interface2d':
            _constraints.append(dde.icbc.Interface2DBC(_Interface2DGeomAdapter(geomtime), _bc_val_fn(_bval),
                                                         _bc_loc_fn(_bloc), _bc_loc_fn(_bloc2), direction=_bdir))
        else:  # 'dirichlet' and any unrecognized type fall back to Dirichlet
            _constraints.append(dde.icbc.DirichletBC(geomtime, _bc_val_fn(_bval), _bc_loc_fn(_bloc), component=_bcomp))

    # ── IC (independent of the BC panel -- same IC config as always;
    # skipped entirely for a steady-state problem, which has no initial
    # time slice to speak of -- dde.icbc.IC()'s on_initial() would have
    # nothing to match against anyway once geomtime is just a plain
    # spatial geometry, see above) ──
    # v19: the Inverse panel's separate "IC type" (expression/file)
    # selector was removed -- confusing to have two places to set the
    # IC when the Initial Condition panel above already covers both
    # expression and file-based ICs, for every problem type and
    # dimension. This branch is permanently disabled (never True) so
    # every problem -- inverse included -- now goes through that one
    # panel's own logic just below. _ic_xt/_ic_u above are kept only
    # so a config saved before this change (inverse_ic_type could
    # still be "File (x, t, u)" in it) still loads without error.
    if _is_steady:
        pass
    elif False:
        _constraints.append(dde.icbc.PointSetBC(_ic_xt, _ic_u, component=0))
    else:
        for _oi in range({config.num_outputs}):
            _comp = _oi
            if {config.forward_ic_from_file} and _oi == 0:
                # Load IC from file -- see _load_ic_from_file() above for
                # the expected column layout per dimension.
                _ic_xyt, _ic_vals = _load_ic_from_file({repr(config.forward_ic_file)})
                _constraints.append(dde.icbc.PointSetBC(_ic_xyt, _ic_vals, component=0))
                _ic_file_disp = {repr(config.forward_ic_file)}
                print(f"IC loaded from file: {{_ic_file_disp}} — {{len(_ic_xyt)}} points")
            elif _oi < len(_ic_active_list) and _ic_active_list[_oi].strip() == "True":
                _ic_expr = _ic_expressions[_oi].strip() if _oi < len(_ic_expressions) else "np.zeros_like(x[:,0])"
                def _make_ic(expr, comp):
                    def _ic_fn(x):
                        if x.ndim == 1:
                            x = x.reshape(-1, 1)
                        return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                    return dde.icbc.IC(geomtime, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                _constraints.append(_make_ic(_ic_expr, _comp))
else:
    # Legacy per-side BCs, for configs saved before the Boundary
    # Conditions panel existed (custom_bc_json empty).
    # v19: the Inverse panel's separate "IC type" (expression/file)
    # selector was removed -- confusing to have two places to set the
    # IC when the Initial Condition panel above already covers both
    # expression and file-based ICs, for every problem type and
    # dimension. This branch is permanently disabled (never True) so
    # every problem -- inverse included -- now goes through that one
    # panel's own logic just below. _ic_xt/_ic_u above are kept only
    # so a config saved before this change (inverse_ic_type could
    # still be "File (x, t, u)" in it) still loads without error.
    if False:
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

            # ── IC (skipped entirely for a steady-state problem, same
            # reasoning/fix as the Boundary-Conditions-panel-active branch
            # above: a steady-state geomtime is just a plain spatial
            # geometry with no on_initial() to match against, so building a
            # dde.icbc.IC() here crashed with "'Rectangle' object has no
            # attribute 'on_initial'" the moment any output's IC was
            # active -- caught via an actual exec-level steady-state
            # training run through this exact legacy branch, which only
            # the custom_bc_json-empty path reaches (any config built by
            # the current GUI always sends a non-empty custom_bc_json,
            # even "[]", routing to the other branch instead -- so this hit
            # only an old pre-Boundary-Conditions-panel config, loaded with
            # steady_state=True and one of its legacy ic_active entries
            # still "True") ──────────────────────────────────────────
            if _is_steady:
                pass
            elif {config.forward_ic_from_file} and _oi == 0:
                # Load IC from file -- see _load_ic_from_file() above for
                # the expected column layout per dimension.
                _ic_xyt, _ic_vals = _load_ic_from_file({repr(config.forward_ic_file)})
                _constraints.append(dde.icbc.PointSetBC(_ic_xyt, _ic_vals, component=0))
                _ic_file_disp = {repr(config.forward_ic_file)}
                print(f"IC loaded from file: {{_ic_file_disp}} — {{len(_ic_xyt)}} points")
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
_ic_a  = "{config.ic_active}".split(",")
_wm_list = [float(v) for v in "{config.loss_weights_multi}".split(",") if v.strip()] if "{config.loss_weights_multi}".strip() else []

_pde_w = []
_wi = 0
for _oi_w in range(_n_out_w):
    _pde_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1

if _custom_bc_entries or _bc_panel_data_present:
    _bc_entry_w = []
    for _bi_w in range(len(_custom_bc_entries)):
        _bc_entry_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1

    _ic_w = []
    for _oi_w in range(_n_out_w):
        if not _is_steady and ((_oi_w == 0 and {config.forward_ic_from_file}) or (_oi_w < len(_ic_a) and _ic_a[_oi_w].strip() == "True")):
            _ic_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _ic_w.append(None); _wi += 1 if (not _is_steady and _wi < len(_wm_list)) else 0

    _multi_weights = list(_pde_w) + list(_bc_entry_w)
    for _oi_w in range(_n_out_w):
        if _ic_w[_oi_w] is not None:
            _multi_weights.append(_ic_w[_oi_w])

    print(f"  PDE weights: {{_pde_w}}")
    print(f"  BC weights ({{len(_custom_bc_entries)}} rows from the Boundary Conditions panel): {{_bc_entry_w}}")
    print(f"  IC weights: {{[w for w in _ic_w if w is not None]}}")
else:
    # Legacy per-side weights, for configs saved before the Boundary
    # Conditions panel existed (custom_bc_json empty).
    _bc_la = "{config.bc_left_active}".split(",")
    _bc_ra = "{config.bc_right_active}".split(",")
    _bc_ba = "{config.bc_bottom_active}".split(",")
    _bc_ta = "{config.bc_top_active}".split(",")

    _bcl_w = []; _bcr_w = []; _bcb_w = []; _bct_w = []; _ic_w = []
    for _oi_w in range(_n_out_w):
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
        # Same _is_steady guard as the constraints block above and as the
        # Boundary-Conditions-panel-active branch's own _ic_w loop just
        # above (which already checks "not _is_steady and (...)") -- a
        # steady-state problem builds no IC constraint at all now, so a
        # weight entry for one here (this legacy branch forgot the check
        # before this fix) would desync _multi_weights from the real
        # constraints list by one slot, and shift every weight after it
        # for any later output too.
        if (not _is_steady) and ((_oi_w == 0 and {config.forward_ic_from_file}) or (_oi_w < len(_ic_a) and _ic_a[_oi_w].strip() == "True")):
            _ic_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _ic_w.append(None); _wi += 1 if (not _is_steady and _wi < len(_wm_list)) else 0

    _bc_left_deriv_active  = "{config.bc_left_deriv}".split(",")
    _bc_bottom_deriv_active = "{config.bc_bottom_deriv}".split(",")
    _multi_weights = []
    for _oi_w in range(_n_out_w):
        _multi_weights.append(_pde_w[_oi_w])
    for _oi_w in range(_n_out_w):
        if _bcl_w[_oi_w] is not None:
            _multi_weights.append(_bcl_w[_oi_w])
            # Add extra weight for derivative periodic BC if enabled -- this
            # must check the LEFT edge's own periodic/derivative flags, not
            # the bottom edge's (a copy-paste bug: this block was gated on
            # _bbt_check/_bc_bottom_deriv_active, the exact same variables
            # the bottom block below correctly uses for itself, so a
            # left-side periodic+derivative BC never got its extra weight
            # slot at all while a bottom-side one could spuriously trigger
            # this block instead, desyncing every loss weight after it).
            _blt_is_per = _oi_w < len(_blt_check) and _blt_check[_oi_w].strip() == "Periodic"
            _bld_is_active = _oi_w < len(_bc_left_deriv_active) and _bc_left_deriv_active[_oi_w].strip() == "True"
            if _blt_is_per and _bld_is_active:
                _multi_weights.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        if _bcr_w[_oi_w] is not None: _multi_weights.append(_bcr_w[_oi_w])
        if _bcb_w[_oi_w] is not None:
            _multi_weights.append(_bcb_w[_oi_w])
            # Add extra weight for derivative periodic BC if enabled -- same
            # fix as the left-edge block above: must actually consume the
            # next GUI-provided weight from _wm_list and advance the shared
            # _wi cursor, not hardcode 1.0. A hardcoded weight here silently
            # discards whatever weight the user set for this BC, AND (since
            # _wi was never advanced) desyncs every weight slot still to be
            # read after it for this or a later output.
            _bbt_is_per = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"
            _bbd_is_active = _oi_w < len(_bc_bottom_deriv_active) and _bc_bottom_deriv_active[_oi_w].strip() == "True"
            if _bbt_is_per and _bbd_is_active:
                _multi_weights.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        if _bct_w[_oi_w] is not None: _multi_weights.append(_bct_w[_oi_w])
        if _ic_w[_oi_w]  is not None: _multi_weights.append(_ic_w[_oi_w])

    print(f"  PDE weights: {{_pde_w}}")
    print(f"  BC left: {{[w for w in _bcl_w if w is not None]}}")
    print(f"  BC right: {{[w for w in _bcr_w if w is not None]}}")
    if _is_2d:
        print(f"  BC bottom: {{[w for w in _bcb_w if w is not None]}}")
        print(f"  BC top: {{[w for w in _bct_w if w is not None]}}")
    print(f"  IC weights: {{[w for w in _ic_w if w is not None]}}")

if _problem_type == "Inverse":
    # One observation-loss weight per measured-data file, same order as
    # the PointSetBC constraints appended just below.
    _multi_weights = _multi_weights + [_e[3] for _e in _obs_entries]

print(f"Loss weights: {{_multi_weights}} ({{len(_multi_weights)}} terms for {{len(_constraints)}} constraints)")

# ── Data ─────────────────────────────────────────────────────
# Steady-state problems have no time axis/Initial Condition, so they use a
# plain dde.data.PDE (spatial geometry only, no num_initial=) instead of
# dde.data.TimePDE -- geomtime is already just an alias for geom in that
# case (see the geometry-construction section above).
if _problem_type == "Inverse":
    _obs_bcs = [
        dde.icbc.PointSetOperatorBC(_e[0], _e[1], _make_obs_func(_e[4], _obs_output_names_list))
        if _e[4] else dde.icbc.PointSetBC(_e[0], _e[1], component=_e[2])
        for _e in _obs_entries
    ]
    _constraints.extend(_obs_bcs)
    _obs_anchors = np.vstack([_e[0] for _e in _obs_entries]) if _obs_entries else None
    if _is_steady:
        data = dde.data.PDE(
            geomtime, pde, _constraints,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_test={config.num_test},
            train_distribution="{config.point_distribution}",
            anchors=_obs_anchors
        )
    else:
        data = dde.data.TimePDE(
            geomtime, pde, _constraints,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_initial={config.num_initial}, num_test={config.num_test},
            train_distribution="{config.point_distribution}",
            anchors=_obs_anchors
        )
else:
    if _is_steady:
        data = dde.data.PDE(
            geomtime, pde, _constraints,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_test={config.num_test},
            train_distribution="{config.point_distribution}",
            anchors=None
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
    # NOTE: _loss_weights (and the four branches below that mutate it --
    # ic_weight/pde_weight/bc_left_weight/bc_right_weight) predate this
    # codebase's multi-output loss-weights system and are currently dead:
    # model.compile() below passes loss_weights=_multi_weights (the real,
    # general per-PDE/per-BC/per-IC weights list, built earlier in this
    # function), never _loss_weights -- so sweeping any of these four
    # parametric options would silently do nothing. Harmless today since
    # Parametric Study has no GUI controls at all yet (nothing can reach
    # this code path), but worth fixing for real -- route the swept value
    # into the matching slot of _multi_weights instead -- before ever
    # wiring up a GUI for it, rather than leaving a "parametric sweep"
    # feature that quietly does nothing for 4 of its 6 options.
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

    net = _apply_net_transforms(_make_net(_layers, "{config.activation}", "{config.kernel_initializer}", {config.weight_decay}))
    model = dde.Model(data, net)

    model.compile(
        "{config.optimizer}", lr=_lr, loss="{config.loss_type}",
        loss_weights=_multi_weights,
        external_trainable_variables=_inv_vars if _problem_type == "Inverse" else None
    )

    _iters = int(_pval) if (_parametric and _param_name == "phase1_iterations" and _pval is not None) else {config.iterations}

    # ── Training callbacks (opt-in: EarlyStopping/PDEPointResampler/
    # ModelCheckpoint/Timer) — see _build_train_cbs_code() in codegen.py
    # for exactly what's included and why; empty list when none are
    # enabled, which every mainline .train() call below appends as a no-op.
{_train_cbs_code}

    # ── IC Pre-Training ───────────────────────────────────────
    if {config.ic_pretrain} and not {config.time_adaptive}:
        print(f"\\n=== IC Pre-Training: {config.ic_pretrain_iterations} iterations (IC loss only) ===")
        # Build IC-only geometry and data — no domain/BC points, only IC
        _ic_geom_pre = _build_geom()
        _ic_td_pre  = dde.geometry.TimeDomain({config.t_min}, {config.t_max})
        _ic_gt_pre  = dde.geometry.GeometryXTime(_ic_geom_pre, _ic_td_pre)
        _ic_ics_pre = []
        if {config.forward_ic_from_file}:
            # Load IC from file for pre-training -- see _load_ic_from_file()
            # near the top of this script for the expected column layout.
            _ic_pre_xyt, _ic_pre_vals = _load_ic_from_file({repr(config.forward_ic_file)})
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
        
        # IC pre-training — dummy PDE (zero residual, same output count as
        # the real network), IC points only. Real boundary conditions are
        # deliberately left OUT of this dataset entirely, not just given a
        # 0 loss weight: with no boundary points sampled here (num_boundary
        # stays 0 -- only the initial-time slice matters for this step), a
        # BC term's collocation match is usually empty, and DeepXDE's
        # MSE(empty, empty) is NaN -- which a 0 weight does NOT neutralize
        # (0 * nan is still nan), silently corrupting every other loss term
        # and the whole gradient. Excluding these terms sidesteps that trap
        # while still training only the initial condition, as intended.
        # num_initial/num_test use their own IC-pre-training-specific config
        # fields (not the main run's), and must be nonzero -- with them at
        # 0, no points are ever sampled at all and construction itself
        # fails with an unrelated-looking IndexError deep inside DeepXDE.
        def _pde_dummy_pre(x, y):
            return [y[:, _oi_d:_oi_d+1] * 0 for _oi_d in range({config.num_outputs})]
        _ic_pre_constraints = _ic_ics_pre
        _data_pre = dde.data.TimePDE(
            _ic_gt_pre, _pde_dummy_pre, _ic_pre_constraints,
            num_domain=0, num_boundary=0,
            num_initial={_ic_pretrain_num_initial_safe}, num_test={config.ic_pretrain_num_test},
            train_distribution="{config.point_distribution}",
            anchors=None
        )
        _model_pre = dde.Model(_data_pre, net)
        _ic_only_weights = [0.0] * {config.num_outputs} + [1000.0] * len(_ic_ics_pre)
        print(f"  IC-only weights: {{_ic_only_weights}} — dummy PDE, IC points only")
        _model_pre.compile("{config.ic_pretrain_optimizer}", lr={config.ic_pretrain_lr},
                           loss="{config.ic_pretrain_loss}", loss_weights=_ic_only_weights)
        _ic_pre_save_dir = _os.path.join({repr(config.save_dir)}, "ic_pretrain")
        _os.makedirs(_ic_pre_save_dir, exist_ok=True)
        if {config.ic_pretrain_restore} and {repr(config.ic_pretrain_restore_path)} and _os.path.exists({repr(config.ic_pretrain_restore_path)}):
            print("  Restoring IC pre-train model from: " + {repr(config.ic_pretrain_restore_path)})
            _ic_ckpt = torch.load({repr(config.ic_pretrain_restore_path)}, map_location="cpu")
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
                    _inv_vars, period=1000, filename="/tmp/param_history.txt",
                    precision=6
                )
                loss_history, train_state = model.train(iterations=_iters, display_every={config.loss_display_every}, callbacks=[_var_cb] + _print_cbs + _save_cbs + _train_cbs)
                # A plain Adam-only Inverse run (scheduler off, optimizer2
                # "none") is a terminal phase just like the Forward case
                # below -- nothing runs afterward to save the model, so it
                # must be saved right here or no checkpoint is ever written
                # at all for this configuration (previously this branch had
                # no save of any kind, not even a config dump).
                if _use_save and "{config.optimizer2}" == "none":
                    _adam_model_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}")
                    model.save(_adam_model_path)
                    _adam_cfg_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}.json")
                    with open(_adam_cfg_path, "w") as _acf:
                        _json.dump(_model_config, _acf, indent=2)
                    print(f"Adam model saved: {{_adam_model_path}}.pt")
                    print(f"Adam config saved to: {{_adam_cfg_path}}")
            else:
                # Scheduler phases below define all training — skip this
                # standalone _iters-iteration pass so training only runs
                # for the iteration counts configured in the scheduler.
                print(f"  (Scheduler enabled — skipping standalone {{_iters}}-iteration pass; "
                      f"training runs only for the phases below)")
        else:
            if not _sched_active:
                # Mirrors the Inverse branch above (which already trains its
                # Phase 1 here when the Training Phases scheduler is off) --
                # previously this Forward branch never called model.train()
                # under any condition, so a Forward problem with the
                # scheduler off either silently skipped Phase 1 entirely
                # (going straight into Phase 2 from a random, untrained
                # network, when optimizer2 was set) or left
                # loss_history/train_state unbound and crashed with a
                # confusing NameError at dde.saveplot() (when optimizer2
                # was also "none").
                loss_history, train_state = model.train(iterations=_iters, display_every={config.loss_display_every}, callbacks=_train_cbs)
            if _use_save:
                # A plain Adam-only run (scheduler off, optimizer2 "none")
                # is itself the terminal phase -- the scheduler-phase loop
                # and the Phase 2 branch below each save their own final
                # checkpoint, but neither runs in this configuration, so
                # the checkpoint must be saved here or none is ever written
                # (this used to only write the config JSON, never the
                # actual model.save() checkpoint -- a silent data-loss bug:
                # Model Restore would find nothing for this exact setup).
                if not _sched_active and "{config.optimizer2}" == "none":
                    _adam_model_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}")
                    model.save(_adam_model_path)
                    print(f"Adam model saved: {{_adam_model_path}}.pt")
                else:
                    pass  # model saved after scheduler phases / Phase 2 below
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
                # The GUI computes each phase's weight string from the Boundary
                # Conditions panel's row count at the time the phase was set up;
                # if the panel (or geometry) changed afterwards without the
                # scheduler panel being rebuilt, this can end up a different
                # length than the constraints actually being trained on, which
                # DeepXDE has no graceful way to handle (a hard crash multiplying
                # mismatched-length tensors). Fall back to the freshly computed,
                # always-correct-length weights rather than crash mid-training.
                _sp_expected_len = {config.num_outputs} + len(_constraints)
                if len(_sp_weights) != _sp_expected_len:
                    print(f"  ⚠️ Scheduler Phase {{_sp_i+1}}: configured weights have {{len(_sp_weights)}} "
                          f"entries but {{_sp_expected_len}} loss terms are active "
                          f"({{len(_constraints)}} constraints + {config.num_outputs} PDE) — the Boundary "
                          f"Conditions panel or geometry likely changed after this phase's weights were set. "
                          f"Using the freshly computed weights for this phase instead.")
                    _sp_weights = list(_multi_weights)
                _sp_ext_vars = _inv_vars if _problem_type == "Inverse" else None
                if _sp['optimizer'] == 'lbfgs':
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter=_sp['iterations'],
                        maxfun=int(_sp['iterations']*1.25), maxls={config.lbfgs_maxls})
                    _lbfgs_float = "{config.lbfgs_float_type}"
                    model.compile("L-BFGS", loss=_sp.get('loss', '{config.loss_type}'),
                                  loss_weights=_sp_weights, external_trainable_variables=_sp_ext_vars)
                    if _problem_type == "Inverse":
                        for _icb in _print_cbs: _icb.set_offset(_sched_cum_iters)
                        for _icb in _save_cbs: _icb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            _inv_vars, period=200,
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            display_every={config.loss_display_every}, callbacks=[_sp_var_cb] + _print_cbs + _save_cbs + _train_cbs)
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _sp_parsed = _split_param_history_line(_spl)
                                    if _sp_parsed is not None:
                                        _sp_it, _sp_raw = _sp_parsed
                                        _fa.write(f"{{_sp_it}} [{{_sp_raw}}]\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(display_every={config.loss_display_every}, callbacks=_train_cbs)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_lbfgs-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} L-BFGS model saved: {{_nta_save_path}}.pt")
                elif _sp['optimizer'] == 'nncg':
                    # NNCG (deepxde>=1.13.0, version-guarded above): exact-case
                    # "NNCG" name required, no lr= (ignored with a warning --
                    # NysNewtonCG's own step size is configured via
                    # dde.optimizers.set_NNCG_options(), left at its defaults
                    # here since this app doesn't expose those knobs), and no
                    # LR decay (decay is meaningless for a Newton-CG method).
                    model.compile("NNCG", loss=_sp.get('loss', '{config.loss_type}'),
                                  loss_weights=_sp_weights, external_trainable_variables=_sp_ext_vars)
                    if _problem_type == "Inverse":
                        for _icb in _print_cbs: _icb.set_offset(_sched_cum_iters)
                        for _icb in _save_cbs: _icb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            _inv_vars, period=1000,
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            iterations=_sp['iterations'], display_every={config.loss_display_every},
                            callbacks=[_sp_var_cb] + _print_cbs + _save_cbs + _train_cbs)
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _sp_parsed = _split_param_history_line(_spl)
                                    if _sp_parsed is not None:
                                        _sp_it, _sp_raw = _sp_parsed
                                        _fa.write(f"{{_sp_it}} [{{_sp_raw}}]\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(iterations=_sp['iterations'], display_every={config.loss_display_every}, callbacks=_train_cbs)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_nncg-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} NNCG model saved: {{_nta_save_path}}.pt")
                else:
                    _sp_decay = None
                    _sp_decay_type = _sp.get('decay_type', 'none')
                    if _sp_decay_type and _sp_decay_type != 'none':
                        _sp_p1 = _sp.get('decay_p1', 0) or 0
                        _sp_p2 = _sp.get('decay_p2', 0) or 0
                        if _sp_decay_type == 'step':
                            _sp_decay = ('step', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'cosine':
                            _sp_decay = ('cosine', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'exponential':
                            _sp_decay = ('exponential', float(_sp_p1))
                    model.compile(_sp['optimizer'], lr=_sp['lr'], decay=_sp_decay,
                                  loss=_sp.get('loss', '{config.loss_type}'), loss_weights=_sp_weights,
                                  external_trainable_variables=_sp_ext_vars)
                    if _problem_type == "Inverse":
                        for _icb in _print_cbs: _icb.set_offset(_sched_cum_iters)
                        for _icb in _save_cbs: _icb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            _inv_vars, period=1000,
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            iterations=_sp['iterations'], display_every={config.loss_display_every},
                            callbacks=[_sp_var_cb] + _print_cbs + _save_cbs + _train_cbs)
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _sp_parsed = _split_param_history_line(_spl)
                                    if _sp_parsed is not None:
                                        _sp_it, _sp_raw = _sp_parsed
                                        _fa.write(f"{{_sp_it}} [{{_sp_raw}}]\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(iterations=_sp['iterations'], display_every={config.loss_display_every}, callbacks=_train_cbs)
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
                    _var_cb2 = dde.callbacks.VariableValue(
                        _inv_vars, period=200, filename="/tmp/param_history_phase2.txt",
                        precision=6
                    )
                    model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_phase2_weights,
                                  external_trainable_variables=_inv_vars)
                    for _icb in _print_cbs: _icb.set_offset(_iters)
                    loss_history, train_state = model.train(display_every={config.loss_display_every}, callbacks=[_var_cb2] + _print_cbs + _train_cbs)
                    # Append L-BFGS history to each variable's own convergence file
                    if _param_save_period > 0:
                        try:
                            with open("/tmp/param_history_phase2.txt", "r") as _lf:
                                for _ll in _lf:
                                    _l_parsed = _split_param_history_line(_ll)
                                    if _l_parsed is None:
                                        continue
                                    _lbfgs_iter, _lbfgs_raw = _l_parsed
                                    try:
                                        _lbfgs_vals = [float(v) for v in _lbfgs_raw.split(",") if v.strip()]
                                    except ValueError:
                                        continue
                                    for _lv_path, _lv_val in zip(_param_save_paths, _lbfgs_vals):
                                        if not _lv_path:
                                            continue
                                        with open(_lv_path, "a") as _pf:
                                            _pf.write(f"{{_lbfgs_iter}},{{_lv_val:.8f}}\\n")
                        except Exception as _le:
                            print(f"Could not save L-BFGS history: {{_le}}")
                    try:
                        with open("/tmp/param_history_phase2.txt", "r") as _f2:
                            _p2_lines = [l.strip() for l in _f2 if l.strip()]
                        with open("/tmp/param_history.txt", "a") as _fa:
                            for _p2l in _p2_lines:
                                _p2_parsed = _split_param_history_line(_p2l)
                                if _p2_parsed is not None:
                                    _new_iter, _p2_raw = _p2_parsed
                                    _fa.write(f"{{_new_iter}} [{{_p2_raw}}]\\n")
                    except Exception as _ae:
                        print(f"Could not append phase 2 history: {{_ae}}")
                else:
                    model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_phase2_weights)
                    loss_history, train_state = model.train(display_every={config.loss_display_every}, callbacks=_train_cbs)
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
                                  external_trainable_variables=_inv_vars)
                else:
                    model.compile("{config.optimizer2}", lr=_lr, loss="{config.loss_type}",
                                  loss_weights=_phase2_weights)
                loss_history, train_state = model.train(iterations={config.iterations2}, display_every={config.loss_display_every}, callbacks=_train_cbs)

    # ── Results-field extractor (_extract_plot_field) ──────────
    # Moved up from the "Plot solution" section below (same "not
    # time_adaptive" condition -- defined here, unconditionally on RAR,
    # instead of duplicated in both places) so each RAR round's own
    # solution snapshot can use it too; the final plot-type dispatch
    # down there still only runs once, at the very end, and now just
    # reuses this same definition instead of redefining it.
    if not {config.time_adaptive}:
        _plot_idx = {config.plot_output_idx}
        _plot_type = "{config.plot_type}"
        _plot_custom_expr = "{plot_custom_expr_converted}"
        _plot_custom_label = "{plot_custom_label_resolved}"
        _plot_title_override = {plot_title_override!r}
        _plot_xlabel_override = {plot_xlabel_override!r}
        _plot_ylabel_override = {plot_ylabel_override!r}
        _plot_output_names_list = {repr(config.output_names)}.split(",")
        _plot_n_out = len(_plot_output_names_list)
        _plot_dim = "3D" if _is_3d else ("2D" if _is_2d else "1D")
        _PLOT_TORCH_MATH_NS = {{
            "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
            "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
            "arcsin": torch.asin, "arccos": torch.acos, "arctan": torch.atan,
            "exp": torch.exp, "log": torch.log, "log10": torch.log10,
            "sqrt": torch.sqrt, "abs": torch.abs, "ceil": torch.ceil, "floor": torch.floor,
            "pi": np.pi,
        }}

        def _plot_custom_op(_pf_inputs, _pf_outputs):
            _pf_dvars = _tm_build_dvars(_pf_inputs, _pf_outputs, _plot_n_out,
                                         _plot_output_names_list, _is_steady, _plot_dim,
                                         _plot_custom_expr)
            _pf_ns = dict(_pf_dvars)
            _pf_ns.update(_PLOT_TORCH_MATH_NS)
            _pf_ns["torch"] = torch
            return eval(_plot_custom_expr, _pf_ns)

        def _extract_plot_field(_x_grid, _pf_model=None):
            # Pick the field to plot from a model.predict() call: either one
            # raw output column (_plot_idx, the pre-existing default), or --
            # when a custom expression is configured -- a derived field
            # built from ALL of this problem's outputs AND their
            # derivatives (du_x, du_xx, du_xy, ...), evaluated through
            # DeepXDE's own dde.Model.predict(x, operator=...) built-in.
            # _pf_model defaults to the module-level `model` but can be
            # overridden. Used here for each RAR round's own snapshot,
            # and below for the main Results panel plot and Inline Error
            # Analysis -- so every one of them shows/compares the exact
            # same field.
            _pf_m = _pf_model if _pf_model is not None else model
            if _plot_custom_expr.strip():
                return _pf_m.predict(_x_grid, operator=_plot_custom_op)[:, 0]
            return _pf_m.predict(_x_grid)[:, _plot_idx]

    # ── RAR Loop ─────────────────────────────────────────────
    if "{config.adapt_method}" == "RAR" and not {config.time_adaptive}:
        # ── Per-round RAR diagnostics ──────────────────────────
        # For every RAR round: a static snapshot of whatever field is
        # configured above (raw output or derivative-aware Custom
        # expression -- same _extract_plot_field every other plot uses),
        # a lightweight comparison against one Error Analysis reference
        # file if any are configured (not the full multi-file/multi-time
        # sweep Inline Error Analysis runs once at the end -- just one
        # representative reference, so this doesn't meaningfully slow
        # RAR training down), and a 3-color collocation-points scatter
        # (original domain points vs. points added in EARLIER rounds vs.
        # points added THIS round) so sharp-residual/refinement patterns
        # across rounds are visible afterward, not just the final model.
        # Saved under this run's own solution_results/rar_rounds/
        # round_NN/ -- the same per-run-unique save_dir every other plot
        # already uses (see Round 19's _on_done() fix).
        _rar_rounds_dir = _os.path.join(_sol_dir, "rar_rounds")
        _rar_orig_pts = np.array(data.train_x_all, copy=True)
        _rar_added_so_far = np.empty((0, _rar_orig_pts.shape[1]))
        _rar_ea_entries = {config.ea_files}

        def _rar_save_round_diagnostics(_rar_idx, _rar_new_pts):
            _rd = _os.path.join(_rar_rounds_dir, f"round_{{_rar_idx:02d}}")
            _os.makedirs(_rd, exist_ok=True)
            _res = {config.plot_resolution}

            # Per-round model checkpoint -- same convention Time-Adaptive's
            # per-step diagnostics already use (a model file saved
            # alongside that unit's own plots), closing a gap RAR's own
            # per-round diagnostics had since they were first added
            # (Round 20): this round's trained model is NOT saved anywhere
            # unless it happens to also be the FINAL round's model (saved
            # separately, after the whole RAR loop ends, as this run's one
            # overall model). Without this, every earlier round's model is
            # unrecoverable once RAR moves on -- only its plots/metrics
            # survive, not the weights themselves.
            if _use_save:
                try:
                    model.save(_os.path.join(_rd, "model"))
                    print(f"  [RAR round {{_rar_idx}}] model saved: {{_os.path.join(_rd, 'model')}}.pt")
                except Exception as _rar_err_save:
                    print(f"  [RAR round {{_rar_idx}}] model save failed: {{_rar_err_save}}")

            if not _is_2d and not _is_3d:
                # -- 1D: own two-file structure (solution snapshot +
                # line-comparison against one reference, if configured).
                # Axis orientation matches the main Plot Settings "Swap
                # axes" choice -- the same convention the Surface plot type
                # and every Error Analysis comparison plot already use
                # (t on the x-axis, x on the y-axis, by default).
                try:
                    _xr = np.linspace({config.x_min}, {config.x_max}, _res)
                    _tr = np.linspace({config.t_min}, {config.t_max}, _res)
                    _Xr, _Tr = np.meshgrid(_xr, _tr)
                    _field_r = _extract_plot_field(np.column_stack([_Xr.ravel(), _Tr.ravel()])).reshape(_res, _res)
                    _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5))
                    if {config.plot_swap_xt}:
                        _im_r = _ax_r.contourf(_Tr, _Xr, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                        _ax_r.set_xlabel("t"); _ax_r.set_ylabel("x")
                    else:
                        _im_r = _ax_r.contourf(_Xr, _Tr, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                        _ax_r.set_xlabel("x"); _ax_r.set_ylabel("t")
                    _fig_r.colorbar(_im_r, ax=_ax_r)
                    _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot")
                    plt.tight_layout()
                    plt.savefig(_os.path.join(_rd, "solution_plot.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close(_fig_r)
                    print(f"  [RAR round {{_rar_idx}}] solution snapshot saved: {{_os.path.join(_rd, 'solution_plot.png')}}")
                except Exception as _rar_err1:
                    print(f"  [RAR round {{_rar_idx}}] solution snapshot failed: {{_rar_err1}}")

                try:
                    _fig_c, _ax_c = plt.subplots(figsize=_plot_figsize(6.5, 5))
                    if {config.plot_swap_xt}:
                        if len(_rar_orig_pts):
                            _ax_c.scatter(_rar_orig_pts[:, 1], _rar_orig_pts[:, 0], s=6, c="#adb5bd", alpha=0.6, label="Original points")
                        if len(_rar_added_so_far):
                            _ax_c.scatter(_rar_added_so_far[:, 1], _rar_added_so_far[:, 0], s=10, c="#fd7e14", label="Added in earlier rounds")
                        _ax_c.scatter(_rar_new_pts[:, 1], _rar_new_pts[:, 0], s=14, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                        _ax_c.set_xlabel("t"); _ax_c.set_ylabel("x")
                    else:
                        if len(_rar_orig_pts):
                            _ax_c.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], s=6, c="#adb5bd", alpha=0.6, label="Original points")
                        if len(_rar_added_so_far):
                            _ax_c.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], s=10, c="#fd7e14", label="Added in earlier rounds")
                        _ax_c.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], s=14, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                        _ax_c.set_xlabel("x"); _ax_c.set_ylabel("t")
                    _ax_c.legend(loc="best", fontsize=8)
                    _ax_c.set_title(f"RAR round {{_rar_idx}} -- collocation points")
                    plt.tight_layout()
                    plt.savefig(_os.path.join(_rd, "collocation_points.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close(_fig_c)
                    print(f"  [RAR round {{_rar_idx}}] collocation-points plot saved: {{_os.path.join(_rd, 'collocation_points.png')}}")
                except Exception as _rar_err2:
                    print(f"  [RAR round {{_rar_idx}}] collocation-points plot failed: {{_rar_err2}}")

                if _rar_ea_entries:
                    try:
                        _best_r = min(_rar_ea_entries, key=lambda _e: abs(_e[0] - {config.t_max}))
                        _ref_t_r, _ref_fp_r = _best_r[0], _best_r[1]
                        _ref_sel_r = _best_r[2] if len(_best_r) >= 3 else None
                        _ref_d_r = np.loadtxt(_ref_fp_r)
                        if _ref_d_r.ndim == 1:
                            _ref_d_r = _ref_d_r.reshape(1, -1)
                        _ref_xyz_r = _ref_d_r[:, :1]; _ref_u_r = _ref_d_r[:, 2]
                        _ref_grid_r = np.column_stack([_ref_xyz_r, np.full(len(_ref_d_r), _ref_t_r)])
                        if _ref_sel_r is None:
                            _ref_pred_r = _extract_plot_field(_ref_grid_r)
                        elif isinstance(_ref_sel_r, int):
                            _ref_pred_r = model.predict(_ref_grid_r)[:, _ref_sel_r]
                        else:
                            _ref_expr_r = _ref_sel_r[0]
                            def _rar_ea_op(_rea_i, _rea_o):
                                _rea_dv = _tm_build_dvars(_rea_i, _rea_o, _plot_n_out, _plot_output_names_list,
                                                           _is_steady, _plot_dim, _ref_expr_r)
                                _rea_ns = dict(_rea_dv); _rea_ns.update(_PLOT_TORCH_MATH_NS); _rea_ns["torch"] = torch
                                return eval(_ref_expr_r, _rea_ns)
                            _ref_pred_r = model.predict(_ref_grid_r, operator=_rar_ea_op)[:, 0]
                        _err_r = _ref_pred_r - _ref_u_r
                        _l2_r = float(np.linalg.norm(_err_r) / (np.linalg.norm(_ref_u_r) + 1e-12))
                        _mse_r = float(np.mean(_err_r ** 2))
                        print(f"  [RAR round {{_rar_idx}}] vs reference t={{_ref_t_r:.3g}}: L2 rel error = {{_l2_r:.4e}}, MSE = {{_mse_r:.4e}}")
                        _fig_e, _ax_e = plt.subplots(figsize=_plot_figsize(6.5, 5))
                        _ord_e = np.argsort(_ref_xyz_r[:, 0])
                        _ax_e.plot(_ref_xyz_r[_ord_e, 0], _ref_u_r[_ord_e], label="Reference", color="#2f9e44")
                        _ax_e.plot(_ref_xyz_r[_ord_e, 0], _ref_pred_r[_ord_e], label="PINN", color="#1971c2", linestyle="--")
                        _ax_e.legend(loc="best", fontsize=8)
                        _ax_e.set_xlabel("x"); _ax_e.set_ylabel("u")
                        _ax_e.set_title(f"RAR round {{_rar_idx}} vs reference t={{_ref_t_r:.3g}} (L2={{_l2_r:.3e}})")
                        plt.tight_layout()
                        plt.savefig(_os.path.join(_rd, "error_compare.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                        plt.close(_fig_e)
                    except Exception as _rar_err3:
                        print(f"  [RAR round {{_rar_idx}}] error compare failed: {{_rar_err3}}")
                return

            # -- 2D/3D: solution snapshot merged with the error comparison --
            # One single file: PINN | Reference | |Error| (same colormap
            # convention the full Inline Error Analysis already uses --
            # PINN and Reference share one color scale, |Error| gets its
            # own, "inferno") whenever a reference file is configured;
            # just the PINN prediction alone, same as before, when it isn't.
            try:
                _ref_ok_r = False
                _ref_t_r = {config.t_max}
                if _rar_ea_entries:
                    try:
                        _best_r = min(_rar_ea_entries, key=lambda _e: abs(_e[0] - {config.t_max}))
                        _ref_t_r, _ref_fp_r = _best_r[0], _best_r[1]
                        _ref_sel_r = _best_r[2] if len(_best_r) >= 3 else None
                        _ref_d_r = np.loadtxt(_ref_fp_r)
                        if _ref_d_r.ndim == 1:
                            _ref_d_r = _ref_d_r.reshape(1, -1)
                        if _is_3d:
                            _ref_xyz_r = _ref_d_r[:, :3]; _ref_u_r = _ref_d_r[:, 4]
                        else:
                            _ref_xyz_r = _ref_d_r[:, :2]; _ref_u_r = _ref_d_r[:, 3]
                        _ref_grid_r = np.column_stack([_ref_xyz_r, np.full(len(_ref_d_r), _ref_t_r)])
                        if _ref_sel_r is None:
                            _ref_pred_pts_r = _extract_plot_field(_ref_grid_r)
                        elif isinstance(_ref_sel_r, int):
                            _ref_pred_pts_r = model.predict(_ref_grid_r)[:, _ref_sel_r]
                        else:
                            _ref_expr_r = _ref_sel_r[0]
                            def _rar_ea_op2(_rea_i, _rea_o):
                                _rea_dv = _tm_build_dvars(_rea_i, _rea_o, _plot_n_out, _plot_output_names_list,
                                                           _is_steady, _plot_dim, _ref_expr_r)
                                _rea_ns = dict(_rea_dv); _rea_ns.update(_PLOT_TORCH_MATH_NS); _rea_ns["torch"] = torch
                                return eval(_ref_expr_r, _rea_ns)
                            _ref_pred_pts_r = model.predict(_ref_grid_r, operator=_rar_ea_op2)[:, 0]
                        _err_pts_r = _ref_pred_pts_r - _ref_u_r
                        _l2_r = float(np.linalg.norm(_err_pts_r) / (np.linalg.norm(_ref_u_r) + 1e-12))
                        _mse_r = float(np.mean(_err_pts_r ** 2))
                        print(f"  [RAR round {{_rar_idx}}] vs reference t={{_ref_t_r:.3g}}: L2 rel error = {{_l2_r:.4e}}, MSE = {{_mse_r:.4e}}")
                        _ref_ok_r = True
                    except Exception as _rar_ref_err:
                        print(f"  [RAR round {{_rar_idx}}] reference compare failed, showing prediction only: {{_rar_ref_err}}")
                        _ref_ok_r = False

                if _is_3d:
                    if _ref_ok_r:
                        _fig_r = plt.figure(figsize=_plot_figsize(15, 5))
                        _abs_err_r = np.abs(_err_pts_r)
                        _vmin3_r = min(_ref_pred_pts_r.min(), _ref_u_r.min())
                        _vmax3_r = max(_ref_pred_pts_r.max(), _ref_u_r.max())
                        _cols3_r = [
                            (_ref_pred_pts_r, f"PINN  t={{_ref_t_r:.3g}}  L2={{_l2_r:.2e}}", _vmin3_r, _vmax3_r, "{config.plot_colormap}"),
                            (_ref_u_r, f"Reference  t={{_ref_t_r:.3g}}", _vmin3_r, _vmax3_r, "{config.plot_colormap}"),
                            (_abs_err_r, f"|Error|  Max={{_abs_err_r.max():.2e}}", None, None, "inferno"),
                        ]
                        for _ci_r, (_vals3_r, _ttl3_r, _vmin_c3_r, _vmax_c3_r, _cmap_c3_r) in enumerate(_cols3_r):
                            _ax3_r = _fig_r.add_subplot(1, 3, _ci_r + 1, projection="3d")
                            _sc3_r = _ax3_r.scatter(_ref_xyz_r[:, 0], _ref_xyz_r[:, 1], _ref_xyz_r[:, 2], c=_vals3_r,
                                                     cmap=_cmap_c3_r, s=10, vmin=_vmin_c3_r, vmax=_vmax_c3_r)
                            _fig_r.colorbar(_sc3_r, ax=_ax3_r, shrink=0.6, pad=0.12)
                            _ax3_r.set_title(_ttl3_r, fontsize=10)
                            _ax3_r.set_xlabel("x"); _ax3_r.set_ylabel("y"); _ax3_r.set_zlabel("z")
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} -- solution vs reference", fontsize=12, fontweight="bold")
                    else:
                        _xp = np.linspace({config.x_min}, {config.x_max}, _res)
                        _yp = np.linspace({config.y_min}, {config.y_max}, _res)
                        _Xg, _Yg = np.meshgrid(_xp, _yp)
                        _z_mid_r = ({config.z_min} + {config.z_max}) / 2.0
                        _grid_r = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _z_mid_r), np.full(_Xg.size, {config.t_max})])
                        _field_r = _extract_plot_field(_grid_r).reshape(_res, _res)
                        _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                        _im_r = _ax_r.contourf(_Xg, _Yg, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                        _ax_r.set_xlabel("x"); _ax_r.set_ylabel("y"); _ax_r.set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im_r, ax=_ax_r)
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot (z={{_z_mid_r:.3g}}, t={config.t_max:.3g})")
                else:
                    _xp = np.linspace({config.x_min}, {config.x_max}, _res)
                    _yp = np.linspace({config.y_min}, {config.y_max}, _res)
                    _Xg, _Yg = np.meshgrid(_xp, _yp)
                    _inside_r = geom.inside(np.column_stack([_Xg.ravel(), _Yg.ravel()])).reshape(_res, _res)
                    if _ref_ok_r:
                        from scipy.interpolate import griddata as _gd_r
                        _grid_pinn_r = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _ref_t_r)])
                        _u_pinn_g_r = _extract_plot_field(_grid_pinn_r).reshape(_res, _res)
                        _u_pinn_g_r = np.where(_inside_r, _u_pinn_g_r, np.nan)
                        _u_ref_g_r = _gd_r(_ref_xyz_r, _ref_u_r, (_Xg, _Yg), method="linear", fill_value=0.0)
                        _u_ref_g_r = np.where(_inside_r, _u_ref_g_r, np.nan)
                        _u_err_g_r = np.abs(_u_pinn_g_r - _u_ref_g_r)
                        _vmin2_r = np.nanmin([_u_pinn_g_r, _u_ref_g_r]); _vmax2_r = np.nanmax([_u_pinn_g_r, _u_ref_g_r])
                        if _vmax2_r - _vmin2_r < 1e-12:
                            _vmax2_r = _vmin2_r + 1e-12
                        # contourf's own vmin/vmax kwargs do NOT restrict
                        # an integer `levels=N` -- it still auto-ranges
                        # each panel to that panel's own data, so PINN and
                        # Reference would silently get two different color
                        # scales despite passing the same vmin/vmax here.
                        # Passing the level BOUNDARIES explicitly (instead
                        # of a level count) is what actually makes both
                        # panels share one scale.
                        _levels2_r = np.linspace(_vmin2_r, _vmax2_r, 41)
                        _fig_r, _axes2_r = plt.subplots(1, 3, figsize=_plot_figsize(15, 5))
                        _im0_r = _axes2_r[0].contourf(_Xg, _Yg, _u_pinn_g_r, levels=_levels2_r, cmap="{config.plot_colormap}")
                        _axes2_r[0].set_title(f"PINN  t={{_ref_t_r:.3g}}  L2={{_l2_r:.2e}}", fontsize=10)
                        _axes2_r[0].set_xlabel("x"); _axes2_r[0].set_ylabel("y"); _axes2_r[0].set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im0_r, ax=_axes2_r[0])
                        _im1_r = _axes2_r[1].contourf(_Xg, _Yg, _u_ref_g_r, levels=_levels2_r, cmap="{config.plot_colormap}")
                        _axes2_r[1].set_title(f"Reference  t={{_ref_t_r:.3g}}", fontsize=10)
                        _axes2_r[1].set_xlabel("x"); _axes2_r[1].set_ylabel("y"); _axes2_r[1].set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im1_r, ax=_axes2_r[1])
                        _im2_r = _axes2_r[2].contourf(_Xg, _Yg, _u_err_g_r, levels={config.plot_levels}, cmap="inferno")
                        _axes2_r[2].set_title(f"|Error|  Max={{np.nanmax(_u_err_g_r):.2e}}", fontsize=10)
                        _axes2_r[2].set_xlabel("x"); _axes2_r[2].set_ylabel("y"); _axes2_r[2].set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im2_r, ax=_axes2_r[2])
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} -- solution vs reference", fontsize=12, fontweight="bold")
                    else:
                        _grid_r = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, {config.t_max})])
                        _field_r = _extract_plot_field(_grid_r).reshape(_res, _res)
                        _field_r = np.where(_inside_r, _field_r, np.nan)
                        _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                        _im_r = _ax_r.contourf(_Xg, _Yg, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                        _ax_r.set_xlabel("x"); _ax_r.set_ylabel("y"); _ax_r.set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im_r, ax=_ax_r)
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot (t={config.t_max:.3g})")
                plt.tight_layout()
                plt.savefig(_os.path.join(_rd, "solution_plot.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                plt.close(_fig_r)
                print(f"  [RAR round {{_rar_idx}}] solution snapshot saved: {{_os.path.join(_rd, 'solution_plot.png')}}")
            except Exception as _rar_err1:
                print(f"  [RAR round {{_rar_idx}}] solution snapshot failed: {{_rar_err1}}")

            # -- 2D/3D: collocation points (original vs. earlier rounds vs. this round) --
            try:
                from mpl_toolkits.mplot3d import Axes3D as _Axes3D_unused  # noqa: F401 -- registers the 3D projection
                if _is_3d:
                    _fig_c = plt.figure(figsize=_plot_figsize(7, 6))
                    _ax_c = _fig_c.add_subplot(111, projection="3d")
                    # 4 real coordinate dims (x,y,z,t) can't all be axes on
                    # one static 3D plot -- spatial (x,y,z) get the 3 plot
                    # axes, and t is folded into per-category alpha instead
                    # (lighter = earlier in the time domain) rather than
                    # dropped entirely.
                    _t_lo_r, _t_hi_r = {config.t_min}, {config.t_max}
                    _span_t_r = (_t_hi_r - _t_lo_r) or 1.0
                    def _rar_alpha(_pts):
                        return float(np.clip(0.25 + 0.65 * (np.mean(_pts[:, 3]) - _t_lo_r) / _span_t_r, 0.15, 0.95)) if len(_pts) else 1.0
                    _orig_alpha, _added_alpha = _rar_alpha(_rar_orig_pts), _rar_alpha(_rar_added_so_far)
                    if len(_rar_orig_pts):
                        _ax_c.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], _rar_orig_pts[:, 2], s=5, c="#adb5bd", alpha=_orig_alpha, label="Original points")
                    if len(_rar_added_so_far):
                        _ax_c.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], _rar_added_so_far[:, 2], s=9, c="#fd7e14", alpha=_added_alpha, label="Added in earlier rounds")
                    _ax_c.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], _rar_new_pts[:, 2], s=12, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                    _ax_c.set_xlabel("x"); _ax_c.set_ylabel("y"); _ax_c.set_zlabel("z")
                    _ax_c.legend(loc="best", fontsize=8)
                    _ax_c.set_title(f"RAR round {{_rar_idx}} -- collocation points")
                    plt.tight_layout()
                else:
                    # 2D: the existing 3D (x, y, t) view, plus a plain 2D
                    # (x, y) view collapsing time -- side by side -- so the
                    # spatial concentration of refinement is readable at a
                    # glance without having to rotate a 3D plot.
                    _fig_c = plt.figure(figsize=_plot_figsize(13, 6))
                    _ax_c = _fig_c.add_subplot(1, 2, 1, projection="3d")
                    if len(_rar_orig_pts):
                        _ax_c.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], _rar_orig_pts[:, 2], s=5, c="#adb5bd", alpha=0.5, label="Original points")
                    if len(_rar_added_so_far):
                        _ax_c.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], _rar_added_so_far[:, 2], s=9, c="#fd7e14", alpha=0.85, label="Added in earlier rounds")
                    _ax_c.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], _rar_new_pts[:, 2], s=12, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                    _ax_c.set_xlabel("x"); _ax_c.set_ylabel("y"); _ax_c.set_zlabel("t")
                    _ax_c.legend(loc="best", fontsize=8)
                    _ax_c.set_title("3D view (x, y, t)")
                    _ax_c2 = _fig_c.add_subplot(1, 2, 2)
                    if len(_rar_orig_pts):
                        _ax_c2.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], s=6, c="#adb5bd", alpha=0.5, label="Original points")
                    if len(_rar_added_so_far):
                        _ax_c2.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], s=10, c="#fd7e14", alpha=0.85, label="Added in earlier rounds")
                    _ax_c2.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], s=14, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                    _ax_c2.set_xlabel("x"); _ax_c2.set_ylabel("y"); _ax_c2.set_aspect("equal", adjustable="box")
                    _ax_c2.legend(loc="best", fontsize=8)
                    _ax_c2.set_title("Spatial view (x, y) -- all times")
                    _fig_c.suptitle(f"RAR round {{_rar_idx}} -- collocation points", fontsize=12, fontweight="bold")
                    plt.tight_layout()
                plt.savefig(_os.path.join(_rd, "collocation_points.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                plt.close(_fig_c)
                print(f"  [RAR round {{_rar_idx}}] collocation-points plot saved: {{_os.path.join(_rd, 'collocation_points.png')}}")
            except Exception as _rar_err2:
                print(f"  [RAR round {{_rar_idx}}] collocation-points plot failed: {{_rar_err2}}")

        print("\\n=== Starting RAR Adaptive Refinement ===")
        for rar_cycle in range({config.rar_cycles}):
            print(f"\\n--- RAR Cycle {{rar_cycle+1}}/{config.rar_cycles} ---")
            if _is_3d:
                x_cand = np.random.uniform({config.x_min}, {config.x_max}, {config.rar_candidates})
                y_cand = np.random.uniform({config.y_min}, {config.y_max}, {config.rar_candidates})
                z_cand = np.random.uniform({config.z_min}, {config.z_max}, {config.rar_candidates})
                t_cand = np.random.uniform({config.t_min}, {config.t_max}, {config.rar_candidates})
                xt_cand = np.column_stack([x_cand, y_cand, z_cand, t_cand])
            elif _is_2d:
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
                # Restrict point selection to one equation's residual
                # (config.rar_output_selector >= 0 -- "Points from:" in the
                # RAR settings panel; equation order == output order for
                # every current template) instead of the combined sum --
                # -1 (the default) always falls through to the original,
                # unchanged combined behavior.
                if 0 <= {config.rar_output_selector} < len(_rar_res):
                    residuals = np.abs(_rar_res[{config.rar_output_selector}]).flatten()
                else:
                    residuals = np.sum([np.abs(r).flatten() for r in _rar_res], axis=0)
            else:
                residuals = np.abs(_rar_res).flatten()
            top_idx    = np.argsort(residuals)[-{config.rar_add_points}:]
            new_points = xt_cand[top_idx]
            print(f"Max residual: {{residuals.max():.4e}}, Mean: {{residuals.mean():.4e}}")
            data.add_anchors(new_points)
            model.compile("{config.optimizer}", lr=_lr, loss="{config.loss_type}", loss_weights=_multi_weights)
            loss_history, train_state = model.train(iterations={config.rar_adam_iters}, display_every={config.loss_display_every})
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
                loss_history, train_state = model.train(display_every={config.loss_display_every})
                if "{config.lbfgs_float_type}" == "float64":
                    dde.config.set_default_float("float32")
                    model.net.float()
                    print("  [L-BFGS] Restored to float32")
            _rar_save_round_diagnostics(rar_cycle + 1, new_points)
            _rar_added_so_far = np.vstack([_rar_added_so_far, new_points])
        print("\\n=== RAR Complete ===")

    # ── Save paths ────────────────────────────────────────────
    if _parametric and _pval is not None:
        _run_dir = _os.path.join(_save_dir if _use_save else "/tmp", f"{{_param_name}}_{{_pval}}")
        _os.makedirs(_run_dir, exist_ok=True)
        _run_loss_path     = _os.path.join(_run_dir, "loss_plot.png")
        _run_solution_path = _os.path.join(_run_dir, "solution_plot.{_sol_ext}")
        _run_log_path      = _os.path.join(_run_dir, "training_log.txt")
    else:
        _run_loss_path     = _loss_path
        _run_solution_path = _solution_path
        _run_log_path      = _log_path

    # ── Plot loss (Loss Plot Settings -- see _loss_plot_runtime_code) ────
    if not {config.time_adaptive}:
        dde.saveplot(loss_history, train_state, issave=False, isplot=False)
        train_loss = loss_history.loss_train
        test_loss  = loss_history.loss_test
        steps      = loss_history.steps

        title_str = f"Loss — {{_param_name}}={{_pval}}" if _parametric else "Training & Test Loss"
        _make_loss_plot(train_loss, test_loss, steps, _run_loss_path, title_str)

        if _run_log_path:
            with open(_run_log_path, "w") as f:
                for i, l in zip(steps, train_loss):
                    f.write(f"Step {{i}}: {{list(l)}}\\n")

        _final_loss = sum(loss_history.loss_train[-1])

        # Parameter history (inverse only) -- one subplot per trainable
        # variable, all sharing the single param_plot.png output path
        # (same filename as the single-variable version, for anything
        # downstream that expects it there).
        if _problem_type == "Inverse":
            try:
                _ph_iters = []
                _ph_vals_by_var = [[] for _ in _inv_var_names]
                with open("/tmp/param_history.txt", "r") as _phf:
                    for _line in _phf:
                        _parsed_line = _split_param_history_line(_line)
                        if _parsed_line is None:
                            continue
                        _it, _raw = _parsed_line
                        try:
                            _vals = [float(v) for v in _raw.split(",") if v.strip()]
                        except ValueError:
                            continue
                        if len(_vals) != len(_inv_var_names):
                            continue
                        _ph_iters.append(_it)
                        for _vi in range(len(_inv_var_names)):
                            _ph_vals_by_var[_vi].append(_vals[_vi])
                _ph_iters_arr = np.array(_ph_iters)
                _n_ivars = len(_inv_var_names)
                _fig, _iv_axes = plt.subplots(_n_ivars, 1, figsize=(_plot_figsize(6, 3.2)[0], _plot_figsize(6, 3.2)[1] * _n_ivars), squeeze=False)
                for _vi, _vname in enumerate(_inv_var_names):
                    _ax = _iv_axes[_vi][0]
                    _vvals = np.array(_ph_vals_by_var[_vi])
                    _final_val = _vvals[-1]
                    if {config.inv_param_log_scale} and np.all(_vvals > 0):
                        _ax.semilogy(_ph_iters_arr, _vvals, color="#69db7c", linewidth=1.5)
                        _ax.set_ylabel(_plot_ylabel_override or f"log({{_vname}})")
                    else:
                        _ax.plot(_ph_iters_arr, _vvals, color="#69db7c", linewidth=1.5)
                        _ax.set_ylabel(_plot_ylabel_override or _vname)
                    _ax.axhline(y=_final_val, color="#ff8787", linestyle="--", alpha=0.5, label=f"Final = {{_final_val:.6f}}")
                    _true_val = _inv_var_trues[_vi] if _vi < len(_inv_var_trues) else None
                    if _true_val is not None:
                        _ax.axhline(y=_true_val, color="#ffd43b", linestyle="--", alpha=0.8, label=f"True = {{_true_val:.6f}}")
                    _ax.set_xlabel(_plot_xlabel_override or "Iteration")
                    _ax.set_title(_plot_title_override or f"Inferred Parameter: {{_vname}}")
                    _ax.legend(); _ax.grid(True, alpha=0.3)
                    print(f"\\n=== {{_vname}} final value: {{_final_val:.6f}} ===")
                plt.tight_layout()
                plt.savefig("/tmp/param_plot.png", dpi=100)
                if _use_save:
                    import shutil as _sp
                    _sp.copy("/tmp/param_plot.png", _os.path.join(
                        _run_dir if (_parametric and _pval is not None) else _save_dir, "param_plot.png"))
                plt.close(_fig)
                for _vi, _vname in enumerate(_inv_var_names):
                    print(f"Final inferred {{_vname}}: {{_ph_vals_by_var[_vi][-1]:.6f}}")
            except Exception as _e:
                print(f"Could not plot parameter history: {{_e}}")

        _summary.append((_pval, _final_loss))
        print(f"Final loss: {{_final_loss:.4e}}")

    # ── Plot solution ─────────────────────────────────────────
    # _plot_idx/_plot_type/_plot_custom_expr/_extract_plot_field etc. are
    # already defined above (see "Results-field extractor"), unconditionally
    # on "not time_adaptive" -- which this whole section already requires
    # too, so they're guaranteed to exist here without redefining them.
    if not {config.time_adaptive}:
        if _problem_type == "Inverse" and _plot_type == "Parameter Convergence":
            # Parameter Convergence isn't a spatial plot at all -- it's the
            # iteration-vs-inferred-value chart already built just above
            # (right after training) from this run's own parameter
            # history. Use that chart as the "solution" plot instead of
            # generating a spatial one, since that's what was asked to be
            # shown by default for an Inverse problem.
            import shutil as _shutil_pc
            if _os.path.exists("/tmp/param_plot.png"):
                _shutil_pc.copy("/tmp/param_plot.png", _run_solution_path)

        elif (not _is_steady) and _plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)"):
            import matplotlib.animation as _anim
            _n_frames = max(2, {config.num_timesteps_anim})
            _fps = {config.plot_fps}
            _t_frames = np.linspace({config.t_min}, {config.t_max}, _n_frames)
            _y_mid_gif = {_line_slice_y} if (_is_2d or _is_3d) else 0.0
            _z_mid_gif = {_line_slice_z} if _is_3d else 0.0

            if _plot_type == "Line Animation (GIF)":
                _x_gif = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
                _all_u_gif = []
                for _tv in _t_frames:
                    if _is_3d:
                        _xt_gif = np.column_stack([_x_gif, np.full_like(_x_gif, _y_mid_gif), np.full_like(_x_gif, _z_mid_gif), np.full_like(_x_gif, _tv)])
                    elif _is_2d:
                        _xt_gif = np.column_stack([_x_gif, np.full_like(_x_gif, _y_mid_gif), np.full_like(_x_gif, _tv)])
                    else:
                        _xt_gif = np.column_stack([_x_gif, np.full_like(_x_gif, _tv)])
                    _all_u_gif.append(_extract_plot_field(_xt_gif).flatten())
                _u_min_gif = min(_u.min() for _u in _all_u_gif)
                _u_max_gif = max(_u.max() for _u in _all_u_gif)
                _fig_gif, _ax_gif = plt.subplots(figsize=_plot_figsize(7, 5))
                _ax_gif.set_xlim(_plot_x_min, _plot_x_max)
                _ax_gif.set_ylim(_u_min_gif - 0.05*abs(_u_min_gif) - 1e-9, _u_max_gif + 0.05*abs(_u_max_gif) + 1e-9)
                _ylabel_gif = (f"u(x,y={{_y_mid_gif:.3g}},z={{_z_mid_gif:.3g}},t)" if _is_3d else f"u(x,y={{_y_mid_gif:.3g}},t)") if (_is_2d or _is_3d) else "u(x,t)"
                _ax_gif.set_xlabel(_plot_xlabel_override or "x"); _ax_gif.set_ylabel(_plot_ylabel_override or _ylabel_gif)
                if _plot_title_override:
                    _ax_gif.set_title(_plot_title_override)
                _line_gif, = _ax_gif.plot([], [], color="#4dabf7", linewidth={config.plot_linewidth_anim})
                # Positioned just above the axes (not set_title()) and
                # styled to match it (black, centered, top) -- same visual
                # place the 2D/3D Surface Animation branches below put their
                # own per-frame "t = ..." via ax.set_title(). A real
                # set_title() isn't used here because this animation runs
                # with blit=True (see FuncAnimation below): blitting only
                # redraws the specific artists returned by _update_gif, and
                # a title isn't redrawn under blit, so it would never
                # update frame to frame. A Text artist positioned in axes
                # coordinates just above y=1.0 renders in the same place a
                # title would, while still being blit-compatible. When a
                # title override is also set (just above, a real ax.set_title()
                # call that blit never touches again, so it's safe to combine
                # with this blitted artist), that same "just above the axes"
                # spot is exactly where the override title renders too -- so
                # this moves inside the axes (top-left corner) instead,
                # matching "Export as DeepXDE Script"'s own Line Animation
                # GIF time-indicator placement, to avoid the two overlapping
                # into illegible text.
                _time_txt_x, _time_txt_y, _time_txt_ha = (0.02, 0.95, 'left') if _plot_title_override else (0.5, 1.02, 'center')
                _time_txt_gif = _ax_gif.text(_time_txt_x, _time_txt_y, '', transform=_ax_gif.transAxes,
                                              color='black', ha=_time_txt_ha, fontsize=11)
                _ax_gif.grid(True, alpha=0.2)
                def _init_gif():
                    _line_gif.set_data([], []); _time_txt_gif.set_text(''); return _line_gif, _time_txt_gif
                def _update_gif(i):
                    _line_gif.set_data(_x_gif, _all_u_gif[i])
                    _time_txt_gif.set_text(f"t = {{_t_frames[i]:.3f}}")
                    return _line_gif, _time_txt_gif
                _ani_gif = _anim.FuncAnimation(_fig_gif, _update_gif, init_func=_init_gif, frames=_n_frames, interval=100, blit=True)
                _ani_gif.save(_run_solution_path, writer='pillow', fps=_fps)
                plt.close(_fig_gif)

            else:  # Surface Animation (GIF)
                if _is_3d:
                    _res3a = 28
                    _bbox3a = np.asarray(geom.bbox)
                    _cx0a, _cy0a, _cz0a = _bbox3a[0]; _cx1a, _cy1a, _cz1a = _bbox3a[1]
                    _xg3a = np.linspace(_cx0a, _cx1a, _res3a)
                    _yg3a = np.linspace(_cy0a, _cy1a, _res3a)
                    _zg3a = np.linspace(_cz0a, _cz1a, _res3a)
                    _Xxy3a, _Yxy3a = np.meshgrid(_xg3a, _yg3a)
                    _Xxz3a, _Zxz3a = np.meshgrid(_xg3a, _zg3a)
                    _Yyz3a, _Zyz3a = np.meshgrid(_yg3a, _zg3a)
                    _faces3a = [
                        (_Xxy3a, _Yxy3a, np.full_like(_Xxy3a, _cz0a)),
                        (_Xxy3a, _Yxy3a, np.full_like(_Xxy3a, _cz1a)),
                        (_Xxz3a, np.full_like(_Xxz3a, _cy0a), _Zxz3a),
                        (_Xxz3a, np.full_like(_Xxz3a, _cy1a), _Zxz3a),
                        (np.full_like(_Yyz3a, _cx0a), _Yyz3a, _Zyz3a),
                        (np.full_like(_Yyz3a, _cx1a), _Yyz3a, _Zyz3a),
                    ]
                    _all_frames_gif = []
                    for _tv in _t_frames:
                        _frame_faces_gif = []
                        for _fX3a, _fY3a, _fZ3a in _faces3a:
                            _fpts3a = np.column_stack([_fX3a.ravel(), _fY3a.ravel(), _fZ3a.ravel(), np.full(_fX3a.size, _tv)])
                            _frame_faces_gif.append(_extract_plot_field(_fpts3a).reshape(_fX3a.shape))
                        _all_frames_gif.append(_frame_faces_gif)
                    if {config.plot_auto_range}:
                        _v_min_gif = min(_f.min() for _frame in _all_frames_gif for _f in _frame)
                        _v_max_gif = max(_f.max() for _frame in _all_frames_gif for _f in _frame)
                    else:
                        _v_min_gif = {config.plot_vmin}
                        _v_max_gif = {config.plot_vmax}
                    _fig_gif = plt.figure(figsize=_plot_figsize(7, 5))
                    _ax_gif = _fig_gif.add_subplot(111, projection='3d')
                    _norm3a = plt.Normalize(vmin=_v_min_gif, vmax=_v_max_gif)
                    _cmap_obj3a = plt.get_cmap("{config.plot_colormap}")
                    _sm3a = plt.cm.ScalarMappable(cmap=_cmap_obj3a, norm=_norm3a)
                    if {config.plot_colorbar}: _fig_gif.colorbar(_sm3a, ax=_ax_gif, shrink=0.6, pad=0.12)
                    def _update_gif(i):
                        _ax_gif.cla()
                        for _fi3a, (_fX3a, _fY3a, _fZ3a) in enumerate(_faces3a):
                            _ax_gif.plot_surface(_fX3a, _fY3a, _fZ3a, facecolors=_cmap_obj3a(_norm3a(_all_frames_gif[i][_fi3a])),
                                             rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
                        _ax_gif.set_xlabel(_plot_xlabel_override or "x"); _ax_gif.set_ylabel(_plot_ylabel_override or "y"); _ax_gif.set_zlabel("z")
                        _ax_gif.set_title(_plot_title_override or f"t = {{_t_frames[i]:.3f}}")
                        try:
                            _ax_gif.set_box_aspect((_cx1a - _cx0a, _cy1a - _cy0a, _cz1a - _cz0a))
                        except Exception:
                            pass
                    _ani_gif = _anim.FuncAnimation(_fig_gif, _update_gif, frames=_n_frames, interval=150)
                    _ani_gif.save(_run_solution_path, writer='pillow', fps=_fps)
                    plt.close(_fig_gif)
                else:
                    _x_anim_gif = np.linspace(_plot_x_min, _plot_x_max, 80)
                    _all_frames_gif = []
                    if _is_2d:
                        _y_anim_gif = np.linspace(_plot_y_min, _plot_y_max, 80)
                        _Xg_gif, _Yg_gif = np.meshgrid(_x_anim_gif, _y_anim_gif)
                        for _tv in _t_frames:
                            _xyt_gif = np.column_stack([_Xg_gif.ravel(), _Yg_gif.ravel(), np.full(_Xg_gif.size, _tv)])
                            _pred_gif = _extract_plot_field(_xyt_gif).reshape(80, 80)
                            _all_frames_gif.append((_Xg_gif, _Yg_gif, _pred_gif))
                    else:
                        # Same configurable axis orientation as the Standard
                        # path's static 1D "Surface" plot (Plot Settings ->
                        # "Swap axes") -- there's no real t-variation within
                        # a single frame (each frame is one instant, u(x) at
                        # that t, drawn as a color band against a cosmetic
                        # t-range axis for width), but the swap still needs
                        # honoring so a GIF frame's orientation matches the
                        # static Surface plot's, instead of always being
                        # x-on-horizontal/t-on-vertical regardless of the
                        # setting. Same no-reshape-of-Z trick: swapping which
                        # of _X_anim_gif/_T_anim_gif is stored first is
                        # enough, since contourf reads each array's own
                        # coordinate values rather than assuming an axis order.
                        _t_anim_gif = np.linspace({config.t_min}, {config.t_max}, 80)
                        _X_anim_gif, _T_anim_gif = np.meshgrid(_x_anim_gif, _t_anim_gif)
                        for _tv in _t_frames:
                            _xt_gif2 = np.vstack([_X_anim_gif.ravel(), np.full(_X_anim_gif.size, _tv)]).T
                            _pred_gif = _extract_plot_field(_xt_gif2).reshape(80, 80)
                            if {config.plot_swap_xt}:
                                _all_frames_gif.append((_T_anim_gif, _X_anim_gif, _pred_gif))
                            else:
                                _all_frames_gif.append((_X_anim_gif, _T_anim_gif, _pred_gif))
                    if {config.plot_auto_range}:
                        _v_min_gif = min(_f[2].min() for _f in _all_frames_gif)
                        _v_max_gif = max(_f[2].max() for _f in _all_frames_gif)
                    else:
                        _v_min_gif = {config.plot_vmin}
                        _v_max_gif = {config.plot_vmax}
                    if _is_2d:
                        _xlabel_gif, _ylabel_gif = "x", "y"
                    elif {config.plot_swap_xt}:
                        _xlabel_gif, _ylabel_gif = "t", "x"
                    else:
                        _xlabel_gif, _ylabel_gif = "x", "t"
                    _fig_gif, _ax_gif = plt.subplots(figsize=_plot_figsize(7, 5))
                    from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable_gif
                    _div_gif = _make_axes_locatable_gif(_ax_gif)
                    _cax_gif = _div_gif.append_axes("right", size="5%", pad=0.1)
                    _sm_gif = plt.cm.ScalarMappable(cmap="{config.plot_colormap}", norm=plt.Normalize(vmin=_v_min_gif, vmax=_v_max_gif))
                    if {config.plot_colorbar}: _fig_gif.colorbar(_sm_gif, cax=_cax_gif)
                    def _update_gif(i):
                        _ax_gif.cla()
                        _Xp_gif, _Yp_gif, _Zp_gif = _all_frames_gif[i]
                        _ax_gif.contourf(_Xp_gif, _Yp_gif, _Zp_gif, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_v_min_gif, vmax=_v_max_gif)
                        _ax_gif.set_xlabel(_plot_xlabel_override or _xlabel_gif)
                        _ax_gif.set_ylabel(_plot_ylabel_override or _ylabel_gif)
                        _ax_gif.set_title(_plot_title_override or f"t = {{_t_frames[i]:.3f}}")
                    _ani_gif = _anim.FuncAnimation(_fig_gif, _update_gif, frames=_n_frames, interval=150)
                    _ani_gif.save(_run_solution_path, writer='pillow', fps=_fps)
                    plt.close(_fig_gif)

        elif _is_2d and _is_steady and _plot_type == "Line (time steps)":
            # Steady 2D "Line (time steps)": no time axis to step through,
            # so this is a single curve u(x) at the configured y slice --
            # same "one curve regardless of plot type" simplification the
            # steady-1D branch below already uses. Previously this
            # selection silently fell through to the x-y heatmap branch
            # just below (plot_type was never checked for 2D/3D at all).
            _x_l2ds = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
            _xy_l2ds = np.column_stack([_x_l2ds, np.full_like(_x_l2ds, {_line_slice_y})])
            _u_l2ds = _extract_plot_field(_xy_l2ds).flatten()
            out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
            ax.plot(_x_l2ds, _u_l2ds, color="#4dabf7", linewidth={config.plot_linewidth})
            ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or f"{{out_name}}(x,y={_line_slice_y:.3g})")
            ax.set_title(_plot_title_override or f"PINN Solution — {{out_name}}(x,y={_line_slice_y:.3g})")
            ax.grid(True, alpha=0.2)
            plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        elif _is_2d and _is_steady:
            # Steady-state 2D: a single static x-y heatmap of u(x,y) -- no
            # time axis, so no time snapshots and no time column on the
            # model input (matches the 2-column network input built for
            # steady 2D problems).
            _res_2d = {config.plot_resolution}
            _xp = np.linspace(_plot_x_min, _plot_x_max, _res_2d)
            _yp = np.linspace(_plot_y_min, _plot_y_max, _res_2d)
            _Xg, _Yg = np.meshgrid(_xp, _yp)
            _inside_2d = geom.inside(np.column_stack([_Xg.ravel(), _Yg.ravel()])).reshape(_res_2d, _res_2d)
            _vmin_2d = None if {config.plot_auto_range} else {config.plot_vmin}
            _vmax_2d = None if {config.plot_auto_range} else {config.plot_vmax}

            _XY = np.column_stack([_Xg.ravel(), _Yg.ravel()])
            _pred = _extract_plot_field(_XY).reshape(_res_2d, _res_2d)
            _pred = np.where(_inside_2d, _pred, np.nan)
            fig, ax = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
            im = ax.contourf(_Xg, _Yg, _pred, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_2d, vmax=_vmax_2d)
            ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or "y")
            ax.set_aspect("equal", adjustable="box")
            if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
            out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            fig.suptitle(_plot_title_override or f"PINN Solution — {{out_name}}(x,y)", fontsize=12)
            plt.tight_layout()
            plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        elif _is_2d and _plot_type == "Line (time steps)":
            # 2D "Line (time steps)": overlaid u(x) curves at several times,
            # all at the configured y slice -- same convention the 1D
            # "Line (time steps)" branch further below already uses
            # (num_timesteps_line curves, color-graded by time, legend by
            # t). Previously this selection silently fell through to the
            # x-y heatmap branch just below (plot_type was never checked
            # for 2D/3D at all).
            _n_steps_l2d = {config.num_timesteps_line}
            _x_l2d = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
            _t_steps_l2d = np.linspace({config.t_min}, {config.t_max}, _n_steps_l2d)
            fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
            _colors_l2d = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, _n_steps_l2d))
            out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            for _i_l2d, _tv_l2d in enumerate(_t_steps_l2d):
                _xyt_l2d = np.column_stack([_x_l2d, np.full_like(_x_l2d, {_line_slice_y}), np.full_like(_x_l2d, _tv_l2d)])
                _u_l2d = _extract_plot_field(_xyt_l2d).flatten()
                ax.plot(_x_l2d, _u_l2d, color=_colors_l2d[_i_l2d], linewidth={config.plot_linewidth}, label=f"t = {{_tv_l2d:.3f}}")
            ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or f"{{out_name}}(x,y={_line_slice_y:.3g},t)")
            ax.set_title(_plot_title_override or f"PINN Solution — {{out_name}}(x,y={_line_slice_y:.3g},t)")
            ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
            plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        elif _is_2d:
            # 2D: x-y heatmaps at user-selected number of time snapshots
            _n_snaps = {config.plot_n_2d_snapshots}
            _t_snaps = np.linspace({config.t_min}, {config.t_max}, _n_snaps)
            _res_2d = {config.plot_resolution}
            _xp = np.linspace(_plot_x_min, _plot_x_max, _res_2d)
            _yp = np.linspace(_plot_y_min, _plot_y_max, _res_2d)
            _Xg, _Yg = np.meshgrid(_xp, _yp)
            # Outside the shape's own boundary (only differs from the bbox
            # for Disk/Ellipse/Triangle/Polygon), mask the grid so the plot
            # doesn't show extrapolated model output past the true domain.
            _inside_2d = geom.inside(np.column_stack([_Xg.ravel(), _Yg.ravel()])).reshape(_res_2d, _res_2d)
            _vmin_2d = None if {config.plot_auto_range} else {config.plot_vmin}
            _vmax_2d = None if {config.plot_auto_range} else {config.plot_vmax}

            fig, axes = plt.subplots(1, _n_snaps, figsize=(_plot_figsize(5, 5)[0]*_n_snaps, _plot_figsize(5, 5)[1]))
            if _n_snaps == 1: axes = [axes]
            for _ai, _tv in enumerate(_t_snaps):
                _XYT = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _tv)])
                _pred = _extract_plot_field(_XYT).reshape(_res_2d, _res_2d)
                _pred = np.where(_inside_2d, _pred, np.nan)
                im = axes[_ai].contourf(_Xg, _Yg, _pred, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_2d, vmax=_vmax_2d)
                axes[_ai].set_title(f"t = {{_tv:.3f}}")
                axes[_ai].set_xlabel(_plot_xlabel_override or "x"); axes[_ai].set_ylabel(_plot_ylabel_override or "y")
                if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
            out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            fig.suptitle(_plot_title_override or f"PINN Solution — {{out_name}}(x,y,t)", fontsize=12)
            plt.tight_layout()
            plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        elif _is_3d and _is_steady and _plot_type == "Line (time steps)":
            # Steady 3D "Line (time steps)": no time axis, so a single
            # curve u(x) at the configured y and z slice -- same "one
            # curve" simplification the steady-2D Line branch above uses.
            _x_l3ds = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
            _xyz_l3ds = np.column_stack([_x_l3ds, np.full_like(_x_l3ds, {_line_slice_y}), np.full_like(_x_l3ds, {_line_slice_z})])
            _u_l3ds = _extract_plot_field(_xyz_l3ds).flatten()
            out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
            ax.plot(_x_l3ds, _u_l3ds, color="#4dabf7", linewidth={config.plot_linewidth})
            ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or f"{{out_name}}(x,y={_line_slice_y:.3g},z={_line_slice_z:.3g})")
            ax.set_title(_plot_title_override or f"PINN Solution — {{out_name}}(x,y={_line_slice_y:.3g},z={_line_slice_z:.3g})")
            ax.grid(True, alpha=0.2)
            plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        elif _is_3d and _is_steady:
            # Steady-state 3D: same "x-y heatmap at the z mid-plane"
            # simplification as the time-dependent 3D branch below, but a
            # single static plot -- no time axis, no time snapshots, and a
            # 3-column (x,y,z) model input rather than 4.
            _res_3d = {config.plot_resolution}
            _xp3 = np.linspace(_plot_x_min, _plot_x_max, _res_3d)
            _yp3 = np.linspace(_plot_y_min, _plot_y_max, _res_3d)
            _Xg3, _Yg3 = np.meshgrid(_xp3, _yp3)
            _z_mid = (_plot_z_min + _plot_z_max) / 2.0
            _inside_3d = geom.inside(np.column_stack(
                [_Xg3.ravel(), _Yg3.ravel(), np.full(_Xg3.size, _z_mid)])).reshape(_res_3d, _res_3d)
            _vmin_3d = None if {config.plot_auto_range} else {config.plot_vmin}
            _vmax_3d = None if {config.plot_auto_range} else {config.plot_vmax}

            _XYZ = np.column_stack([_Xg3.ravel(), _Yg3.ravel(), np.full(_Xg3.size, _z_mid)])
            _pred3 = _extract_plot_field(_XYZ).reshape(_res_3d, _res_3d)
            _pred3 = np.where(_inside_3d, _pred3, np.nan)
            fig, ax = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
            im = ax.contourf(_Xg3, _Yg3, _pred3, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_3d, vmax=_vmax_3d)
            ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or "y")
            ax.set_aspect("equal", adjustable="box")
            if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
            out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            fig.suptitle(_plot_title_override or f"PINN Solution — {{out_name}}(x,y,z={{_z_mid:.3g}})", fontsize=12)
            plt.tight_layout()
            plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        elif _is_3d and _plot_type == "Line (time steps)":
            # 3D "Line (time steps)": overlaid u(x) curves at several
            # times, all at the configured y and z slice -- same
            # convention as the 2D Line branch above and the 1D one
            # further below. Previously this selection silently fell
            # through to the x-y heatmap branch just below.
            _n_steps_l3d = {config.num_timesteps_line}
            _x_l3d = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
            _t_steps_l3d = np.linspace({config.t_min}, {config.t_max}, _n_steps_l3d)
            fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
            _colors_l3d = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, _n_steps_l3d))
            out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            for _i_l3d, _tv_l3d in enumerate(_t_steps_l3d):
                _xyzt_l3d = np.column_stack([_x_l3d, np.full_like(_x_l3d, {_line_slice_y}), np.full_like(_x_l3d, {_line_slice_z}), np.full_like(_x_l3d, _tv_l3d)])
                _u_l3d = _extract_plot_field(_xyzt_l3d).flatten()
                ax.plot(_x_l3d, _u_l3d, color=_colors_l3d[_i_l3d], linewidth={config.plot_linewidth}, label=f"t = {{_tv_l3d:.3f}}")
            ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or f"{{out_name}}(x,y={_line_slice_y:.3g},z={_line_slice_z:.3g},t)")
            ax.set_title(_plot_title_override or f"PINN Solution — {{out_name}}(x,y={_line_slice_y:.3g},z={_line_slice_z:.3g},t)")
            ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
            plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        elif _is_3d:
            # 3D: no volumetric renderer yet -- plot an x-y heatmap at the
            # domain's z mid-plane, at the same time snapshots the 2D case
            # uses, so the run finishes with a solution plot instead of
            # crashing. Full 3D visualization (multiple slices / isosurfaces)
            # is a possible future addition once there's a template that
            # needs it.
            _n_snaps = {config.plot_n_2d_snapshots}
            _t_snaps = np.linspace({config.t_min}, {config.t_max}, _n_snaps)
            _res_3d = {config.plot_resolution}
            _xp3 = np.linspace(_plot_x_min, _plot_x_max, _res_3d)
            _yp3 = np.linspace(_plot_y_min, _plot_y_max, _res_3d)
            _Xg3, _Yg3 = np.meshgrid(_xp3, _yp3)
            _z_mid = (_plot_z_min + _plot_z_max) / 2.0
            # Mask points outside the true 3D shape at this z-slice (only
            # differs from the bbox for Sphere) -- same reasoning as the 2D
            # branch above.
            _inside_3d = geom.inside(np.column_stack(
                [_Xg3.ravel(), _Yg3.ravel(), np.full(_Xg3.size, _z_mid)])).reshape(_res_3d, _res_3d)
            _vmin_3d = None if {config.plot_auto_range} else {config.plot_vmin}
            _vmax_3d = None if {config.plot_auto_range} else {config.plot_vmax}

            fig, axes = plt.subplots(1, _n_snaps, figsize=(_plot_figsize(5, 5)[0]*_n_snaps, _plot_figsize(5, 5)[1]))
            if _n_snaps == 1: axes = [axes]
            for _ai, _tv in enumerate(_t_snaps):
                _XYZT = np.column_stack([_Xg3.ravel(), _Yg3.ravel(), np.full(_Xg3.size, _z_mid), np.full(_Xg3.size, _tv)])
                _pred3 = _extract_plot_field(_XYZT).reshape(_res_3d, _res_3d)
                _pred3 = np.where(_inside_3d, _pred3, np.nan)
                im = axes[_ai].contourf(_Xg3, _Yg3, _pred3, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_3d, vmax=_vmax_3d)
                axes[_ai].set_title(f"t = {{_tv:.3f}}, z = {{_z_mid:.3g}} (mid-plane)")
                axes[_ai].set_xlabel(_plot_xlabel_override or "x"); axes[_ai].set_ylabel(_plot_ylabel_override or "y")
                if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
            out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            fig.suptitle(_plot_title_override or f"PINN Solution — {{out_name}}(x,y,z={{_z_mid:.3g}},t)", fontsize=12)
            plt.tight_layout()
            plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        elif _is_steady:
            # Steady-state 1D: a single static curve u(x) -- no time axis
            # at all, so neither of the time-dependent 1D plot types
            # ("Surface" over x,t / "Line (time steps)") apply.
            _res_1d = {config.plot_resolution}
            _x_1d = np.linspace({config.x_min}, {config.x_max}, _res_1d)
            _u_1d = _extract_plot_field(_x_1d.reshape(-1, 1)).flatten()
            _out_name_1d = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
            fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
            ax.plot(_x_1d, _u_1d, color="#4dabf7", linewidth={config.plot_linewidth})
            ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or f"{{_out_name_1d}}(x)")
            ax.set_title(_plot_title_override or (f"PINN Solution — {{_param_name}}={{_pval}}" if _parametric else "PINN Solution"))
            ax.grid(True, alpha=0.2)
            plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        else:
            # 1D plot (_plot_type already computed above)
            if _plot_type == "Surface":
                _res = {config.plot_resolution}
                _x_s = np.linspace({config.x_min}, {config.x_max}, _res)
                _t_s = np.linspace({config.t_min}, {config.t_max}, _res)
                _Xs, _Ts = np.meshgrid(_x_s, _t_s)
                _XTs = np.vstack([_Xs.ravel(), _Ts.ravel()]).T
                _u_s = _extract_plot_field(_XTs).reshape(_res, _res)
                _out_name_1dt = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                _vmin_s = None if {config.plot_auto_range} else {config.plot_vmin}
                _vmax_s = None if {config.plot_auto_range} else {config.plot_vmax}
                print(f"Plot settings: cmap={config.plot_colormap}, levels={config.plot_levels}, dpi={config.plot_dpi}, res={config.plot_resolution}")
                fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
                # Axis orientation is configurable (Plot Settings ->
                # "Swap axes"); contourf just needs its three arrays to
                # line up element-for-element, so swapping which of
                # _Xs/_Ts is passed first -- with no reshape/transpose of
                # _u_s -- is enough to flip which one lands on the x-axis.
                if {config.plot_swap_xt}:
                    im = ax.contourf(_Ts, _Xs, _u_s, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_s, vmax=_vmax_s)
                    ax.set_xlabel(_plot_xlabel_override or "t"); ax.set_ylabel(_plot_ylabel_override or "x")
                else:
                    im = ax.contourf(_Xs, _Ts, _u_s, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_s, vmax=_vmax_s)
                    ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or "t")
                if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
                ax.set_title(_plot_title_override or (f"PINN Solution — {{_out_name_1dt}}(x,t) — {{_param_name}}={{_pval}}" if _parametric
                              else f"PINN Solution — {{_out_name_1dt}}(x,t)"))
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

            elif _plot_type == "Line (time steps)":
                n_steps_plot = {config.num_timesteps_line}
                _x_l = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                t_steps_plot = np.linspace({config.t_min}, {config.t_max}, n_steps_plot)
                fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
                colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_plot))
                _out_name_line = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                for i, t_val in enumerate(t_steps_plot):
                    xt = np.column_stack([_x_l, np.full_like(_x_l, t_val)])
                    u_line = _extract_plot_field(xt).flatten()
                    ax.plot(_x_l, u_line, color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{t_val:.3f}}")
                ax.set_xlabel(_plot_xlabel_override or "x"); ax.set_ylabel(_plot_ylabel_override or f"{{_out_name_line}}(x,t)")
                ax.set_title(_plot_title_override or (f"PINN Solution — {{_param_name}}={{_pval}}" if _parametric else "PINN Solution"))
                ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

        import shutil as _shutil
        if _run_loss_path != "/tmp/loss_plot.png":
            _shutil.copy(_run_loss_path, "/tmp/loss_plot.png")
        _tmp_solution_path = "/tmp/solution_plot.{_sol_ext}"
        if _run_solution_path != _tmp_solution_path and _os.path.exists(_run_solution_path):
            _shutil.copy(_run_solution_path, _tmp_solution_path)

        # ── Inline Error Analysis ─────────────────────────────
        if {config.ea_files}:
            from scipy.interpolate import interp1d as _interp1d
            _ea_dir = _os.path.join(_save_dir if _use_save else "/tmp", "error_analysis")
            _os.makedirs(_ea_dir, exist_ok=True)

            # Normalize `ea_files` entries to (time, path, output_selector)
            # 3-tuples -- output_selector is None (implicit default: falls
            # back to whatever single field the Results panel/plot_custom_expr
            # is already configured to show, exactly like before this
            # feature existed), an int (a raw output column index), or a
            # [expr, label] pair (a custom derived field of its own, reusing
            # the same expression mechanism as the Results panel's
            # "Custom..." plot field). Older saved configs only ever wrote
            # 2-tuples (no selector at all) -- padded with None here so they
            # keep behaving exactly as before.
            _ea_files_norm = []
            for _ea_entry in {config.ea_files}:
                if len(_ea_entry) >= 3:
                    _ea_tv0, _ea_fp0, _ea_sel0 = _ea_entry[0], _ea_entry[1], _ea_entry[2]
                else:
                    _ea_tv0, _ea_fp0 = _ea_entry[0], _ea_entry[1]
                    _ea_sel0 = None
                if {config.t_min} - 1e-10 <= _ea_tv0 <= {config.t_max} + 1e-10:
                    _ea_files_norm.append((_ea_tv0, _ea_fp0, _ea_sel0))
            print("\\n=== Running Error Analysis ===")
            print(f"  Filtering to t=[{config.t_min}, {config.t_max}]: {{len(_ea_files_norm)}} files")

            # Group reference files by which model output they belong to --
            # each distinct output gets its own full metrics/line/surface
            # report below. When every file resolves to the same (usually
            # the implicit default) output, there's exactly one group and
            # every filename below is unsuffixed -- byte-for-byte the same
            # output as before this feature existed.
            _ea_groups = {{}}
            for _ea_tv0, _ea_fp0, _ea_sel0 in _ea_files_norm:
                _ea_gkey = repr(_ea_sel0)
                if _ea_gkey not in _ea_groups:
                    _ea_groups[_ea_gkey] = {{"sel": _ea_sel0, "files": []}}
                _ea_groups[_ea_gkey]["files"].append((_ea_tv0, _ea_fp0))
            _ea_multi_output = len(_ea_groups) > 1

            def _ea_group_label(_ea_sel):
                if _ea_sel is None:
                    return ""
                if isinstance(_ea_sel, int):
                    return (_plot_output_names_list[_ea_sel].strip()
                            if 0 <= _ea_sel < len(_plot_output_names_list) else f"out{{_ea_sel}}")
                _ea_lbl = (_ea_sel[1] or "").strip() if len(_ea_sel) > 1 else ""
                return _ea_lbl if _ea_lbl else "custom"

            def _ea_extract(_ea_grid, _ea_sel, _ea_model=None):
                # _ea_sel is None: delegate to _extract_plot_field, so the
                # default plot field (including a derivative-aware custom
                # expression, if configured) is also what Error Analysis
                # compares a reference dataset against. _ea_sel an int:
                # a specific raw output column was picked for this
                # reference file group. _ea_sel a (expr, label) tuple: a
                # per-group custom expression of its own -- now also
                # derivative-aware (du_x, du_xx, ...) via the same
                # operator-based mechanism _extract_plot_field itself uses,
                # reusing Training Monitors' own _tm_build_dvars rather than
                # a second derivative-building copy. This used to be
                # algebraic-only (NumPy eval on a pre-computed prediction);
                # that was this feature's one remaining flagged gap from the
                # round that added derivative-aware custom plotting.
                _ea_m = _ea_model if _ea_model is not None else model
                if _ea_sel is None:
                    return _extract_plot_field(_ea_grid, _ea_m)
                if isinstance(_ea_sel, int):
                    return _ea_m.predict(_ea_grid)[:, _ea_sel]
                _ea_expr = _ea_sel[0]

                def _ea_custom_op(_ea_inputs, _ea_outputs):
                    _ea_dvars = _tm_build_dvars(_ea_inputs, _ea_outputs, _plot_n_out,
                                                 _plot_output_names_list, _is_steady, _plot_dim,
                                                 _ea_expr)
                    _ea_ns = dict(_ea_dvars)
                    _ea_ns.update(_PLOT_TORCH_MATH_NS)
                    _ea_ns["torch"] = torch
                    return eval(_ea_expr, _ea_ns)

                return _ea_m.predict(_ea_grid, operator=_ea_custom_op)[:, 0]

            for _ea_group_key, _ea_group in _ea_groups.items():
                _ea_files = _ea_group["files"]
                _ea_sel = _ea_group["sel"]
                _ea_suffix = f"_{{_ea_group_label(_ea_sel)}}" if _ea_multi_output else ""
                if _ea_multi_output:
                    print(f"  ── Output group: {{_ea_group_label(_ea_sel) or 'default'}} ({{len(_ea_files)}} files) ──")

                # Load all ground truth files
                _ea_times = []; _ea_x_refs = []; _ea_y_refs = []; _ea_z_refs = []; _ea_u_refs = []
                for _ea_tv, _ea_fp in _ea_files:
                    _ea_d = np.loadtxt(_ea_fp)
                    if _ea_d.ndim == 1: _ea_d = _ea_d.reshape(1, -1)
                    if _is_steady and _is_3d:
                        # Steady 3D format: x, y, z, u — no time column at all.
                        _ea_idx = np.lexsort((_ea_d[:, 2], _ea_d[:, 1], _ea_d[:, 0]))
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(_ea_d[_ea_idx, 1])
                        _ea_z_refs.append(_ea_d[_ea_idx, 2])
                        _ea_u_refs.append(_ea_d[_ea_idx, 3])
                        _ea_times.append(0.0)
                        print(f"  Loaded ground truth (steady-state): {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    elif _is_steady and _is_2d:
                        # Steady 2D format: x, y, u — no time column at all.
                        _ea_idx = np.lexsort((_ea_d[:, 1], _ea_d[:, 0]))
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(_ea_d[_ea_idx, 1])
                        _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_u_refs.append(_ea_d[_ea_idx, 2])
                        _ea_times.append(0.0)
                        print(f"  Loaded ground truth (steady-state): {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    elif _is_3d:
                        # 3D format: x, y, z, t, u — sort by x, y, z
                        _ea_idx = np.lexsort((_ea_d[:, 2], _ea_d[:, 1], _ea_d[:, 0]))
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(_ea_d[_ea_idx, 1])
                        _ea_z_refs.append(_ea_d[_ea_idx, 2])
                        _ea_u_refs.append(_ea_d[_ea_idx, 4])
                        _detected_t = float(_ea_d[0, 3])
                        _ea_times.append(_detected_t)
                        print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    elif _is_2d:
                        # 2D format: x, y, t, u — sort by x then y
                        _ea_idx = np.lexsort((_ea_d[:, 1], _ea_d[:, 0]))
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(_ea_d[_ea_idx, 1])
                        _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_u_refs.append(_ea_d[_ea_idx, 3])
                        _detected_t = float(_ea_d[0, 2])
                        _ea_times.append(_detected_t)
                        print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    else:
                        # 1D format: x, t, u — sort by x
                        _ea_idx = np.argsort(_ea_d[:, 0])
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_u_refs.append(_ea_d[_ea_idx, 2])
                        _ea_times.append(float(_ea_tv))
                        print(f"  Loaded ground truth t={{_ea_tv:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                # Drop reference points with no solution value (NaN) -- e.g. grid
                # points outside a non-rectangular geometry like L-Shape's missing
                # quadrant or Disk/Sphere's bounding-box corners. A no-op for
                # reference files that don't have any (the usual case).
                for _ei in range(len(_ea_u_refs)):
                    _ea_valid = ~np.isnan(_ea_u_refs[_ei])
                    if not _ea_valid.all():
                        _n_dropped = int((~_ea_valid).sum())
                        _ea_x_refs[_ei] = _ea_x_refs[_ei][_ea_valid]
                        _ea_y_refs[_ei] = _ea_y_refs[_ei][_ea_valid]
                        _ea_z_refs[_ei] = _ea_z_refs[_ei][_ea_valid]
                        _ea_u_refs[_ei] = _ea_u_refs[_ei][_ea_valid]
                        print(f"  Dropped {{_n_dropped}} NaN reference point(s) outside the geometry")
                # Sort all loaded data by time value — outside the loop
                _ea_sort_idx = np.argsort(_ea_times)
                _ea_times  = [_ea_times[_i]  for _i in _ea_sort_idx]
                _ea_x_refs = [_ea_x_refs[_i] for _i in _ea_sort_idx]
                _ea_y_refs = [_ea_y_refs[_i] for _i in _ea_sort_idx]
                _ea_z_refs = [_ea_z_refs[_i] for _i in _ea_sort_idx]
                _ea_u_refs = [_ea_u_refs[_i] for _i in _ea_sort_idx]
                _ea_n_t = len(_ea_times)
                _ea_u_pinns = [None] * _ea_n_t
                _ea_metrics = [None] * _ea_n_t

                if not {config.time_adaptive}:
                    # ── Non-adaptive: use single model ───────────────
                    for _ei, _ea_tv in enumerate(_ea_times):
                        _ea_xf = _ea_x_refs[_ei]
                        if _is_steady and _is_3d:
                            _ea_xt = np.column_stack([_ea_xf, _ea_y_refs[_ei], _ea_z_refs[_ei]])
                        elif _is_steady and _is_2d:
                            _ea_xt = np.column_stack([_ea_xf, _ea_y_refs[_ei]])
                        elif _is_3d:
                            _ea_yf = _ea_y_refs[_ei]
                            _ea_zf = _ea_z_refs[_ei]
                            _ea_xt = np.column_stack([_ea_xf, _ea_yf, _ea_zf, np.full_like(_ea_xf, _ea_tv)])
                        elif _is_2d:
                            _ea_yf = _ea_y_refs[_ei]
                            _ea_xt = np.column_stack([_ea_xf, _ea_yf, np.full_like(_ea_xf, _ea_tv)])
                        else:
                            _ea_xt = np.column_stack([_ea_xf, np.full_like(_ea_xf, _ea_tv)])
                        _ea_u_pinns[_ei] = _ea_extract(_ea_xt, _ea_sel, model).flatten()
                        if _is_steady:
                            print(f"  PINN predicted (steady-state): {{len(_ea_xf)}} points")
                        else:
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
                        _step_net   = _apply_net_transforms(_make_net(_step_layers, _step_act, "Glorot uniform"))
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
                            _ea_u_pinns[_ei] = _ea_extract(_ea_xt, _ea_sel, _step_model).flatten()
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

                _ea_metrics_path = _os.path.join(_ea_dir, f"error_metrics{{_ea_suffix}}.txt")
                with open(_ea_metrics_path, "w") as _emf:
                    _emf.write("t,L2_relative,MSE,Max_error,Mean_abs_error\\n")
                    for _ea_tv, _l2, _mse, _mx, _ma in _ea_metrics:
                        _emf.write(f"{{_ea_tv:.6f}},{{_l2:.6e}},{{_mse:.6e}},{{_mx:.6e}},{{_ma:.6e}}\\n")
                print(f"  Metrics saved: {{_ea_metrics_path}}")

                # ── Line comparison ───────────────────────────────────
                if {config.ea_do_line}:
                    _ea_ncols = min(4, _ea_n_t)
                    _ea_nrows = (_ea_n_t + _ea_ncols - 1) // _ea_ncols
                    fig, axes = plt.subplots(_ea_nrows, _ea_ncols, figsize=(4*_ea_ncols, 3.5*_ea_nrows), squeeze=False)
                    _ea_line_suptitle = "PINN vs Ground Truth — Line Comparison"
                    if _is_3d:
                        _ea_line_suptitle += f" (y={_line_slice_y:.3g}, z={_line_slice_z:.3g})"
                    elif _is_2d:
                        _ea_line_suptitle += f" (y={_line_slice_y:.3g})"
                    fig.suptitle(_ea_line_suptitle, fontsize=13, fontweight='bold')
                    _ea_ax_flat = axes.flatten()
                    for _ei in range(_ea_n_t):
                        ax = _ea_ax_flat[_ei]
                        _xv = _ea_x_refs[_ei]
                        if _is_3d:
                            # For 3D line plot: extract the reference points
                            # nearest the same (y, z) slice the Line plot
                            # itself uses (PINNConfig.line_slice_y/_z),
                            # widening the tolerance band if too few
                            # reference points happen to fall near it.
                            _yv = _ea_y_refs[_ei]; _zv = _ea_z_refs[_ei]
                            _y_mid = {_line_slice_y}
                            _z_mid = {_line_slice_z}
                            _y_tol = ({config.y_max} - {config.y_min}) / 20.0
                            _z_tol = ({config.z_max} - {config.z_min}) / 20.0
                            _mid_mask = (np.abs(_yv - _y_mid) < _y_tol) & (np.abs(_zv - _z_mid) < _z_tol)
                            if _mid_mask.sum() < 5:
                                _y_tol2 = ({config.y_max} - {config.y_min}) / 5.0
                                _z_tol2 = ({config.z_max} - {config.z_min}) / 5.0
                                _mid_mask = (np.abs(_yv - _y_mid) < _y_tol2) & (np.abs(_zv - _z_mid) < _z_tol2)
                            if _mid_mask.sum() < 2:
                                _mid_mask = np.ones_like(_xv, dtype=bool)  # fall back to all points
                            _ea_sort = np.argsort(_xv[_mid_mask])
                            _xv_s   = _xv[_mid_mask][_ea_sort]
                            _gt_s   = _ea_u_refs[_ei][_mid_mask][_ea_sort]
                            _pinn_s = _ea_u_pinns[_ei][_mid_mask][_ea_sort]
                        elif _is_2d:
                            # For 2D line plot: extract the reference points
                            # nearest the same y slice the Line plot itself
                            # uses (PINNConfig.line_slice_y).
                            _yv = _ea_y_refs[_ei]
                            _y_mid = {_line_slice_y}
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
                        # Steady-state (e.g. a Poisson equation) has no time
                        # axis at all -- every reference file is really just
                        # a single snapshot at a placeholder t=0, so showing
                        # "t = 0.000" here would be meaningless noise rather
                        # than a real time coordinate.
                        ax.set_title((f"L2 = {{_l2:.2e}}" if _is_steady else f"t = {{_ea_tv:.3f}}  |  L2 = {{_l2:.2e}}"), fontsize=10)
                        ax.set_xlabel("x"); ax.set_ylabel("u(x)" if _is_steady else "u(x,t)"); ax.grid(True, alpha=0.3)
                    for _ej in range(_ea_n_t, len(_ea_ax_flat)):
                        _ea_ax_flat[_ej].set_visible(False)
                    handles, labels = _ea_ax_flat[0].get_legend_handles_labels()
                    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
                               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
                    plt.tight_layout(rect=[0, 0.06, 1, 1])
                    _ea_lp = _os.path.join(_ea_dir, f"line_comparison{{_ea_suffix}}.png")
                    plt.savefig(_ea_lp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                    print(f"  Line comparison saved: {{_ea_lp}}")

                # ── Surface comparison ────────────────────────────────
                if {config.ea_do_surface}:
                    _ea_did_surface = True
                    if _is_3d and _geom_type != "Sphere":
                        # Box-shaped 3D geometry (Cuboid): a genuine smooth
                        # surface (PINN | Ground Truth | Error), like a COMSOL
                        # surface plot. Each of the geometry's 6 flat faces
                        # (from geom.bbox) is predicted on a fine regular grid;
                        # ground truth is interpolated (griddata) from reference
                        # points near that face onto the same grid; and each
                        # face is drawn with plot_surface's per-quad facecolors
                        # -- unlike a scatter of discrete points, adjacent
                        # same-ish-colored grid quads blend into a continuous-
                        # looking colored surface, matching the smoothness of
                        # the 2D contourf plots above.
                        from scipy.interpolate import griddata as _gd3
                        fig = plt.figure(figsize=(15, 4.5 * _ea_n_t))
                        fig.suptitle("PINN vs Ground Truth — 3D Surface Comparison", fontsize=13, fontweight='bold')
                        _geom_bbox_ea = np.asarray(geom.bbox)
                        _bx0, _by0, _bz0 = _geom_bbox_ea[0]
                        _bx1, _by1, _bz1 = _geom_bbox_ea[1]
                        _res3_ea = 36
                        _xg3_ea = np.linspace(_bx0, _bx1, _res3_ea)
                        _yg3_ea = np.linspace(_by0, _by1, _res3_ea)
                        _zg3_ea = np.linspace(_bz0, _bz1, _res3_ea)
                        _Xxy_ea, _Yxy_ea = np.meshgrid(_xg3_ea, _yg3_ea)   # z-faces (free: x,y)
                        _Xxz_ea, _Zxz_ea = np.meshgrid(_xg3_ea, _zg3_ea)   # y-faces (free: x,z)
                        _Yyz_ea, _Zyz_ea = np.meshgrid(_yg3_ea, _zg3_ea)   # x-faces (free: y,z)
                        # (fixed axis idx into x/y/z, fixed value, X, Y, Z grids, free-axis idx pair)
                        _faces3_ea = [
                            (2, _bz0, _Xxy_ea, _Yxy_ea, np.full_like(_Xxy_ea, _bz0), (0, 1)),
                            (2, _bz1, _Xxy_ea, _Yxy_ea, np.full_like(_Xxy_ea, _bz1), (0, 1)),
                            (1, _by0, _Xxz_ea, np.full_like(_Xxz_ea, _by0), _Zxz_ea, (0, 2)),
                            (1, _by1, _Xxz_ea, np.full_like(_Xxz_ea, _by1), _Zxz_ea, (0, 2)),
                            (0, _bx0, np.full_like(_Yyz_ea, _bx0), _Yyz_ea, _Zyz_ea, (1, 2)),
                            (0, _bx1, np.full_like(_Yyz_ea, _bx1), _Yyz_ea, _Zyz_ea, (1, 2)),
                        ]
                        _face_tol_ea = (
                            max((_bx1 - _bx0) * 0.02, 1e-6),
                            max((_by1 - _by0) * 0.02, 1e-6),
                            max((_bz1 - _bz0) * 0.02, 1e-6),
                        )
                        for _ei, _ea_tv in enumerate(_ea_times):
                            _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                            _bxr_ea = _ea_x_refs[_ei]; _byr_ea = _ea_y_refs[_ei]; _bzr_ea = _ea_z_refs[_ei]
                            _ur_ea = _ea_u_refs[_ei]
                            _all_coords_ea = (_bxr_ea, _byr_ea, _bzr_ea)
                            _pinn_faces_ea = []
                            _gt_faces_ea = []
                            for _fax_ea, _fval_ea, _fX_ea, _fY_ea, _fZ_ea, _free_idx_ea in _faces3_ea:
                                _fpts_ea = np.column_stack([_fX_ea.ravel(), _fY_ea.ravel(), _fZ_ea.ravel(), np.full(_fX_ea.size, _ea_tv)])
                                _fpinn_ea = _ea_extract(_fpts_ea, _ea_sel, model).reshape(_fX_ea.shape)
                                _near_mask_ea = np.abs(_all_coords_ea[_fax_ea] - _fval_ea) < _face_tol_ea[_fax_ea]
                                if _near_mask_ea.sum() < 4:
                                    _near_mask_ea = np.ones_like(_bxr_ea, dtype=bool)
                                _free0_ea = _all_coords_ea[_free_idx_ea[0]][_near_mask_ea]
                                _free1_ea = _all_coords_ea[_free_idx_ea[1]][_near_mask_ea]
                                _u_near_ea = _ur_ea[_near_mask_ea]
                                _face_arrs_ea = (_fX_ea, _fY_ea, _fZ_ea)
                                _gridA_ea = _face_arrs_ea[_free_idx_ea[0]]
                                _gridB_ea = _face_arrs_ea[_free_idx_ea[1]]
                                _fgt_ea = _gd3(np.column_stack([_free0_ea, _free1_ea]), _u_near_ea, (_gridA_ea, _gridB_ea), method='linear')
                                _nan_mask_ea = np.isnan(_fgt_ea)
                                if _nan_mask_ea.any():
                                    _fgt_nn_ea = _gd3(np.column_stack([_free0_ea, _free1_ea]), _u_near_ea, (_gridA_ea, _gridB_ea), method='nearest')
                                    _fgt_ea[_nan_mask_ea] = _fgt_nn_ea[_nan_mask_ea]
                                _pinn_faces_ea.append(_fpinn_ea)
                                _gt_faces_ea.append(_fgt_ea)
                            _vmin3_ea = min(min(_f.min() for _f in _pinn_faces_ea), min(_f.min() for _f in _gt_faces_ea))
                            _vmax3_ea = max(max(_f.max() for _f in _pinn_faces_ea), max(_f.max() for _f in _gt_faces_ea))
                            _err_faces_ea = [np.abs(_pf_ea - _gf_ea) for _pf_ea, _gf_ea in zip(_pinn_faces_ea, _gt_faces_ea)]
                            _vmax_err_ea = max(_f.max() for _f in _err_faces_ea)
                            # See the matching comment on the Line comparison
                            # title above -- steady-state has no time axis,
                            # so "t=..." is dropped from every column title.
                            _cols_ea = [
                                (_pinn_faces_ea, (f"PINN  L2={{_l2:.2e}}" if _is_steady else f"PINN  t={{_ea_tv:.3f}}  L2={{_l2:.2e}}"), _vmin3_ea, _vmax3_ea, '{config.plot_colormap}'),
                                (_gt_faces_ea,   ("Ground Truth" if _is_steady else f"Ground Truth  t={{_ea_tv:.3f}}"),         _vmin3_ea, _vmax3_ea, '{config.plot_colormap}'),
                                (_err_faces_ea,  (f"|Error|  Max={{_mx:.2e}}" if _is_steady else f"|Error|  t={{_ea_tv:.3f}}  Max={{_mx:.2e}}"), 0.0, _vmax_err_ea, 'inferno'),
                            ]
                            for _col_ea, (_face_vals_ea, _ttl_ea, _vmin_c_ea, _vmax_c_ea, _cmap_c_ea) in enumerate(_cols_ea):
                                _ax3_ea = fig.add_subplot(_ea_n_t, 3, _ei * 3 + _col_ea + 1, projection='3d')
                                _norm3_ea = plt.Normalize(vmin=_vmin_c_ea, vmax=_vmax_c_ea)
                                _cmap_obj_ea = plt.get_cmap(_cmap_c_ea)
                                for _face_i_ea, (_fax_ea, _fval_ea, _fX_ea, _fY_ea, _fZ_ea, _free_idx_ea) in enumerate(_faces3_ea):
                                    _ax3_ea.plot_surface(
                                        _fX_ea, _fY_ea, _fZ_ea,
                                        facecolors=_cmap_obj_ea(_norm3_ea(_face_vals_ea[_face_i_ea])),
                                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False,
                                    )
                                _sm3_ea = plt.cm.ScalarMappable(cmap=_cmap_obj_ea, norm=_norm3_ea)
                                fig.colorbar(_sm3_ea, ax=_ax3_ea, shrink=0.6, pad=0.12)
                                _ax3_ea.set_title(_ttl_ea, fontsize=10)
                                _ax3_ea.set_xlabel("x"); _ax3_ea.set_ylabel("y"); _ax3_ea.set_zlabel("z")
                                try:
                                    _ax3_ea.set_box_aspect((_bx1 - _bx0, _by1 - _by0, _bz1 - _bz0))
                                except Exception:
                                    pass  # older matplotlib without set_box_aspect -- cosmetic only, safe to skip
                        plt.tight_layout()
                    elif _is_3d:
                        # Non-box 3D geometry (Sphere): no flat-face
                        # parameterization to grid-interpolate onto, so fall
                        # back to a genuine boundary-point scatter -- real
                        # (x, y, z) points that already lie on the geometry's
                        # own boundary (geom.on_boundary()), predicted and
                        # compared directly (no interpolation needed), with a
                        # wireframe box from geom.bbox for spatial context.
                        fig = plt.figure(figsize=(15, 4.5 * _ea_n_t))
                        fig.suptitle("PINN vs Ground Truth — 3D Surface Comparison", fontsize=13, fontweight='bold')
                        _geom_bbox_ea = np.asarray(geom.bbox)
                        _bx0, _by0, _bz0 = _geom_bbox_ea[0]
                        _bx1, _by1, _bz1 = _geom_bbox_ea[1]
                        _corners3_ea = [(_bx0,_by0,_bz0),(_bx1,_by0,_bz0),(_bx1,_by1,_bz0),(_bx0,_by1,_bz0),
                                         (_bx0,_by0,_bz1),(_bx1,_by0,_bz1),(_bx1,_by1,_bz1),(_bx0,_by1,_bz1)]
                        _edges3_ea = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                        for _ei, _ea_tv in enumerate(_ea_times):
                            _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                            _bxr_ea = _ea_x_refs[_ei]; _byr_ea = _ea_y_refs[_ei]; _bzr_ea = _ea_z_refs[_ei]
                            _bnd_mask_ea = geom.on_boundary(np.column_stack([_bxr_ea, _byr_ea, _bzr_ea]))
                            # A Cartesian reference grid only reliably
                            # intersects a FLAT boundary (a Cuboid's faces,
                            # handled by the box-face branch above) in large
                            # numbers; a CURVED boundary is essentially never
                            # landed on exactly by an axis-aligned grid, so
                            # on_boundary() can come back with only a handful
                            # of coincidental exact matches (e.g. a unit
                            # sphere's 6 axis points) even when the dataset
                            # has thousands of valid points. Falling back
                            # whenever the boundary subset is a small
                            # fraction of the data -- not just when it's
                            # nearly empty -- catches that case too.
                            if _bnd_mask_ea.sum() < 4 or _bnd_mask_ea.sum() < 0.05 * len(_bxr_ea):
                                _bnd_mask_ea = np.ones_like(_bxr_ea, dtype=bool)  # shape has no clean boundary subset -- use all ref points
                            _bx_ea = _bxr_ea[_bnd_mask_ea]; _by_ea = _byr_ea[_bnd_mask_ea]; _bz_ea = _bzr_ea[_bnd_mask_ea]
                            _gt_b_ea = _ea_u_refs[_ei][_bnd_mask_ea]
                            if len(_bx_ea) > 3000:
                                _ea_sub = np.random.default_rng(0).choice(len(_bx_ea), size=3000, replace=False)
                                _bx_ea, _by_ea, _bz_ea, _gt_b_ea = _bx_ea[_ea_sub], _by_ea[_ea_sub], _bz_ea[_ea_sub], _gt_b_ea[_ea_sub]
                            _pinn_b_pts_ea = (np.column_stack([_bx_ea, _by_ea, _bz_ea]) if _is_steady else
                                              np.column_stack([_bx_ea, _by_ea, _bz_ea, np.full_like(_bx_ea, _ea_tv)]))
                            _pinn_b_ea = _ea_extract(_pinn_b_pts_ea, _ea_sel, model).flatten()
                            _err_b_ea = np.abs(_pinn_b_ea - _gt_b_ea)
                            _vmin3_ea = min(_pinn_b_ea.min(), _gt_b_ea.min())
                            _vmax3_ea = max(_pinn_b_ea.max(), _gt_b_ea.max())
                            # See the matching comment on the Line comparison
                            # title above -- steady-state has no time axis,
                            # so "t=..." is dropped from every column title.
                            _cols_ea = [
                                (_pinn_b_ea, (f"PINN  L2={{_l2:.2e}}" if _is_steady else f"PINN  t={{_ea_tv:.3f}}  L2={{_l2:.2e}}"), _vmin3_ea, _vmax3_ea, '{config.plot_colormap}'),
                                (_gt_b_ea,   ("Ground Truth" if _is_steady else f"Ground Truth  t={{_ea_tv:.3f}}"),         _vmin3_ea, _vmax3_ea, '{config.plot_colormap}'),
                                (_err_b_ea,  (f"|Error|  Max={{_mx:.2e}}" if _is_steady else f"|Error|  t={{_ea_tv:.3f}}  Max={{_mx:.2e}}"), None, None, 'inferno'),
                            ]
                            for _col_ea, (_vals_ea, _ttl_ea, _vmin_c_ea, _vmax_c_ea, _cmap_c_ea) in enumerate(_cols_ea):
                                _ax3_ea = fig.add_subplot(_ea_n_t, 3, _ei * 3 + _col_ea + 1, projection='3d')
                                for _i3_ea, _j3_ea in _edges3_ea:
                                    _p0_ea, _p1_ea = _corners3_ea[_i3_ea], _corners3_ea[_j3_ea]
                                    _ax3_ea.plot([_p0_ea[0], _p1_ea[0]], [_p0_ea[1], _p1_ea[1]], [_p0_ea[2], _p1_ea[2]],
                                                 color='#888888', linewidth=0.8, alpha=0.6)
                                _sc3_ea = _ax3_ea.scatter(_bx_ea, _by_ea, _bz_ea, c=_vals_ea, cmap=_cmap_c_ea, s=14,
                                                           vmin=_vmin_c_ea, vmax=_vmax_c_ea)
                                fig.colorbar(_sc3_ea, ax=_ax3_ea, shrink=0.6, pad=0.12)
                                _ax3_ea.set_title(_ttl_ea, fontsize=10)
                                _ax3_ea.set_xlabel("x"); _ax3_ea.set_ylabel("y"); _ax3_ea.set_zlabel("z")
                                try:
                                    _ax3_ea.set_box_aspect((_bx1 - _bx0, _by1 - _by0, _bz1 - _bz0))
                                except Exception:
                                    pass  # older matplotlib without set_box_aspect -- cosmetic only, safe to skip
                        plt.tight_layout()
                    elif _is_2d:
                        # 2D: 3 columns (PINN | FEM | Error), one row per time snapshot
                        fig, axes = plt.subplots(_ea_n_t, 3,
                                                 figsize=(15, 4*_ea_n_t), squeeze=False)
                        fig.suptitle("PINN vs Ground Truth — 2D Heatmaps", fontsize=13, fontweight='bold')
                        _res_ea = {config.plot_resolution}
                        _xg_ea = np.linspace({config.x_min}, {config.x_max}, _res_ea)
                        _yg_ea = np.linspace({config.y_min}, {config.y_max}, _res_ea)
                        _Xg_ea, _Yg_ea = np.meshgrid(_xg_ea, _yg_ea)
                        # Outside the shape's own boundary (only differs from
                        # the bbox for Disk/Ellipse/Triangle/Polygon), mask the
                        # grid to NaN so neither the PINN prediction nor the
                        # griddata-interpolated ground truth shows fabricated
                        # values there -- griddata's Delaunay triangulation
                        # otherwise happily interpolates straight across a
                        # concave notch (e.g. the L-Shape's missing quadrant),
                        # matching the same masking the main solution plot uses.
                        _inside_2d_ea = geom.inside(np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel()])).reshape(_res_ea, _res_ea)
                        from scipy.interpolate import griddata as _gd
                        for _ei, _ea_tv in enumerate(_ea_times):
                            _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                            # PINN prediction on grid
                            _xy_grid = (np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel()]) if _is_steady else
                                        np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel(), np.full(_Xg_ea.size, _ea_tv)]))
                            _u_pinn_grid = _ea_extract(_xy_grid, _ea_sel, model).reshape(_res_ea, _res_ea)
                            _u_pinn_grid = np.where(_inside_2d_ea, _u_pinn_grid, np.nan)
                            # FEM interpolated onto same grid
                            _u_fem_grid = _gd(
                                np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei]]),
                                _ea_u_refs[_ei],
                                (_Xg_ea, _Yg_ea), method='linear', fill_value=0.0)
                            _u_fem_grid = np.where(_inside_2d_ea, _u_fem_grid, np.nan)
                            _u_err_grid = np.abs(_u_pinn_grid - _u_fem_grid)
                            _vmin_ea = np.nanmin([_u_pinn_grid, _u_fem_grid])
                            _vmax_ea = np.nanmax([_u_pinn_grid, _u_fem_grid])
                            if _vmax_ea - _vmin_ea < 1e-12:
                                _vmax_ea = _vmin_ea + 1e-12
                            # contourf's own vmin/vmax kwargs do NOT constrain
                            # an integer `levels=N` -- each panel still
                            # auto-ranges to its own data, so PINN and Ground
                            # Truth silently got two different color scales
                            # despite passing the same vmin/vmax here. Passing
                            # explicit level BOUNDARIES instead of a count is
                            # what actually makes both panels share one scale.
                            _levels_ea2d = np.linspace(_vmin_ea, _vmax_ea, 41)
                            # Column 0: PINN
                            im0 = axes[_ei][0].contourf(_Xg_ea, _Yg_ea, _u_pinn_grid, levels=_levels_ea2d,
                                                         cmap='{config.plot_colormap}')
                            # Steady-state has no time axis -- see the
                            # matching comment on the Line comparison title
                            # above -- so "t=..." is dropped from every
                            # column title here too.
                            axes[_ei][0].set_title((f"PINN  L2={{_l2:.2e}}" if _is_steady else f"PINN  t={{_ea_tv:.3f}}  L2={{_l2:.2e}}"), fontsize=10)
                            axes[_ei][0].set_xlabel("x"); axes[_ei][0].set_ylabel("y")
                            fig.colorbar(im0, ax=axes[_ei][0])
                            # Column 1: FEM
                            im1 = axes[_ei][1].contourf(_Xg_ea, _Yg_ea, _u_fem_grid, levels=_levels_ea2d,
                                                         cmap='{config.plot_colormap}')
                            axes[_ei][1].set_title(("Ground Truth" if _is_steady else f"Ground Truth  t={{_ea_tv:.3f}}"), fontsize=10)
                            axes[_ei][1].set_xlabel("x"); axes[_ei][1].set_ylabel("y")
                            fig.colorbar(im1, ax=axes[_ei][1])
                            # Column 2: Absolute error
                            im2 = axes[_ei][2].contourf(_Xg_ea, _Yg_ea, _u_err_grid, levels={config.plot_levels}, cmap='{config.plot_colormap}')
                            axes[_ei][2].set_title((f"|Error|  Max={{_mx:.2e}}" if _is_steady else f"|Error|  t={{_ea_tv:.3f}}  Max={{_mx:.2e}}"), fontsize=10)
                            axes[_ei][2].set_xlabel("x"); axes[_ei][2].set_ylabel("y")
                            fig.colorbar(im2, ax=axes[_ei][2])
                        plt.tight_layout()
                    elif len(_ea_times) < 2:
                        # A 1D x-t surface needs at least 2 distinct time
                        # snapshots to form a non-degenerate grid -- e.g. only
                        # one reference file provided so far. Skip gracefully
                        # instead of crashing matplotlib's contourf on a (1, N)
                        # array; the line comparison above already covers this
                        # single snapshot.
                        _ea_did_surface = False
                        print("  Skipping surface comparison — need at least 2 time snapshots for a 1D x-t surface plot")
                    else:
                        # 1D: standard x vs t surface
                        _ea_x_common = np.linspace({config.x_min}, {config.x_max}, 300)
                        _ea_t_arr = np.array(_ea_times)
                        _ea_U_pinn = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                        _ea_U_fem  = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                        if not {config.time_adaptive}:
                            for _ei, _ea_tv in enumerate(_ea_times):
                                _ea_xt_c = np.column_stack([_ea_x_common, np.full_like(_ea_x_common, _ea_tv)])
                                _ea_U_pinn[_ei, :] = _ea_extract(_ea_xt_c, _ea_sel, model).flatten()
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
                        if _ea_vmax - _ea_vmin < 1e-12:
                            _ea_vmax = _ea_vmin + 1e-12
                        _levels_ea1d = np.linspace(_ea_vmin, _ea_vmax, 41)  # see the matching comment above -- contourf ignores vmin/vmax with an integer `levels=N`
                        fig, axes_s = plt.subplots(1, 3, figsize=(15, 5))
                        fig.suptitle("PINN vs Ground Truth — Surface Comparison", fontsize=13, fontweight='bold')
                        im0 = axes_s[0].contourf(_ea_Tg, _ea_Xg, _ea_U_pinn, levels=_levels_ea1d, cmap='{config.plot_colormap}')
                        axes_s[0].set_title("PINN  u(x,t)"); axes_s[0].set_xlabel("t"); axes_s[0].set_ylabel("x")
                        fig.colorbar(im0, ax=axes_s[0])
                        im1 = axes_s[1].contourf(_ea_Tg, _ea_Xg, _ea_U_fem, levels=_levels_ea1d, cmap='{config.plot_colormap}')
                        axes_s[1].set_title("Ground Truth  u(x,t)"); axes_s[1].set_xlabel("t"); axes_s[1].set_ylabel("x")
                        fig.colorbar(im1, ax=axes_s[1])
                        im2 = axes_s[2].contourf(_ea_Tg, _ea_Xg, _ea_U_err, levels={config.plot_levels}, cmap='{config.plot_colormap}')
                        axes_s[2].set_title("Error  |PINN - Ground Truth|"); axes_s[2].set_xlabel("t"); axes_s[2].set_ylabel("x")
                        fig.colorbar(im2, ax=axes_s[2])
                        plt.tight_layout()
                    if _ea_did_surface:
                        _ea_sp = _os.path.join(_ea_dir, f"surface_comparison{{_ea_suffix}}.png")
                        plt.savefig(_ea_sp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                        print(f"  Surface comparison saved: {{_ea_sp}}")
                print(f"  Group '{{_ea_group_label(_ea_sel) or 'default'}}' analysis complete")
            print("=== Error Analysis Complete ===")

        # ── Export solution data ──────────────────────────────
        if _problem_type != "Inverse":
            _data_dir = _os.path.join(
                _run_dir if (_parametric and _pval is not None)
                else (_sol_dir if _use_save else "/tmp"),
                "solution_data"
            )
            _os.makedirs(_data_dir, exist_ok=True)
            _x_export = np.linspace(_plot_x_min, _plot_x_max, {config.export_grid_size})
            _t_export = np.linspace({config.t_min}, {config.t_max}, {config.export_t_steps})

            if _is_steady and _is_2d:
                # No time axis -- one export, x,y columns only (matches the
                # 2-column network input built for steady 2D problems).
                _y_export = np.linspace(_plot_y_min, _plot_y_max, {config.export_grid_size})
                _Xe, _Ye = np.meshgrid(_x_export, _y_export)
                _XY_exp = np.column_stack([_Xe.ravel(), _Ye.ravel()])
                _u_exp = model.predict(_XY_exp)
                _header = "x,y," + ",".join(_out_names_list)
                _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), _u_exp])
                _fname = _os.path.join(_data_dir, "solution.txt")
                np.savetxt(_fname, _out, header=_header, delimiter=",", comments="")
            elif _is_steady and _is_3d:
                _y_export = np.linspace(_plot_y_min, _plot_y_max, {config.export_grid_size})
                _z_export = np.linspace(_plot_z_min, _plot_z_max, {config.export_grid_size})
                _Xe, _Ye, _Ze = np.meshgrid(_x_export, _y_export, _z_export)
                _XYZ_exp = np.column_stack([_Xe.ravel(), _Ye.ravel(), _Ze.ravel()])
                _u_exp = model.predict(_XYZ_exp)
                _header = "x,y,z," + ",".join(_out_names_list)
                _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), _Ze.ravel(), _u_exp])
                _fname = _os.path.join(_data_dir, "solution.txt")
                np.savetxt(_fname, _out, header=_header, delimiter=",", comments="")
            elif _is_steady:
                _u_exp_all = model.predict(_x_export.reshape(-1, 1))
                _header_cols = "x," + ",".join(_out_names_list)
                _out = np.column_stack([_x_export, _u_exp_all])
                _fname = _os.path.join(_data_dir, "solution.txt")
                np.savetxt(_fname, _out, header=_header_cols, delimiter=",", comments="")
            elif _is_2d:
                _y_export = np.linspace(_plot_y_min, _plot_y_max, {config.export_grid_size})
                for _t_val in _t_export:
                    _Xe, _Ye = np.meshgrid(_x_export, _y_export)
                    _XYT_exp = np.column_stack([_Xe.ravel(), _Ye.ravel(), np.full(_Xe.size, _t_val)])
                    _u_exp = model.predict(_XYT_exp)
                    _header = "x,y,t," + ",".join(_out_names_list)
                    _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), np.full(_Xe.size, _t_val), _u_exp])
                    _fname = _os.path.join(_data_dir, f"solution_t{{_t_val:.4f}}.txt")
                    np.savetxt(_fname, _out, header=_header, delimiter=",", comments="")
            elif _is_3d:
                # Same "x,y,z,t,<outputs>" column convention _auto_configure_ea
                # (main_window.py) already expects when scanning a template's
                # ref_dir for t_*.txt ground-truth files -- so a COMSOL
                # export dropped in with matching columns will line up with
                # this automatically once it's provided.
                _y_export = np.linspace(_plot_y_min, _plot_y_max, {config.export_grid_size})
                _z_export = np.linspace(_plot_z_min, _plot_z_max, {config.export_grid_size})
                for _t_val in _t_export:
                    _Xe, _Ye, _Ze = np.meshgrid(_x_export, _y_export, _z_export)
                    _XYZT_exp = np.column_stack([_Xe.ravel(), _Ye.ravel(), _Ze.ravel(), np.full(_Xe.size, _t_val)])
                    _u_exp = model.predict(_XYZT_exp)
                    _header = "x,y,z,t," + ",".join(_out_names_list)
                    _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), _Ze.ravel(), np.full(_Xe.size, _t_val), _u_exp])
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

            dim_str = "3D" if _is_3d else ("2D" if _is_2d else "1D")
            _steps_desc = "steady (single export)" if _is_steady else f"t_steps={config.export_t_steps}"
            print(f"Solution data saved to: {{_data_dir}} ({{dim_str}}, grid={config.export_grid_size}, {{_steps_desc}})")

# ── End parametric loop ───────────────────────────────────────

# ── Parametric summary plot ───────────────────────────────────
if _parametric and len(_summary) > 0:
    labels = [str(v) for v, _ in _summary]
    losses = [l for _, l in _summary]
    fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
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
    if _is_3d:
        # grid_size is a per-axis resolution, so this cubes it (e.g. 51 ->
        # ~132k points) -- much bigger than the 2D case's square. Left to
        # the user's own ta_grid_size choice (the GUI's smallest preset,
        # 11, keeps this to ~1.3k points) rather than silently overriding
        # it here.
        _xg_ta = np.linspace({config.x_min}, {config.x_max}, grid_size)
        _yg_ta = np.linspace({config.y_min}, {config.y_max}, grid_size)
        _zg_ta = np.linspace({config.z_min}, {config.z_max}, grid_size)
        _Xmesh, _Ymesh, _Zmesh = np.meshgrid(_xg_ta, _yg_ta, _zg_ta)
        x_grid = np.column_stack([_Xmesh.ravel(), _Ymesh.ravel(), _Zmesh.ravel()])
        x = x_grid  # (N*N*N, 3)
    elif _is_2d:
        _xg_ta = np.linspace({config.x_min}, {config.x_max}, grid_size)
        _yg_ta = np.linspace({config.y_min}, {config.y_max}, grid_size)
        _Xmesh, _Ymesh = np.meshgrid(_xg_ta, _yg_ta)
        x_grid = np.column_stack([_Xmesh.ravel(), _Ymesh.ravel()])
        x = x_grid  # (N*N, 2)
    else:
        x_grid = np.linspace({config.x_min}, {config.x_max}, grid_size)

    all_x = []; all_t = []; all_u = []

    if _is_2d or _is_3d:
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

    _plot_idx = {config.plot_output_idx}
    _plot_custom_expr = "{plot_custom_expr_converted}"
    _plot_output_names_list = {repr(config.output_names)}.split(",")
    _plot_n_out = len(_plot_output_names_list)
    _plot_dim = "3D" if _is_3d else ("2D" if _is_2d else "1D")
    _PLOT_TORCH_MATH_NS = {{
        "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
        "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
        "arcsin": torch.asin, "arccos": torch.acos, "arctan": torch.atan,
        "exp": torch.exp, "log": torch.log, "log10": torch.log10,
        "sqrt": torch.sqrt, "abs": torch.abs, "ceil": torch.ceil, "floor": torch.floor,
        "pi": np.pi,
    }}

    def _plot_custom_op(_pf_inputs, _pf_outputs):
        _pf_dvars = _tm_build_dvars(_pf_inputs, _pf_outputs, _plot_n_out,
                                     _plot_output_names_list, _is_steady, _plot_dim,
                                     _plot_custom_expr)
        _pf_ns = dict(_pf_dvars)
        _pf_ns.update(_PLOT_TORCH_MATH_NS)
        _pf_ns["torch"] = torch
        return eval(_plot_custom_expr, _pf_ns)

    def _extract_plot_field(_x_grid, _pf_model):
        # Time-Adaptive's own copy of the Standard-path helper of the same
        # name (out of scope here -- it's defined inside `if not
        # {config.time_adaptive}:`, a separate top-level block this one
        # never runs alongside). Same logic: a derived custom field (e.g.
        # |h| = sqrt(u**2+v**2) for 1D Schrodinger), now including
        # derivatives (du_x, du_xx, du_xy, ...) via the same
        # dde.Model.predict(x, operator=...) mechanism the Standard path
        # uses, when one is configured, else the single raw output column
        # -- so every prediction call in this Time-Adaptive loop plots/
        # animates the same field the Standard path and the main Results
        # panel would, instead of indexing a column that may not even
        # exist (e.g. plot_output_idx pointing past the last real output
        # when "Custom..." is selected). Unlike the Standard path there's
        # no single module-level `model` here -- each step has its own
        # restored model -- so the caller's model object is always passed
        # in explicitly rather than defaulted.
        if _plot_custom_expr.strip():
            return _pf_model.predict(_x_grid, operator=_plot_custom_op)[:, 0]
        return _pf_model.predict(_x_grid)[:, _plot_idx]

    # Accumulates the loss history across EVERY optimizer phase of EVERY
    # time sub-domain, so the loss plot below (see "_ta_all_train_loss")
    # shows the full Time-Adaptive run instead of just the LAST phase of
    # the LAST sub-domain. lh_i/ts_i get overwritten by each new
    # model_i.train() call below -- there can be several per sub-domain
    # (one per optimizer-scheduler phase), plus one more for the legacy
    # Adam-then-L-BFGS path -- so without this, only whichever call
    # happened to run last for the very last step_i would ever reach the
    # plot. Each phase's own step counter restarts at 0 (DeepXDE's
    # LossHistory numbers every train() call's steps from scratch), so
    # _ta_loss_offset -- the cumulative iteration count so far across
    # every phase and every sub-domain -- is added to keep the x-axis
    # continuously increasing across the whole run. Same "steps[-1] added
    # to a running total" trick already used just for the printed
    # cumulative-iteration numbers in the scheduler-phase logging below
    # (see "_sp_iters"/"_sched_cum_iters" elsewhere in this file) --
    # applied here to the loss arrays themselves, not just a printed
    # number.
    _ta_all_train_loss = []
    _ta_all_test_loss = []
    _ta_all_steps = []
    _ta_loss_offset = 0
    _ta_step_boundaries = []  # cumulative-iteration marks where one time sub-domain ends and the next begins

    def _ta_accumulate_loss(_lh):
        global _ta_loss_offset
        _ta_all_steps.extend([_s + _ta_loss_offset for _s in _lh.steps])
        _ta_all_train_loss.extend(_lh.loss_train)
        _ta_all_test_loss.extend(_lh.loss_test)
        _ta_loss_offset += (_lh.steps[-1] if _lh.steps else 0)

    for step_i, (t0, t1) in enumerate(_ta_flat_intervals):
        print(f"\\n--- Time step {{step_i+1}}/{{n_steps}}: t = {{t0:.4f}} to {{t1:.4f}} ---")

        geom_i = _build_geom()
        time_i     = dde.geometry.TimeDomain(t0, t1)
        geomtime_i = dde.geometry.GeometryXTime(geom_i, time_i)

        _constraints_i = []
        # The Boundary Conditions panel (custom_bc_json, already parsed into
        # _custom_bc_entries above) is now the source of truth here too --
        # same as the main (non-time-adaptive) path. The legacy per-side
        # scheme below only runs for configs saved before the panel existed.
        if _custom_bc_entries or _bc_panel_data_present:
            for _bi_ta, _be_ta in enumerate(_custom_bc_entries):
                _btype_ta = _be_ta.get('type', 'dirichlet')
                _bcomp_ta = int(_be_ta.get('component', 0) or 0)
                _bloc_ta  = _be_ta.get('location', '')
                _bval_ta  = _be_ta.get('value', '0')
                _baxis_ta = _be_ta.get('axis', 'x')
                _bderiv_ta = int(_be_ta.get('deriv_order', 0) or 0)
                _bpts_file_ta = _be_ta.get('points_file', '')
                _bloc2_ta = _be_ta.get('location2', '')
                _bdir_ta  = _be_ta.get('direction', 'normal')
                _blabel_ta = f"BC {{_bi_ta + 1}} ({{_btype_ta}})"

                if _btype_ta == 'neumann':
                    _constraints_i.append(dde.icbc.NeumannBC(geomtime_i, _bc_val_fn(_bval_ta), _bc_loc_fn(_bloc_ta), component=_bcomp_ta))
                elif _btype_ta == 'robin':
                    _constraints_i.append(dde.icbc.RobinBC(geomtime_i, _bc_robin_fn(_bval_ta, _bcomp_ta), _bc_loc_fn(_bloc_ta), component=_bcomp_ta))
                elif _btype_ta == 'periodic':
                    _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, _axis_idx_map.get(_baxis_ta, 0), _bc_loc_fn(_bloc_ta),
                                                                derivative_order=_bderiv_ta, component=_bcomp_ta))
                elif _btype_ta == 'pointset':
                    _pts_ta, _pvals_ta = _bc_load_points_file(_bpts_file_ta, _blabel_ta)
                    _constraints_i.append(dde.icbc.PointSetBC(_pts_ta, _pvals_ta, component=_bcomp_ta))
                elif _btype_ta == 'pointset_operator':
                    _pts_ta, _pvals_ta = _bc_load_points_file(_bpts_file_ta, _blabel_ta)
                    _constraints_i.append(dde.icbc.PointSetOperatorBC(_pts_ta, _pvals_ta, _bc_operator_fn(_bval_ta)))
                elif _btype_ta == 'operator':
                    _constraints_i.append(dde.icbc.OperatorBC(geomtime_i, _bc_operator_fn(_bval_ta), _bc_loc_fn(_bloc_ta)))
                elif _btype_ta == 'interface2d':
                    _constraints_i.append(dde.icbc.Interface2DBC(_Interface2DGeomAdapter(geomtime_i), _bc_val_fn(_bval_ta),
                                                                    _bc_loc_fn(_bloc_ta), _bc_loc_fn(_bloc2_ta), direction=_bdir_ta))
                else:  # 'dirichlet' and any unrecognized type fall back to Dirichlet
                    _constraints_i.append(dde.icbc.DirichletBC(geomtime_i, _bc_val_fn(_bval_ta), _bc_loc_fn(_bloc_ta), component=_bcomp_ta))

            # IC (independent of the BC panel -- same per-step IC scheme as always)
            for _oi_ta in range({config.num_outputs}):
                _comp_ta = _oi_ta
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
                    if _is_2d or _is_3d:
                        # x_grid is already an (N, 2) or (N, 3) array of
                        # spatial points either way, so one column_stack
                        # covers both.
                        _xt_ic_2d = np.column_stack([x_grid, np.full(len(x_grid), t0)])
                        _xyt_ic_anchor = _xt_ic_2d
                        _constraints_i.append(dde.icbc.PointSetBC(_xt_ic_2d, prev_u[:, _oi_ta:_oi_ta+1], component=_oi_ta))
                    else:
                        xt_ic = np.column_stack([x_grid, np.full_like(x_grid.ravel(), t0)])
                        _xyt_ic_anchor = xt_ic
                        _constraints_i.append(dde.icbc.PointSetBC(xt_ic, prev_u[:, _oi_ta:_oi_ta+1], component=_oi_ta))
        else:
            # Legacy per-side BCs, for configs saved before the Boundary
            # Conditions panel existed (custom_bc_json empty).
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
                    if _is_2d or _is_3d:
                        _xt_ic_2d = np.column_stack([x_grid, np.full(len(x_grid), t0)])
                        _xyt_ic_anchor = _xt_ic_2d
                        _constraints_i.append(dde.icbc.PointSetBC(_xt_ic_2d, prev_u[:, _oi_ta:_oi_ta+1], component=_oi_ta))
                    else:
                        xt_ic = np.column_stack([x_grid, np.full_like(x_grid.ravel(), t0)])
                        _xyt_ic_anchor = xt_ic
                        _constraints_i.append(dde.icbc.PointSetBC(xt_ic, prev_u[:, _oi_ta:_oi_ta+1], component=_oi_ta))

        data_i = dde.data.TimePDE(
            geomtime_i, pde, _constraints_i,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_initial={config.num_initial}, num_test={config.num_test},
            # forward_ic_from_file only matters for step 0 (loading the
            # true initial condition from a file there); every later step's
            # IC constraint comes from the previous window's own predicted
            # solution (prev_u) via _xyt_ic_anchor, regardless of that flag.
            # Gating this on the bare flag (rather than "step 0 AND the
            # flag") silently dropped the previous-window anchor points for
            # EVERY step after the first whenever forward_ic_from_file was
            # on, weakening (though not removing -- the PointSetBC loss
            # term above is still added either way) how strongly later
            # windows are forced to match their own true starting point.
            anchors=None if (step_i == 0 and {config.forward_ic_from_file}) else (_xyt_ic_anchor if (step_i > 0 and (_is_2d or _is_3d)) else None)
        )

        net_i   = _apply_net_transforms(_make_net({config.layers}, "{config.activation}", "{config.kernel_initializer}", {config.weight_decay}))
        model_i = dde.Model(data_i, net_i)

        # ── Training callbacks (opt-in, fresh instances each step) ─
{_train_cbs_code_ta}

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
            _ic_geom_pt = _build_geom()
            _ic_td_pt = dde.geometry.TimeDomain(t0, t1)
            _ic_gt_pt = dde.geometry.GeometryXTime(_ic_geom_pt, _ic_td_pt)
            # Only IC constraint
            _ic_constraints_pt = []
            if {config.forward_ic_from_file}:
                _ic_ta_xyt, _ic_ta_vals = _load_ic_from_file({repr(config.forward_ic_file)})
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
            # Only IC constraints included -- see the matching comment in
            # the Standard-path IC pre-training block above for why real
            # BCs are left out entirely rather than just zero-weighted
            # (an empty-match BC term's NaN loss isn't neutralized by a 0
            # weight), and why num_initial/num_test can't stay 0 here.
            _ic_pre_constraints_ta = _ic_constraints_pt
            _data_pt = dde.data.TimePDE(
                _ic_gt_pt, _pde_dummy_ta, _ic_pre_constraints_ta,
                num_domain=0, num_boundary=0,
                num_initial={_ic_pretrain_num_initial_safe}, num_test={config.ic_pretrain_num_test},
                train_distribution="{config.point_distribution}",
                anchors=None
            )
            _net_pt = model_i.net
            _model_pt = dde.Model(_data_pt, _net_pt)
            # IC-only weights: PDE=0, IC=1000
            _ic_only_w_ta = [0.0] * {config.num_outputs} + [1000.0] * len(_ic_constraints_pt)
            _model_pt.compile("{config.ic_pretrain_optimizer}", lr={config.ic_pretrain_lr},
                              loss="{config.ic_pretrain_loss}", loss_weights=_ic_only_w_ta)
            _ic_pre_save_dir_ta = _os.path.join({repr(config.save_dir)}, "ic_pretrain")
            _os.makedirs(_ic_pre_save_dir_ta, exist_ok=True)
            if {config.ic_pretrain_restore} and {repr(config.ic_pretrain_restore_path)} and _os.path.exists({repr(config.ic_pretrain_restore_path)}):
                print("  Restoring IC pre-train model from: " + {repr(config.ic_pretrain_restore_path)})
                _ic_ckpt_ta = torch.load({repr(config.ic_pretrain_restore_path)}, map_location="cpu")
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

        # ── Phase 1 (Adam) ──────────────────────────────────────
        # Mirrors the Standard (non-Time-Adaptive) path's own unconditional
        # Phase 1 training call above (model.train(iterations=_iters, ...)
        # under "if not _sched_active:") -- this per-step loop previously
        # had no equivalent: it only ever compiled/trained model_i inside
        # the scheduler-phases block below or the legacy optimizer2
        # L-BFGS-only branch's elif, with nothing to fall back on when
        # neither applied. A legacy/hand-edited config with the (GUI-
        # hidden, always-on) scheduler off, an empty scheduler_phases, and
        # optimizer2 left at "none" would then never compile or train
        # model_i at all for this step -- model_i.predict(...) ran
        # immediately after on an untrained, randomly-initialized network,
        # and that garbage became the next step's initial condition, with
        # no exception raised anywhere.
        if not ({config.optimizer_scheduler} and {len(config.scheduler_phases) > 0}):
            model_i.compile("{config.optimizer}", lr={config.learning_rate},
                            loss="{config.loss_type}", loss_weights=_multi_weights)
            lh_i, ts_i = model_i.train(iterations={config.iterations}, display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
            _ta_accumulate_loss(lh_i)
            print(f"  Adam phase done. Steps: {{len(lh_i.steps)}}")
            if _use_save:
                _step_dir_adam = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                _os.makedirs(_step_dir_adam, exist_ok=True)
                _step_adam_path = _os.path.join(_step_dir_adam, "model_adam")
                model_i.save(_step_adam_path)
                print(f"Step Adam model saved: {{_step_adam_path}}.pt")

        # ── Optimizer Scheduler phases ────────────────────────
        if {config.optimizer_scheduler} and {len(config.scheduler_phases) > 0}:
            import json as _json
            _sched_phases = _json.loads({repr(config.scheduler_phases)})
            for _sp_i, _sp in enumerate(_sched_phases):
                print(f"  === Scheduler Phase {{_sp_i+1}}: {{_sp['optimizer']}} {{_sp['iterations']}} iters ===")
                _sp_weights = [float(w) for w in _sp['weights'].split(',') if w.strip()]
                # Same safety net as the main (non-Time-Adaptive) scheduler loop:
                # each step rebuilds its own _constraints_i, which can be a
                # different length than what the GUI's scheduler-phase weight
                # string was built from (e.g. a step boundary changing which IC
                # constraint applies) -- fall back to uniform weights of the
                # right length rather than crash mid-training.
                _sp_expected_len_i = {config.num_outputs} + len(_constraints_i)
                if len(_sp_weights) != _sp_expected_len_i:
                    print(f"  ⚠️ Scheduler Phase {{_sp_i+1}} (step {{step_i+1}}): configured weights have "
                          f"{{len(_sp_weights)}} entries but {{_sp_expected_len_i}} loss terms are active this "
                          f"step ({{len(_constraints_i)}} constraints + {config.num_outputs} PDE) — using "
                          f"uniform weights for this phase instead of crashing.")
                    _sp_weights = [1.0] * _sp_expected_len_i
                if _sp['optimizer'] == 'lbfgs':
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter=_sp['iterations'],
                        maxfun=int(_sp['iterations']*1.25), maxls={config.lbfgs_maxls})
                    _lbfgs_float = "{config.lbfgs_float_type}"
                    model_i.compile("L-BFGS", loss=_sp.get('loss', '{config.loss_type}'),
                                    loss_weights=_sp_weights)
                    lh_i, ts_i = model_i.train(display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
                    _ta_accumulate_loss(lh_i)
                    if _use_save:
                        _sp_step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                        _os.makedirs(_sp_step_dir, exist_ok=True)
                        _sp_save_path = _os.path.join(_sp_step_dir, f"model_lbfgs-phase{{_sp_i+1}}")
                        model_i.save(_sp_save_path)
                        print(f"  Phase {{_sp_i+2}} L-BFGS model saved: {{_sp_save_path}}.pt")
                elif _sp['optimizer'] == 'nncg':
                    # See the matching NNCG branch in the Standard scheduler
                    # loop above for why: exact-case "NNCG" name, no lr=, no
                    # decay (NNCG_options is left at its defaults -- not
                    # exposed in this app's UI).
                    model_i.compile("NNCG", loss=_sp.get('loss', '{config.loss_type}'),
                                    loss_weights=_sp_weights)
                    lh_i, ts_i = model_i.train(iterations=_sp['iterations'], display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
                    _ta_accumulate_loss(lh_i)
                    if _use_save:
                        _sp_iters = lh_i.steps[-1] if lh_i.steps else _sp['iterations']
                        _sp_step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                        _os.makedirs(_sp_step_dir, exist_ok=True)
                        _sp_save_path = _os.path.join(_sp_step_dir, f"model_nncg-phase{{_sp_i+1}}")
                        model_i.save(_sp_save_path)
                        print(f"  Phase {{_sp_i+2}} NNCG model saved: {{_sp_save_path}}.pt")
                else:
                    _sp_decay = None
                    _sp_decay_type = _sp.get('decay_type', 'none')
                    if _sp_decay_type and _sp_decay_type != 'none':
                        _sp_p1 = _sp.get('decay_p1', 0) or 0
                        _sp_p2 = _sp.get('decay_p2', 0) or 0
                        if _sp_decay_type == 'step':
                            _sp_decay = ('step', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'cosine':
                            _sp_decay = ('cosine', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'exponential':
                            _sp_decay = ('exponential', float(_sp_p1))
                    model_i.compile(_sp['optimizer'], lr=_sp['lr'], decay=_sp_decay,
                                    loss=_sp.get('loss', '{config.loss_type}'), loss_weights=_sp_weights)
                    lh_i, ts_i = model_i.train(iterations=_sp['iterations'], display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
                    _ta_accumulate_loss(lh_i)
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
            lh_i, ts_i = model_i.train(display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
            _ta_accumulate_loss(lh_i)
            print(f"  L-BFGS phase done. Steps: {{len(lh_i.steps)}}")

            if _use_save:
                _step_dir_lbfgs = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                _os.makedirs(_step_dir_lbfgs, exist_ok=True)
                _step_lbfgs_path = _os.path.join(_step_dir_lbfgs, "model_lbfgs")
                model_i.save(_step_lbfgs_path)
                print(f"Step L-BFGS model saved: {{_step_lbfgs_path}}.pt")

        if _is_3d:
            # 3D: predict on x-y-z grid at t=t1, store as flat array for PointSetBC next step
            _x_pred = np.linspace({config.x_min}, {config.x_max}, grid_size)
            _y_pred = np.linspace({config.y_min}, {config.y_max}, grid_size)
            _z_pred = np.linspace({config.z_min}, {config.z_max}, grid_size)
            _Xp, _Yp, _Zp = np.meshgrid(_x_pred, _y_pred, _z_pred)
            _xyzt_pred = np.column_stack([_Xp.ravel(), _Yp.ravel(), _Zp.ravel(), np.full(_Xp.size, t1)])
            prev_u  = model_i.predict(_xyzt_pred)
            x_grid  = _xyzt_pred[:, :3]  # store (x,y,z) triples for next step's PointSetBC
        elif _is_2d:
            # 2D: predict on x-y grid at t=t1, store as flat array for PointSetBC next step
            _x_pred = np.linspace({config.x_min}, {config.x_max}, grid_size)
            _y_pred = np.linspace({config.y_min}, {config.y_max}, grid_size)
            _Xp, _Yp = np.meshgrid(_x_pred, _y_pred)
            _xyt_pred = np.column_stack([_Xp.ravel(), _Yp.ravel(), np.full(_Xp.size, t1)])
            prev_u  = model_i.predict(_xyt_pred)
            x_grid  = _xyt_pred[:, :2]  # store (x,y) pairs for next step's PointSetBC
        else:
            x_pred  = np.linspace({config.x_min}, {config.x_max}, grid_size)
            t_pred  = np.full_like(x_pred, t1)
            xt_pred = np.column_stack([x_pred, t_pred])
            prev_u  = model_i.predict(xt_pred)
            x_grid  = x_pred.reshape(-1, 1)

        if _is_3d:
            # 3D: no volumetric renderer for the stitched plot either (same
            # limit as the Standard path's own 3D solution plot) -- fix
            # BOTH y and z at their domain mid-points and store an x-t
            # slice, exactly like the 2D branch below fixes y alone. This
            # keeps all_x/all_t/all_u in the same shape regardless of
            # dimension, so the final combined plot needs no 3D-specific
            # branch of its own.
            x_plot = np.linspace({config.x_min}, {config.x_max}, 100)
            t_plot = np.linspace(t0, t1, 50)
            y_mid  = {_line_slice_y}
            z_mid  = {_line_slice_z}
            Xp, Tp = np.meshgrid(x_plot, t_plot)
            XYZTp  = np.column_stack([Xp.ravel(), np.full(Xp.size, y_mid), np.full(Xp.size, z_mid), Tp.ravel()])
            Up     = _extract_plot_field(XYZTp, model_i).reshape(50, 100)
            all_x.append(Xp); all_t.append(Tp); all_u.append(Up)
        elif not _is_2d:
            x_plot = np.linspace({config.x_min}, {config.x_max}, 100)
            t_plot = np.linspace(t0, t1, 50)
            Xp, Tp = np.meshgrid(x_plot, t_plot)
            XTp    = np.vstack([Xp.ravel(), Tp.ravel()]).T
            Up     = _extract_plot_field(XTp, model_i).reshape(50, 100)
            all_x.append(Xp); all_t.append(Tp); all_u.append(Up)
        else:
            # 2D: store mid-y slice for combined plot
            x_plot = np.linspace({config.x_min}, {config.x_max}, 100)
            t_plot = np.linspace(t0, t1, 50)
            y_mid  = {_line_slice_y}
            Xp, Tp = np.meshgrid(x_plot, t_plot)
            XYTp   = np.column_stack([Xp.ravel(), np.full(Xp.size, y_mid), Tp.ravel()])
            Up     = _extract_plot_field(XYTp, model_i).reshape(50, 100)
            all_x.append(Xp); all_t.append(Tp); all_u.append(Up)

        print(f"Step {{step_i+1}} done. Final train loss: {{sum(lh_i.loss_train[-1]):.4e}}")
        _ta_step_boundaries.append(_ta_loss_offset)

        # ── Save step plot & models ───────────────────────────
        if _use_save:
            _step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
            _os.makedirs(_step_dir, exist_ok=True)
            _plot_type_step = "{config.plot_type}"
            if _plot_type_step not in ("Surface", "Line (time steps)"):
                # GIF animations and Parameter Convergence aren't wired up
                # for Time-Adaptive per-step preview images yet -- fall
                # back to the static Surface heatmap rather than silently
                # skipping this step's preview entirely.
                _plot_type_step = "Surface"
            _x_plot_step = np.linspace({config.x_min}, {config.x_max}, 100)
            _t_plot_step = np.linspace(t0, t1, 50)
            _Xp_step, _Tp_step = np.meshgrid(_x_plot_step, _t_plot_step)
            if _is_3d:
                _y_mid_step = {_line_slice_y}
                _z_mid_step = {_line_slice_z}
                _XTp_step = np.column_stack([_Xp_step.ravel(), np.full(_Xp_step.size, _y_mid_step),
                                              np.full(_Xp_step.size, _z_mid_step), _Tp_step.ravel()])
            elif _is_2d:
                _y_mid_step = {_line_slice_y}
                _XTp_step = np.column_stack([_Xp_step.ravel(), np.full(_Xp_step.size, _y_mid_step), _Tp_step.ravel()])
            else:
                _XTp_step = np.vstack([_Xp_step.ravel(), _Tp_step.ravel()]).T
            _Up_step = _extract_plot_field(_XTp_step, model_i).reshape(50, 100)

            if _plot_type_step == "Surface" or _plot_type_step.startswith("📊"):
                _vmin_step = None if {config.plot_auto_range} else {config.plot_vmin}
                _vmax_step = None if {config.plot_auto_range} else {config.plot_vmax}
                _step_fname = _os.path.join(_step_dir, f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}.png")
                if _is_3d:
                    # 3D: same "x-y heatmap at the z mid-plane" convention as
                    # the Standard (non-Time-Adaptive) 3D solution plot --
                    # at t=t0 and t=t1.
                    _res_step = {config.plot_resolution}
                    _xg_s = np.linspace({config.x_min}, {config.x_max}, _res_step)
                    _yg_s = np.linspace({config.y_min}, {config.y_max}, _res_step)
                    _Xg_s, _Yg_s = np.meshgrid(_xg_s, _yg_s)
                    _z_mid_s = {_line_slice_z}
                    fig, axes = plt.subplots(1, 2, figsize=_plot_figsize(10, 4))
                    for _ai, _tv_s in enumerate([t0, t1]):
                        _xyt_s = np.column_stack([_Xg_s.ravel(), _Yg_s.ravel(), np.full(_Xg_s.size, _z_mid_s), np.full(_Xg_s.size, _tv_s)])
                        _U_s = _extract_plot_field(_xyt_s, model_i).reshape(_res_step, _res_step)
                        im = axes[_ai].contourf(_Xg_s, _Yg_s, _U_s, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_step, vmax=_vmax_step)
                        if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
                        axes[_ai].set_xlabel("x"); axes[_ai].set_ylabel("y")
                        axes[_ai].set_title(f"t = {{_tv_s:.4f}}  (z={{_z_mid_s:.3g}})")
                    fig.suptitle(f"Step {{step_i+1}}: t = {{t0:.4f}} → {{t1:.4f}}", fontsize=11)
                    plt.tight_layout()
                elif _is_2d:
                    # 2D: x-y heatmaps at t=t0 and t=t1
                    _res_step = {config.plot_resolution}
                    _xg_s = np.linspace({config.x_min}, {config.x_max}, _res_step)
                    _yg_s = np.linspace({config.y_min}, {config.y_max}, _res_step)
                    _Xg_s, _Yg_s = np.meshgrid(_xg_s, _yg_s)
                    fig, axes = plt.subplots(1, 2, figsize=_plot_figsize(10, 4))
                    for _ai, _tv_s in enumerate([t0, t1]):
                        _xyt_s = np.column_stack([_Xg_s.ravel(), _Yg_s.ravel(), np.full(_Xg_s.size, _tv_s)])
                        _U_s = _extract_plot_field(_xyt_s, model_i).reshape(_res_step, _res_step)
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
                    _Us2 = _extract_plot_field(_XTs2, model_i).reshape(_res_step, _res_step)
                    fig, ax = plt.subplots(figsize=_plot_figsize(7, 4))
                    # Same configurable axis orientation as the Standard
                    # (non-Time-Adaptive) static Surface plot's own "Swap
                    # axes" setting (plot_swap_xt, default True -- x-axis=t,
                    # y-axis=domain x) -- this per-step preview used to
                    # always hardcode the opposite (x-axis=x, y-axis=t),
                    # ignoring the setting entirely, so a Time-Adaptive run's
                    # step-by-step plots looked inconsistent with every
                    # other 1D surface plot in the app. No reshape of
                    # _Us2 needed, just swapping which of _Xs2/_Ts2 is
                    # passed first to contourf (same trick used by the
                    # Standard plot and its GIF-animation sibling).
                    if {config.plot_swap_xt}:
                        im = ax.contourf(_Ts2, _Xs2, _Us2, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_step, vmax=_vmax_step)
                        ax.set_xlabel("t"); ax.set_ylabel("x")
                    else:
                        im = ax.contourf(_Xs2, _Ts2, _Us2, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_step, vmax=_vmax_step)
                        ax.set_xlabel("x"); ax.set_ylabel("t")
                    if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
                    ax.set_title(f"Step {{step_i+1}}: t = {{t0:.4f}} → {{t1:.4f}}")
                    plt.tight_layout()
                plt.savefig(_step_fname, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _plot_type_step.startswith("Line"):
                n_steps_plot = {config.num_timesteps_line}
                _x_l2 = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                _t_line = np.linspace(t0, t1, n_steps_plot)
                fig, ax = plt.subplots(figsize=_plot_figsize(8, 4))
                colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_plot))
                # A line plot is inherently 1D -- for 2D/3D this is a slice
                # along x at the domain's other mid-point(s) (same "fix the
                # other axes at their mid-point" convention used throughout
                # this Time-Adaptive loop for 2D/3D previews).
                if _is_3d:
                    _y_mid_l2 = {_line_slice_y}
                    _z_mid_l2 = {_line_slice_z}
                elif _is_2d:
                    _y_mid_l2 = {_line_slice_y}
                for _ci, _tv in enumerate(_t_line):
                    if _is_3d:
                        _xt_line = np.column_stack([_x_l2, np.full_like(_x_l2, _y_mid_l2),
                                                     np.full_like(_x_l2, _z_mid_l2), np.full_like(_x_l2, _tv)])
                    elif _is_2d:
                        _xt_line = np.column_stack([_x_l2, np.full_like(_x_l2, _y_mid_l2), np.full_like(_x_l2, _tv)])
                    else:
                        _xt_line = np.column_stack([_x_l2, np.full_like(_x_l2, _tv)])
                    _u_line = _extract_plot_field(_xt_line, model_i).flatten()
                    ax.plot(_x_l2, _u_line, color=colors[_ci], linewidth={config.plot_linewidth}, label=f"t={{_tv:.3f}}")
                _ylabel_l2 = (f"u(x,y={_line_slice_y:.3g},z={_line_slice_z:.3g})" if _is_3d
                              else (f"u(x,y={_line_slice_y:.3g})" if _is_2d else "u"))
                ax.set_xlabel("x"); ax.set_ylabel(_ylabel_l2)
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
            # y_min/y_max/z_min/z_max added alongside x_min/x_max so a 2D/3D
            # Time-Adaptive step's own model_config-equivalent is complete
            # on its own (previously only x_min/x_max were saved here --
            # harmless for 1D, since restore already defaults those two,
            # but silently wrong for 2D/3D if this file were ever used as a
            # restore config directly, since the y/z bounds would fall back
            # to the [0,1] default instead of this run's real domain).
            _step_cfg = {{
                "step": step_i + 1,
                "t0": t0, "t1": t1,
                "layers": {config.layers},
                "activation": "{config.activation}",
                "x_min": {config.x_min}, "x_max": {config.x_max},
                "y_min": {config.y_min}, "y_max": {config.y_max},
                "z_min": {config.z_min}, "z_max": {config.z_max},
                "t_min": t0, "t_max": t1,
                "problem_dim": "{config.problem_dim}",
                "loss_type": "{config.loss_type}",
                "optimizer": "{config.optimizer}",
                "optimizer2": "{config.optimizer2}",
                "output_names": {repr(config.output_names)},
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
    if _plot_type_ta not in ("Surface", "Line (time steps)"):
        # Same fallback reasoning as the per-step preview above -- GIF
        # animations and Parameter Convergence aren't wired up for
        # Time-Adaptive's final solution plot yet.
        _plot_type_ta = "Surface"
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
            fig, axes = plt.subplots(1, _n_snaps_ta, figsize=(_plot_figsize(5, 5)[0]*_n_snaps_ta, _plot_figsize(5, 5)[1]))
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
                _sn_ta = _apply_net_transforms(_make_net(_sc_ta.get("layers",{config.layers}), _sc_ta.get("activation","{config.activation}"), "Glorot uniform"))
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
            # 1D, and 3D too (its per-step loop above already fixed y and z
            # at their domain mid-points and stitched an x-t slice into
            # X_full/T_full/U_full, same shape as the 1D case) -- no
            # volumetric renderer, so this mid-plane slice is the final
            # solution plot for 3D as well.
            fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
            # Same configurable axis orientation as the Standard path's 1D
            # "Surface" plot -- see the matching comment there.
            if {config.plot_swap_xt}:
                im = ax.contourf(T_full, X_full, U_full, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_ta, vmax=_vmax_ta)
                ax.set_xlabel("t"); ax.set_ylabel("x")
            else:
                im = ax.contourf(X_full, T_full, U_full, levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=_vmin_ta, vmax=_vmax_ta)
                ax.set_xlabel("x"); ax.set_ylabel("t")
            if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
            ax.set_title("Time-Adaptive PINN Solution")
            plt.tight_layout(); plt.savefig(_ta_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
    elif _plot_type_ta.startswith("Line"):
        n_ts   = {config.num_timesteps_line}
        t_vals = np.linspace(_ta_flat_intervals[0][0], _ta_flat_intervals[-1][1], n_ts)
        fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
        colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_ts))
        for ci, tv in enumerate(t_vals):
            idx = np.argmin(np.abs(T_full[:,0] - tv))
            ax.plot(X_full[idx,:], U_full[idx,:], color=colors[ci], linewidth={config.plot_linewidth}, label=f"t={{tv:.3f}}")
        _ylabel_ta_line = (f"u(x,y={_line_slice_y:.3g},z={_line_slice_z:.3g},t)" if _is_3d
                           else (f"u(x,y={_line_slice_y:.3g},t)" if _is_2d else "u(x,t)"))
        ax.set_xlabel("x"); ax.set_ylabel(_ylabel_ta_line)
        ax.set_title("Time-Adaptive PINN — Line Plot")
        ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
        plt.tight_layout(); plt.savefig(_ta_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

    # Full Time-Adaptive run -- every optimizer phase of every time
    # sub-domain, concatenated with a running iteration offset (see
    # "_ta_accumulate_loss" above) -- not just whichever phase happened
    # to run last for the very last sub-domain.
    train_loss_ta = _ta_all_train_loss; test_loss_ta = _ta_all_test_loss
    steps_ta = _ta_all_steps
    # Light vertical markers at each time sub-domain's boundary, so a
    # jump/kink in the loss (a brand-new network starting that
    # sub-domain, warm-started only via its IC) is visually distinguishable
    # from an actual optimizer-related loss spike. Skip the very last
    # boundary (the run's own end -- nothing follows it worth marking) and
    # don't bother with a boundary at 0 if the first sub-domain starts
    # right there.
    def _ta_loss_boundary_markers(_ax):
        for _tb in _ta_step_boundaries[:-1]:
            if _tb > 0:
                _ax.axvline(_tb, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    _make_loss_plot(train_loss_ta, test_loss_ta, steps_ta, _ta_loss_path,
                     f"Loss — All {{n_steps}} Time Sub-domain(s)",
                     xlabel="Iteration (cumulative across all time sub-domains)",
                     extra_fn=_ta_loss_boundary_markers)

    # ── Time Adaptive Error Analysis ──────────────────────────
    if {config.ea_files}:
        from scipy.interpolate import interp1d as _interp1d
        import glob as _ea_glob, json as _ea_json
        _ea_dir = _os.path.join(_save_dir if _use_save else "/tmp", "error_analysis")
        _os.makedirs(_ea_dir, exist_ok=True)

        # Same (time, path, output_selector) normalization and per-output
        # grouping as the Standard Error Analysis path above -- see its own
        # comment for the full rationale. Kept as its own copy here since
        # Time-Adaptive is a separate top-level branch that never runs
        # alongside the Standard one.
        _ea_files_norm = []
        for _ea_entry in {config.ea_files}:
            if len(_ea_entry) >= 3:
                _ea_tv0, _ea_fp0, _ea_sel0 = _ea_entry[0], _ea_entry[1], _ea_entry[2]
            else:
                _ea_tv0, _ea_fp0 = _ea_entry[0], _ea_entry[1]
                _ea_sel0 = None
            if {config.t_min} - 1e-10 <= _ea_tv0 <= {config.t_max} + 1e-10:
                _ea_files_norm.append((_ea_tv0, _ea_fp0, _ea_sel0))
        print(f"  Filtering to t=[{config.t_min}, {config.t_max}]: {{len(_ea_files_norm)}} files")
        print("\\n=== Running Time-Adaptive Error Analysis ===")

        _ea_groups = {{}}
        for _ea_tv0, _ea_fp0, _ea_sel0 in _ea_files_norm:
            _ea_gkey = repr(_ea_sel0)
            if _ea_gkey not in _ea_groups:
                _ea_groups[_ea_gkey] = {{"sel": _ea_sel0, "files": []}}
            _ea_groups[_ea_gkey]["files"].append((_ea_tv0, _ea_fp0))
        _ea_multi_output = len(_ea_groups) > 1

        def _ea_group_label(_ea_sel):
            if _ea_sel is None:
                return ""
            if isinstance(_ea_sel, int):
                return (_plot_output_names_list[_ea_sel].strip()
                        if 0 <= _ea_sel < len(_plot_output_names_list) else f"out{{_ea_sel}}")
            _ea_lbl = (_ea_sel[1] or "").strip() if len(_ea_sel) > 1 else ""
            return _ea_lbl if _ea_lbl else "custom"

        def _ea_extract(_ea_grid, _ea_sel, _ea_model):
            # TA has no single module-level `model` (each step restores
            # its own) -- unlike the Standard path's _ea_extract, the
            # model object here is always required, never defaulted.
            # _ea_sel a (expr, label) tuple is now derivative-aware too --
            # see the matching comment on the Standard path's own
            # _ea_extract above.
            if _ea_sel is None:
                return _extract_plot_field(_ea_grid, _ea_model)
            if isinstance(_ea_sel, int):
                return _ea_model.predict(_ea_grid)[:, _ea_sel]
            _ea_expr = _ea_sel[0]

            def _ea_custom_op(_ea_inputs, _ea_outputs):
                _ea_dvars = _tm_build_dvars(_ea_inputs, _ea_outputs, _plot_n_out,
                                             _plot_output_names_list, _is_steady, _plot_dim,
                                             _ea_expr)
                _ea_ns = dict(_ea_dvars)
                _ea_ns.update(_PLOT_TORCH_MATH_NS)
                _ea_ns["torch"] = torch
                return eval(_ea_expr, _ea_ns)

            return _ea_model.predict(_ea_grid, operator=_ea_custom_op)[:, 0]

        for _ea_group_key, _ea_group in _ea_groups.items():
            _ea_files = _ea_group["files"]
            _ea_sel = _ea_group["sel"]
            _ea_suffix = f"_{{_ea_group_label(_ea_sel)}}" if _ea_multi_output else ""
            if _ea_multi_output:
                print(f"  ── Output group: {{_ea_group_label(_ea_sel) or 'default'}} ({{len(_ea_files)}} files) ──")

            # Load all ground truth files
            _ea_times = []; _ea_x_refs = []; _ea_y_refs = []; _ea_z_refs = []; _ea_u_refs = []
            for _ea_tv, _ea_fp in _ea_files:
                _ea_d = np.loadtxt(_ea_fp)
                if _ea_d.ndim == 1: _ea_d = _ea_d.reshape(1, -1)
                if _is_3d:
                    # 3D format: x, y, z, t, u -- same as the Standard path's
                    # own non-steady 3D Error Analysis loader.
                    _ea_idx = np.lexsort((_ea_d[:, 2], _ea_d[:, 1], _ea_d[:, 0]))
                    _ea_x_refs.append(_ea_d[_ea_idx, 0])
                    _ea_y_refs.append(_ea_d[_ea_idx, 1])
                    _ea_z_refs.append(_ea_d[_ea_idx, 2])
                    _ea_u_refs.append(_ea_d[_ea_idx, 4])
                    _detected_t = float(_ea_d[0, 3])
                    _ea_times.append(_detected_t)
                    print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                elif _is_2d:
                    _ea_idx = np.lexsort((_ea_d[:, 1], _ea_d[:, 0]))
                    _ea_x_refs.append(_ea_d[_ea_idx, 0])
                    _ea_y_refs.append(_ea_d[_ea_idx, 1])
                    _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                    _ea_u_refs.append(_ea_d[_ea_idx, 3])
                    _detected_t = float(_ea_d[0, 2])
                    _ea_times.append(_detected_t)
                    print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                else:
                    _ea_idx = np.argsort(_ea_d[:, 0])
                    _ea_x_refs.append(_ea_d[_ea_idx, 0])
                    _ea_y_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                    _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                    _ea_u_refs.append(_ea_d[_ea_idx, 2])
                    _ea_times.append(float(_ea_tv))
                    print(f"  Loaded ground truth t={{_ea_tv:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
            # Sort by time
            _ea_sort_idx = np.argsort(_ea_times)
            _ea_times  = [_ea_times[_i]  for _i in _ea_sort_idx]
            _ea_x_refs = [_ea_x_refs[_i] for _i in _ea_sort_idx]
            _ea_y_refs = [_ea_y_refs[_i] for _i in _ea_sort_idx]
            _ea_z_refs = [_ea_z_refs[_i] for _i in _ea_sort_idx]
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

                if _is_3d:
                    _step_geom = dde.geometry.Cuboid([{config.x_min}, {config.y_min}, {config.z_min}],
                                                       [{config.x_max}, {config.y_max}, {config.z_max}])
                elif _is_2d:
                    _step_geom = dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])
                else:
                    _step_geom = dde.geometry.Interval({config.x_min}, {config.x_max})
                _step_td    = dde.geometry.TimeDomain(_t0_i, _t1_i)
                _step_gt    = dde.geometry.GeometryXTime(_step_geom, _step_td)
                def _step_pde(x, y): return y[:, 0:1] * 0
                _step_data  = dde.data.TimePDE(_step_gt, _step_pde, [], num_domain=100, num_test=100)
                _step_net   = _apply_net_transforms(_make_net(_step_layers, _step_act, "Glorot uniform"))
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
                    if _is_3d:
                        _ea_yf = _ea_y_refs[_ei]; _ea_zf = _ea_z_refs[_ei]
                        _ea_xt = np.column_stack([_ea_xf, _ea_yf, _ea_zf, np.full_like(_ea_xf, _ea_tv)])
                    elif _is_2d:
                        _ea_yf = _ea_y_refs[_ei]
                        _ea_xt = np.column_stack([_ea_xf, _ea_yf, np.full_like(_ea_xf, _ea_tv)])
                    else:
                        _ea_xt = np.column_stack([_ea_xf, np.full_like(_ea_xf, _ea_tv)])
                    _ea_u_pinns[_ei] = _ea_extract(_ea_xt, _ea_sel, _step_model).flatten()
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

            _ea_metrics_path = _os.path.join(_ea_dir, f"error_metrics{{_ea_suffix}}.txt")
            with open(_ea_metrics_path, "w") as _emf:
                _emf.write("t,L2_relative,MSE,Max_error,Mean_abs_error\\n")
                for _ea_tv, _l2, _mse, _mx, _ma in _ea_metrics:
                    _emf.write(f"{{_ea_tv:.6f}},{{_l2:.6e}},{{_mse:.6e}},{{_mx:.6e}},{{_ma:.6e}}\\n")
            print(f"  Metrics saved: {{_ea_metrics_path}}")

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
                _ea_lp = _os.path.join(_ea_dir, f"line_comparison{{_ea_suffix}}.png")
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
                            _sn2 = _apply_net_transforms(_make_net(_sc2.get("layers",{config.layers}), _sc2.get("activation","{config.activation}"), "Glorot uniform"))
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
                                _u_pinn_grid = _ea_extract(_xyt_grid, _ea_sel, _sm2).reshape(_res_ea, _res_ea)
                            else:
                                _u_pinn_grid = np.zeros((_res_ea, _res_ea))
                        else:
                            _u_pinn_grid = np.zeros((_res_ea, _res_ea))
                        _u_fem_grid = _gd(np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei]]),
                                          _ea_u_refs[_ei], (_Xg_ea, _Yg_ea), method='linear', fill_value=0.0)
                        _u_err_grid = np.abs(_u_pinn_grid - _u_fem_grid)
                        _vmin_ea = min(_u_pinn_grid.min(), _u_fem_grid.min())
                        _vmax_ea = max(_u_pinn_grid.max(), _u_fem_grid.max())
                        if _vmax_ea - _vmin_ea < 1e-12:
                            _vmax_ea = _vmin_ea + 1e-12
                        _levels_ea2d = np.linspace(_vmin_ea, _vmax_ea, 41)  # contourf ignores vmin/vmax with an integer `levels=N` -- see the Standard-path comment on the same pattern above
                        im0 = axes[_ei][0].contourf(_Xg_ea, _Yg_ea, _u_pinn_grid, levels=_levels_ea2d, cmap='{config.plot_colormap}')
                        axes[_ei][0].set_title(f"PINN  t={{_ea_tv:.3f}}  L2={{_l2:.2e}}", fontsize=10)
                        axes[_ei][0].set_xlabel("x"); axes[_ei][0].set_ylabel("y")
                        fig.colorbar(im0, ax=axes[_ei][0])
                        im1 = axes[_ei][1].contourf(_Xg_ea, _Yg_ea, _u_fem_grid, levels=_levels_ea2d, cmap='{config.plot_colormap}')
                        axes[_ei][1].set_title(f"Ground Truth  t={{_ea_tv:.3f}}", fontsize=10)
                        axes[_ei][1].set_xlabel("x"); axes[_ei][1].set_ylabel("y")
                        fig.colorbar(im1, ax=axes[_ei][1])
                        im2 = axes[_ei][2].contourf(_Xg_ea, _Yg_ea, _u_err_grid, levels={config.plot_levels}, cmap='{config.plot_colormap}')
                        axes[_ei][2].set_title(f"|Error|  t={{_ea_tv:.3f}}  Max={{_mx:.2e}}", fontsize=10)
                        axes[_ei][2].set_xlabel("x"); axes[_ei][2].set_ylabel("y")
                        fig.colorbar(im2, ax=axes[_ei][2])
                    plt.tight_layout()
                elif _is_3d:
                    # 3D: same "boundary-point scatter" convention as the
                    # Standard path's non-box-geometry 3D surface comparison
                    # -- no flat-face grid to heatmap onto in general, so
                    # compare real (x, y, z) points that lie on the
                    # geometry's own boundary. Reuses the per-time PINN
                    # predictions already computed above (_ea_u_pinns) --
                    # each one came from that time's own correctly-restored
                    # step model -- instead of reloading a step model again
                    # here just to predict on a dense grid, as the 2D branch
                    # above does.
                    _ea_geom_ta = _build_geom()
                    _ea_bbox_ta = np.asarray(_ea_geom_ta.bbox)
                    _bx0_ta, _by0_ta, _bz0_ta = _ea_bbox_ta[0]
                    _bx1_ta, _by1_ta, _bz1_ta = _ea_bbox_ta[1]
                    _corners_ta = [(_bx0_ta,_by0_ta,_bz0_ta),(_bx1_ta,_by0_ta,_bz0_ta),(_bx1_ta,_by1_ta,_bz0_ta),(_bx0_ta,_by1_ta,_bz0_ta),
                                    (_bx0_ta,_by0_ta,_bz1_ta),(_bx1_ta,_by0_ta,_bz1_ta),(_bx1_ta,_by1_ta,_bz1_ta),(_bx0_ta,_by1_ta,_bz1_ta)]
                    _edges_ta = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                    fig = plt.figure(figsize=(15, 4.5 * _ea_n_t))
                    fig.suptitle("PINN vs Ground Truth — 3D Comparison", fontsize=13, fontweight='bold')
                    for _ei, _ea_tv in enumerate(_ea_times):
                        _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                        _bnd_ta = _ea_geom_ta.on_boundary(np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei], _ea_z_refs[_ei]]))
                        # A Cartesian reference grid only reliably intersects a
                        # FLAT boundary (a Cuboid's faces) in large numbers; a
                        # CURVED boundary (a Sphere) is essentially never
                        # landed on exactly by an axis-aligned grid, so
                        # on_boundary() can come back with only a handful of
                        # coincidental exact matches (e.g. a unit sphere's 6
                        # axis points) even when the dataset has thousands of
                        # valid points -- falling back whenever the boundary
                        # subset is a small fraction of the data (not just
                        # when it's nearly empty) catches that case too.
                        if _bnd_ta.sum() < 4 or _bnd_ta.sum() < 0.05 * len(_ea_x_refs[_ei]):
                            _bnd_ta = np.ones_like(_ea_x_refs[_ei], dtype=bool)
                        _bx_ta, _by_ta, _bz_ta = _ea_x_refs[_ei][_bnd_ta], _ea_y_refs[_ei][_bnd_ta], _ea_z_refs[_ei][_bnd_ta]
                        _gt_b_ta = _ea_u_refs[_ei][_bnd_ta]
                        _pinn_b_ta = _ea_u_pinns[_ei][_bnd_ta]
                        if len(_bx_ta) > 3000:
                            _ea_sub_ta = np.random.default_rng(0).choice(len(_bx_ta), size=3000, replace=False)
                            _bx_ta, _by_ta, _bz_ta = _bx_ta[_ea_sub_ta], _by_ta[_ea_sub_ta], _bz_ta[_ea_sub_ta]
                            _gt_b_ta, _pinn_b_ta = _gt_b_ta[_ea_sub_ta], _pinn_b_ta[_ea_sub_ta]
                        _err_b_ta = np.abs(_pinn_b_ta - _gt_b_ta)
                        _cols_ta = [(_pinn_b_ta, f"PINN  t={{_ea_tv:.3f}}  L2={{_l2:.2e}}"),
                                    (_gt_b_ta, f"Ground Truth  t={{_ea_tv:.3f}}"),
                                    (_err_b_ta, f"|Error|  Max={{_mx:.2e}}")]
                        for _ci_ta, (_vals_ta, _ttl_ta) in enumerate(_cols_ta):
                            _ax3_ta = fig.add_subplot(_ea_n_t, 3, _ei * 3 + _ci_ta + 1, projection='3d')
                            for _i3_ta, _j3_ta in _edges_ta:
                                _p0_ta, _p1_ta = _corners_ta[_i3_ta], _corners_ta[_j3_ta]
                                _ax3_ta.plot([_p0_ta[0], _p1_ta[0]], [_p0_ta[1], _p1_ta[1]], [_p0_ta[2], _p1_ta[2]],
                                             color='#888888', linewidth=0.8, alpha=0.6)
                            _sc3_ta = _ax3_ta.scatter(_bx_ta, _by_ta, _bz_ta, c=_vals_ta,
                                                       cmap='{config.plot_colormap}' if _ci_ta < 2 else 'inferno', s=14)
                            fig.colorbar(_sc3_ta, ax=_ax3_ta, shrink=0.6, pad=0.12)
                            _ax3_ta.set_title(_ttl_ta, fontsize=10)
                            _ax3_ta.set_xlabel("x"); _ax3_ta.set_ylabel("y"); _ax3_ta.set_zlabel("z")
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
                    if _ea_vmax - _ea_vmin < 1e-12:
                        _ea_vmax = _ea_vmin + 1e-12
                    _levels_ea1d = np.linspace(_ea_vmin, _ea_vmax, 41)  # contourf ignores vmin/vmax with an integer `levels=N` -- see the Standard-path comment on the same pattern above
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle("PINN vs Ground Truth — Surface Comparison", fontsize=13, fontweight='bold')
                    im0 = axes[0].contourf(_ea_Tg, _ea_Xg, _ea_U_pinn, levels=_levels_ea1d, cmap='{config.plot_colormap}')
                    axes[0].set_title("PINN  u(x,t)"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
                    fig.colorbar(im0, ax=axes[0])
                    im1 = axes[1].contourf(_ea_Tg, _ea_Xg, _ea_U_fem, levels=_levels_ea1d, cmap='{config.plot_colormap}')
                    axes[1].set_title("Ground Truth  u(x,t)"); axes[1].set_xlabel("t"); axes[1].set_ylabel("x")
                    fig.colorbar(im1, ax=axes[1])
                    im2 = axes[2].contourf(_ea_Tg, _ea_Xg, _ea_U_err, levels={config.plot_levels}, cmap='{config.plot_colormap}')
                    axes[2].set_title("Error  |PINN - Ground Truth|"); axes[2].set_xlabel("t"); axes[2].set_ylabel("x")
                    fig.colorbar(im2, ax=axes[2])
                    plt.tight_layout()
                _ea_sp = _os.path.join(_ea_dir, f"surface_comparison{{_ea_suffix}}.png")
                plt.savefig(_ea_sp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                print(f"  Surface comparison saved: {{_ea_sp}}")

            print(f"  Group '{{_ea_group_label(_ea_sel) or 'default'}}' analysis complete")
        print("=== Time-Adaptive Error Analysis Complete ===")

{_tm_plot_code}
print("DONE")
"""
    return script


# =====================================================================
# generate_clean_script -- a SECOND, independent code generator.
#
# generate_script() above is what Solve actually runs: deliberately
# generic/defensive, built to work for any possible GUI configuration via
# runtime "if" branches over every option this app exposes, with app
# bookkeeping (training_log.txt, model_config.json, per-phase JSON dumps)
# for the Restore & Visualization panel. It must stay exactly as-is.
#
# This function is for "Export as DeepXDE Script" instead: a short,
# tutorial-style, plain DeepXDE/PyTorch + matplotlib script containing
# ONLY the pieces actually configured for THIS problem -- an unticked
# feature (RAR, IC pre-training, Training Callbacks, weight decay,
# input/output transforms, Error Analysis, Inverse) is left out of the
# file entirely, not hidden behind a False branch. Everything that is
# fully known at export time (geometry type, which PDE derivative terms
# are referenced, the BC/IC list, loss weights, the training-phase list)
# is resolved right here in Python and written into the script as plain,
# literal DeepXDE calls instead of runtime dispatch code, so the result
# reads like a hand-written example: something to read, run, and edit.
# =====================================================================

import ast as _clean_ast
import json as _clean_json


def _clean_needed_derivs(oname, oi, n_out, pde_check, is_2d, is_3d, is_steady=False):
    """Host-time equivalent of _pde_standard's runtime substring checks
    above -- returns an ordered list of (varname, rhs_code) for exactly
    the derivative terms this output's PDE expression(s) actually
    reference, resolved now instead of re-checked every time the
    generated script's pde() function runs."""

    def has(suffix):
        return f"d{oname}_{suffix}" in pde_check

    def hess(i, j):
        if n_out == 1:
            return f"dde.grad.hessian(y, x, i={i}, j={j})"
        return f"dde.grad.hessian(y, x, component={oi}, i={i}, j={j})"

    out = []
    if is_3d:
        out.append((f"d{oname}_x", f"dde.grad.jacobian(y, x, i={oi}, j=0)"))
        out.append((f"d{oname}_y", f"dde.grad.jacobian(y, x, i={oi}, j=1)"))
        out.append((f"d{oname}_z", f"dde.grad.jacobian(y, x, i={oi}, j=2)"))
        if not is_steady:
            out.append((f"d{oname}_t", f"dde.grad.jacobian(y, x, i={oi}, j=3)"))
        second = [("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("tt", 3, 3),
                  ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2),
                  ("xt", 0, 3), ("yt", 1, 3), ("zt", 2, 3)]
        have2 = set()
        for nm, i, j in second:
            if has(nm):
                out.append((f"d{oname}_{nm}", hess(i, j)))
                have2.add(nm)
        fourth = [("xxxx", "xx", 0, 0), ("yyyy", "yy", 1, 1), ("zzzz", "zz", 2, 2),
                  ("xxyy", "xx", 1, 1), ("xxzz", "xx", 2, 2), ("yyzz", "yy", 2, 2),
                  ("xxtt", "xx", 3, 3), ("yytt", "yy", 3, 3), ("zztt", "zz", 3, 3)]
        for nm, base, i, j in fourth:
            if has(nm) and base in have2:
                out.append((f"d{oname}_{nm}", f"dde.grad.hessian(d{oname}_{base}, x, i={i}, j={j})"))
    elif is_2d:
        out.append((f"d{oname}_x", f"dde.grad.jacobian(y, x, i={oi}, j=0)"))
        out.append((f"d{oname}_y", f"dde.grad.jacobian(y, x, i={oi}, j=1)"))
        if not is_steady:
            out.append((f"d{oname}_t", f"dde.grad.jacobian(y, x, i={oi}, j=2)"))
        need_xx = has("xx") or has("xxxx") or has("xxyy") or has("xxtt")
        need_yy = has("yy") or has("yyyy") or has("xxyy") or has("yytt")
        if need_xx:
            out.append((f"d{oname}_xx", hess(0, 0)))
        if need_yy:
            out.append((f"d{oname}_yy", hess(1, 1)))
        if has("xy"):
            out.append((f"d{oname}_xy", hess(0, 1)))
        if has("tt"):
            out.append((f"d{oname}_tt", hess(2, 2)))
        if has("xt"):
            out.append((f"d{oname}_xt", hess(0, 2)))
        if has("yt"):
            out.append((f"d{oname}_yt", hess(1, 2)))
        if has("xxxx") and need_xx:
            out.append((f"d{oname}_xxxx", f"dde.grad.hessian(d{oname}_xx, x, i=0, j=0)"))
        if has("yyyy") and need_yy:
            out.append((f"d{oname}_yyyy", f"dde.grad.hessian(d{oname}_yy, x, i=1, j=1)"))
        if has("xxyy") and need_xx:
            out.append((f"d{oname}_xxyy", f"dde.grad.hessian(d{oname}_xx, x, i=1, j=1)"))
        if has("xxtt") and need_xx:
            out.append((f"d{oname}_xxtt", f"dde.grad.hessian(d{oname}_xx, x, i=2, j=2)"))
        if has("yytt") and need_yy:
            out.append((f"d{oname}_yytt", f"dde.grad.hessian(d{oname}_yy, x, i=2, j=2)"))
    else:
        out.append((f"d{oname}_x", f"dde.grad.jacobian(y, x, i={oi}, j=0)"))
        if not is_steady:
            out.append((f"d{oname}_t", f"dde.grad.jacobian(y, x, i={oi}, j=1)"))
        need_xx = has("xx") or has("xxx") or has("xxxx") or has("xxtt")
        need_tt = has("tt") or has("tttt") or has("xxtt")
        if need_xx:
            out.append((f"d{oname}_xx", hess(0, 0)))
        if need_tt:
            out.append((f"d{oname}_tt", hess(1, 1)))
        if has("xt"):
            out.append((f"d{oname}_xt", hess(0, 1)))
        # Third-order pure x-derivative (KdV's dispersive u_xxx term): one
        # more jacobian pass on top of u_xx, not a hessian (that would give
        # the 4th-order xxxx term below instead).
        if has("xxx") and need_xx:
            out.append((f"d{oname}_xxx", f"dde.grad.jacobian(d{oname}_xx, x, i=0, j=0)"))
        if has("xxxx") and need_xx:
            out.append((f"d{oname}_xxxx", f"dde.grad.hessian(d{oname}_xx, x, i=0, j=0)"))
        if has("xxtt") and need_xx:
            out.append((f"d{oname}_xxtt", f"dde.grad.hessian(d{oname}_xx, x, i=1, j=1)"))
        if has("tttt") and need_tt:
            out.append((f"d{oname}_tttt", f"dde.grad.hessian(d{oname}_tt, x, i=1, j=1)"))
    return out


def _clean_geom_line(config, is_2d, is_3d, tri_verts, poly_verts):
    """One literal geom = dde.geometry.XXX(...) line -- geometry type is
    fixed per-problem, so no runtime dispatch is needed."""
    gt = config.geometry_type or "Rectangle"
    wrap_needed = gt in ("Disk", "Ellipse", "Triangle", "Polygon", "Sphere")
    if gt == "Custom":
        inner, wrap_needed = _build_custom_geom_code(config)
    elif gt == "Disk":
        inner = f"dde.geometry.Disk([{config.geom_center_x}, {config.geom_center_y}], {config.geom_radius})"
    elif gt == "Ellipse":
        inner = (f"dde.geometry.Ellipse([{config.geom_center_x}, {config.geom_center_y}], "
                  f"{config.geom_semi_major}, {config.geom_semi_minor}, {config.geom_angle})")
    elif gt == "Triangle":
        inner = f"dde.geometry.Triangle({tri_verts[0]}, {tri_verts[1]}, {tri_verts[2]})"
    elif gt == "Polygon":
        inner = f"dde.geometry.Polygon({poly_verts})"
    elif gt == "Sphere":
        inner = (f"dde.geometry.Sphere([{config.geom_center_x}, {config.geom_center_y}, "
                  f"{config.geom_center_z}], {config.geom_radius})")
    elif is_2d:
        inner = f"dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])"
    elif is_3d:
        inner = (f"dde.geometry.Cuboid([{config.x_min}, {config.y_min}, {config.z_min}], "
                  f"[{config.x_max}, {config.y_max}, {config.z_max}])")
    else:
        inner = f"dde.geometry.Interval({config.x_min}, {config.x_max})"
    if wrap_needed:
        return f"geom = _DTypeSafeGeom({inner})", True
    return f"geom = {inner}", False


def _clean_coord_unpack(indent, is_batch, is_2d, is_3d):
    """`x = X[:, 0]` (batch) or `x = X[0]` (single point), plus y/z when
    the problem has them -- shared by every BC location/value helper
    below so an expression like "x" or "np.isclose(x, 0)" the user typed
    means exactly what it looks like."""
    lines = []
    idx = ":, 0" if is_batch else "0"
    lines.append(f"{indent}x = X[{idx}]")
    if is_2d or is_3d:
        idx = ":, 1" if is_batch else "1"
        lines.append(f"{indent}y = X[{idx}]")
    if is_3d:
        idx = ":, 2" if is_batch else "2"
        lines.append(f"{indent}z = X[{idx}]")
    return lines


def _clean_bc_row_code(i, entry, is_2d, is_3d, n_coord_cols):
    """Everything needed for one Boundary Conditions panel row: the small
    helper function(s) its location/value expression needs (the
    expression text itself is spliced straight into the function body --
    no eval() indirection, since the whole point is a script someone can
    read and edit) plus the one dde.icbc.XxxBC(...) call that adds it to
    `constraints`. Returns (def_lines, append_lines)."""
    btype = entry.get('type', 'dirichlet')
    comp = int(entry.get('component', 0) or 0)
    loc = (entry.get('location', '') or '').strip() or 'True'
    val = (entry.get('value', '') or '').strip() or '0'
    axis = entry.get('axis', 'x')
    deriv = int(entry.get('deriv_order', 0) or 0)
    pts_file = entry.get('points_file', '')
    loc2 = (entry.get('location2', '') or '').strip() or 'True'
    direction = entry.get('direction', 'normal')
    axis_idx_map = {"x": 0, "y": 1, "z": 2}

    loc_name = f"_bc{i}_loc"
    loc2_name = f"_bc{i}_loc2"
    val_name = f"_bc{i}_val"
    defs, append = [], []

    def loc_fn(name, expr):
        L = [f"def {name}(X, on_boundary):", "    if not on_boundary:", "        return False"]
        L += _clean_coord_unpack("    ", False, is_2d, is_3d)
        L.append(f"    return bool({expr})")
        return L

    def val_fn(name, expr):
        L = [f"def {name}(X):"]
        L += _clean_coord_unpack("    ", True, is_2d, is_3d)
        L.append(f"    _r = np.asarray({expr}, dtype=float)")
        L.append("    return np.full((len(X), 1), float(_r)) if _r.ndim == 0 else _r.reshape(-1, 1)")
        return L

    if btype == 'neumann':
        defs += loc_fn(loc_name, loc) + val_fn(val_name, val)
        append.append(f"constraints.append(dde.icbc.NeumannBC(geomtime, {val_name}, {loc_name}, component={comp}))")
    elif btype == 'robin':
        rob_name = f"_bc{i}_robin"
        L = [f"def {rob_name}(X, Y):"]
        L += _clean_coord_unpack("    ", True, is_2d, is_3d)
        L.append(f"    u = Y[:, {comp}:{comp}+1]")
        L.append(f"    _r = {val}")
        L.append("    if isinstance(_r, torch.Tensor):")
        L.append("        return _r if _r.dim() == 2 else _r.reshape(-1, 1)")
        L.append("    _arr = np.asarray(_r, dtype=float)")
        L.append("    if _arr.ndim == 0:")
        L.append("        return Y.new_full((len(X), 1), float(_arr))")
        L.append("    return Y.new_tensor(_arr.reshape(-1, 1))")
        defs += L + loc_fn(loc_name, loc)
        append.append(f"constraints.append(dde.icbc.RobinBC(geomtime, {rob_name}, {loc_name}, component={comp}))")
    elif btype == 'periodic':
        defs += loc_fn(loc_name, loc)
        append.append(f"constraints.append(dde.icbc.PeriodicBC(geomtime, {axis_idx_map.get(axis, 0)}, "
                       f"{loc_name}, derivative_order={deriv}, component={comp}))")
    elif btype == 'pointset':
        append.append(f'_pts{i}, _pvals{i} = _load_bc_points({pts_file!r}, {n_coord_cols})')
        append.append(f"constraints.append(dde.icbc.PointSetBC(_pts{i}, _pvals{i}, component={comp}))")
    elif btype == 'pointset_operator':
        op_name = f"_bc{i}_op"
        defs += [f"def {op_name}(inputs, outputs, X):", f"    return {val}"]
        append.append(f'_pts{i}, _pvals{i} = _load_bc_points({pts_file!r}, {n_coord_cols})')
        append.append(f"constraints.append(dde.icbc.PointSetOperatorBC(_pts{i}, _pvals{i}, {op_name}))")
    elif btype == 'operator':
        op_name = f"_bc{i}_op"
        defs += [f"def {op_name}(inputs, outputs, X):", f"    return {val}"] + loc_fn(loc_name, loc)
        append.append(f"constraints.append(dde.icbc.OperatorBC(geomtime, {op_name}, {loc_name}))")
    elif btype == 'interface2d':
        defs += loc_fn(loc_name, loc) + loc_fn(loc2_name, loc2) + val_fn(val_name, val)
        append.append(f"constraints.append(dde.icbc.Interface2DBC(_Interface2DGeomAdapter(geomtime), {val_name}, "
                       f"{loc_name}, {loc2_name}, direction={direction!r}))")
    else:  # 'dirichlet' and any unrecognized type fall back to Dirichlet
        defs += loc_fn(loc_name, loc) + val_fn(val_name, val)
        append.append(f"constraints.append(dde.icbc.DirichletBC(geomtime, {val_name}, {loc_name}, component={comp}))")

    return defs, append


def _clean_active_ic_outputs(config, n_out, ic_active_list):
    """Which output indices actually get an Initial Condition, in order --
    used both to build the IC constraints and to know how many IC weight
    slots the loss-weight list needs."""
    active = []
    for oi in range(n_out):
        if oi == 0 and config.forward_ic_from_file:
            active.append(oi)
        elif oi < len(ic_active_list) and ic_active_list[oi].strip() == "True":
            active.append(oi)
    return active


def _clean_loss_weights(config, n_out, bc_entries, ic_active_list, obs_weights, is_steady=False):
    """Fully resolves the loss-weight list at export time -- same slot
    ordering as the running app's runtime reconstruction (PDE per output,
    then one per Boundary Conditions panel row, then one per active IC,
    then one per Inverse observation file), but computed once now instead
    of on every run. Returns (weights, expected_len) -- expected_len is
    how many terms a Training Phase's own weight string must have to be
    used as-is (see _clean_phase_weights below). A steady-state problem
    has no Initial Condition at all, so it never gets an IC weight slot."""
    wm_list = [float(v) for v in (config.loss_weights_multi or "").split(",") if v.strip()]
    wi = 0
    pde_w = []
    for _ in range(n_out):
        pde_w.append(wm_list[wi] if wi < len(wm_list) else 1.0)
        wi += 1
    bc_w = []
    for _ in range(len(bc_entries)):
        bc_w.append(wm_list[wi] if wi < len(wm_list) else 1.0)
        wi += 1
    ic_w = []
    if not is_steady:
        for oi in range(n_out):
            active = (oi == 0 and config.forward_ic_from_file) or (oi < len(ic_active_list) and ic_active_list[oi].strip() == "True")
            if active:
                ic_w.append(wm_list[wi] if wi < len(wm_list) else 1.0)
                wi += 1
            elif wi < len(wm_list):
                wi += 1
    weights = pde_w + bc_w + ic_w + list(obs_weights)
    expected_len = n_out + len(bc_entries) + len(ic_w) + len(obs_weights)
    return weights, expected_len


def _clean_phase_weights(phase, expected_len, fallback):
    try:
        w = [float(x) for x in str(phase.get('weights', '')).split(',') if x.strip()]
    except (TypeError, ValueError):
        w = []
    if len(w) != expected_len:
        w = list(fallback)
    return w


def _clean_sched_phases(config, fallback_weights):
    """The literal, ordered list of training phases this run actually
    uses -- straight from the Training Phases scheduler when it's set up
    (the normal case for anything built in the GUI), or synthesized from
    the older single/second-optimizer fields for a config saved before
    the scheduler existed."""
    try:
        phases = _clean_json.loads(config.scheduler_phases) if config.scheduler_phases else []
    except (ValueError, TypeError):
        phases = []
    if not phases:
        w_str = ",".join(str(w) for w in fallback_weights)
        phases = [{"optimizer": config.optimizer, "iterations": config.iterations,
                   "lr": config.learning_rate, "loss": config.loss_type, "weights": w_str}]
        if config.optimizer2 and config.optimizer2 != "none":
            phases.append({"optimizer": config.optimizer2, "iterations": config.iterations2,
                            "lr": config.learning_rate, "loss": config.loss_type, "weights": w_str})
    return phases


def _clean_phase_train_lines(sp, w, ext_vars_kw, cbs_kw, model_name, data_name, config, indent=""):
    """The literal model.compile()/model.train() call(s) for exactly one
    training phase -- the "literal sequential calls" style: read top to
    bottom as exactly what happens, no phases-list loop at runtime."""
    opt = sp.get('optimizer', 'adam')
    iters = int(sp.get('iterations', 1000) or 0)
    loss = sp.get('loss') or config.loss_type
    L = []
    if opt == 'lbfgs':
        L.append(f"dde.optimizers.set_LBFGS_options(maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol}, "
                  f"gtol={config.lbfgs_gtol}, maxiter={iters}, maxfun={int(iters * 1.25)}, maxls={config.lbfgs_maxls})")
        L.append(f"{model_name}.compile(\"L-BFGS\", loss={loss!r}, loss_weights={w}{ext_vars_kw})")
        L.append(f"loss_history, train_state = {model_name}.train(display_every={config.loss_display_every}{cbs_kw})")
    elif opt == 'nncg':
        L.append(f"{model_name}.compile(\"NNCG\", loss={loss!r}, loss_weights={w}{ext_vars_kw})")
        L.append(f"loss_history, train_state = {model_name}.train(iterations={iters}, display_every={config.loss_display_every}{cbs_kw})")
    else:
        decay = None
        dtype = sp.get('decay_type', 'none')
        if dtype and dtype != 'none':
            p1 = sp.get('decay_p1', 0) or 0
            p2 = sp.get('decay_p2', 0) or 0
            if dtype == 'step':
                decay = ('step', int(p1), float(p2))
            elif dtype == 'cosine':
                decay = ('cosine', int(p1), float(p2))
            elif dtype == 'exponential':
                decay = ('exponential', float(p1))
        lr = sp.get('lr', config.learning_rate)
        L.append(f"{model_name}.compile({opt!r}, lr={lr}, decay={decay!r}, loss={loss!r}, "
                  f"loss_weights={w}{ext_vars_kw})")
        L.append(f"loss_history, train_state = {model_name}.train(iterations={iters}, display_every={config.loss_display_every}{cbs_kw})")
    return [f"{indent}{ln}" for ln in L]


def _clean_loss_series_lines(config, train_rows_expr, test_rows_expr, steps_expr):
    """Builds the plain `plt.<semilogy|plot>(...)` lines for one "Export as
    DeepXDE Script" loss plot, honoring the Round 28 Loss Plot Settings
    (config.loss_plot_mode/loss_plot_log_y/loss_plot_linewidth) -- the same
    settings generate_script()'s _make_loss_plot applies, but resolved HERE,
    directly into fixed plt calls at codegen time, since generate_clean_script
    deliberately bakes every option into plain literal calls rather than a
    shared runtime helper/dispatch (see its own docstring and the
    plot_title_override precedent above). The surrounding xlabel/title/
    legend/savefig lines at each call site are unaffected and stay as they
    were; only the data-series lines themselves are replaced by this.

    Mirrors _loss_plot_runtime_code's own term-labeling rule: only the
    leading len(output_names) PDE terms get a real name ("PDE (u)"), every
    later term (BC/IC, in whatever order they were added) is labeled
    generically as "Constraint N" -- DeepXDE never tracks per-term Test
    loss, so "individual"/"all" only ever add TRAIN per-term lines.

    Keeps the pre-existing `train_loss = [sum(l) for l in <rows>]` /
    `test_loss = [...]` intermediate-variable assignment (rather than
    summing inline in the plt call) so the default "train_test" mode's
    output is byte-for-byte identical to the pre-Round-28 code -- a
    verified no-op for any config that doesn't touch the new setting,
    and what tests/test_save_and_export_fixes.py's own literal
    "plt.semilogy(..., train_loss" substring check still expects."""
    mode = config.loss_plot_mode
    plotfn = "semilogy" if config.loss_plot_log_y else "plot"
    lw = config.loss_plot_linewidth
    lines = [
        f"train_loss = [sum(l) for l in {train_rows_expr}]",
        f"test_loss = [sum(l) for l in {test_rows_expr}]",
    ]
    if mode in ("train_test", "all"):
        lines.append(f'plt.{plotfn}({steps_expr}, train_loss, label="Train loss", color="#4dabf7", linewidth={lw})')
        lines.append(f'plt.{plotfn}({steps_expr}, test_loss, label="Test loss", color="#ff8787", linestyle="--", linewidth={lw})')
    if mode in ("individual", "all"):
        pde_names = [n.strip() or f"u{i+1}" for i, n in enumerate((config.output_names or "u").split(","))]
        lines.append(f"_loss_pde_names = {pde_names!r}")
        lines.append(f"_loss_term_rows = {train_rows_expr}")
        lines.append("_loss_n_terms = len(_loss_term_rows[0]) if _loss_term_rows else 0")
        lines.append('_loss_term_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(_loss_n_terms, 1)))')
        lines.append("for _li in range(_loss_n_terms):")
        lines.append('    _loss_lbl = f"PDE ({_loss_pde_names[_li]})" if _li < len(_loss_pde_names) else f"Constraint {_li - len(_loss_pde_names) + 1}"')
        lines.append(f'    plt.{plotfn}({steps_expr}, [r[_li] if _li < len(r) else float("nan") for r in _loss_term_rows], '
                      f'label=_loss_lbl, color=_loss_term_colors[_li], linewidth=max({lw} - 0.5, 0.5), alpha=0.85)')
    return lines


def generate_clean_script(config):
    """Builds the "Export as DeepXDE Script" file: a short, plain,
    tutorial-style DeepXDE/PyTorch + matplotlib script for exactly this
    problem -- only the features actually configured (weight decay, RAR,
    Training Callbacks, IC Pre-Training, Inverse variables/observations,
    input/output transforms, Time-Adaptive stepping, Error Analysis) are
    written in at all, each as plain DeepXDE calls rather than runtime
    "if" dispatch over every option this app exposes. See generate_script()
    above for the generator Solve itself actually runs -- that one stays
    untouched; this is a second, independent generator."""
    is_2d = config.problem_dim == "2D"
    is_3d = config.problem_dim == "3D"
    # Steady-state (time-independent) problem -- see generate_script()'s
    # _is_steady for the full rationale; this generator mirrors the same
    # dispatch (plain dde.data.PDE, no GeometryXTime/Initial Condition/
    # time-derivative terms) for the "Export as DeepXDE Script" output.
    is_steady = bool(config.steady_state)
    problem_type = config.problem_type
    n_out = config.num_outputs
    out_names = [n.strip() for n in config.output_names.split(",")]
    while len(out_names) < n_out:
        out_names.append(f"u{len(out_names)}")
    n_coord = 3 if is_3d else (2 if is_2d else 1)      # spatial dims
    n_coord_cols = n_coord + 1                          # + time, for points files
    sol_ext = "gif" if (not is_steady and config.plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)")) else "png"
    is_inverse = problem_type == "Inverse"

    # Same y/z slice-value precomputation as generate_script() above -- see
    # the matching comment there. Every line-type plot in this generator
    # reads it instead of hardcoding the domain midpoint independently.
    line_slice_y = (config.y_min + config.y_max) / 2.0 if config.line_slice_y_auto else config.line_slice_y
    line_slice_z = (config.z_min + config.z_max) / 2.0 if config.line_slice_z_auto else config.line_slice_z

    # Optional title/axis-label overrides -- same blank-means-default
    # convention as generate_script() and the Restore & Visualize script
    # builders. Unlike generate_script()'s runtime `_plot_title_override
    # or <default>` (where out_name is only known at the generated
    # script's own runtime), out_name below is a plain Python variable
    # already known here at codegen time -- same as the restore builders
    # -- so the override is resolved immediately via `title_override or
    # <default>` in this function's own Python, baking in the final
    # fixed string.
    plot_title_override = (getattr(config, "plot_title_override", "") or "").strip()
    plot_xlabel_override = (getattr(config, "plot_xlabel_override", "") or "").strip()
    plot_ylabel_override = (getattr(config, "plot_ylabel_override", "") or "").strip()

    # ---- PDE expressions (host-resolved derivative terms) --------------
    pde_exprs = [_simplify_pde_expr(e) for e in config.pde_expressions.split("|")]
    pde_check_str = "|".join(pde_exprs)
    pde_body = []
    for oi, oname in enumerate(out_names[:n_out]):
        pde_body.append(f"    {oname} = y[:, {oi}:{oi + 1}]")
        for varname, rhs in _clean_needed_derivs(oname, oi, n_out, pde_check_str, is_2d, is_3d, is_steady):
            pde_body.append(f"    {varname} = {rhs}")
    if n_out == 1:
        pde_body.append(f"    return {pde_exprs[0].strip()}")
    else:
        exprs = [e.strip() for e in pde_exprs[:n_out]]
        pde_body.append("    return [")
        for e in exprs:
            pde_body.append(f"        {e},")
        pde_body.append("    ]")
    pde_block = "def pde(x, y):\n" + "\n".join(pde_body)

    # ---- Geometry --------------------------------------------------------
    tri_verts = _parse_vertex_list(config.geom_triangle_vertices)
    if len(tri_verts) != 3:
        tri_verts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    poly_verts = _parse_vertex_list(config.geom_polygon_vertices)
    if len(poly_verts) < 3:
        poly_verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    geom_line, needs_dtype_wrap = _clean_geom_line(config, is_2d, is_3d, tri_verts, poly_verts)

    # ---- Boundary conditions (custom_bc_json is the live source) --------
    try:
        bc_entries = _clean_json.loads(config.custom_bc_json) if config.custom_bc_json else []
    except (ValueError, TypeError):
        bc_entries = []
    bc_used_types = {e.get('type', 'dirichlet') for e in bc_entries}
    needs_points_loader = bool({'pointset', 'pointset_operator'} & bc_used_types)
    needs_interface_adapter = 'interface2d' in bc_used_types

    bc_defs, bc_appends = [], []
    for i, entry in enumerate(bc_entries):
        d, a = _clean_bc_row_code(i, entry, is_2d, is_3d, n_coord_cols)
        bc_defs += d
        bc_appends += a

    # ---- Initial condition(s) --------------------------------------------
    ic_exprs = [_simplify_expr(e, is_2d, is_3d) for e in config.ic_expressions.split("|")]
    ic_active_list = config.ic_active.split(",")
    active_ic_outputs = _clean_active_ic_outputs(config, n_out, ic_active_list)
    needs_ic_file_loader = bool(config.forward_ic_from_file)

    ic_defs, ic_appends = [], []
    for oi in (range(n_out) if not is_steady else []):
        if oi == 0 and config.forward_ic_from_file:
            ic_appends.append(f'_ic_xt, _ic_vals = _load_ic_from_file({repr(config.forward_ic_file)})')
            ic_appends.append("constraints.append(dde.icbc.PointSetBC(_ic_xt, _ic_vals, component=0))")
        elif oi in active_ic_outputs:
            expr = ic_exprs[oi].strip() if oi < len(ic_exprs) else "np.zeros_like(x[:, 0])"
            ic_defs.append(f"def _ic{oi}_fn(x):")
            ic_defs.append(f"    return np.reshape({expr}, (-1, 1))")
            ic_appends.append(f"constraints.append(dde.icbc.IC(geomtime, _ic{oi}_fn, "
                               f"lambda x, on_initial: on_initial, component={oi}))")

    # ---- Inverse: trainable variables + observation data -----------------
    inv_vars_parsed = _parse_inverse_variables(config) if is_inverse else []
    obs_files_parsed = _parse_inverse_obs_files(config) if is_inverse else []
    obs_weights = [e['weight'] for e in obs_files_parsed] if is_inverse else []

    # ---- Loss weights (fully resolved now, not reconstructed at runtime) -
    multi_weights, expected_len = _clean_loss_weights(config, n_out, bc_entries, ic_active_list, obs_weights, is_steady)

    # ---- Training-phase list, resolved to literal per-phase weights ------
    sched_phases = _clean_sched_phases(config, multi_weights)
    for sp in sched_phases:
        sp['_w'] = _clean_phase_weights(sp, expected_len, multi_weights)
    uses_nncg = any(sp.get('optimizer') == 'nncg' for sp in sched_phases)
    total_iters = sum(int(sp.get('iterations', 0) or 0) for sp in sched_phases)

    # ---- Training callbacks (opt-in only) --------------------------------
    any_cbs = bool(config.cb_early_stopping or config.cb_point_resampler
                   or config.cb_model_checkpoint or config.cb_timer
                   or config.training_monitors_enabled)
    cbs_code = ""
    if any_cbs:
        _tm_clean_cbs_code = _build_training_monitors_code(config, "train_cbs", "(save_dir or '.')", indent=0)
        cbs_code = (
            _build_train_cbs_code(config, "train_cbs", indent=0)
            + ("\n" + _tm_clean_cbs_code if _tm_clean_cbs_code else "")
        ).replace("_os.path.join", "os.path.join")

    var_cb_period = max(1, total_iters // 200) if total_iters else 100
    callbacks_parts = []
    if is_inverse:
        callbacks_parts.append("[var_cb]")
    if any_cbs:
        callbacks_parts.append("train_cbs")
    if callbacks_parts:
        cbs_kw = ", callbacks=" + " + ".join(callbacks_parts)
    else:
        cbs_kw = ""
    ext_vars_kw = ", external_trainable_variables=inv_vars" if is_inverse else ""

    # ---- Network architecture: FNN or DeepXDE's built-in PFNN class -------
    # Same shared helper as generate_script() (see
    # _net_construction_helper_code()'s docstring) -- _make_net(...)
    # replaces every bare dde.nn.FNN(...) call below, including the
    # weight-decay arg this used to splice in directly (now passed through
    # as a plain float; _make_net silently drops it for PFNN).
    net_helper_code = _net_construction_helper_code(config.network_type)

    # ---- Optional input/output transforms --------------------------------
    has_in_transform = bool(config.input_transform_enabled and config.input_transform_scale)
    has_out_transform = bool(config.output_transform_enabled and config.output_transform_scale)
    needs_transform_helper = has_in_transform or has_out_transform

    # ---- RAR / IC Pre-Training / Time-Adaptive flags ----------------------
    use_rar = (config.adapt_method == "RAR") and not config.time_adaptive
    use_ic_pretrain = bool(config.ic_pretrain)
    use_ta = bool(config.time_adaptive)

    # ---- Error Analysis ----------------------------------------------------
    try:
        ea_files = _clean_ast.literal_eval(config.ea_files) if config.ea_files else []
    except (ValueError, SyntaxError):
        ea_files = []
    # ea_files entries are (time, path, output_selector) 3-tuples since the
    # v40 multi-output Error Analysis feature (see the matching
    # normalization in generate_script()'s two Error Analysis paths above);
    # older saved configs only ever wrote 2-tuples (no selector), padded
    # with None here the same way, so they keep behaving exactly as before.
    # Unpacking as a plain 2-tuple here (as before v40) raised "too many
    # values to unpack" for any template with reference data the moment
    # "Export as DeepXDE Script" was used. This generator still doesn't
    # have per-output grouping/routing (a separate, larger, still-open
    # gap) -- it always predicts a single field, below -- but that one
    # field can now be either a raw column OR a custom derived expression
    # (see _extract_plot_field, added below), matching what the Results
    # panel/main Solve path already show. A file is kept here only if it
    # belongs to that same field: selector is None (the implicit-default
    # case -- always follows whichever field this script is actually
    # plotting), or an int matching plot_output_idx when no custom field
    # is configured, or a [expr, label] pair whose expression matches this
    # problem's own custom plot expression when one is. A named file for a
    # genuinely different output/expression is correctly dropped rather
    # than silently compared against the wrong prediction.
    _plot_custom_expr_val = (config.plot_custom_expr or "").strip()

    def _ea_sel_matches(_sel):
        if _sel is None:
            return True
        if _plot_custom_expr_val:
            return (isinstance(_sel, (list, tuple)) and len(_sel) >= 1
                    and str(_sel[0]).strip() == _plot_custom_expr_val)
        return isinstance(_sel, int) and _sel == config.plot_output_idx

    _ea_files_norm = []
    for _entry in ea_files:
        if len(_entry) >= 3:
            _tv0, _fp0, _sel0 = _entry[0], _entry[1], _entry[2]
        else:
            _tv0, _fp0, _sel0 = _entry[0], _entry[1], None
        _ea_files_norm.append((_tv0, _fp0, _sel0))
    ea_files = [
        (tv, fp) for tv, fp, sel in _ea_files_norm
        if _ea_sel_matches(sel)
        and config.t_min - 1e-10 <= tv <= config.t_max + 1e-10
    ]
    use_ea = bool(ea_files) and (config.ea_do_line or config.ea_do_surface)

    use_save = bool((config.save_dir or "").strip())

    # =====================================================================
    # Assemble the script
    # =====================================================================
    parts = []

    dim_label = "3D" if is_3d else ("2D" if is_2d else "1D")
    title = f"{dim_label} {problem_type} PINN" + (" (Time-Adaptive)" if use_ta else "")
    parts.append(f'''"""
{title}, exported from PINNStudio.

Outputs: {", ".join(out_names[:n_out])}
Domain:  x in [{config.x_min}, {config.x_max}]''' +
                 (f", y in [{config.y_min}, {config.y_max}]" if is_2d or is_3d else "") +
                 (f", z in [{config.z_min}, {config.z_max}]" if is_3d else "") +
                 ("" if is_steady else f'''
Time:    t in [{config.t_min}, {config.t_max}]''') +
                 f'''

Edit anything below and re-run -- this is a plain script, not tied to
the PINNStudio GUI.
"""
import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt''')

    if uses_nncg:
        parts.append('''_dxde_ver_parts = dde.__version__.split(".")[:3]
try:
    _dxde_ver = tuple(int(p) for p in _dxde_ver_parts)
except ValueError:
    _dxde_ver = None
if _dxde_ver is not None and _dxde_ver < (1, 13, 0):
    raise SystemExit(
        f"The NNCG optimizer requires deepxde>=1.13.0 (you have {dde.__version__}). "
        f"Please upgrade: pip install --upgrade deepxde"
    )''')

    parts.append(f'dde.config.set_default_float("{config.float_type}")')

    if config.use_random_seed:
        parts.append(f'dde.config.set_random_seed({config.random_seed})')

    save_dir_repr = repr(config.save_dir) if use_save else repr("")
    parts.append(f'''save_dir = {save_dir_repr}
if save_dir:
    os.makedirs(save_dir, exist_ok=True)''')

    if needs_ic_file_loader:
        parts.append(f'''def _load_ic_from_file(path):
    """Loads an Initial Condition from a plain, header-less file: one
    coordinate column per spatial axis, then time, then value -- e.g.
    "x, t, u" in 1D. Only the rows already at t = {config.t_min} (the
    domain's start time) are used."""
    raw = np.loadtxt(path)
    n_coord = {n_coord}
    t_col, v_col = n_coord, n_coord + 1
    mask = np.abs(raw[:, t_col] - {config.t_min}) < 1e-8
    coords = raw[mask, :n_coord]
    t0 = np.full((int(mask.sum()), 1), {config.t_min})
    return np.hstack([coords, t0]), raw[mask, v_col:v_col + 1]''')

    if needs_points_loader:
        parts.append(f'''def _load_bc_points(path, n_coord_cols={n_coord_cols}):
    """Loads a points file for a PointSetBC / PointSetOperatorBC row:
    n_coord_cols coordinate (+ time) columns, then the target value."""
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"Boundary condition points file not found: {{path!r}}")
    data = np.loadtxt(path, delimiter=",") if path.lower().endswith(".csv") else np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :n_coord_cols], data[:, n_coord_cols:n_coord_cols + 1]''')

    if needs_interface_adapter:
        parts.append('''class _Interface2DGeomAdapter:
    """Interface2DBC expects boundary_normal() without GeometryXTime's
    extra (always-zero) time-normal column."""
    def __init__(self, gt):
        self._gt = gt
    def on_boundary(self, x):
        return self._gt.on_boundary(x)
    def boundary_normal(self, x):
        return self._gt.boundary_normal(x)[:, :-1]''')

    if needs_dtype_wrap:
        parts.append('''class _DTypeSafeGeom:
    """Some DeepXDE geometries (Disk/Ellipse/Triangle/Polygon/Sphere on
    this installed version) return float64 points even when the network
    is float32 -- cast every sampled point array to the configured float
    type so training doesn't hit a dtype-mismatch error."""
    def __init__(self, geom):
        self._geom = geom
    def __getattr__(self, name):
        return getattr(self._geom, name)
    def random_points(self, n, random="pseudo"):
        return self._geom.random_points(n, random=random).astype(dde.config.real(np))
    def uniform_points(self, n, boundary=True):
        return self._geom.uniform_points(n, boundary=boundary).astype(dde.config.real(np))
    def random_boundary_points(self, n, random="pseudo"):
        return self._geom.random_boundary_points(n, random=random).astype(dde.config.real(np))
    def uniform_boundary_points(self, n):
        return self._geom.uniform_boundary_points(n).astype(dde.config.real(np))''')

    parts.append(net_helper_code)
    # Training Monitors' own _tm_build_dvars/_tm_hess helpers are also what
    # the custom plot-field expression (_extract_plot_field, below) routes
    # through once configured -- same derivative-aware mechanism Round 17
    # wired into generate_script()/the Restore scripts, now also reaching
    # this exporter. Embed the helper whenever either needs it, not just
    # for Training Monitors, so a custom expression works even if Training
    # Monitors itself is off.
    if (any_cbs and config.training_monitors_enabled) or _plot_custom_expr_val:
        parts.append(_training_monitor_runtime_code())

    if needs_transform_helper:
        tlines = ["def _apply_transforms(net):"]
        if has_in_transform:
            tlines.append(f"    in_scale = {list(config.input_transform_scale)}")
            tlines.append(f"    in_shift = {list(config.input_transform_shift)}")
            tlines.append("    def _input_transform(x):")
            tlines.append("        sc = torch.tensor(in_scale, dtype=x.dtype, device=x.device)")
            tlines.append("        sh = torch.tensor(in_shift, dtype=x.dtype, device=x.device)")
            tlines.append("        return x * sc + sh")
            tlines.append("    net.apply_feature_transform(_input_transform)")
        if has_out_transform:
            tlines.append(f"    out_scale = {list(config.output_transform_scale)}")
            tlines.append(f"    out_shift = {list(config.output_transform_shift)}")
            tlines.append("    def _output_transform(x, y):")
            tlines.append("        sc = torch.tensor(out_scale, dtype=y.dtype, device=y.device)")
            tlines.append("        sh = torch.tensor(out_shift, dtype=y.dtype, device=y.device)")
            tlines.append("        return y * sc + sh")
            tlines.append("    net.apply_output_transform(_output_transform)")
        tlines.append("    return net")
        parts.append("\n".join(tlines))

    parts.append(pde_block)

    if bc_defs:
        parts.append("\n".join(bc_defs))

    if ic_defs:
        parts.append("\n".join(ic_defs))

    if is_inverse:
        inv_lines = []
        for name, init, _true in inv_vars_parsed:
            inv_lines.append(f"{name} = dde.Variable({init})")
        inv_lines.append("inv_vars = [" + ", ".join(n for n, _i, _t in inv_vars_parsed) + "]")
        inv_lines.append('''
def _load_obs_data(path):
    try:
        return np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception:
        try:
            return np.loadtxt(path, skiprows=1)
        except Exception:
            return np.loadtxt(path, delimiter=",")
''')
        # Number of leading coordinate columns in a measured-data file --
        # x/y/z plus time, EXCEPT for a steady-state problem, which has no
        # time axis at all (same distinction the BC-panel machinery above
        # already makes). The pre-existing is_3d/is_2d-only ladder never
        # checked is_steady, so a steady 2D/3D template's observed-data
        # file (x, y, u -- no time column) would have been sliced as if
        # its value column were "t" and there were no value column at
        # all. Known at generation time here, so this resolves to a
        # literal slice, not a runtime branch.
        _n_obs_cols = (3 if is_3d else (2 if is_2d else 1)) if is_steady else (4 if is_3d else (3 if is_2d else 2))
        obs_lines = ["obs_entries = []  # (xt, u, output_idx, weight, custom_expr), one per measured-data file"]
        for _oi_f, of in enumerate(obs_files_parsed):
            # of['path']!r (not r"...") -- same Windows-trailing-backslash
            # SyntaxError this file's other path sites were already fixed
            # for; this one is generate_clean_script()'s own separate
            # Inverse observation-file loading path and was missed then.
            obs_lines.append(f"_of_data = _load_obs_data({of['path']!r})")
            obs_lines.append(f"_of_xt, _of_u = _of_data[:, 0:{_n_obs_cols}], _of_data[:, {_n_obs_cols}:{_n_obs_cols + 1}]")
            obs_lines.append(f"obs_entries.append((_of_xt, _of_u, {of['output_idx']}, {of['weight']}, {of.get('custom_expr', '')!r}))")
            if of.get('custom_expr'):
                # This measured-data file's value is a DERIVED field (e.g.
                # 1D Schrodinger's |h| = sqrt(u**2+v**2)), not a single raw
                # output column -- same expression syntax as the Results
                # panel's "Custom..." plot field, evaluated here on live
                # torch tensors (gradient-safe) rather than a numpy array
                # after predict(), via dde.icbc.PointSetOperatorBC instead
                # of PointSetBC(component=...).
                # n.strip()!r (not a manually-quoted f-string) so an output
                # name containing a quote can't break this dict literal --
                # same repr()-escaping convention used for free-text fields
                # elsewhere in this file.
                _bindings = ", ".join(f'{n.strip()!r}: outputs[:, {_i}:{_i + 1}]' for _i, n in enumerate(out_names))
                obs_lines.append(f'''def _obs_func_{_oi_f}(inputs, outputs, X):
    _ns_obs = {{"sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
                "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
                "arcsin": torch.arcsin, "arccos": torch.arccos, "arctan": torch.arctan,
                "exp": torch.exp, "log": torch.log, "log10": torch.log10,
                # +1e-12 guard baked into sqrt itself -- same fix and
                # rationale as generate_script()'s own _make_obs_func (this
                # is generate_clean_script()'s separate copy of the same
                # training-time observation-matching function, which
                # DeepXDE differentiates through via PointSetOperatorBC;
                # the same u=v=0 singularity applies here identically).
                "sqrt": lambda _v: torch.sqrt(_v + 1e-12), "abs": torch.abs, "ceil": torch.ceil,
                "floor": torch.floor, "pi": np.pi, "torch": torch,
                {_bindings}}}
    _r = eval({of['custom_expr']!r}, _ns_obs)
    return _r if _r.dim() == 2 else _r.reshape(-1, 1)''')
        # Built as literal per-entry statements (not a generic runtime
        # dispatch) so the exported script stays plain, readable code --
        # matching this generator's own "resolve everything at export
        # time" convention -- rather than looking up each _obs_func_N by
        # constructed name at runtime.
        obs_lines.append("obs_bcs = []")
        for _oi_f, of in enumerate(obs_files_parsed):
            if of.get('custom_expr'):
                obs_lines.append(f"obs_bcs.append(dde.icbc.PointSetOperatorBC("
                                  f"obs_entries[{_oi_f}][0], obs_entries[{_oi_f}][1], _obs_func_{_oi_f}))")
            else:
                obs_lines.append(f"obs_bcs.append(dde.icbc.PointSetBC("
                                  f"obs_entries[{_oi_f}][0], obs_entries[{_oi_f}][1], "
                                  f"component=obs_entries[{_oi_f}][2]))")
        obs_lines.append("obs_anchors = np.vstack([e[0] for e in obs_entries]) if obs_entries else None")
        parts.append("\n".join(inv_lines))
        parts.append("\n".join(obs_lines))

    net_line = (f'net = _make_net({list(config.layers)}, "{config.activation}", '
                f'"{config.kernel_initializer}", {config.weight_decay})')
    if needs_transform_helper:
        net_line += "\nnet = _apply_transforms(net)"

    if not use_ta:
        # =================================================================
        # Standard (non-Time-Adaptive) training path
        # =================================================================
        if is_steady:
            # No time axis at all -- geomtime is just an alias for geom (every
            # dde.icbc.XxxBC constructor accepts any Geometry, not specifically
            # a GeometryXTime), so every BC-construction line above is reused
            # unchanged. See generate_script()'s _is_steady for the same trick.
            geom_lines = [geom_line, "geomtime = geom"]
        else:
            geom_lines = [geom_line,
                          f"timedomain = dde.geometry.TimeDomain({config.t_min}, {config.t_max})",
                          "geomtime = dde.geometry.GeometryXTime(geom, timedomain)"]
        parts.append("\n".join(geom_lines))

        constraints_lines = ["constraints = []"] + bc_appends + ic_appends
        if is_inverse:
            constraints_lines.append("constraints.extend(obs_bcs)")
        parts.append("\n".join(constraints_lines))

        anchors_arg = "obs_anchors" if is_inverse else "None"
        if is_steady:
            data_lines = [f'''data = dde.data.PDE(
    geomtime, pde, constraints,
    num_domain={config.num_domain}, num_boundary={config.num_boundary},
    num_test={config.num_test},
    train_distribution="{config.point_distribution}",
    anchors={anchors_arg},
)''']
        else:
            data_lines = [f'''data = dde.data.TimePDE(
    geomtime, pde, constraints,
    num_domain={config.num_domain}, num_boundary={config.num_boundary},
    num_initial={config.num_initial}, num_test={config.num_test},
    train_distribution="{config.point_distribution}",
    anchors={anchors_arg},
)''']
        parts.append("\n".join(data_lines))
        parts.append(net_line)

        if use_ic_pretrain and not is_steady:
            ic_pretrain_lines = [f'''# ── IC Pre-Training: {config.ic_pretrain_iterations} iterations, IC loss only ──
# A dummy zero-residual PDE and an IC-only dataset (no domain/boundary
# points at all -- an empty-match BC term's loss is NaN even at weight
# 0, so real BCs are left out of this dataset entirely, not zero-weighted).
_ic_pre_geomtime = dde.geometry.GeometryXTime({geom_line.split("=", 1)[1].strip()}, dde.geometry.TimeDomain({config.t_min}, {config.t_max}))
_ic_pre_constraints = []''']
            # Re-use the same IC construction, bound to the pre-training geomtime.
            pretrain_ic_appends = [ln.replace("geomtime", "_ic_pre_geomtime") if "geomtime" in ln else ln
                                    for ln in ic_appends]
            pretrain_ic_appends = [ln.replace("constraints.append", "_ic_pre_constraints.append")
                                    for ln in pretrain_ic_appends]
            ic_pretrain_lines[0] += "\n" + "\n".join(pretrain_ic_appends)
            ic_only_weights = [0.0] * n_out + [1000.0] * len(active_ic_outputs)
            ic_pretrain_lines.append(f'''def _pde_dummy_pre(x, y):
    return [y[:, i:i + 1] * 0 for i in range({n_out})]

_ic_pretrain_num_initial = {max(1, config.ic_pretrain_num_initial)}
_data_pre = dde.data.TimePDE(
    _ic_pre_geomtime, _pde_dummy_pre, _ic_pre_constraints,
    num_domain=0, num_boundary=0,
    num_initial=_ic_pretrain_num_initial, num_test={config.ic_pretrain_num_test},
    train_distribution="{config.point_distribution}",
)
_model_pre = dde.Model(_data_pre, net)
_model_pre.compile("{config.ic_pretrain_optimizer}", lr={config.ic_pretrain_lr},
                    loss="{config.ic_pretrain_loss}", loss_weights={ic_only_weights})
_ic_lh, _ = _model_pre.train(iterations={config.ic_pretrain_iterations}, display_every=10000)
print(f"IC pre-training done. Final IC loss: {{sum(_ic_lh.loss_train[-1]):.4e}}")''')
            parts.append("\n".join(ic_pretrain_lines))

        parts.append("model = dde.Model(data, net)")
        if cbs_code:
            parts.append(cbs_code)
        if is_inverse:
            hist_path = 'os.path.join(save_dir, "param_history.txt") if save_dir else "/tmp/param_history.txt"'
            parts.append(f'''var_history_path = {hist_path}
var_cb = dde.callbacks.VariableValue(inv_vars, period={var_cb_period}, filename=var_history_path, precision=8)''')

        phase_lines = []
        for pi, sp in enumerate(sched_phases):
            phase_lines.append(f"# ── Phase {pi + 1}: {sp.get('optimizer', 'adam')}, "
                                f"{int(sp.get('iterations', 0) or 0)} iterations ──")
            phase_lines += _clean_phase_train_lines(sp, sp['_w'], ext_vars_kw, cbs_kw, "model", "data", config)
        parts.append("\n".join(phase_lines))

        if use_rar:
            rar_lines = [f'''# ── RAR: residual-based adaptive refinement ──
for rar_cycle in range({config.rar_cycles}):''']
            if is_3d:
                rar_lines.append(f'''    x_cand = np.random.uniform({config.x_min}, {config.x_max}, {config.rar_candidates})
    y_cand = np.random.uniform({config.y_min}, {config.y_max}, {config.rar_candidates})
    z_cand = np.random.uniform({config.z_min}, {config.z_max}, {config.rar_candidates})
    t_cand = np.random.uniform({config.t_min}, {config.t_max}, {config.rar_candidates})
    xt_cand = np.column_stack([x_cand, y_cand, z_cand, t_cand])''')
            elif is_2d:
                rar_lines.append(f'''    x_cand = np.random.uniform({config.x_min}, {config.x_max}, {config.rar_candidates})
    y_cand = np.random.uniform({config.y_min}, {config.y_max}, {config.rar_candidates})
    t_cand = np.random.uniform({config.t_min}, {config.t_max}, {config.rar_candidates})
    xt_cand = np.column_stack([x_cand, y_cand, t_cand])''')
            else:
                rar_lines.append(f'''    x_cand = np.random.uniform({config.x_min}, {config.x_max}, {config.rar_candidates})
    t_cand = np.random.uniform({config.t_min}, {config.t_max}, {config.rar_candidates})
    xt_cand = np.column_stack([x_cand, t_cand])''')
            rar_lines.append(f'''    res = model.predict(xt_cand, operator=pde)
    residuals = np.sum([np.abs(r).flatten() for r in res], axis=0) if isinstance(res, list) else np.abs(res).flatten()
    top_idx = np.argsort(residuals)[-{config.rar_add_points}:]
    data.add_anchors(xt_cand[top_idx])
    model.compile("{config.optimizer}", lr={config.learning_rate}, loss="{config.loss_type}", loss_weights={multi_weights})
    loss_history, train_state = model.train(iterations={config.rar_adam_iters}, display_every={config.loss_display_every})''')
            if config.rar_lbfgs_iters > 0:
                rar_lines.append(f'''    dde.optimizers.set_LBFGS_options(maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                                     gtol={config.lbfgs_gtol}, maxiter={config.rar_lbfgs_iters},
                                     maxfun={config.lbfgs_maxfun}, maxls={config.lbfgs_maxls})
    model.compile("L-BFGS", loss="{config.loss_type}", loss_weights={multi_weights})
    loss_history, train_state = model.train(display_every={config.loss_display_every})''')
            parts.append("\n".join(rar_lines))

        if use_save:
            parts.append('model.save(os.path.join(save_dir, "model"))')

    else:
        # =================================================================
        # Time-Adaptive training path -- the domain's time range is split
        # into sequential sub-intervals; each one trains its own model,
        # seeded from the previous interval's predicted solution as its
        # Initial Condition (and, if Transfer Learning is on, from its
        # network weights too). Matches this app's real Time-Adaptive
        # support: 1D, 2D, and 3D (a 3D solution is visualized the same way
        # as the Standard path's own 3D plot -- an x-t slice at the domain's
        # y/z mid-points, since there's no volumetric renderer here).
        # =================================================================
        try:
            ta_groups = _clean_json.loads(config.ta_step_groups) if config.ta_step_groups else []
        except (ValueError, TypeError):
            ta_groups = []
        if not ta_groups:
            ta_groups = [{"t_start": config.t_min, "t_end": config.t_max, "steps": config.ta_num_steps}]
        intervals = []
        for grp in ta_groups:
            dt = (grp['t_end'] - grp['t_start']) / grp['steps']
            for gi in range(grp['steps']):
                t0 = grp['t_start'] + gi * dt
                intervals.append((t0, t0 + dt))
        parts.append(f"intervals = {intervals}  # (t0, t1) per Time-Adaptive step")

        geom_expr = geom_line.split("=", 1)[1].strip()
        ta_lines = [f'''def _build_geom():
    return {geom_expr}
''']
        if is_3d:
            # grid_size is a per-axis resolution, so this cubes it.
            ta_lines.append(f'''grid_size = {config.ta_grid_size}
_xg = np.linspace({config.x_min}, {config.x_max}, grid_size)
_yg = np.linspace({config.y_min}, {config.y_max}, grid_size)
_zg = np.linspace({config.z_min}, {config.z_max}, grid_size)
_Xg, _Yg, _Zg = np.meshgrid(_xg, _yg, _zg)
x_grid = np.column_stack([_Xg.ravel(), _Yg.ravel(), _Zg.ravel()])''')
        elif is_2d:
            ta_lines.append(f'''grid_size = {config.ta_grid_size}
_xg = np.linspace({config.x_min}, {config.x_max}, grid_size)
_yg = np.linspace({config.y_min}, {config.y_max}, grid_size)
_Xg, _Yg = np.meshgrid(_xg, _yg)
x_grid = np.column_stack([_Xg.ravel(), _Yg.ravel()])''')
        else:
            ta_lines.append(f'''grid_size = {config.ta_grid_size}
x_grid = np.linspace({config.x_min}, {config.x_max}, grid_size).reshape(-1, 1)''')

        if config.forward_ic_from_file:
            ta_lines.append(f'_ic_ta_xt, prev_u = _load_ic_from_file({repr(config.forward_ic_file)})')
        else:
            # ic_exprs[0] already expects spatial columns only (x, or x & y) --
            # exactly what x_grid holds, so no time column is needed here.
            ic0_expr = ic_exprs[0].strip() if ic_exprs else "np.zeros_like(x[:, 0])"
            ta_lines.append(f'''def _prev_u0(x):
    return np.reshape({ic0_expr}, (-1, 1))
prev_u = _prev_u0(x_grid)''')

        ta_lines.append('''
all_x, all_t, all_u = [], [], []   # accumulated per-step (X, T, U) grids for the final stitched plot
ta_step_models = []                # [(t0, t1, model_i)] -- used for Error Analysis, if configured
prev_net = None

# Accumulates the loss history across EVERY optimizer phase of EVERY time
# sub-domain, so the final loss plot shows the full Time-Adaptive run
# instead of just the last phase of the last sub-domain -- loss_history
# gets overwritten by each new model_i.train() call below (there can be
# several per sub-domain, one per scheduler phase), so without this only
# whichever call happened to run last for the very last step would ever
# reach the plot. Each phase's own step counter restarts at 0, so
# _ta_loss_offset (the cumulative iteration count so far) is added to keep
# the x-axis continuously increasing across the whole run.
_ta_all_steps, _ta_all_train_loss, _ta_all_test_loss = [], [], []
_ta_loss_offset = 0''')
        parts.append("\n".join(ta_lines))

        loop_lines = ["for step_i, (t0, t1) in enumerate(intervals):",
                      "    geom_i = _build_geom()",
                      "    timedomain_i = dde.geometry.TimeDomain(t0, t1)",
                      "    geomtime_i = dde.geometry.GeometryXTime(geom_i, timedomain_i)",
                      ""]
        # Reuse the shared BC construction, bound to this step's geomtime.
        step_bc_appends = [ln.replace("geomtime", "geomtime_i") for ln in bc_appends]
        loop_lines.append("    constraints_i = []")
        for ln in step_bc_appends:
            loop_lines.append(f"    {ln.replace('constraints.append', 'constraints_i.append')}")
        loop_lines.append("")
        loop_lines.append("    if step_i == 0:")
        step0_ic_appends = [ln.replace("geomtime", "geomtime_i").replace("constraints.append", "constraints_i.append")
                             for ln in ic_appends]
        for ln in step0_ic_appends:
            loop_lines.append(f"        {ln}")
        loop_lines.append("    else:")
        loop_lines.append("        _xt_ic_i = np.column_stack([x_grid, np.full(len(x_grid), t0)])")
        for _oi_ic in range(n_out):
            loop_lines.append(
                f"        constraints_i.append(dde.icbc.PointSetBC(_xt_ic_i, "
                f"prev_u[:, {_oi_ic}:{_oi_ic + 1}], component={_oi_ic}))"
            )
        loop_lines.append("")
        anchors_i = "obs_anchors" if is_inverse else "None"
        loop_lines.append(f'''    data_i = dde.data.TimePDE(
        geomtime_i, pde, constraints_i,
        num_domain={config.num_domain}, num_boundary={config.num_boundary},
        num_initial={config.num_initial}, num_test={config.num_test},
        train_distribution="{config.point_distribution}",
        anchors={anchors_i},
    )
    net_i = _make_net({list(config.layers)}, "{config.activation}", "{config.kernel_initializer}", {config.weight_decay})''')
        if needs_transform_helper:
            loop_lines.append("    net_i = _apply_transforms(net_i)")
        if config.ta_transfer_learning:
            loop_lines.append('''    if prev_net is not None:
        net_i.load_state_dict(prev_net.state_dict())''')

        if use_ic_pretrain:
            ic_only_weights = [0.0] * n_out + [1000.0] * len(active_ic_outputs)
            step0_pretrain_ic = [ln.replace("geomtime", "_ic_pre_geomtime_i")
                                  .replace("constraints.append", "_ic_pre_constraints_i.append")
                                  for ln in ic_appends]
            loop_lines.append(f'''    if step_i == 0:
        _ic_pre_geomtime_i = dde.geometry.GeometryXTime(_build_geom(), dde.geometry.TimeDomain(t0, t1))
        _ic_pre_constraints_i = []''')
            for ln in step0_pretrain_ic:
                loop_lines.append(f"        {ln}")
            loop_lines.append(f'''        def _pde_dummy_pre(x, y):
            return [y[:, i:i + 1] * 0 for i in range({n_out})]
        _data_pre_i = dde.data.TimePDE(
            _ic_pre_geomtime_i, _pde_dummy_pre, _ic_pre_constraints_i,
            num_domain=0, num_boundary=0,
            num_initial={max(1, config.ic_pretrain_num_initial)}, num_test={config.ic_pretrain_num_test},
            train_distribution="{config.point_distribution}",
        )
        _model_pre_i = dde.Model(_data_pre_i, net_i)
        _model_pre_i.compile("{config.ic_pretrain_optimizer}", lr={config.ic_pretrain_lr},
                              loss="{config.ic_pretrain_loss}", loss_weights={ic_only_weights})
        _model_pre_i.train(iterations={config.ic_pretrain_iterations}, display_every=10000)''')

        loop_lines.append("    model_i = dde.Model(data_i, net_i)")
        if is_inverse:
            loop_lines.append("    if step_i == 0:")
            hist_path = 'os.path.join(save_dir, "param_history.txt") if save_dir else "/tmp/param_history.txt"'
            loop_lines.append(f'''        var_history_path = {hist_path}
        var_cb = dde.callbacks.VariableValue(inv_vars, period={var_cb_period}, filename=var_history_path, precision=8)''')
        for pi, sp in enumerate(sched_phases):
            loop_lines.append(f"    # ── Phase {pi + 1}: {sp.get('optimizer', 'adam')}, "
                               f"{int(sp.get('iterations', 0) or 0)} iterations ──")
            loop_lines += _clean_phase_train_lines(sp, sp['_w'], ext_vars_kw, cbs_kw, "model_i", "data_i", config,
                                                     indent="    ")
            loop_lines.append(
                "    _ta_all_steps.extend([_s + _ta_loss_offset for _s in loss_history.steps]); "
                "_ta_all_train_loss.extend(loss_history.loss_train); "
                "_ta_all_test_loss.extend(loss_history.loss_test); "
                "_ta_loss_offset += (loss_history.steps[-1] if loss_history.steps else 0)"
            )
        loop_lines.append("    ta_step_models.append((t0, t1, model_i))")
        loop_lines.append("    prev_net = net_i")
        if is_3d:
            loop_lines.append('''    _xyzt_pred = np.column_stack([x_grid, np.full(len(x_grid), t1)])
    prev_u = model_i.predict(_xyzt_pred)
    _x_plot = np.linspace({0}, {1}, 100)
    _t_plot = np.linspace(t0, t1, 50)
    _y_mid = ({2} + {3}) / 2.0
    _z_mid = ({4} + {5}) / 2.0
    _Xp, _Tp = np.meshgrid(_x_plot, _t_plot)
    _XYZTp = np.column_stack([_Xp.ravel(), np.full(_Xp.size, _y_mid), np.full(_Xp.size, _z_mid), _Tp.ravel()])
    _Up = model_i.predict(_XYZTp)[:, {6}].reshape(50, 100)
    all_x.append(_Xp); all_t.append(_Tp); all_u.append(_Up)'''.format(
                config.x_min, config.x_max, line_slice_y, line_slice_y, line_slice_z, line_slice_z, config.plot_output_idx))
        elif is_2d:
            loop_lines.append('''    _xyt_pred = np.column_stack([x_grid, np.full(len(x_grid), t1)])
    prev_u = model_i.predict(_xyt_pred)
    _x_plot = np.linspace({0}, {1}, 100)
    _t_plot = np.linspace(t0, t1, 50)
    _y_mid = ({2} + {3}) / 2.0
    _Xp, _Tp = np.meshgrid(_x_plot, _t_plot)
    _XYTp = np.column_stack([_Xp.ravel(), np.full(_Xp.size, _y_mid), _Tp.ravel()])
    _Up = model_i.predict(_XYTp)[:, {4}].reshape(50, 100)
    all_x.append(_Xp); all_t.append(_Tp); all_u.append(_Up)'''.format(
                config.x_min, config.x_max, line_slice_y, line_slice_y, config.plot_output_idx))
        else:
            loop_lines.append('''    _xt_pred = np.column_stack([x_grid.ravel(), np.full(grid_size, t1)])
    prev_u = model_i.predict(_xt_pred)
    _x_plot = np.linspace({0}, {1}, 100)
    _t_plot = np.linspace(t0, t1, 50)
    _Xp, _Tp = np.meshgrid(_x_plot, _t_plot)
    _XTp = np.vstack([_Xp.ravel(), _Tp.ravel()]).T
    _Up = model_i.predict(_XTp)[:, {2}].reshape(50, 100)
    all_x.append(_Xp); all_t.append(_Tp); all_u.append(_Up)'''.format(
                config.x_min, config.x_max, config.plot_output_idx))
        loop_lines.append(f'    print(f"Step {{step_i + 1}}/{{len(intervals)}} done. '
                           f'Final train loss: {{sum(loss_history.loss_train[-1]):.4e}}")')
        parts.append("\n".join(loop_lines))

        parts.append('''model = ta_step_models[-1][2]  # last step's model, for Error Analysis / ad-hoc predict() calls''')
        if use_save:
            parts.append('model.save(os.path.join(save_dir, "model"))')

    # =========================================================================
    # Plots
    # =========================================================================
    parts.append(f'''sol_dir = save_dir if save_dir else "/tmp"
loss_path = os.path.join(sol_dir, "loss_plot.png")
solution_path = os.path.join(sol_dir, "solution_plot.{sol_ext}")''')

    if use_ta:
        # Full Time-Adaptive run -- every optimizer phase of every time
        # sub-domain, concatenated with a running iteration offset (see
        # "_ta_all_steps"/"_ta_loss_offset" above) -- not just whichever
        # phase happened to run last for the very last sub-domain. This used
        # to plot only loss_history (the last phase's own, restarting-at-0
        # history), which is what the old "last Time-Adaptive step's loss
        # curve" comment here was flagging as a known limitation.
        loss_comment = "  # full run: every phase of every time sub-domain, stitched together"
        _loss_series = "\n".join(_clean_loss_series_lines(config, "_ta_all_train_loss", "_ta_all_test_loss", "_ta_all_steps"))
        parts.append(f'''# ── Loss plot ──{loss_comment}
plt.figure(figsize=(7, 5))
{_loss_series}
plt.xlabel("Iteration (cumulative across all time sub-domains)"); plt.ylabel("Loss")
plt.title("Training & Test Loss — All Time Sub-domains")
plt.legend(); plt.tight_layout()
plt.savefig(loss_path, dpi={config.plot_dpi})
plt.close()
print(f"Loss plot saved: {{loss_path}}")''')
    else:
        _loss_series = "\n".join(_clean_loss_series_lines(config, "loss_history.loss_train", "loss_history.loss_test", "loss_history.steps"))
        parts.append(f'''# ── Loss plot ──
plt.figure(figsize=(7, 5))
{_loss_series}
plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title("Training & Test Loss")
plt.legend(); plt.tight_layout()
plt.savefig(loss_path, dpi={config.plot_dpi})
plt.close()
print(f"Loss plot saved: {{loss_path}}")''')

    if is_inverse:
        parts.append(f'''# ── Parameter convergence plot ──
param_iters, param_vals = [], [[] for _ in inv_vars]
with open(var_history_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or "[" not in line:
            continue
        it_str, rest = line.split(None, 1)
        vals = [float(v) for v in rest.strip("[]").split(",") if v.strip()]
        if len(vals) != len(inv_vars):
            continue
        param_iters.append(int(it_str))
        for vi, v in enumerate(vals):
            param_vals[vi].append(v)
param_iters = np.array(param_iters)
_inv_true_vals = {[t for _n, _i, t in inv_vars_parsed]!r}
fig, axes = plt.subplots(len(inv_vars), 1, figsize=(6, 3.2 * len(inv_vars)), squeeze=False)
for vi, name in enumerate({[n for n, _i, _t in inv_vars_parsed]!r}):
    ax = axes[vi][0]
    vals = np.array(param_vals[vi])
    final_val = vals[-1] if len(vals) else float("nan")
    ax.plot(param_iters, vals, color="#69db7c", linewidth=1.5)
    ax.axhline(y=final_val, color="#ff8787", linestyle="--", alpha=0.5, label=f"Final = {{final_val:.6f}}")
    true_val = _inv_true_vals[vi] if vi < len(_inv_true_vals) else None
    if true_val is not None:
        ax.axhline(y=true_val, color="#ffd43b", linestyle="--", alpha=0.8, label=f"True = {{true_val:.6f}}")
    ax.set_xlabel("Iteration"); ax.set_ylabel(name)
    ax.set_title(f"Inferred Parameter: {{name}}")
    ax.legend(); ax.grid(True, alpha=0.3)
    print(f"Final inferred {{name}}: {{final_val:.6f}}")
plt.tight_layout()
param_plot_path = os.path.join(sol_dir, "param_convergence.png")
plt.savefig(param_plot_path, dpi=100)
plt.close(fig)''')

    # ── Result plot -- exactly one of these, matching this problem's
    # dimension / Time-Adaptive status / selected Plot Type. No runtime
    # dispatch: which branch applies is already known now.
    plot_idx = config.plot_output_idx
    out_name = out_names[plot_idx] if plot_idx < len(out_names) else out_names[0]
    if _plot_custom_expr_val:
        out_name = (config.plot_custom_label or "").strip() or "custom"

    # ── Field to plot/analyze: one raw output column, or -- when a custom
    # expression is configured, e.g. |h| = sqrt(u**2+v**2) for 1D
    # Schrodinger's complex-valued u,v outputs, or a derivative like du_x --
    # a derived scalar field built from ALL of this problem's outputs AND
    # their derivatives (same d{{name}}_x syntax the Custom PDE box, Training
    # Monitors, and the Restore tab/live Results panel's own custom plotting
    # already use), evaluated through DeepXDE's own dde.Model.predict(x,
    # operator=...) built-in, reusing Training Monitors' own _tm_build_dvars
    # (embedded above whenever a custom expression is configured). Mirrors
    # generate_script()'s own _extract_plot_field helper (minus its multi-
    # output Error-Analysis routing, which this single-field generator
    # doesn't have) -- without this, an exported script for a custom-field
    # template always plotted and error-analyzed the raw first output
    # instead of the derived field the GUI itself shows, and (before this
    # round) couldn't reference derivatives at all -- Round 17 added
    # derivative support to the Restore tab and the live Results panel but
    # explicitly left this exporter on the old NumPy-only path; this closes
    # that gap.
    _plot_dim_clean = "3D" if is_3d else ("2D" if is_2d else "1D")
    parts.append(f'''_plot_custom_expr = {_plot_custom_expr_val!r}
_plot_output_names = {out_names!r}
_plot_n_out = len(_plot_output_names)
_plot_is_steady = {is_steady}
_plot_dim = {_plot_dim_clean!r}
_PLOT_TORCH_MATH_NS = {{
    "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
    "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
    "arcsin": torch.asin, "arccos": torch.acos, "arctan": torch.atan,
    "exp": torch.exp, "log": torch.log, "log10": torch.log10,
    "sqrt": torch.sqrt, "abs": torch.abs, "ceil": torch.ceil, "floor": torch.floor,
    "pi": np.pi,
}}
def _plot_custom_op(_pf_inputs, _pf_outputs):
    _pf_dvars = _tm_build_dvars(_pf_inputs, _pf_outputs, _plot_n_out,
                                 _plot_output_names, _plot_is_steady, _plot_dim,
                                 _plot_custom_expr)
    _pf_ns = dict(_pf_dvars)
    _pf_ns.update(_PLOT_TORCH_MATH_NS)
    _pf_ns["torch"] = torch
    return eval(_plot_custom_expr, _pf_ns)

def _extract_plot_field(_x_grid, _pf_model=None):
    _pf_m = _pf_model if _pf_model is not None else model
    if _plot_custom_expr:
        return _pf_m.predict(_x_grid, operator=_plot_custom_op)[:, 0]
    return _pf_m.predict(_x_grid)[:, {plot_idx}]''')

    if use_ta:
        parts.append(f'''# ── Result plot: stitched Time-Adaptive solution ──
Xg = np.concatenate([a for a in all_x], axis=0)
Tg = np.concatenate([a for a in all_t], axis=0)
Ug = np.concatenate([a for a in all_u], axis=0)
plt.figure(figsize=(7, 5))
if {config.plot_swap_xt}:
    plt.contourf(Tg, Xg, Ug, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    plt.xlabel("t"); plt.ylabel("x")
else:
    plt.contourf(Xg, Tg, Ug, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    plt.xlabel("x"); plt.ylabel("t")
if {config.plot_colorbar}:
    plt.colorbar(label="{out_name}")
plt.title("PINN Solution (Time-Adaptive)")
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_inverse and config.plot_type == "Parameter Convergence":
        parts.append('''# ── Result plot: parameter convergence (Inverse default) ──
import shutil
shutil.copy(param_plot_path, solution_path)
print(f"Solution plot saved: {solution_path}")''')

    elif is_2d and is_steady and config.plot_type == "Line (time steps)":
        # Steady 2D "Line (time steps)": no time axis, so a single curve
        # u(x) at the configured y slice -- previously this selection
        # silently fell through to the x-y heatmap branch just below
        # (plot_type was never checked for 2D/3D at all in this
        # generator either).
        parts.append(f'''# ── Result plot: line at y={line_slice_y:.3g} (steady, no time axis) ──
res = {config.plot_resolution}
x_l2ds = np.linspace({config.x_min}, {config.x_max}, res)
xy_l2ds = np.column_stack([x_l2ds, np.full_like(x_l2ds, {line_slice_y})])
u_l2ds = _extract_plot_field(xy_l2ds).flatten()
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_l2ds, u_l2ds, color="#4dabf7", linewidth={config.plot_linewidth})
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, y={line_slice_y:.3g})")!r})
ax.set_title({(plot_title_override or "PINN Solution")!r})
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_2d and is_steady:
        parts.append(f'''# ── Result plot: steady-state 2D heatmap (no time axis) ──
res = {config.plot_resolution}
xp = np.linspace({config.x_min}, {config.x_max}, res)
yp = np.linspace({config.y_min}, {config.y_max}, res)
Xg, Yg = np.meshgrid(xp, yp)
inside = geom.inside(np.column_stack([Xg.ravel(), Yg.ravel()])).reshape(res, res)
xy = np.column_stack([Xg.ravel(), Yg.ravel()])
pred = _extract_plot_field(xy).reshape(res, res)
pred = np.where(inside, pred, np.nan)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.contourf(Xg, Yg, pred, levels={config.plot_levels}, cmap="{config.plot_colormap}")
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or "y")!r}); ax.set_aspect("equal", adjustable="box")
if {config.plot_colorbar}:
    fig.colorbar(im, ax=ax)
fig.suptitle({(plot_title_override or f"PINN Solution — {out_name}(x, y)")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_3d and is_steady and config.plot_type == "Line (time steps)":
        # Steady 3D "Line (time steps)": single curve u(x) at the
        # configured y and z slice.
        parts.append(f'''# ── Result plot: line at y={line_slice_y:.3g}, z={line_slice_z:.3g} (steady, no time axis) ──
res = {config.plot_resolution}
x_l3ds = np.linspace({config.x_min}, {config.x_max}, res)
xyz_l3ds = np.column_stack([x_l3ds, np.full_like(x_l3ds, {line_slice_y}), np.full_like(x_l3ds, {line_slice_z})])
u_l3ds = _extract_plot_field(xyz_l3ds).flatten()
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_l3ds, u_l3ds, color="#4dabf7", linewidth={config.plot_linewidth})
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, y={line_slice_y:.3g}, z={line_slice_z:.3g})")!r})
ax.set_title({(plot_title_override or "PINN Solution")!r})
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_3d and is_steady:
        parts.append(f'''# ── Result plot: steady-state 3D heatmap at the z mid-plane (no time axis) ──
res = {config.plot_resolution}
xp = np.linspace({config.x_min}, {config.x_max}, res)
yp = np.linspace({config.y_min}, {config.y_max}, res)
Xg, Yg = np.meshgrid(xp, yp)
z_mid = ({config.z_min} + {config.z_max}) / 2.0
inside = geom.inside(np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_mid)])).reshape(res, res)
xyz = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_mid)])
pred = _extract_plot_field(xyz).reshape(res, res)
pred = np.where(inside, pred, np.nan)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.contourf(Xg, Yg, pred, levels={config.plot_levels}, cmap="{config.plot_colormap}")
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or "y")!r}); ax.set_aspect("equal", adjustable="box")
if {config.plot_colorbar}:
    fig.colorbar(im, ax=ax)
fig.suptitle({plot_title_override!r} if {bool(plot_title_override)} else f"PINN Solution — {out_name}(x, y, z={{z_mid:.3g}})")
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_steady:
        # Steady-state 1D: a single static curve u(x) -- no time axis.
        parts.append(f'''# ── Result plot: steady-state 1D curve (no time axis) ──
res = {config.plot_resolution}
x_1d = np.linspace({config.x_min}, {config.x_max}, res)
u_1d = _extract_plot_field(x_1d.reshape(-1, 1)).flatten()
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_1d, u_1d, color="#4dabf7", linewidth={config.plot_linewidth})
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x)")!r})
ax.set_title({(plot_title_override or "PINN Solution")!r})
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif config.plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)"):
        n_frames = max(2, config.num_timesteps_anim)
        anim_kind = "line" if config.plot_type == "Line Animation (GIF)" else "surface"
        vrange = ("v_min, v_max = None, None" if config.plot_auto_range
                  else f"v_min, v_max = {config.plot_vmin}, {config.plot_vmax}")
        if anim_kind == "line":
            # 2D/3D previously fell through to the "else" (surface
            # animation) branch below instead -- plot_type was only
            # checked against ("Line Animation (GIF)", "Surface Animation
            # (GIF)") as a pair, never distinguishing anim_kind for 2D/3D,
            # so picking "Line Animation (GIF)" on a 2D/3D problem
            # silently produced a Surface Animation GIF instead. Now
            # handled directly, at the configured y/z slice (same
            # PINNConfig.line_slice_y/_z generate_script() uses).
            if is_3d:
                _xt_build = (f'xt = np.column_stack([x_line, np.full_like(x_line, {line_slice_y}), '
                             f'np.full_like(x_line, {line_slice_z}), np.full_like(x_line, tv)])')
                _ylabel_anim = f"{out_name}(x, y={line_slice_y:.3g}, z={line_slice_z:.3g}, t)"
            elif is_2d:
                _xt_build = f'xt = np.column_stack([x_line, np.full_like(x_line, {line_slice_y}), np.full_like(x_line, tv)])'
                _ylabel_anim = f"{out_name}(x, y={line_slice_y:.3g}, t)"
            else:
                _xt_build = 'xt = np.column_stack([x_line, np.full_like(x_line, tv)])'
                _ylabel_anim = f"{out_name}(x, t)"
            parts.append(f'''# ── Result plot: line animation (GIF) ──
import matplotlib.animation as animation
t_frames = np.linspace({config.t_min}, {config.t_max}, {n_frames})
x_line = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
frames_u = []
for tv in t_frames:
    {_xt_build}
    frames_u.append(_extract_plot_field(xt).flatten())
u_min = min(u.min() for u in frames_u); u_max = max(u.max() for u in frames_u)
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim({config.x_min}, {config.x_max})
ax.set_ylim(u_min - 0.05 * abs(u_min) - 1e-9, u_max + 0.05 * abs(u_max) + 1e-9)
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or _ylabel_anim)!r}); ax.grid(True, alpha=0.2)
{f"ax.set_title({plot_title_override!r})" if plot_title_override else "# (no title override set)"}
line, = ax.plot([], [], color="#4dabf7", linewidth={config.plot_linewidth_anim})
time_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="#ff8787")
def _update(i):
    line.set_data(x_line, frames_u[i])
    time_txt.set_text(f"t = {{t_frames[i]:.3f}}")
    return line, time_txt
ani = animation.FuncAnimation(fig, _update, frames={n_frames}, interval=100, blit=True)
ani.save(solution_path, writer="pillow", fps={config.plot_fps})
plt.close(fig)
print(f"Solution plot saved: {{solution_path}}")''')
        else:
            parts.append(f'''# ── Result plot: surface animation (GIF) ──
import matplotlib.animation as animation
t_frames = np.linspace({config.t_min}, {config.t_max}, {n_frames})
res = 80
x_a = np.linspace({config.x_min}, {config.x_max}, res)
frames = []''')
            if is_3d:
                parts.append(f'''# 3D: same "x-y heatmap at the z mid-plane" convention as the static
# 3D solution plot below -- no volumetric GIF renderer.
y_a = np.linspace({config.y_min}, {config.y_max}, res)
z_mid_a = ({config.z_min} + {config.z_max}) / 2.0
Xa, Ya = np.meshgrid(x_a, y_a)
for tv in t_frames:
    xyzt = np.column_stack([Xa.ravel(), Ya.ravel(), np.full(Xa.size, z_mid_a), np.full(Xa.size, tv)])
    frames.append(_extract_plot_field(xyzt).reshape(res, res))
{vrange}
if v_min is None:
    v_min = min(f.min() for f in frames); v_max = max(f.max() for f in frames)
fig, ax = plt.subplots(figsize=(7, 5))
def _update(i):
    ax.cla()
    ax.contourf(Xa, Ya, frames[i], levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=v_min, vmax=v_max)
    ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or "y")!r}); ax.set_title({plot_title_override!r} if {bool(plot_title_override)} else f"t = {{t_frames[i]:.3f}}")
ani = animation.FuncAnimation(fig, _update, frames={n_frames}, interval=150)
ani.save(solution_path, writer="pillow", fps={config.plot_fps})
plt.close(fig)
print(f"Solution plot saved: {{solution_path}}")''')
            elif is_2d:
                parts.append(f'''y_a = np.linspace({config.y_min}, {config.y_max}, res)
Xa, Ya = np.meshgrid(x_a, y_a)
for tv in t_frames:
    xyt = np.column_stack([Xa.ravel(), Ya.ravel(), np.full(Xa.size, tv)])
    frames.append(_extract_plot_field(xyt).reshape(res, res))
{vrange}
if v_min is None:
    v_min = min(f.min() for f in frames); v_max = max(f.max() for f in frames)
fig, ax = plt.subplots(figsize=(7, 5))
def _update(i):
    ax.cla()
    ax.contourf(Xa, Ya, frames[i], levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=v_min, vmax=v_max)
    ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or "y")!r}); ax.set_title({plot_title_override!r} if {bool(plot_title_override)} else f"t = {{t_frames[i]:.3f}}")
ani = animation.FuncAnimation(fig, _update, frames={n_frames}, interval=150)
ani.save(solution_path, writer="pillow", fps={config.plot_fps})
plt.close(fig)
print(f"Solution plot saved: {{solution_path}}")''')
            else:
                parts.append(f'''t_a = np.linspace({config.t_min}, {config.t_max}, res)
Xa, Ta = np.meshgrid(x_a, t_a)
for tv in t_frames:
    xt = np.vstack([Xa.ravel(), np.full(Xa.size, tv)]).T
    frames.append(_extract_plot_field(xt).reshape(res, res))
{vrange}
if v_min is None:
    v_min = min(f.min() for f in frames); v_max = max(f.max() for f in frames)
fig, ax = plt.subplots(figsize=(7, 5))
# Same configurable axis orientation as the static "Surface" plot above
# (Plot Settings -> "Swap axes") -- no reshape of frames[i] needed, just
# swapping which of Xa/Ta is passed first to contourf (see the static
# Surface block's comment for why that alone is enough to flip axes).
def _update(i):
    ax.cla()
    if {config.plot_swap_xt}:
        ax.contourf(Ta, Xa, frames[i], levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=v_min, vmax=v_max)
        ax.set_xlabel({(plot_xlabel_override or "t")!r}); ax.set_ylabel({(plot_ylabel_override or "x")!r})
    else:
        ax.contourf(Xa, Ta, frames[i], levels={config.plot_levels}, cmap="{config.plot_colormap}", vmin=v_min, vmax=v_max)
        ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or "t")!r})
    ax.set_title({plot_title_override!r} if {bool(plot_title_override)} else f"t = {{t_frames[i]:.3f}}")
ani = animation.FuncAnimation(fig, _update, frames={n_frames}, interval=150)
ani.save(solution_path, writer="pillow", fps={config.plot_fps})
plt.close(fig)
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_2d and config.plot_type == "Line (time steps)":
        # 2D "Line (time steps)": overlaid u(x) curves at several times,
        # all at the configured y slice -- same convention as the 1D
        # "Line (time steps)" branch further below, and as
        # generate_script()'s own already-fixed equivalent. Previously
        # this selection silently fell through to the x-y heatmap branch
        # just below.
        parts.append(f'''# ── Result plot: line, several time steps, at y={line_slice_y:.3g} ──
n_steps_l2d = {config.num_timesteps_line}
x_l2d = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
t_steps_l2d = np.linspace({config.t_min}, {config.t_max}, n_steps_l2d)
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_l2d))
for i, tv in enumerate(t_steps_l2d):
    xyt = np.column_stack([x_l2d, np.full_like(x_l2d, {line_slice_y}), np.full_like(x_l2d, tv)])
    u_line = _extract_plot_field(xyt).flatten()
    ax.plot(x_l2d, u_line, color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{tv:.3f}}")
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, y={line_slice_y:.3g}, t)")!r})
ax.set_title({(plot_title_override or "PINN Solution")!r})
ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_2d:
        parts.append(f'''# ── Result plot: 2D snapshots ──
n_snaps = {config.plot_n_2d_snapshots}
t_snaps = np.linspace({config.t_min}, {config.t_max}, n_snaps)
res = {config.plot_resolution}
xp = np.linspace({config.x_min}, {config.x_max}, res)
yp = np.linspace({config.y_min}, {config.y_max}, res)
Xg, Yg = np.meshgrid(xp, yp)
inside = geom.inside(np.column_stack([Xg.ravel(), Yg.ravel()])).reshape(res, res)
fig, axes = plt.subplots(1, n_snaps, figsize=(5 * n_snaps, 5))
if n_snaps == 1:
    axes = [axes]
for ai, tv in enumerate(t_snaps):
    xyt = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tv)])
    pred = _extract_plot_field(xyt).reshape(res, res)
    pred = np.where(inside, pred, np.nan)
    im = axes[ai].contourf(Xg, Yg, pred, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    axes[ai].set_title(f"t = {{tv:.3f}}"); axes[ai].set_xlabel({(plot_xlabel_override or "x")!r}); axes[ai].set_ylabel({(plot_ylabel_override or "y")!r})
    if {config.plot_colorbar}:
        fig.colorbar(im, ax=axes[ai])
fig.suptitle({(plot_title_override or f"PINN Solution — {out_name}(x, y, t)")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_3d and config.plot_type == "Line (time steps)":
        # 3D "Line (time steps)": overlaid u(x) curves at several times,
        # at the configured y and z slice.
        parts.append(f'''# ── Result plot: line, several time steps, at y={line_slice_y:.3g}, z={line_slice_z:.3g} ──
n_steps_l3d = {config.num_timesteps_line}
x_l3d = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
t_steps_l3d = np.linspace({config.t_min}, {config.t_max}, n_steps_l3d)
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_l3d))
for i, tv in enumerate(t_steps_l3d):
    xyzt = np.column_stack([x_l3d, np.full_like(x_l3d, {line_slice_y}), np.full_like(x_l3d, {line_slice_z}), np.full_like(x_l3d, tv)])
    u_line = _extract_plot_field(xyzt).flatten()
    ax.plot(x_l3d, u_line, color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{tv:.3f}}")
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, y={line_slice_y:.3g}, z={line_slice_z:.3g}, t)")!r})
ax.set_title({(plot_title_override or "PINN Solution")!r})
ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif is_3d:
        parts.append(f'''# ── Result plot: 3D snapshots at the domain's z mid-plane ──
n_snaps = {config.plot_n_2d_snapshots}
t_snaps = np.linspace({config.t_min}, {config.t_max}, n_snaps)
res = {config.plot_resolution}
xp = np.linspace({config.x_min}, {config.x_max}, res)
yp = np.linspace({config.y_min}, {config.y_max}, res)
Xg, Yg = np.meshgrid(xp, yp)
z_mid = ({config.z_min} + {config.z_max}) / 2.0
inside = geom.inside(np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_mid)])).reshape(res, res)
fig, axes = plt.subplots(1, n_snaps, figsize=(5 * n_snaps, 5))
if n_snaps == 1:
    axes = [axes]
for ai, tv in enumerate(t_snaps):
    xyzt = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_mid), np.full(Xg.size, tv)])
    pred = _extract_plot_field(xyzt).reshape(res, res)
    pred = np.where(inside, pred, np.nan)
    im = axes[ai].contourf(Xg, Yg, pred, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    axes[ai].set_title(f"t = {{tv:.3f}}, z = {{z_mid:.3g}}"); axes[ai].set_xlabel({(plot_xlabel_override or "x")!r}); axes[ai].set_ylabel({(plot_ylabel_override or "y")!r})
    if {config.plot_colorbar}:
        fig.colorbar(im, ax=axes[ai])
fig.suptitle({plot_title_override!r} if {bool(plot_title_override)} else f"PINN Solution — {out_name}(x, y, z={{z_mid:.3g}}, t)")
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    elif config.plot_type == "Line (time steps)":
        parts.append(f'''# ── Result plot: line, several time steps ──
n_steps_plot = {config.num_timesteps_line}
x_l = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
t_steps = np.linspace({config.t_min}, {config.t_max}, n_steps_plot)
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_plot))
for i, tv in enumerate(t_steps):
    xt = np.column_stack([x_l, np.full_like(x_l, tv)])
    u_line = _extract_plot_field(xt).flatten()
    ax.plot(x_l, u_line, color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{tv:.3f}}")
ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, t)")!r})
ax.set_title({(plot_title_override or "PINN Solution")!r})
ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    else:  # "Surface" -- 1D x-t heatmap
        parts.append(f'''# ── Result plot: surface (x-t heatmap) ──
res = {config.plot_resolution}
x_s = np.linspace({config.x_min}, {config.x_max}, res)
t_s = np.linspace({config.t_min}, {config.t_max}, res)
Xs, Ts = np.meshgrid(x_s, t_s)
xts = np.vstack([Xs.ravel(), Ts.ravel()]).T
u_s = _extract_plot_field(xts).reshape(res, res)
fig, ax = plt.subplots(figsize=(7, 5))
if {config.plot_swap_xt}:
    im = ax.contourf(Ts, Xs, u_s, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    ax.set_xlabel({(plot_xlabel_override or "t")!r}); ax.set_ylabel({(plot_ylabel_override or "x")!r})
else:
    im = ax.contourf(Xs, Ts, u_s, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    ax.set_xlabel({(plot_xlabel_override or "x")!r}); ax.set_ylabel({(plot_ylabel_override or "t")!r})
if {config.plot_colorbar}:
    fig.colorbar(im, ax=ax)
ax.set_title({(plot_title_override or "PINN Solution")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    # =========================================================================
    # Error Analysis (only emitted when reference/ground-truth files are
    # configured for this problem)
    # =========================================================================
    if use_ea:
        ea_lines = [f'''# ── Error Analysis: compare against reference data ──
ea_dir = os.path.join(sol_dir, "error_analysis")
os.makedirs(ea_dir, exist_ok=True)
ea_files = {ea_files}
ea_times, ea_x_refs, ea_y_refs, ea_z_refs, ea_u_refs = [], [], [], [], []
for tv, fp in ea_files:
    d = np.loadtxt(fp)
    if d.ndim == 1:
        d = d.reshape(1, -1)''']
        # Same distinction as _n_obs_coord_cols/_n_coord_cols_bc elsewhere in
        # this file: a steady-state reference file (PINNStudio's own
        # "solution.txt", written by _auto_configure_ea's is_steady branch
        # in main_window.py) has no time column at all -- x,y,u for 2D /
        # x,y,z,u for 3D -- one fewer column than the time-dependent layout,
        # with every column after the coordinates shifted down by one.
        # Getting this wrong doesn't raise anywhere near the mistake: it
        # reads a steady 2D file's u column as if it were a time column and
        # crashes with an out-of-bounds IndexError on the (nonexistent)
        # column after it instead.
        if is_3d:
            if is_steady:
                ea_lines.append('''    idx = np.lexsort((d[:, 2], d[:, 1], d[:, 0]))
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(d[idx, 1]); ea_z_refs.append(d[idx, 2])
    ea_u_refs.append(d[idx, 3]); ea_times.append(float(tv))''')
            else:
                ea_lines.append('''    idx = np.lexsort((d[:, 2], d[:, 1], d[:, 0]))
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(d[idx, 1]); ea_z_refs.append(d[idx, 2])
    ea_u_refs.append(d[idx, 4]); ea_times.append(float(d[0, 3]))''')
        elif is_2d:
            if is_steady:
                ea_lines.append('''    idx = np.lexsort((d[:, 1], d[:, 0]))
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(d[idx, 1]); ea_z_refs.append(np.zeros_like(d[idx, 0]))
    ea_u_refs.append(d[idx, 2]); ea_times.append(float(tv))''')
            else:
                ea_lines.append('''    idx = np.lexsort((d[:, 1], d[:, 0]))
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(d[idx, 1]); ea_z_refs.append(np.zeros_like(d[idx, 0]))
    ea_u_refs.append(d[idx, 3]); ea_times.append(float(d[0, 2]))''')
        else:
            if is_steady:
                ea_lines.append('''    idx = np.argsort(d[:, 0])
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(np.zeros_like(d[idx, 0])); ea_z_refs.append(np.zeros_like(d[idx, 0]))
    ea_u_refs.append(d[idx, 1]); ea_times.append(float(tv))''')
            else:
                ea_lines.append('''    idx = np.argsort(d[:, 0])
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(np.zeros_like(d[idx, 0])); ea_z_refs.append(np.zeros_like(d[idx, 0]))
    ea_u_refs.append(d[idx, 2]); ea_times.append(float(tv))''')
        # Drop reference points with no solution value (NaN) -- e.g. grid
        # points outside a non-rectangular geometry like L-Shape's missing
        # quadrant or Disk's bounding-box corners. A no-op for reference
        # files that don't have any (the usual case).
        ea_lines.append('''for _ei in range(len(ea_u_refs)):
    _ea_valid = ~np.isnan(ea_u_refs[_ei])
    if not _ea_valid.all():
        _n_dropped = int((~_ea_valid).sum())
        ea_x_refs[_ei] = ea_x_refs[_ei][_ea_valid]
        ea_y_refs[_ei] = ea_y_refs[_ei][_ea_valid]
        ea_z_refs[_ei] = ea_z_refs[_ei][_ea_valid]
        ea_u_refs[_ei] = ea_u_refs[_ei][_ea_valid]
        print(f"  Dropped {_n_dropped} NaN reference point(s) outside the geometry")''')
        ea_lines.append('''order = np.argsort(ea_times)
ea_times  = [ea_times[i] for i in order]
ea_x_refs = [ea_x_refs[i] for i in order]
ea_y_refs = [ea_y_refs[i] for i in order]
ea_z_refs = [ea_z_refs[i] for i in order]
ea_u_refs = [ea_u_refs[i] for i in order]
n_t = len(ea_times)''')

        if use_ta:
            ea_lines.append(f'''def _predict_at_time(xt_no_time, tv):
    for t0, t1, m in ta_step_models:
        if t0 - 1e-9 <= tv <= t1 + 1e-9:
            return _extract_plot_field(np.column_stack([xt_no_time, np.full(len(xt_no_time), tv)]), m).flatten()
    return _extract_plot_field(
        np.column_stack([xt_no_time, np.full(len(xt_no_time), tv)]), ta_step_models[-1][2]).flatten()

ea_u_pinns = []
for i, tv in enumerate(ea_times):''')
            if is_3d:
                ea_lines.append('''    coords = np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i]])
    ea_u_pinns.append(_predict_at_time(coords, tv))''')
            elif is_2d:
                ea_lines.append('''    coords = np.column_stack([ea_x_refs[i], ea_y_refs[i]])
    ea_u_pinns.append(_predict_at_time(coords, tv))''')
            else:
                ea_lines.append('''    coords = ea_x_refs[i].reshape(-1, 1)
    ea_u_pinns.append(_predict_at_time(coords, tv))''')
        else:
            ea_lines.append("ea_u_pinns = []")
            ea_lines.append("for i, tv in enumerate(ea_times):")
            # A steady-state model was trained on (x[, y[, z]]) alone -- no
            # time input at all -- so predict() must not be fed a time
            # column here either (same distinction as the ground-truth
            # loader above); doing so is what produced the earlier
            # "mat1 and mat2 shapes cannot be multiplied" crash.
            if is_3d:
                if is_steady:
                    ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i]])
    ea_u_pinns.append(_extract_plot_field(xt).flatten())''')
                else:
                    ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i], np.full_like(ea_x_refs[i], tv)])
    ea_u_pinns.append(_extract_plot_field(xt).flatten())''')
            elif is_2d:
                if is_steady:
                    ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], ea_y_refs[i]])
    ea_u_pinns.append(_extract_plot_field(xt).flatten())''')
                else:
                    ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], ea_y_refs[i], np.full_like(ea_x_refs[i], tv)])
    ea_u_pinns.append(_extract_plot_field(xt).flatten())''')
            else:
                if is_steady:
                    ea_lines.append(f'''    xt = ea_x_refs[i].reshape(-1, 1)
    ea_u_pinns.append(_extract_plot_field(xt).flatten())''')
                else:
                    ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], np.full_like(ea_x_refs[i], tv)])
    ea_u_pinns.append(_extract_plot_field(xt).flatten())''')

        ea_lines.append('''
# ── Metrics ──
ea_metrics = []
for i, tv in enumerate(ea_times):
    up, uf = ea_u_pinns[i], ea_u_refs[i]
    l2 = np.linalg.norm(up - uf) / (np.linalg.norm(uf) + 1e-10)
    mse = np.mean((up - uf) ** 2)
    mx = np.max(np.abs(up - uf))
    ma = np.mean(np.abs(up - uf))
    ea_metrics.append((tv, l2, mse, mx, ma))
    print(f"  t={tv:.4f} -- L2={l2:.4e}, MSE={mse:.4e}, Max={mx:.4e}, MeanAbs={ma:.4e}")
with open(os.path.join(ea_dir, "error_metrics.txt"), "w") as f:
    f.write("t,L2_relative,MSE,Max_error,Mean_abs_error\\n")
    for tv, l2, mse, mx, ma in ea_metrics:
        f.write(f"{tv:.6f},{l2:.6e},{mse:.6e},{mx:.6e},{ma:.6e}\\n")''')

        if config.ea_do_line:
            if is_2d or is_3d:
                # Extract the reference points nearest the same slice the
                # Line plot itself uses (PINNConfig.line_slice_y/_z),
                # widening the tolerance band if too few reference points
                # fall near it -- previously this searched around the
                # domain midpoint unconditionally, and for the 3D case
                # the y_mid/y_tol computation was accidentally built from
                # config.x_min/x_max instead of config.y_min/y_max (a
                # copy-paste bug, fixed here along with the slice-value
                # change).
                if not is_3d:
                    mid_slice = f'''
    y_mid = {line_slice_y}
    y_tol = ({config.y_max} - {config.y_min}) / 20.0
    mask = np.abs(ea_y_refs[i] - y_mid) < y_tol
    if mask.sum() < 5:
        mask = np.abs(ea_y_refs[i] - y_mid) < ({config.y_max} - {config.y_min}) / 5.0'''
                else:
                    mid_slice = f'''
    y_mid = {line_slice_y}; z_mid = {line_slice_z}
    y_tol = ({config.y_max} - {config.y_min}) / 20.0; z_tol = ({config.z_max} - {config.z_min}) / 20.0
    mask = (np.abs(ea_y_refs[i] - y_mid) < y_tol) & (np.abs(ea_z_refs[i] - z_mid) < z_tol)
    if mask.sum() < 5:
        y_tol2 = ({config.y_max} - {config.y_min}) / 5.0; z_tol2 = ({config.z_max} - {config.z_min}) / 5.0
        mask = (np.abs(ea_y_refs[i] - y_mid) < y_tol2) & (np.abs(ea_z_refs[i] - z_mid) < z_tol2)'''
                mid_slice += "\n    if mask.sum() < 2:\n        mask = np.ones_like(ea_x_refs[i], dtype=bool)"
                sort_line = "    order_i = np.argsort(ea_x_refs[i][mask]); xv, gt, pn = ea_x_refs[i][mask][order_i], ea_u_refs[i][mask][order_i], ea_u_pinns[i][mask][order_i]"
            else:
                mid_slice = ""
                sort_line = "    order_i = np.argsort(ea_x_refs[i]); xv, gt, pn = ea_x_refs[i][order_i], ea_u_refs[i][order_i], ea_u_pinns[i][order_i]"
            # Steady-state (e.g. a Poisson equation) has no time axis at
            # all -- every reference file is really just a single snapshot
            # at a placeholder t=0, so "t = 0.000" in the title would be
            # meaningless noise rather than a real time coordinate.
            _ea_line_title = 'f"L2 = {l2:.2e}"' if is_steady else 'f"t = {tv:.3f}  |  L2 = {l2:.2e}"'
            if is_3d:
                _ea_line_ylabel = repr(f"{out_name}(x, y={line_slice_y:.3g}, z={line_slice_z:.3g})") if is_steady else repr(f"{out_name}(x, y={line_slice_y:.3g}, z={line_slice_z:.3g}, t)")
                _ea_line_suptitle = f"PINN vs Ground Truth -- Line Comparison (y={line_slice_y:.3g}, z={line_slice_z:.3g})"
            elif is_2d:
                _ea_line_ylabel = repr(f"{out_name}(x, y={line_slice_y:.3g})") if is_steady else repr(f"{out_name}(x, y={line_slice_y:.3g}, t)")
                _ea_line_suptitle = f"PINN vs Ground Truth -- Line Comparison (y={line_slice_y:.3g})"
            else:
                _ea_line_ylabel = repr(f"{out_name}(x)") if is_steady else repr(f"{out_name}(x, t)")
                _ea_line_suptitle = "PINN vs Ground Truth -- Line Comparison"
            ea_lines.append(f'''
# ── Line comparison ──
ncols = min(4, n_t); nrows = (n_t + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
fig.suptitle("{_ea_line_suptitle}", fontsize=13, fontweight="bold")
axf = axes.flatten()
for i, tv in enumerate(ea_times):
    ax = axf[i]{mid_slice}
{sort_line}
    tv_r, l2, mse, mx, ma = ea_metrics[i]
    ax.plot(xv, gt, color="#4dabf7", linewidth=2.0, label="Ground Truth")
    ax.plot(xv, pn, color="#ff6b6b", linewidth=2.0, linestyle="--", label="PINN")
    ax.set_title({_ea_line_title}, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel({_ea_line_ylabel}); ax.grid(True, alpha=0.3)
for j in range(n_t, len(axf)):
    axf[j].set_visible(False)
handles, labels = axf[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(ea_dir, "line_comparison.png"), dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print("  Line comparison saved.")''')

        if config.ea_do_surface and not (is_2d or is_3d):
            ea_lines.append(f'''
# ── Surface comparison (1D: x-t) ──
from scipy.interpolate import interp1d
x_common = np.linspace({config.x_min}, {config.x_max}, 300)
t_arr = np.array(ea_times)
U_pinn = np.zeros((len(t_arr), len(x_common)))
U_ref  = np.zeros((len(t_arr), len(x_common)))
for i, tv in enumerate(ea_times):
    fi_p = interp1d(ea_x_refs[i], ea_u_pinns[i], kind="linear", fill_value="extrapolate")
    fi_r = interp1d(ea_x_refs[i], ea_u_refs[i], kind="linear", fill_value="extrapolate")
    U_pinn[i, :] = fi_p(x_common); U_ref[i, :] = fi_r(x_common)
Xg, Tg = np.meshgrid(x_common, t_arr)
U_err = np.abs(U_pinn - U_ref)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("PINN vs Ground Truth -- Surface Comparison", fontsize=13, fontweight="bold")
im0 = axes[0].contourf(Tg, Xg, U_pinn, levels={config.plot_levels}, cmap="{config.plot_colormap}")
axes[0].set_title("PINN"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x"); fig.colorbar(im0, ax=axes[0])
im1 = axes[1].contourf(Tg, Xg, U_ref, levels={config.plot_levels}, cmap="{config.plot_colormap}")
axes[1].set_title("Ground Truth"); axes[1].set_xlabel("t"); axes[1].set_ylabel("x"); fig.colorbar(im1, ax=axes[1])
im2 = axes[2].contourf(Tg, Xg, U_err, levels={config.plot_levels}, cmap="{config.plot_colormap}")
axes[2].set_title("|Error|"); axes[2].set_xlabel("t"); axes[2].set_ylabel("x"); fig.colorbar(im2, ax=axes[2])
plt.tight_layout()
plt.savefig(os.path.join(ea_dir, "surface_comparison.png"), dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print("  Surface comparison saved.")''')
        elif config.ea_do_surface and is_2d:
            # See the matching comment on the Line comparison title above --
            # steady-state has no time axis, so "t=..." is dropped here too.
            _ea_pinn_title = 'f"PINN  L2={l2:.2e}"' if is_steady else 'f"PINN  t={tv:.3f}  L2={l2:.2e}"'
            _ea_gt_title = '"Ground Truth"' if is_steady else 'f"Ground Truth  t={tv:.3f}"'
            ea_lines.append(f'''
# ── Surface comparison (2D heatmaps) ──
from scipy.interpolate import griddata
res_ea = {config.plot_resolution}
xg_ea = np.linspace({config.x_min}, {config.x_max}, res_ea)
yg_ea = np.linspace({config.y_min}, {config.y_max}, res_ea)
Xg_ea, Yg_ea = np.meshgrid(xg_ea, yg_ea)
fig, axes = plt.subplots(n_t, 3, figsize=(15, 4 * n_t), squeeze=False)
fig.suptitle("PINN vs Ground Truth -- 2D Heatmaps", fontsize=13, fontweight="bold")
for i, tv in enumerate(ea_times):
    tv_r, l2, mse, mx, ma = ea_metrics[i]
    xyt_grid = np.column_stack([Xg_ea.ravel(), Yg_ea.ravel()]{"" if is_steady else " + [np.full(Xg_ea.size, tv)]"})
    u_pinn_grid = _extract_plot_field(xyt_grid).reshape(res_ea, res_ea)
    u_ref_grid = griddata(np.column_stack([ea_x_refs[i], ea_y_refs[i]]), ea_u_refs[i], (Xg_ea, Yg_ea), method="linear", fill_value=0.0)
    u_err_grid = np.abs(u_pinn_grid - u_ref_grid)
    im0 = axes[i][0].contourf(Xg_ea, Yg_ea, u_pinn_grid, levels=40, cmap="{config.plot_colormap}")
    axes[i][0].set_title({_ea_pinn_title}); fig.colorbar(im0, ax=axes[i][0])
    im1 = axes[i][1].contourf(Xg_ea, Yg_ea, u_ref_grid, levels=40, cmap="{config.plot_colormap}")
    axes[i][1].set_title({_ea_gt_title}); fig.colorbar(im1, ax=axes[i][1])
    im2 = axes[i][2].contourf(Xg_ea, Yg_ea, u_err_grid, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    axes[i][2].set_title(f"|Error|  Max={{mx:.2e}}"); fig.colorbar(im2, ax=axes[i][2])
plt.tight_layout()
plt.savefig(os.path.join(ea_dir, "surface_comparison.png"), dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print("  Surface comparison saved.")''')
        elif config.ea_do_surface and is_3d:
            # Same steady-vs-time-dependent distinction as the metrics
            # block above (see the "mat1 and mat2 shapes cannot be
            # multiplied" comment there): a steady-state model takes no
            # time input at all, so predict() must not be fed a time
            # column here either. This also now routes through
            # _extract_plot_field like every other prediction site in
            # this function, instead of a raw [:, plot_idx] column --
            # previously a custom derived plot field (e.g. a future
            # 3D template's own |h|-style field) would have been silently
            # ignored in this one plot while the metrics/line-comparison
            # plots above already used it correctly.
            _ea_3d_predict_line = (
                "    pinn_b = _extract_plot_field(np.column_stack([bx, by, bz])).flatten()"
                if is_steady else
                "    pinn_b = _extract_plot_field(np.column_stack([bx, by, bz, np.full_like(bx, tv)])).flatten()"
            )
            # Time-Adaptive never keeps a bare `geom` variable around (it only
            # ever builds per-step geometry via _build_geom()/geom_i, scoped
            # to that step's loop iteration) -- the Standard path's own
            # geom_line assignment is what defines `geom` here.
            _ea_geom_ref = "_build_geom()" if use_ta else "geom"
            # See the matching comment on the Line comparison title above --
            # steady-state has no time axis, so "t=..." is dropped here too.
            _ea_pinn_title_3d = 'f"PINN  L2={l2:.2e}"' if is_steady else 'f"PINN  t={tv:.3f}  L2={l2:.2e}"'
            _ea_gt_title_3d = '"Ground Truth"' if is_steady else 'f"Ground Truth  t={tv:.3f}"'
            ea_lines.append(f'''
# ── Surface comparison (3D: boundary-point scatter vs reference) ──
fig = plt.figure(figsize=(15, 4.5 * n_t))
fig.suptitle("PINN vs Ground Truth -- 3D Comparison", fontsize=13, fontweight="bold")
_ea_geom3d = {_ea_geom_ref}
for i, tv in enumerate(ea_times):
    tv_r, l2, mse, mx, ma = ea_metrics[i]
    bnd = _ea_geom3d.on_boundary(np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i]]))
    # A Cartesian reference grid only reliably intersects a FLAT boundary
    # (e.g. a Cuboid's 6 faces, which are themselves subsets of the grid's
    # own axis-aligned planes) in large numbers. A CURVED boundary (e.g. a
    # Sphere) is essentially never landed on exactly by an axis-aligned
    # grid -- for a unit sphere, only the 6 axis points satisfy x^2+y^2+z^2=1
    # exactly -- so on_boundary() can come back with a handful of
    # coincidental exact matches even when the dataset has thousands of
    # valid points. The old "< 4 points" fallback didn't catch this (6
    # matches is already >= 4), so it silently plotted a near-empty
    # scatter instead of a real comparison. Falling back whenever the
    # boundary subset is a small fraction of the available data -- not
    # just when it's almost empty -- fixes curved geometries without
    # changing anything for flat ones (a Cuboid's face points are normally
    # a large share of its grid, well above this threshold).
    if bnd.sum() < 4 or bnd.sum() < 0.05 * len(ea_x_refs[i]):
        bnd = np.ones_like(ea_x_refs[i], dtype=bool)
    bx, by, bz = ea_x_refs[i][bnd], ea_y_refs[i][bnd], ea_z_refs[i][bnd]
    gt_b = ea_u_refs[i][bnd]
    # Cap how many points actually get scattered -- falling back to a
    # dense volumetric grid (tens of thousands of points) is slow to
    # render and visually saturates a 3D scatter; a few thousand points
    # is still plenty to read the solution's shape clearly.
    if len(bx) > 3000:
        _ea_sub = np.random.default_rng(0).choice(len(bx), size=3000, replace=False)
        bx, by, bz, gt_b = bx[_ea_sub], by[_ea_sub], bz[_ea_sub], gt_b[_ea_sub]
{_ea_3d_predict_line}
    err_b = np.abs(pinn_b - gt_b)
    cols = [(pinn_b, {_ea_pinn_title_3d}), (gt_b, {_ea_gt_title_3d}), (err_b, f"|Error|  Max={{mx:.2e}}")]
    for ci, (vals, ttl) in enumerate(cols):
        ax3 = fig.add_subplot(n_t, 3, i * 3 + ci + 1, projection="3d")
        sc = ax3.scatter(bx, by, bz, c=vals, cmap="{config.plot_colormap}" if ci < 2 else "inferno", s=14)
        fig.colorbar(sc, ax=ax3, shrink=0.6, pad=0.12)
        ax3.set_title(ttl, fontsize=10)
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
plt.tight_layout()
plt.savefig(os.path.join(ea_dir, "surface_comparison.png"), dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print("  Surface comparison saved.")''')

        parts.append("\n".join(ea_lines))

    if any_cbs and config.training_monitors_enabled:
        _tm_clean_plot_code = _build_training_monitors_plot_code(config, "(save_dir or '.')", indent=0).replace("_os.path.join", "os.path.join")
        if _tm_clean_plot_code:
            parts.append(_tm_clean_plot_code)

    parts.append('print("DONE")')
    return "\n\n".join(parts) + "\n"

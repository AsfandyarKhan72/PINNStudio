"""
Parameter Sweep registry: the canonical list of PINNConfig fields a user
can sweep, grouped by category, with get/set accessors that resolve each
field against the CURRENT training architecture.

Why this file exists rather than just reading/writing PINNConfig
attributes directly from the sweep UI: training-relevant fields like
learning rate, iterations, and optimizer live inside the active
Optimizer Scheduler's phases (config.scheduler_phases, a JSON list) for
every template today -- the scheduler is on by default everywhere. The
single-phase legacy fields (config.iterations, config.optimizer, ...)
are only read when the scheduler is OFF, which is not how any built-in
template is configured. A sweep implementation that read/wrote those
legacy fields directly would silently do nothing for virtually every
real run -- precisely the bug the v56 fix (config.validate() checking
the wrong iteration source) already had to correct once. This registry
exists so every call site resolves a parameter the same, correct way,
once, instead of each UI/driver re-deriving "which field actually drives
training right now" and risking getting it wrong.

SCOPE (v1): scalar, unambiguous fields -- per-phase learning rate /
iterations / optimizer choice, weight decay, network width/depth, and
collocation point counts.

Phase-scoped entries (learning rate / iterations / optimizer) are only
offered when the Optimizer Scheduler is actually enabled and has that
many phases configured -- which is the state every template is in by
default. Sweeping those with the scheduler off isn't supported in v1;
is_available() for those entries returns False in that case rather than
silently targeting the inert legacy fields.

SCOPE (v2): Loss weights (PDE/BC/IC). These live in config.loss_weights_multi
(a flat, comma-separated string) in a column order that depends on the
number of outputs, how many rows are in the Boundary Conditions panel
(config.custom_bc_json), and which outputs currently have an Initial
Condition active (config.ic_active / forward_ic_from_file) -- exactly
the kind of config-dependent indexing this module's docstring originally
flagged as needing its own careful, separately-tested pass. See the
"Loss weight helpers" section below:

 - _weight_slot_descriptors(config) is the ONE place that decides this
   order -- PDE (one per output), then one per Boundary Conditions panel
   row, then IC (one per currently-active output, none at all for a
   steady-state problem). It deliberately mirrors, slot for slot,
   codegen.py's own _clean_loss_weights() (used for "Export as DeepXDE
   Script") and the equivalent runtime block inside generate_script()
   (the live Solve/Sweep path) -- both already build PDE-then-BC-then-IC
   in this same order, gated the same way, so this is not a new scheme,
   just the first place that *names* each slot so it can be swept.
 - Observation-file weights (Inverse problems) are NOT included here --
   those already have their own per-file weight spinbox in the Inverse
   Data panel (inverse_obs_files_json), a separate mechanism, and always
   sit after this block in the resolved list, so nothing here needs to
   touch that tail.
 - _weight_set() writes the new value into config.loss_weights_multi AND
   re-syncs every entry in config.scheduler_phases' own per-phase
   "weights" string at that same slot index. This second part is not
   optional: once the Optimizer Scheduler is active (the default for
   every template), codegen.py reads EACH PHASE'S OWN "weights" string
   at training time, not loss_weights_multi directly -- updating only
   loss_weights_multi would silently leave every phase training with its
   stale copy of the old value. In the GUI this never diverges because
   _build_scheduler_phases_json() rebuilds every phase's weights from the
   exact same widgets loss_weights_multi comes from on every single
   _build_config() call -- but a swept config is produced by mutating a
   deep copy directly (sweep_runner.build_runs()), bypassing that GUI
   rebuild entirely, so this module has to do the same resync by hand.
 - Sweeping a loss weight is therefore only offered when there is
   exactly ONE active copy of the weights across the whole run -- the
   scheduler is off (a single implicit phase), or "Same weights (all
   phases)" is checked (config.scheduler_same_weights, the default for
   every template). When per-phase weights are allowed to diverge
   (scheduler_same_weights=False), "the PDE weight" is ambiguous -- which
   phase's copy? -- so is_available() returns False rather than silently
   sweeping only one phase's copy while the others train with whatever
   they already had.

SCOPE (v3): Inverse-problem entries -- only offered when
config.problem_type == "Inverse". Two kinds, one per row already present
in the Inverse panel (main_window.py's inv_var_rows/inv_data_rows):

 - Each trainable (unknown) variable's own "initial guess" value
   (config.inverse_variables_json, falling back to the single legacy
   inverse_param_name/inverse_param_init fields for a config saved before
   multi-variable support existed -- mirrors codegen.py's own
   _parse_inverse_variables() fallback, minus the identifier-sanitizing/
   de-duplication that function additionally does for generated code,
   which sweeping a raw numeric value never needs).
 - Each measured-data file's own "Data loss weight" (config.
   inverse_obs_files_json, falling back to the single legacy
   inverse_data_file/inverse_obs_output_idx/loss_weight_obs fields --
   mirrors codegen.py's own _parse_inverse_obs_files() fallback the same
   way).

Unlike the PDE/BC/IC weights above, neither of these ever needs a
scheduler-phase resync: an observation-file weight is its own separate
loss term, never folded into any scheduler phase's "weights" string
(see that section's own note that observation weights "always sit after
this block ... nothing here needs to touch that tail"), and an initial
guess isn't a loss weight at all -- it only seeds where DeepXDE starts
optimizing that variable from, so there's exactly one copy of it,
always. _inv_var_set_init()/_inv_obs_set_weight() below do keep the
single legacy field (inverse_param_init/loss_weight_obs) in sync for
the FIRST row only, the same "every representation this config might
still be read through stays correct" approach _weight_set() above takes
-- a couple of older call sites (e.g. the primary variable's PDE auto-
substitution default) read that legacy field directly rather than
through inverse_variables_json.
"""
import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional


@dataclass
class SweepParam:
    id: str
    label: str
    category: str
    value_type: str  # "float" | "int" | "categorical"
    choices: Optional[List[str]] = None
    # Suggested default range for the UI to pre-fill a new row with --
    # purely a convenience, never enforced.
    default_min: Optional[float] = None
    default_max: Optional[float] = None
    get_value: Optional[Callable[[Any], Any]] = None
    set_value: Optional[Callable[[Any, Any], None]] = None
    is_available: Optional[Callable[[Any], bool]] = None


# ── Scheduler-phase helpers ───────────────────────────────────────────

def _load_phases(config) -> list:
    try:
        phases = json.loads(config.scheduler_phases) if config.scheduler_phases else []
    except (ValueError, TypeError):
        phases = []
    return phases if isinstance(phases, list) else []


def _save_phases(config, phases: list) -> None:
    config.scheduler_phases = json.dumps(phases)


def sched_active(config) -> bool:
    """Mirrors PINNConfig.validate()'s own _sched_active check (see the
    v56 fix in config.py) -- the scheduler phases are what actually
    drives training whenever this is True, which is the default state
    for every built-in template."""
    return bool(getattr(config, "optimizer_scheduler", False)) and len(_load_phases(config)) > 0


def _phase_available(config, idx: int) -> bool:
    return sched_active(config) and idx < len(_load_phases(config))


def _phase_get(config, idx: int, key: str):
    phases = _load_phases(config)
    if idx < len(phases):
        return phases[idx].get(key)
    return None


def _phase_set(config, idx: int, key: str, value) -> None:
    phases = _load_phases(config)
    if idx < len(phases):
        phases[idx][key] = value
        _save_phases(config, phases)
    # idx out of range: no-op. is_available() guards against this ever
    # being reachable from the UI for a config that doesn't have that
    # many phases.


def num_phases(config) -> int:
    return len(_load_phases(config)) if sched_active(config) else 0


def phase_label(idx: int, config) -> str:
    opt = _phase_get(config, idx, "optimizer") or "?"
    return f"Phase {idx + 1} ({opt})"


# ── Network (layers/neurons) helpers ──────────────────────────────────

def _get_hidden_layers(config) -> int:
    return max(0, len(config.layers) - 2)


def _set_hidden_layers(config, n: int) -> None:
    n = max(0, int(n))
    width = config.layers[1] if len(config.layers) > 2 else 64
    config.layers = [config.layers[0]] + [width] * n + [config.layers[-1]]


def _get_neurons_per_layer(config) -> int:
    return config.layers[1] if len(config.layers) > 2 else 64


def _set_neurons_per_layer(config, w: int) -> None:
    w = max(1, int(w))
    n = max(0, len(config.layers) - 2)
    config.layers = [config.layers[0]] + [w] * n + [config.layers[-1]]


# ── Loss weight helpers ────────────────────────────────────────────────
# See the module docstring's "SCOPE (v2)" section for the full rationale.
# Every function below mirrors a specific piece of codegen.py so there is
# never a second, independently-guessed definition of "what order are
# the loss weights in for this config" anywhere in the codebase.

def _weight_bc_entries(config) -> list:
    """The Boundary Conditions panel rows this config currently has, in
    order -- mirrors codegen.py's own parsing of config.custom_bc_json
    (generate_clean_script's bc_entries / generate_script's
    _custom_bc_entries)."""
    try:
        entries = json.loads(config.custom_bc_json) if config.custom_bc_json else []
    except (ValueError, TypeError):
        entries = []
    return entries if isinstance(entries, list) else []


def _weight_active_ic_outputs(config, n_out: int) -> list:
    """Which output indices get an Initial Condition weight slot, in
    order -- mirrors codegen.py's _clean_active_ic_outputs()/the
    equivalent runtime check in generate_script(): output 0 counts if
    forward_ic_from_file is on, every other output counts if its
    ic_active entry is 'True'. Steady-state problems never have any IC
    slot at all -- callers check config.steady_state themselves before
    calling this, matching codegen.py's own is_steady gating."""
    ic_active_list = (config.ic_active or "").split(",")
    active = []
    for oi in range(n_out):
        if oi == 0 and getattr(config, "forward_ic_from_file", False):
            active.append(oi)
        elif oi < len(ic_active_list) and ic_active_list[oi].strip() == "True":
            active.append(oi)
    return active


def _weight_output_names(config, n_out: int) -> list:
    names = [n.strip() for n in (config.output_names or "").split(",")]
    while len(names) < n_out:
        names.append(f"u{len(names)}")
    return names


def _weight_slot_descriptors(config) -> list:
    """Ordered [{"key", "label", "default_min", "default_max"}, ...] for
    every loss-weight slot THIS config currently has: PDE (one per
    output), then one per Boundary Conditions panel row, then IC (one
    per currently-active output, none at all if steady-state). This is
    the single definition every other function in this section, and
    every SweepParam built from it, resolves against -- see the module
    docstring."""
    n_out = config.num_outputs
    names = _weight_output_names(config, n_out)
    slots = []
    for i in range(n_out):
        slots.append({"key": f"pde_{i}", "label": f"PDE {i + 1} ({names[i]})",
                       "default_min": 0.1, "default_max": 10.0})
    for j, e in enumerate(_weight_bc_entries(config)):
        btype = e.get("type", "dirichlet") if isinstance(e, dict) else "dirichlet"
        comp = e.get("component", 0) if isinstance(e, dict) else 0
        slots.append({"key": f"bc_{j}", "label": f"BC {j + 1} ({btype}, Output {comp})",
                       "default_min": 0.1, "default_max": 10.0})
    if not config.steady_state:
        for i in _weight_active_ic_outputs(config, n_out):
            slots.append({"key": f"ic_{i}", "label": f"IC {i + 1} ({names[i]})",
                           "default_min": 10.0, "default_max": 1000.0})
    return slots


def _weight_index(config, key: str):
    for idx, slot in enumerate(_weight_slot_descriptors(config)):
        if slot["key"] == key:
            return idx
    return None


def _weight_flat_list(config) -> list:
    """The full, ordered flat weight list this config's loss_weights_multi
    SHOULD resolve to -- exactly as many entries as
    _weight_slot_descriptors() returns, read positionally from
    config.loss_weights_multi, falling back to 1.0 for any slot not yet
    present -- the same fallback codegen.py itself uses (its _wm_list
    lookup is always "value if present else 1.0", for every slot kind,
    IC included; 100.0 only ever appears as a GUI new-row convenience
    default, never as codegen's own fallback)."""
    try:
        wm = [v.strip() for v in (config.loss_weights_multi or "").split(",") if v.strip()]
    except AttributeError:
        wm = []
    out = []
    for idx in range(len(_weight_slot_descriptors(config))):
        try:
            out.append(float(wm[idx]) if idx < len(wm) else 1.0)
        except (TypeError, ValueError):
            out.append(1.0)
    return out


def _weight_get(config, key: str):
    idx = _weight_index(config, key)
    if idx is None:
        return None
    return _weight_flat_list(config)[idx]


def _weight_set(config, key: str, value) -> None:
    idx = _weight_index(config, key)
    if idx is None:
        return  # is_available() guards against this being reachable for a
                 # slot this config doesn't currently have.
    values = _weight_flat_list(config)
    values[idx] = float(value)
    config.loss_weights_multi = ",".join(str(v) for v in values)

    # Re-sync every scheduler phase's OWN "weights" string at this same
    # index -- see the module docstring's "SCOPE (v2)" section for why
    # this is required, not optional. A phase whose weights string is
    # shorter than this index (a stale/hand-edited save, or a phase that
    # somehow never matched the current slot count) is left untouched at
    # that phase -- codegen.py's own length check already falls back to
    # the freshly-resolved shared list in that case, which now carries
    # the new value anyway.
    phases = _load_phases(config)
    changed = False
    for ph in phases:
        try:
            ph_w = [float(x) for x in str(ph.get("weights", "")).split(",") if x.strip()]
        except (TypeError, ValueError):
            continue
        if idx < len(ph_w):
            ph_w[idx] = float(value)
            ph["weights"] = ",".join(str(w) for w in ph_w)
            changed = True
    if changed:
        _save_phases(config, phases)


def _weight_available(config) -> bool:
    """See the module docstring's "SCOPE (v2)" section: sweeping a loss
    weight is only unambiguous when there's exactly one active copy of
    the weights for the whole run."""
    return (not sched_active(config)) or bool(getattr(config, "scheduler_same_weights", True))


def _weight_params(config) -> List[SweepParam]:
    params = []
    for slot in _weight_slot_descriptors(config):
        key = slot["key"]
        params.append(SweepParam(
            id=f"weight_{key}", label=slot["label"], category="Loss Weights",
            value_type="float", default_min=slot["default_min"], default_max=slot["default_max"],
            get_value=lambda c, k=key: _weight_get(c, k),
            set_value=lambda c, v, k=key: _weight_set(c, k, v),
            is_available=lambda c: _weight_available(c),
        ))
    return params


# ── Inverse-problem helpers ─────────────────────────────────────────────
# See the module docstring's "SCOPE (v3)" section for the full rationale.

def _inv_vars_list(config) -> list:
    """Ordered list of {"name", "init"} dicts, one per trainable
    (unknown) variable this config currently has -- mirrors codegen.py's
    own _parse_inverse_variables() fallback (a single entry built from
    the legacy inverse_param_name/inverse_param_init fields when
    inverse_variables_json is empty/unparseable)."""
    raw = getattr(config, "inverse_variables_json", "") or ""
    variables = []
    if raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = []
        for i, v in enumerate(parsed or []):
            v = v or {}
            name = str(v.get("name") or f"trainable_variable_{i + 1}").strip() or f"trainable_variable_{i + 1}"
            try:
                init = float(v.get("init", 1.0))
            except (TypeError, ValueError):
                init = 1.0
            variables.append({"name": name, "init": init})
    if not variables:
        variables.append({
            "name": config.inverse_param_name or "trainable_variable_1",
            "init": config.inverse_param_init,
        })
    return variables


def _inv_var_get_init(config, idx: int):
    variables = _inv_vars_list(config)
    return variables[idx]["init"] if idx < len(variables) else None


def _inv_var_set_init(config, idx: int, value) -> None:
    variables = _inv_vars_list(config)
    if idx >= len(variables):
        return  # is_available() guards against this being reachable.
    variables[idx]["init"] = float(value)
    config.inverse_variables_json = json.dumps(variables)
    if idx == 0:
        # Keep the single legacy field in sync too -- see module
        # docstring. Only the primary (first) variable has a legacy
        # field at all.
        config.inverse_param_init = float(value)


def _inv_obs_files_list(config) -> list:
    """Ordered list of {"path", "output_idx", "weight", "custom_expr"}
    dicts, one per measured-data file this config currently has --
    mirrors codegen.py's own _parse_inverse_obs_files() fallback (a
    single entry built from the legacy inverse_data_file/
    inverse_obs_output_idx/loss_weight_obs fields when
    inverse_obs_files_json is empty/unparseable)."""
    raw = getattr(config, "inverse_obs_files_json", "") or ""
    files = []
    if raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = []
        for f in (parsed or []):
            f = f or {}
            try:
                weight = float(f.get("weight", 100.0))
            except (TypeError, ValueError):
                weight = 100.0
            files.append({
                "path": str(f.get("path") or "").strip(),
                "output_idx": f.get("output_idx", 0),
                "weight": weight,
                "custom_expr": f.get("custom_expr", ""),
            })
    if not files:
        files.append({
            "path": config.inverse_data_file or "",
            "output_idx": getattr(config, "inverse_obs_output_idx", 0),
            "weight": config.loss_weight_obs,
            "custom_expr": "",
        })
    return files


def _inv_obs_get_weight(config, idx: int):
    files = _inv_obs_files_list(config)
    return files[idx]["weight"] if idx < len(files) else None


def _inv_obs_set_weight(config, idx: int, value) -> None:
    files = _inv_obs_files_list(config)
    if idx >= len(files):
        return  # is_available() guards against this being reachable.
    files[idx]["weight"] = float(value)
    config.inverse_obs_files_json = json.dumps(files)
    if idx == 0:
        # Keep the single legacy field in sync too -- see module
        # docstring. Only the primary (first) file has a legacy field.
        config.loss_weight_obs = float(value)


def _inv_available(config) -> bool:
    return getattr(config, "problem_type", "Forward") == "Inverse"


def _inv_var_params(config) -> List["SweepParam"]:
    if not _inv_available(config):
        return []
    params = []
    for i, v in enumerate(_inv_vars_list(config)):
        params.append(SweepParam(
            id=f"inv_var_init_{i}", label=f"{v['name']}: initial guess",
            category="Inverse", value_type="float",
            default_min=0.1, default_max=10.0,
            get_value=lambda c, idx=i: _inv_var_get_init(c, idx),
            set_value=lambda c, val, idx=i: _inv_var_set_init(c, idx, val),
            is_available=lambda c: _inv_available(c),
        ))
    return params


def _inv_obs_weight_params(config) -> List["SweepParam"]:
    if not _inv_available(config):
        return []
    params = []
    for i, f in enumerate(_inv_obs_files_list(config)):
        tail = f" ({f['path'].rsplit('/', 1)[-1]})" if f.get("path") else ""
        params.append(SweepParam(
            id=f"inv_obs_weight_{i}", label=f"Obs {i + 1}{tail}: Data loss weight",
            category="Inverse", value_type="float",
            default_min=1.0, default_max=1000.0,
            get_value=lambda c, idx=i: _inv_obs_get_weight(c, idx),
            set_value=lambda c, val, idx=i: _inv_obs_set_weight(c, idx, val),
            is_available=lambda c: _inv_available(c),
        ))
    return params


# ── Static (always-available) registry entries ────────────────────────

_STATIC_PARAMS: List[SweepParam] = [
    SweepParam(
        id="weight_decay", label="Weight decay", category="Network",
        value_type="float", default_min=0.0, default_max=1e-3,
        get_value=lambda c: c.weight_decay,
        set_value=lambda c, v: setattr(c, "weight_decay", float(v)),
        is_available=lambda c: True,
    ),
    SweepParam(
        id="hidden_layers", label="Hidden layers", category="Network",
        value_type="int", default_min=1, default_max=6,
        get_value=_get_hidden_layers,
        set_value=_set_hidden_layers,
        is_available=lambda c: True,
    ),
    SweepParam(
        id="neurons_per_layer", label="Neurons per layer", category="Network",
        value_type="int", default_min=16, default_max=128,
        get_value=_get_neurons_per_layer,
        set_value=_set_neurons_per_layer,
        is_available=lambda c: True,
    ),
    SweepParam(
        id="num_domain", label="Domain points", category="Collocation Points",
        value_type="int", default_min=500, default_max=5000,
        get_value=lambda c: c.num_domain,
        set_value=lambda c, v: setattr(c, "num_domain", int(v)),
        is_available=lambda c: True,
    ),
    SweepParam(
        id="num_boundary", label="Boundary points", category="Collocation Points",
        value_type="int", default_min=100, default_max=1000,
        get_value=lambda c: c.num_boundary,
        set_value=lambda c, v: setattr(c, "num_boundary", int(v)),
        is_available=lambda c: True,
    ),
    SweepParam(
        id="num_initial", label="Initial points", category="Collocation Points",
        value_type="int", default_min=100, default_max=1000,
        get_value=lambda c: c.num_initial,
        set_value=lambda c, v: setattr(c, "num_initial", int(v)),
        # Steady-state problems have no initial condition at all -- this
        # field is unused and would be a no-op sweep for them.
        is_available=lambda c: not c.steady_state,
    ),
    SweepParam(
        id="num_test", label="Test points", category="Collocation Points",
        value_type="int", default_min=500, default_max=5000,
        get_value=lambda c: c.num_test,
        set_value=lambda c, v: setattr(c, "num_test", int(v)),
        is_available=lambda c: True,
    ),
]


def _phase_param(idx: int) -> List[SweepParam]:
    cat = f"Phase {idx + 1}"
    return [
        SweepParam(
            id=f"phase{idx}_lr", label=f"{cat}: Learning rate", category=cat,
            value_type="float", default_min=1e-5, default_max=1e-2,
            get_value=lambda c, i=idx: _phase_get(c, i, "lr"),
            set_value=lambda c, v, i=idx: _phase_set(c, i, "lr", float(v)),
            is_available=lambda c, i=idx: _phase_available(c, i) and
                (_phase_get(c, i, "optimizer") in ("adam", "adamw")),
        ),
        SweepParam(
            id=f"phase{idx}_iterations", label=f"{cat}: Iterations", category=cat,
            value_type="int", default_min=1000, default_max=30000,
            get_value=lambda c, i=idx: _phase_get(c, i, "iterations"),
            set_value=lambda c, v, i=idx: _phase_set(c, i, "iterations", int(v)),
            is_available=lambda c, i=idx: _phase_available(c, i),
        ),
        SweepParam(
            id=f"phase{idx}_optimizer", label=f"{cat}: Optimizer", category=cat,
            value_type="categorical", choices=["adam", "adamw", "lbfgs", "nncg"],
            get_value=lambda c, i=idx: _phase_get(c, i, "optimizer"),
            set_value=lambda c, v, i=idx: _phase_set(c, i, "optimizer", str(v)),
            is_available=lambda c, i=idx: _phase_available(c, i),
        ),
    ]


def available_params(config) -> List[SweepParam]:
    """The full list of sweep-able parameters for the CURRENT state of
    this config -- static entries always included, plus one group of
    phase-scoped entries per currently-configured scheduler phase (none
    at all if the scheduler is off, see module docstring), plus one
    entry per current loss-weight slot (PDE/BC/IC -- none at all when
    per-phase weights are allowed to diverge, see "SCOPE (v2)" above),
    plus one entry per trainable variable's initial guess and one per
    measured-data file's loss weight when this is an Inverse problem
    (see "SCOPE (v3)" above)."""
    params = list(_STATIC_PARAMS)
    for i in range(num_phases(config)):
        params.extend(_phase_param(i))
    params.extend(_weight_params(config))
    params.extend(_inv_var_params(config))
    params.extend(_inv_obs_weight_params(config))
    return [p for p in params if p.is_available(config)]


def get_param(param_id: str, config) -> Optional[SweepParam]:
    for p in available_params(config):
        if p.id == param_id:
            return p
    return None


def categories(config) -> List[str]:
    """Category display order: Network/Collocation first (always
    present), then one group per scheduler phase in training order."""
    seen = []
    for p in available_params(config):
        if p.category not in seen:
            seen.append(p.category)
    return seen

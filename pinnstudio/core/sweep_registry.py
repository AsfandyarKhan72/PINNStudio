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

SCOPE (v1): only scalar, unambiguous fields are registered here --
per-phase learning rate / iterations / optimizer choice, weight decay,
network width/depth, and collocation point counts. Loss weights
(PDE/BC/IC) are deliberately NOT included yet: the active weights for a
problem live in each scheduler phase's own "weights" comma-separated
string, whose column order depends on how many BCs are active, whether
IC is active, and the number of outputs for that specific config --
mapping "the PDE weight" to the right index generically needs its own
careful, separately-tested pass before it's safe to expose, rather than
guessing. A follow-up patch adds it once that mapping is verified.

Phase-scoped entries (learning rate / iterations / optimizer) are only
offered when the Optimizer Scheduler is actually enabled and has that
many phases configured -- which is the state every template is in by
default. Sweeping those with the scheduler off isn't supported in v1;
is_available() for those entries returns False in that case rather than
silently targeting the inert legacy fields.
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
    at all if the scheduler is off, see module docstring)."""
    params = list(_STATIC_PARAMS)
    for i in range(num_phases(config)):
        params.extend(_phase_param(i))
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

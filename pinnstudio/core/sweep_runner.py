"""
Parameter Sweep execution driver (Phase 4): turns a base PINNConfig that
has sweep_enabled=True (and a validated sweep_parameters JSON list) into
a sequence of concrete PINNConfig variants -- per the configured
sweep_mode, "oat" (one-at-a-time) or "grid" (full Cartesian product) --
and runs each one through the exact same subprocess path a normal Solve
uses (pinnstudio.core.runner.run_pinn), so a swept run behaves exactly
like a hand-run Solve with those settings; nothing about training itself
is reimplemented here.

Design: the base config (as built by MainWindow._build_config()) already
holds every field at its current GUI value. sweep_parameters names which
fields to VARY away from that baseline, one combination at a time --
"oat" holds every other swept field at its baseline value while only one
varies per run; "grid" is the Cartesian product across every swept
field's values. Either way, a "baseline" run (the config exactly as
built, completely unmodified) is always included first, so there is
always a same-settings reference point to compare every other run to.

Each concrete config is produced via pinnstudio.core.sweep_registry's
SweepParam.get_value/set_value accessors -- the same ones Phase 2 built
and Phase 3's GUI tab reads from -- so a swept field is always resolved
the one correct way (e.g. a learning rate that actually lives inside the
active Optimizer Scheduler phase), never a legacy/inert field.

Per-run results are intentionally minimal in this v1: each run's final
training loss (parsed from the "Final loss: ..." line every generated
script already prints to stdout) and, when the problem has Error
Analysis reference data configured, the last reported L2 relative error
(read back from that run's own error_analysis/error_metrics*.txt, which
the generated script already writes -- see codegen.py). A richer
results view (saved plots per run, CSV export, etc.) is Phase 5.
"""
import copy
import glob
import itertools
import json
import os
import re
import time

import numpy as np

from pinnstudio.core import sweep_registry as reg
from pinnstudio.core.runner import run_pinn

_FINAL_LOSS_RE = re.compile(r"Final loss:\s*([0-9.eE+-]+)")


def expand_values(entry):
    """Returns the concrete list of values a single sweep_parameters
    entry represents -- its explicit "values" list for mode "list", or
    `n` points spanning [min, max] for "linear"/"log". Mirrors exactly
    the shapes PINNConfig.validate() already accepts (see config.py)."""
    mode = entry.get("mode")
    if mode == "list":
        return list(entry.get("values", []))
    n = int(entry.get("n", 5))
    lo, hi = float(entry["min"]), float(entry["max"])
    if mode == "linear":
        pts = np.linspace(lo, hi, n)
    elif mode == "log":
        pts = np.logspace(np.log10(lo), np.log10(hi), n)
    else:
        raise ValueError(f"unrecognized sweep mode: {mode!r}")
    return [float(v) for v in pts]


def expand_params(base_config):
    """Returns [(SweepParam, [concrete values...]), ...] for every entry
    in base_config.sweep_parameters that still resolves against this
    config, in the same order build_runs() consumes them (and the same
    order a Phase 5 results plot needs to regroup run results back by
    which parameter produced them). Shared by build_runs() below and by
    MainWindow's sweep-results plot (pinnstudio/ui/main_window.py) so
    both always agree on exactly which values a sweep actually used."""
    entries = json.loads(base_config.sweep_parameters or "[]")
    per_param = []
    for e in entries:
        param = reg.get_param(e.get("id"), base_config)
        if param is None:
            continue
        values = expand_values(e)
        if param.value_type == "int":
            values = [int(round(float(v))) for v in values]
        per_param.append((param, values))
    return per_param


def build_runs(base_config):
    """Returns a list of (label, config) pairs for every run this sweep
    implies -- always starting with ("baseline", <base_config exactly as
    given, deep-copied>). Parameter entries whose id no longer resolves
    against this config (e.g. referencing a since-removed scheduler
    phase) are silently skipped here -- PINNConfig.validate() is what's
    responsible for catching that and should already have been called
    before this runs."""
    runs = [("baseline", copy.deepcopy(base_config))]
    per_param = expand_params(base_config)
    if not per_param:
        return runs

    if base_config.sweep_mode == "grid":
        for combo in itertools.product(*(vals for _, vals in per_param)):
            cfg = copy.deepcopy(base_config)
            parts = []
            for (param, _), v in zip(per_param, combo):
                param.set_value(cfg, v)
                parts.append(f"{param.label}={v}")
            runs.append((", ".join(parts), cfg))
    else:  # "oat" -- every other swept field stays at the baseline value
        for param, values in per_param:
            for v in values:
                cfg = copy.deepcopy(base_config)
                param.set_value(cfg, v)
                runs.append((f"{param.label}={v}", cfg))
    return runs


def _read_last_l2(cfg, not_before):
    """Best-effort read of the error_analysis/error_metrics*.txt THIS
    run just wrote -- returns the last row's L2_relative column, or None
    if there's nothing to read (no Error Analysis reference data
    configured for this template at all, in which case nothing is even
    attempted) or nothing NEW to read.

    `not_before` (a time.time() timestamp taken just before this run's
    subprocess started) is required and strictly enforced: the same
    save_dir/error_analysis folder is reused across every run in a
    sweep (most configs leave save_dir blank, which falls back to the
    shared /tmp/error_analysis for every run), so a file left over from
    an earlier run -- or from a completely unrelated Solve run earlier
    in the session -- sitting there un-rewritten would otherwise be
    silently misreported as this run's own result. Only a file whose
    mtime is at or after `not_before` is trusted; anything older is
    treated the same as the file not existing."""
    if not getattr(cfg, "ea_files", None) or cfg.ea_files == "[]":
        return None  # this config has no Error Analysis reference data -- nothing will be written, don't look
    save_dir = getattr(cfg, "save_dir", "") or "/tmp"
    ea_dir = os.path.join(save_dir, "error_analysis")
    candidates = [
        p for p in glob.glob(os.path.join(ea_dir, "error_metrics*.txt"))
        if os.path.exists(p) and os.path.getmtime(p) >= not_before
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    try:
        with open(candidates[-1]) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) < 2:
            return None
        last_row = lines[-1].split(",")
        return float(last_row[1])
    except (OSError, IndexError, ValueError):
        return None


def run_sweep(base_config, on_run_start=None, on_output=None, on_run_done=None, should_stop=None):
    """Sequentially executes every run from build_runs(). Runs are
    deliberately sequential, not parallel: these are real training
    subprocesses (potentially GPU-bound), and running several at once
    would silently oversubscribe whatever GPU/CPU resources are
    available with no way for the user to control it from here.

    on_run_start(i, total, label) fires before each run starts.
    on_output(i, total, line) fires per stdout line from that run (so a
    caller can stream progress the same way a normal Solve does).
    on_run_done(i, total, label, result) fires once a run finishes,
    where result = {"status": "done"|"error", "final_loss": float|None,
    "l2_relative": float|None}.
    should_stop(), if given, is checked before each run and stops the
    sweep early (without running the remaining combinations) if it
    returns True.

    Returns the full list of (label, result) pairs collected so far.
    """
    runs = build_runs(base_config)
    total = len(runs)
    results = []
    for i, (label, cfg) in enumerate(runs):
        if should_stop and should_stop():
            break
        if on_run_start:
            on_run_start(i, total, label)

        log_lines = []

        def _collect(line, _i=i, _total=total):
            log_lines.append(line)
            if on_output:
                on_output(_i, _total, line)

        _run_started_at = time.time()
        status = run_pinn(cfg, on_output=_collect, set_process=None)

        final_loss = None
        for m in _FINAL_LOSS_RE.finditer("\n".join(log_lines)):
            try:
                final_loss = float(m.group(1))
            except ValueError:
                pass  # keep the last successfully-parsed value, if any

        result = {
            "status": "done" if status == "DONE" else "error",
            "final_loss": final_loss,
            "l2_relative": _read_last_l2(cfg, _run_started_at),
        }
        results.append((label, result))
        if on_run_done:
            on_run_done(i, total, label, result)
    return results

"""
Parameter Sweep execution driver (Phase 4): turns a base PINNConfig that
has sweep_enabled=True (and a validated sweep_parameters JSON list) into
a sequence of concrete PINNConfig variants -- per the configured
sweep_mode, "oat" (one-at-a-time), "grid" (full Cartesian product, aka
COMSOL's "All combinations"), or "zip" (COMSOL's "Specified
combinations" -- every parameter's Nth value run together) -- and runs
each one through the exact same subprocess path a normal Solve uses
(pinnstudio.core.runner.run_pinn), so a swept run behaves exactly like a
hand-run Solve with those settings; nothing about training itself is
reimplemented here.

Design: the base config (as built by MainWindow._build_config()) already
holds every field at its current GUI value. sweep_parameters names which
fields to VARY away from that baseline, one combination at a time --
"oat" holds every other swept field at its baseline value while only one
varies per run; "grid" is the Cartesian product across every swept
field's values; "zip" pairs up each parameter's values position-by-
position (P1's 1st value with P2's 1st value, P1's 2nd with P2's 2nd,
...), which only makes sense when every swept parameter has the same
number of values -- PINNConfig.validate() checks that before this runs.
A "baseline" run (the config exactly as built, unmodified) can come
first as a same-settings reference point (sweep_include_baseline; by
default only for One-at-a-time with 2+ parameters and Bayesian search,
where the other runs are measured against it -- see includes_baseline).

Each concrete config is produced via pinnstudio.core.sweep_registry's
SweepParam.get_value/set_value accessors -- the same ones Phase 2 built
and Phase 3's GUI tab reads from -- so a swept field is always resolved
the one correct way (e.g. a learning rate that actually lives inside the
active Optimizer Scheduler phase), never a legacy/inert field.

Per-run results: each run's final training loss (parsed from the
"Final loss: ..." line every generated script already prints to
stdout) and, when the problem has Error Analysis reference data
configured, the Error Analysis metrics averaged over the reference
times (read back from that run's own error_analysis/error_metrics*.txt,
which the generated script already writes -- see codegen.py).

OUTPUT ORGANIZATION: the sweep's folder is named like a normal run's,
"<label>_sweep__YYYY-MM-DD_HH-MM-SS", and every run gets its own
descriptively-named folder under it -- e.g. "run_02__Phase-1-Iterations_2000"
-- with its save_dir pointed there, so the SAME save
pipeline a normal Solve already uses (model checkpoint, loss/solution
plot per plot_type, Error Analysis figures) writes into that run's own
folder automatically; nothing about saving is reimplemented here either.
This deliberately overrides whatever base_config.save_dir (the Setup
tab's own "Save results to") happens to be -- reusing that single path
for every run in the sweep would make every run collide into the same
folder. base_config.sweep_export_mode == "custom" additionally
overrides plot_type/plot_output_idx/plot_custom_expr/plot_custom_label/
export_t_steps for every run from the matching sweep_plot_* fields,
uniformly across the whole sweep -- "same_as_setup" (the default) makes
no change here at all. A sweep_manifest.json, sweep_summary.csv and a
sweep_error_vs_parameter.png summary plot (error and final loss against
the swept value(s)) are written into the sweep's root folder once the
sweep finishes (or is cancelled -- whatever ran so far is still
recorded), so the results are readable later without reopening the GUI.
"""
import copy
import csv
import datetime
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
_FINAL_LOSS_UNW_RE = re.compile(r"Final loss \(unweighted\):\s*([0-9.eE+-]+)")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._=-]+")


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


def includes_baseline(base_config):
    """Whether the sweep starts with a baseline run (the current
    settings, unchanged). sweep_include_baseline "on"/"off" decide;
    "auto" (the default) runs it only where the other runs are measured
    against it: One-at-a-time with 2+ parameters, and Bayesian search
    (its first point)."""
    v = str(getattr(base_config, "sweep_include_baseline", "auto"))
    if v in ("on", "True"):
        return True
    if v in ("off", "False"):
        return False
    mode = getattr(base_config, "sweep_mode", "oat")
    if mode == "bayes":
        return True
    try:
        n = len(json.loads(base_config.sweep_parameters or "[]"))
    except ValueError:
        n = 0
    return mode == "oat" and n >= 2


def _run_label(parts):
    return ", ".join(f"{p.label}={v}" for p, v in parts)


def build_runs_detailed(base_config):
    """Every run this sweep implies, as dicts {"label", "config",
    "values": {param label: value}, "group": index of the parameter a
    One-at-a-time run varies (None otherwise / for the baseline),
    "baseline": bool}. The baseline (if includes_baseline) comes first."""
    per_param = expand_params(base_config)
    runs = []
    if includes_baseline(base_config) or not per_param:
        runs.append({"label": "baseline", "config": copy.deepcopy(base_config), "group": None, "baseline": True,
                     "values": {p.label: p.shown(base_config) for p, _ in per_param}})
    if not per_param:
        return runs

    def _variant(parts, group=None):
        cfg = copy.deepcopy(base_config)
        for param, v in parts:
            param.set_value(cfg, v)
        values = {p.label: p.shown(base_config) for p, _ in per_param}
        values.update({p.label: v for p, v in parts})
        return {"label": _run_label(parts), "config": cfg, "group": group, "baseline": False, "values": values}

    if base_config.sweep_mode == "grid":
        for combo in itertools.product(*(vals for _, vals in per_param)):
            runs.append(_variant(list(zip((p for p, _ in per_param), combo))))
    elif base_config.sweep_mode == "zip":
        # "Specified combinations" (COMSOL's term): every parameter's
        # Nth value, together, as one combination -- P1=[1,2] & P2=[10,20]
        # gives exactly 2 runs, (1,10) and (2,20), never grid mode's 4.
        # zip() truncates to the shortest list if lengths somehow differ
        # (PINNConfig.validate() already rejects that).
        for combo in zip(*(vals for _, vals in per_param)):
            runs.append(_variant(list(zip((p for p, _ in per_param), combo))))
    else:  # "oat" -- every other swept field stays at the baseline value
        for k, (param, values) in enumerate(per_param):
            for v in values:
                runs.append(_variant([(param, v)], group=k))
    return runs


def build_runs(base_config):
    """[(label, config), ...] for every run this sweep implies -- the
    baseline first when includes_baseline(base_config). Parameter
    entries whose id no longer resolves against this config (e.g. a
    since-removed scheduler phase) are skipped; PINNConfig.validate()
    catches those before a sweep starts."""
    return [(r["label"], r["config"]) for r in build_runs_detailed(base_config)]


# ── Bayesian search (Round 75) ──────────────────────────────────────────

def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)



def _in_space(dim, v):
    """Whether value v lies in a Bayesian-search dimension."""
    if dim.kind == "categorical":
        return str(v) in dim.choices
    try:
        return float(dim.low) <= float(v) <= float(dim.high)
    except (TypeError, ValueError):
        return False

class BayesRuns:
    """Run source for sweep_mode "bayes": the baseline (unless turned off),
    then one run at a
    time chosen by pinnstudio.core.bayes_opt from the results so far
    (DeepXDE's Helmholtz HPO example, gp_minimize with expected
    improvement). next_run() gives (label, config); tell(result) records
    the finished run's objective."""

    def __init__(self, base_config):
        from pinnstudio.core.bayes_opt import BayesOptimizer, Dim
        self.base = base_config
        self.params, dims = [], []
        for e in json.loads(base_config.sweep_parameters or "[]"):
            param = reg.get_param(e.get("id"), base_config)
            if param is None:
                continue
            if e.get("mode") in ("linear", "log") and param.value_type in ("float", "int"):
                dims.append(Dim(param.value_type, float(e["min"]), float(e["max"]), log=e.get("mode") == "log"))
            else:
                vals = e.get("values") or expand_values(e)
                dims.append(Dim("categorical", choices=[str(v) for v in vals]))
            self.params.append(param)
        self.dims = dims
        self.total = max(2, int(getattr(base_config, "sweep_bayes_calls", 20) or 20))
        self.objective = getattr(base_config, "sweep_bayes_objective", "loss")
        self.opt = BayesOptimizer(dims, n_initial=int(getattr(base_config, "sweep_bayes_initial", 5) or 5),
                                  seed=int(getattr(base_config, "sweep_bayes_seed", 0) or 0))
        self.history = []          # (label, point, objective)
        self._pending = None
        self.with_baseline = includes_baseline(base_config)

    def _cast(self, param, v):
        if param.value_type == "int":
            return int(round(float(v)))
        if param.value_type == "float":
            return float(v)
        return str(v)

    def values(self, point):
        return {p.label: self._cast(p, v) for p, v in zip(self.params, point)}

    def next_run(self, i):
        if i == 0 and self.with_baseline:
            point = []
            for p, d in zip(self.params, self.dims):
                v = p.get_value(self.base)
                point.append(str(v) if d.kind == "categorical" else (v if v is not None else d.low))
            self._pending = point
            # The baseline teaches the model only if it lies in the search
            # space: an activation not among the choices, or a value outside
            # Min..Max, would be recorded as the nearest choice or edge.
            self._pending_in_space = all(_in_space(d, v) for d, v in zip(self.dims, point))
            return "baseline", copy.deepcopy(self.base)
        self._pending_in_space = True
        point = self.opt.ask()
        cfg = copy.deepcopy(self.base)
        parts = []
        for p, v in zip(self.params, point):
            cv = self._cast(p, v)
            p.set_value(cfg, cv)
            parts.append(f"{p.label}={_fmt(cv)}")
        self._pending = point
        return ", ".join(parts), cfg

    def tell(self, label, result):
        key = {"loss": "final_loss", "loss_unweighted": "final_loss_unweighted"}.get(self.objective, "l2_relative")
        val = result.get(key) if result.get("status") == "done" else None
        if getattr(self, "_pending_in_space", True):
            self.opt.tell(self._pending, val)
        self.history.append((label, list(self._pending), val))

    def best(self):
        ok = [(v, lab, pt) for lab, pt, v in self.history if v is not None and np.isfinite(v)]
        return min(ok, key=lambda r: r[0]) if ok else None

    def write_outputs(self, sweep_root):
        """bayes_search.csv (every run, its objective and the best so far)
        and bayes_convergence.png (best objective vs run)."""
        try:
            with open(os.path.join(sweep_root, "bayes_search.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Run"] + [p.label for p in self.params] + [self._objective_label(), "Best so far"])
                best = float("inf")
                for k, (_lab, pt, v) in enumerate(self.history):
                    if v is not None and np.isfinite(v):
                        best = min(best, v)
                    w.writerow([k] + [_fmt(x) for x in pt] + ["" if v is None else v,
                                                              "" if not np.isfinite(best) else best])
        except OSError:
            pass
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            matplotlib.rcParams["axes.formatter.use_mathtext"] = True
            vals = [v if v is not None and np.isfinite(v) else np.nan for _l, _p, v in self.history]
            best = np.fmin.accumulate(np.where(np.isnan(vals), np.inf, vals))
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.semilogy(range(len(vals)), vals, "o", color="#adb5bd", label="run")
            ax.semilogy(range(len(best)), np.where(np.isinf(best), np.nan, best), "-", color="#1c7ed6",
                        linewidth=2, label="best so far")
            ax.set_xlabel("run"); ax.set_ylabel(self._objective_label())
            ax.set_title("Bayesian search -- convergence"); ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(sweep_root, "bayes_convergence.png"), dpi=150)
            plt.close(fig)
        except Exception:
            pass

    def _objective_label(self):
        return {"loss": "Final loss", "loss_unweighted": "Final loss (unweighted)"}.get(self.objective, "L2 relative error")


# ── Per-run output organization ────────────────────────────────────────

def _slugify(text, maxlen=60):
    """A filesystem-safe folder-name fragment: letters, digits, dots and
    dashes kept, anything else (spaces, colons, commas, parens, ...)
    collapsed to one dash -- "Phase 1: Iterations" -> "Phase-1-Iterations"."""
    text = re.sub(r"[^A-Za-z0-9.+-]+", "-", str(text).strip())
    text = re.sub(r"-+", "-", text).strip("-")
    # no trailing dot or dash: Windows drops a trailing dot from a folder name
    return ((text or "run")[:maxlen].rstrip(".-")) or "run"


def sweep_root_dir(base_config, started_at=None):
    """The folder this sweep's runs are organized under, named like a
    normal run's folder: <sweep_save_dir>/<label>_sweep__YYYY-MM-DD_HH-MM-SS
    (<label> from base_config.sweep_label -- the loaded example or the
    problem's shape -- or just "sweep__<timestamp>" when it is blank).
    `started_at` (a datetime, defaulting to now) fixes the timestamp so
    every run of the same run_sweep() call lands under the same root,
    while a later sweep gets its own. Blank sweep_save_dir falls back to
    ~/PINNStudio_Results/parameter_sweep_results."""
    root = (getattr(base_config, "sweep_save_dir", "") or "").strip()
    if not root:
        root = os.path.join(os.path.expanduser("~"), "PINNStudio_Results", "parameter_sweep_results")
    stamp = (started_at or datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", (getattr(base_config, "sweep_label", "") or "").strip()).strip("_")
    return os.path.join(root, f"{label}_sweep__{stamp}" if label else f"sweep__{stamp}")


def run_folder_name(index, label, total=None):
    """One run's subfolder under sweep_root_dir(): numbered (so the
    order survives alphabetical sorting and two runs never collide) and
    named after what the run set -- "run_00__baseline",
    "run_02__Phase-1-Iterations_2000", "run_05__Learning-rate_0.001__Width_64"
    (one "name_value" part per swept parameter). The number has 2 digits,
    more when a sweep has 100+ runs."""
    width = max(2, len(str(max(0, (total or 0) - 1))))
    if label == "baseline":
        return f"run_{index:0{width}d}__baseline"
    pieces, depth, cur = [], 0, ""
    for ch in label:  # split "A=1, B (x, y)=2" at the commas outside parentheses
        depth += (ch in "([") - (ch in ")]")
        if ch == "," and depth == 0:
            pieces.append(cur.strip()); cur = ""
        else:
            cur += ch
    pieces.append(cur.strip())
    parts = []
    for part in pieces:
        name, _, value = part.rpartition("=")
        parts.append(f"{_slugify(name, 40)}_{_slugify(value, 24)}" if name else _slugify(part))
    # Whole name_value parts only (a cut in the middle would show a wrong
    # value), and "__more" when some did not fit; the log and
    # sweep_summary.csv always have every value.
    kept, n = [], 0
    for part in parts:
        if kept and n + 2 + len(part) > 100:
            kept.append("more")
            break
        kept.append(part[:100])
        n += len(part) + 2
    return f"run_{index:0{width}d}__" + "__".join(kept)


_PARAM_LINE_RE = re.compile(r"^\s*\[([^\]]+)\] Iter \d+: ([-+0-9.eE]+)")


def _read_inverse_params(cfg, log_lines, not_before):
    """An inverse run's parameters: {name: {"value": last value, "true":
    true value or None}}. From each parameter's convergence file in the
    run's folder (written during training), else the last value the run
    printed. {} for a forward run."""
    if getattr(cfg, "problem_type", "Forward") != "Inverse":
        return {}
    # The names the script uses (cleaned up to Python names, made unique),
    # which are the names of its convergence files and log lines.
    from pinnstudio.core.codegen import _parse_inverse_variables
    variables = [{"name": n, "true": t} for n, _init, t in _parse_inverse_variables(cfg)]
    out = {}
    printed = {}
    for line in log_lines:
        m = _PARAM_LINE_RE.match(line)
        if m:
            try:
                printed[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    for v in variables:
        name = str(v.get("name", "")).strip()
        if not name:
            continue
        value = None
        path = os.path.join(cfg.save_dir or "", f"{name}_convergence.txt")
        try:
            if os.path.getmtime(path) >= not_before - 1:
                with open(path, encoding="utf-8") as f:
                    last = [l for l in f if l.strip() and l[0].isdigit()][-1]
                value = float(last.replace(",", " ").split()[-1])
        except (OSError, IndexError, ValueError):
            value = None
        if value is None:
            value = printed.get(name)
        true = v.get("true")
        try:
            true = float(true) if true not in (None, "") else None
        except (TypeError, ValueError):
            true = None
        if value is not None:
            out[name] = {"value": value, "true": true}
    return out


def _param_error(p):
    """Relative error (%) of an inferred parameter, or None without a true value."""
    if p.get("true") in (None, 0):
        return None
    return abs(p["value"] - p["true"]) / abs(p["true"]) * 100.0


def _read_run_metrics(cfg, not_before):
    """Best-effort read of the error_analysis/error_metrics*.txt files THIS
    run just wrote (one per output group), averaged over every row --
    i.e. over all the reference times set in Error Analysis (and over the
    output groups, if several): {"L2_relative": .., "MSE": .., "Max_error":
    .., "Mean_abs_error": .., "n_times": number of reference times}. Only
    the metrics ticked in Error Analysis are in the files. A steady
    problem has one row, so its value is just that row's. {} if there's
    nothing to read -- no Error Analysis reference configured (nothing is
    even looked for) or nothing NEW.

    `not_before` (a time.time() timestamp taken just before this run's
    subprocess started) is strictly enforced: only files whose mtime is
    at or after it are trusted, so a file left over from an earlier run
    or an unrelated Solve in the same folder is never misreported as
    this run's result."""
    if (not getattr(cfg, "ea_files", None) or cfg.ea_files == "[]") and \
            getattr(cfg, "ea_ref_mode", "files") != "formula":
        return {}
    save_dir = getattr(cfg, "save_dir", "") or "/tmp"
    ea_dir = os.path.join(save_dir, "error_analysis")
    files = [
        p for p in glob.glob(os.path.join(ea_dir, "error_metrics*.txt"))
        if os.path.exists(p) and os.path.getmtime(p) >= not_before
    ]
    cols, times = {}, set()
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except OSError:
            continue
        if len(lines) < 2:
            continue
        head = [h.strip() for h in lines[0].split(",")]
        for line in lines[1:]:
            for h, v in zip(head, line.split(",")):
                try:
                    x = float(v)
                except ValueError:
                    continue
                if h == "t":
                    times.add(x)
                elif np.isfinite(x):
                    cols.setdefault(h, []).append(x)
    if not cols:
        return {}
    out = {h: float(np.mean(v)) for h, v in cols.items()}
    out["n_times"] = max(1, len(times))
    return out


def _read_last_l2(cfg, not_before):
    """The run's relative L2 error (mean over the reference times), or None."""
    return _read_run_metrics(cfg, not_before).get("L2_relative")


def _apply_run_output_settings(cfg, base_config, run_dir):
    """Points this run's save_dir at its own folder, and -- only when
    the sweep's export settings are "custom" -- overrides the plot/export
    fields a normal Solve already reads, from the matching sweep_plot_*
    fields on base_config. "same_as_setup" (the default) leaves whatever
    the Setup tab itself configured on cfg completely untouched, so a
    sweep behaves exactly like today unless the user explicitly opts
    into custom per-sweep export settings."""
    cfg.save_dir = run_dir
    if getattr(base_config, "sweep_export_mode", "same_as_setup") == "custom":
        cfg.plot_type = base_config.sweep_plot_type
        cfg.plot_output_idx = base_config.sweep_plot_output_idx
        cfg.plot_custom_expr = base_config.sweep_plot_custom_expr
        cfg.plot_custom_label = base_config.sweep_plot_custom_label
        cfg.export_t_steps = base_config.sweep_export_t_steps


def run_sweep(base_config, on_run_start=None, on_output=None, on_run_done=None,
              should_stop=None, on_sweep_root=None, set_process=None):
    """Sequentially executes every run from build_runs(). Runs are
    deliberately sequential, not parallel: these are real training
    subprocesses (potentially GPU-bound), and running several at once
    would silently oversubscribe whatever GPU/CPU resources are
    available with no way for the user to control it from here.

    Every run's save_dir is pointed at its own folder under
    sweep_root_dir(base_config) before it starts (see that function's
    and _apply_run_output_settings()'s docstrings) -- so each run's
    model/plots/Error Analysis figures land somewhere findable
    afterward, under a name that says which parameter(s)/value(s) that
    run actually used, instead of overwriting the same shared location
    every run the way leaving save_dir alone would.

    on_sweep_root(root_path) fires once, before the first run starts,
    with the sweep's root folder (absolute path) -- so a caller (e.g.
    the GUI) can tell the user where results are landing right away,
    not only once everything is done.
    on_run_start(i, total, label) fires before each run starts.
    on_output(i, total, line) fires per stdout line from that run (so a
    caller can stream progress the same way a normal Solve does).
    on_run_done(i, total, label, result) fires once a run finishes,
    where result = {"status": "done"|"error", "final_loss": float|None,
    "l2_relative": float|None, "save_dir": str}.
    should_stop(), if given, is checked before each run and stops the
    sweep early (without running the remaining combinations) if it
    returns True -- whatever ran before that point is still written to
    sweep_manifest.json/sweep_summary.csv, not discarded.
    set_process(process), if given, is passed straight through to each
    run_pinn() call -- exactly the same hook a normal single Solve uses
    (see runner.run_pinn/SolverThread) to let a caller hard-kill the
    CURRENTLY RUNNING subprocess immediately (e.g. the GUI's Stop
    button), rather than only being able to stop the sweep from
    starting its NEXT run via should_stop().

    Returns the full list of (label, result) pairs collected so far.
    """
    bayes = BayesRuns(base_config) if getattr(base_config, "sweep_mode", "") == "bayes" else None
    runs = None if bayes else build_runs_detailed(base_config)
    total = bayes.total if bayes else len(runs)
    started_at = datetime.datetime.now()
    sweep_root = sweep_root_dir(base_config, started_at)
    os.makedirs(sweep_root, exist_ok=True)
    if on_sweep_root:
        on_sweep_root(sweep_root)

    manifest = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "sweep_mode": base_config.sweep_mode,
        "sweep_parameters": base_config.sweep_parameters,
        "runs": [],
    }
    results = []
    for i in range(total):
        if should_stop and should_stop():
            break
        if bayes:
            label, cfg = bayes.next_run(i)
            info = {"values": bayes.values(bayes._pending), "group": None, "baseline": label == "baseline"}
        else:
            info = runs[i]
            label, cfg = info["label"], info["config"]
        # Numbered from 1 for the swept runs; the baseline, when there is
        # one, is run 0.
        number = i if (bayes.with_baseline if bayes else runs[0]["baseline"]) else i + 1
        run_dir = os.path.join(sweep_root, run_folder_name(number, label, total + 1))
        _apply_run_output_settings(cfg, base_config, run_dir)
        if on_run_start:
            on_run_start(i, total, label)

        log_lines = []

        def _collect(line, _i=i, _total=total):
            log_lines.append(line)
            if on_output:
                on_output(_i, _total, line)

        _run_started_at = time.time()
        status = run_pinn(cfg, on_output=_collect, set_process=set_process)

        final_loss = None
        for m in _FINAL_LOSS_RE.finditer("\n".join(log_lines)):
            try:
                final_loss = float(m.group(1))
            except ValueError:
                pass  # keep the last successfully-parsed value, if any

        final_loss_unw = None
        for m in _FINAL_LOSS_UNW_RE.finditer("\n".join(log_lines)):
            try:
                final_loss_unw = float(m.group(1))
            except ValueError:
                pass

        metrics = _read_run_metrics(cfg, _run_started_at)
        result = {
            "status": "done" if status == "DONE" else "error",
            "final_loss": final_loss,
            "final_loss_unweighted": final_loss_unw,
            "l2_relative": metrics.get("L2_relative"),
            "metrics": metrics,
            "params": _read_inverse_params(cfg, log_lines, _run_started_at),
            "values": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                       for k, v in info["values"].items()},
            "group": info["group"],
            "baseline": info["baseline"],
            "save_dir": run_dir,
        }
        results.append((label, result))
        manifest["runs"].append({"index": i, "label": label, **result})
        if bayes:
            bayes.tell(label, result)
        if on_run_done:
            on_run_done(i, total, label, result)

    if bayes:
        b = bayes.best()
        if b:
            manifest["bayes_best"] = {"label": b[1], "objective": b[0], "objective_name": bayes.objective}
            if on_output:
                on_output(total - 1, total, f"Bayesian search best ({bayes._objective_label()} {b[0]:.4e}): {b[1]}")
        bayes.write_outputs(sweep_root)
    _write_sweep_manifest(sweep_root, manifest)
    _write_sweep_summary_csv(sweep_root, results)
    if not bayes:
        try:
            write_sweep_summary_plot(sweep_root, base_config, results)
        except Exception as exc:  # a summary figure must never fail a finished sweep
            if on_output:
                on_output(max(0, len(results) - 1), total, f"(sweep summary plot skipped: {exc})")
    return results


def _write_sweep_manifest(sweep_root, manifest):
    try:
        with open(os.path.join(sweep_root, "sweep_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except OSError:
        pass  # best-effort -- a sweep's real results/models are already
              # safely on disk per-run regardless of this summary file


_METRIC_COLS = (("L2_relative", "L2 relative error"), ("MSE", "MSE"), ("Max_error", "Max error"),
                ("Mean_abs_error", "Mean abs error"))


def _write_sweep_summary_csv(sweep_root, results):
    """sweep_summary.csv in the sweep's own folder: one row per run --
    its label, every swept parameter's value, status, final loss, the
    Error Analysis metrics it reported, and its folder."""
    params = []
    for _label, r in results:
        for k in (r.get("values") or {}):
            if k not in params:
                params.append(k)
    metrics = [(k, name) for k, name in _METRIC_COLS
               if any((r.get("metrics") or {}).get(k) is not None for _l, r in results)]
    if not metrics:
        metrics = [("L2_relative", "L2 relative error")]
    pnames = []
    for _label, r in results:
        for k in (r.get("params") or {}):
            if k not in pnames:
                pnames.append(k)
    pcols = []
    for k in pnames:
        pcols.append((k, "value", k))
        if any((r.get("params") or {}).get(k, {}).get("true") is not None for _l, r in results):
            pcols.append((k, "error", f"{k} error (%)"))
    try:
        with open(os.path.join(sweep_root, "sweep_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Run"] + params + ["Status", "Final loss", "Final loss (unweighted)"] + [n for _k, n in metrics]
                            + [h for _k, _w, h in pcols] + ["Folder"])
            for label, result in results:
                m = result.get("metrics") or {}
                if result.get("l2_relative") is not None:
                    m = {**m, "L2_relative": result["l2_relative"]}
                vals = result.get("values") or {}
                writer.writerow(
                    [label] + [vals.get(p, "") for p in params]
                    + [result.get("status"), result.get("final_loss") if result.get("final_loss") is not None else "",
                       result.get("final_loss_unweighted") if result.get("final_loss_unweighted") is not None else ""]
                    + [m.get(k, "") for k, _n in metrics]
                    + [_param_cell(result, k, what) for k, what, _h in pcols]
                    + [os.path.basename(result.get("save_dir", ""))])
    except OSError:
        pass


def _param_cell(result, name, what):
    p = (result.get("params") or {}).get(name)
    if not p:
        return ""
    if what == "value":
        return f"{p['value']:.6g}"
    e = _param_error(p)
    return "" if e is None else f"{e:.4g}"


# ── Summary plot: error vs swept parameter ─────────────────────────────

_PARAM_COLORS = ("#c2255c", "#7048e8", "#0c8599", "#e67700")

_METRIC_STYLE = {"L2_relative": ("Relative $L_2$ error", "o", "#1c7ed6"),
                 "MSE": ("MSE", "s", "#e8590c"),
                 "Max_error": ("Max error", "^", "#2f9e44")}


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def write_sweep_summary_plot(sweep_root, base_config, results):
    """sweep_error_vs_parameter.png in the sweep's folder: the Error
    Analysis error(s) of every run against the swept value (top row, log
    scale; the metrics ticked in Error Analysis, each the mean over the
    run's reference times) and the final training
    loss (bottom row). One column per parameter for One-at-a-time; for
    All/Specified combinations the first parameter is on the x axis and
    the other parameters' values give one line each. Numeric values use
    a numeric axis (log when they span 100x or more), others a
    categorical one. The baseline, when run, is a separate marker.
    Returns the path, or None if there was nothing to plot."""
    import matplotlib
    matplotlib.use("Agg")
    with matplotlib.rc_context({"axes.formatter.use_mathtext": True, "axes.formatter.limits": (-2, 5)}):
        return _summary_plot(sweep_root, base_config, results)


def _summary_plot(sweep_root, base_config, results):
    import matplotlib.pyplot as plt

    done = [(lab, r) for lab, r in results if r.get("status") == "done"]
    per_param = expand_params(base_config)
    if not done or not per_param:
        return None
    names = [p.label for p, _ in per_param]
    mode = getattr(base_config, "sweep_mode", "oat")
    metrics = [k for k in _METRIC_STYLE if any(_num((r.get("metrics") or {}).get(k)) is not None for _l, r in done)]
    has_loss = any(_num(r.get("final_loss")) is not None for _l, r in done)
    pkeys = []
    for _l, r in done:
        for k, p in (r.get("params") or {}).items():
            if _param_error(p) is not None and f"param:{k}" not in pkeys:
                pkeys.append(f"param:{k}")
    rows = (([("param", pkeys)] if pkeys else []) + ([("err", metrics)] if metrics else [])
            + ([("loss", ["final_loss"])] if has_loss else []))
    if not rows:
        return None

    def _y(r, key):
        if key.startswith("param:"):
            p = (r.get("params") or {}).get(key[6:])
            e = _param_error(p) if p else None
            return e if e and e > 0 else None
        return _num(r.get("final_loss")) if key == "final_loss" else _num((r.get("metrics") or {}).get(key))

    # Panels: (x parameter index, [(series label, [(x, r), ...]), ...])
    base_runs = [r for _l, r in done if r.get("baseline")]
    swept = [r for _l, r in done if not r.get("baseline")]
    panels = []
    if mode == "oat" and len(names) > 1:
        for k, name in enumerate(names):
            pts = [(r["values"][name], r) for r in swept if r.get("group") == k]
            panels.append((k, [(None, pts)]))
    else:
        series = {}
        for r in swept:
            key = ", ".join(f"{n} = {r['values'].get(n)}" for n in names[1:]) or None
            series.setdefault(key, []).append((r["values"].get(names[0]), r))
        panels.append((0, list(series.items())))

    ncol, nrow = len(panels), len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(6.5, 5.2 * ncol), 3.6 * nrow + 0.6), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for c, (k, series) in enumerate(panels):
        xs_all = [x for _s, pts in series for x, _r in pts] + [r["values"].get(names[k]) for r in base_runs]
        numeric = all(_num(x) is not None for x in xs_all) and xs_all
        cats = [] if numeric else list(dict.fromkeys(str(x) for x in xs_all))

        def _xpos(x):
            return float(x) if numeric else cats.index(str(x))

        for rr, (kind, keys) in enumerate(rows):
            ax = axes[rr][c]
            for si, (sname, pts) in enumerate(series):
                pts = sorted(pts, key=lambda p: _xpos(p[0]))
                for key in keys:
                    xy = [(_xpos(x), _y(r, key)) for x, r in pts if _y(r, key) is not None]
                    if not xy:
                        continue
                    if key.startswith("param:"):
                        ylab, marker, col = key[6:], "D", _PARAM_COLORS[keys.index(key) % len(_PARAM_COLORS)]
                    else:
                        ylab, marker, col = _METRIC_STYLE.get(key, ("Final loss", "o", "#495057"))
                    if len(series) > 1:
                        col = colors[si % len(colors)]
                        lab = sname if (kind == "loss" or len(keys) == 1) else f"{ylab}, {sname}"
                    else:
                        lab = ylab if kind in ("err", "param") and len(keys) > 1 else None
                    ax.plot(*zip(*xy), marker=marker, color=col, linewidth=1.5, markersize=6,
                            label=lab, linestyle="-" if numeric else ":")
            for r in base_runs:
                for key in keys:
                    y = _y(r, key)
                    if y is not None:
                        ax.plot([_xpos(r["values"].get(names[k]))], [y], marker="*", markersize=13,
                                color="#fab005", markeredgecolor="#5c3d00", linestyle="none",
                                label="baseline" if key == keys[0] else None, zorder=5)
            ax.set_yscale("log")
            if numeric:
                fx = [float(x) for x in xs_all]
                if min(fx) > 0 and max(fx) / min(fx) >= 100:
                    ax.set_xscale("log")
            else:
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels(cats, rotation=20 if max(len(x) for x in cats) > 6 else 0)
            ax.grid(True, which="both", alpha=0.25)
            if kind == "param":
                ax.set_ylabel("Parameter error (%)" if len(keys) > 1 else f"{keys[0][6:]} error (%)")
            else:
                ax.set_ylabel("Error" if kind == "err" and len(keys) > 1
                              else _METRIC_STYLE.get(keys[0], ("Final loss",))[0])
            if rr == nrow - 1:
                ax.set_xlabel(names[k])
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best", fontsize=8, framealpha=0.9)
    what = "Parameter and solution error" if pkeys and metrics else ("Error" if (metrics or pkeys) else "Final loss")
    title = f"Parameter sweep: {what.lower()} vs " + (" / ".join(names) if ncol > 1 else names[0])
    if len(title) > 32 + 30 * ncol:          # wider than the figure: break before "vs"
        title = title.replace(" vs ", "\nvs ", 1)
    n_t = max((int((r.get("metrics") or {}).get("n_times", 1)) for _l, r in done), default=1)
    if metrics and n_t > 1:
        title += f"\n(error = mean over the {n_t} reference times)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(sweep_root, "sweep_error_vs_parameter.png")
    fig.savefig(path, dpi=int(getattr(base_config, "plot_dpi", 150) or 150))
    plt.close(fig)
    return path

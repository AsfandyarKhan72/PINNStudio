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
Either way, a "baseline" run (the config exactly as built, completely
unmodified) is always included first, so there is always a same-
settings reference point to compare every other run to.

Each concrete config is produced via pinnstudio.core.sweep_registry's
SweepParam.get_value/set_value accessors -- the same ones Phase 2 built
and Phase 3's GUI tab reads from -- so a swept field is always resolved
the one correct way (e.g. a learning rate that actually lives inside the
active Optimizer Scheduler phase), never a legacy/inert field.

Per-run results: each run's final training loss (parsed from the
"Final loss: ..." line every generated script already prints to
stdout) and, when the problem has Error Analysis reference data
configured, the last reported L2 relative error (read back from that
run's own error_analysis/error_metrics*.txt, which the generated script
already writes -- see codegen.py).

OUTPUT ORGANIZATION: every run (baseline included) gets its own,
descriptively-named folder under sweep_root_dir(base_config) -- e.g.
"run_002_BC_2_dirichlet_Output_0_7p5" for a run that set that BC's
weight to 7.5 -- with its save_dir pointed there, so the SAME save
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
no change here at all. A sweep_manifest.json and sweep_summary.csv are
written into the sweep's root folder once the sweep finishes (or is
cancelled -- whatever ran so far is still recorded), so the run/folder/
result mapping is readable later without reopening the GUI.
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
    elif base_config.sweep_mode == "zip":
        # "Specified combinations" (COMSOL's term): every parameter's
        # Nth value, together, as one combination -- P1=[1,2] & P2=[10,20]
        # gives exactly 2 runs, (1,10) and (2,20), never grid mode's 4.
        # zip() itself truncates to the shortest list if lengths somehow
        # differ (should already be caught by validate()) rather than
        # raising -- consistent with this module leaving that check to
        # PINNConfig.validate(), as its docstring above already says.
        for combo in zip(*(vals for _, vals in per_param)):
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


# ── Bayesian search (Round 75) ──────────────────────────────────────────

def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


class BayesRuns:
    """Run source for sweep_mode "bayes": the baseline, then one run at a
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

    def _cast(self, param, v):
        if param.value_type == "int":
            return int(round(float(v)))
        if param.value_type == "float":
            return float(v)
        return str(v)

    def next_run(self, i):
        if i == 0:
            point = []
            for p, d in zip(self.params, self.dims):
                v = p.get_value(self.base)
                point.append(str(v) if d.kind == "categorical" else (v if v is not None else d.low))
            self._pending = point
            return "baseline", copy.deepcopy(self.base)
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
        key = "final_loss" if self.objective == "loss" else "l2_relative"
        val = result.get(key) if result.get("status") == "done" else None
        self.opt.tell(self._pending, val)
        self.history.append((label, list(self._pending), val))

    def best(self):
        ok = [(v, lab, pt) for lab, pt, v in self.history if v is not None and np.isfinite(v)]
        return min(ok, key=lambda r: r[0]) if ok else None

    def write_outputs(self, sweep_root):
        """bayes_search.csv (every run, its objective and the best so far)
        and bayes_convergence.png (best objective vs run)."""
        try:
            with open(os.path.join(sweep_root, "bayes_search.csv"), "w", newline="") as f:
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
        return "Final loss" if self.objective == "loss" else "L2 relative error"


# ── Per-run output organization ────────────────────────────────────────

def _slugify(text, maxlen=60):
    """Turns a run label like "BC 2 (dirichlet, Output 0)=7.5" into a
    filesystem-safe folder-name fragment like "BC_2_dirichlet_Output_0_7.5"
    -- keeps letters/digits/dot/underscore/dash/equals, collapses
    everything else (spaces, commas, parens, ...) to a single
    underscore, and trims to a sane length so a Grid sweep over several
    parameters at once can't produce an unusably long path."""
    text = _SLUG_RE.sub("_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "run")[:maxlen]


def sweep_root_dir(base_config, started_at=None):
    """The folder this sweep's runs are organized under:
    <sweep_save_dir>/sweep_<timestamp>/ -- `started_at` (a datetime,
    defaulting to now) fixes the timestamp so every run started by the
    same run_sweep() call lands under the same root, while a later sweep
    of the same config gets its own, never silently overwriting the
    first. Falls back to the same ~/PINNStudio_Results convention the
    Setup tab's own "Save results to" field already defaults to (see
    MainWindow._build_ui) when sweep_save_dir is blank -- which it will
    be for any config built before this field existed."""
    root = (getattr(base_config, "sweep_save_dir", "") or "").strip()
    if not root:
        root = os.path.join(os.path.expanduser("~"), "PINNStudio_Results", "parameter_sweep_results")
    stamp = (started_at or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    return os.path.join(root, f"sweep_{stamp}")


def run_folder_name(index, label):
    """The one subfolder name a given run gets, under sweep_root_dir()
    -- numbered (so ordering survives alphabetical sorting and folder
    names can never collide even if two runs' labels happen to slugify
    the same) and, for anything past the baseline, named after exactly
    which parameter(s) and value(s) that run used -- e.g. a weight sweep
    over one BC's weight gets "run_001_BC_2_dirichlet_Output_0_7.5", not
    a generic "run_001" that gives no hint what it actually varied."""
    if index == 0 and label == "baseline":
        return "run_000_baseline"
    return f"run_{index:03d}_{_slugify(label)}"


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
    if (not getattr(cfg, "ea_files", None) or cfg.ea_files == "[]") and \
            getattr(cfg, "ea_ref_mode", "files") != "formula":
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
    runs = None if bayes else build_runs(base_config)
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
        label, cfg = bayes.next_run(i) if bayes else runs[i]
        run_dir = os.path.join(sweep_root, run_folder_name(i, label))
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

        result = {
            "status": "done" if status == "DONE" else "error",
            "final_loss": final_loss,
            "l2_relative": _read_last_l2(cfg, _run_started_at),
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
    return results


def _write_sweep_manifest(sweep_root, manifest):
    try:
        with open(os.path.join(sweep_root, "sweep_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    except OSError:
        pass  # best-effort -- a sweep's real results/models are already
              # safely on disk per-run regardless of this summary file


def _write_sweep_summary_csv(sweep_root, results):
    """Writes the same rows MainWindow._on_export_sweep_csv() lets the
    user save by hand, automatically, into the sweep's own root folder
    -- so a CSV index of every run's status/loss/L2 and which folder it
    landed in always exists afterward, without relying on the user
    remembering to click "Export results (CSV)"."""
    try:
        with open(os.path.join(sweep_root, "sweep_summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Run", "Status", "Final loss", "L2 relative error", "Folder"])
            for label, result in results:
                writer.writerow([
                    label, result.get("status"),
                    result.get("final_loss") if result.get("final_loss") is not None else "",
                    result.get("l2_relative") if result.get("l2_relative") is not None else "",
                    os.path.basename(result.get("save_dir", "")),
                ])
    except OSError:
        pass

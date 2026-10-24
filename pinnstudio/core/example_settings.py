"""Examples -> Settings: Default or Tuned.

Default: a DeepXDE example with DeepXDE's own settings (as it always loaded);
a template taken from a paper solved the way the paper solves it (network,
points, training: paper_settings.py); PINNStudio's own templates as they are.

Tuned: PINNStudio's best for the example, from the tuning study
(benchmarks/README.md): the most accurate run, and of the runs about as
accurate the fastest. Stored in resources/tuned_settings.json as the changes
to make to the example's own settings: for a template, the training methods
study's best setup (SWITCHES) and then one of the tuning study's changes
(apply_variant). An example that needed no tuning loads as it is.
"""
import copy
import json
import os

DEFAULT, TUNED = "Default", "Tuned"
MODES = (DEFAULT, TUNED)
_TUNED_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources",
                           "tuned_settings.json")
LBFGS_PHASE = {"optimizer": "lbfgs", "iterations": 15000, "lr": 0.001, "decay_type": "none"}
IC_WEIGHT = 100.0

# What the tuning study's changes are (shown in the Training Log).
VARIANT_TEXT = {
    "plain": "the base PINN, as the paper's baseline",
    "paper_wz": "Wight & Zhao's method", "paper_mg": "Mattey & Ghosh's bc-PINN",
    "paper_raissi": "Raissi et al.'s settings", "paper_mathias": "Mathias et al.'s settings",
    "iters": "3× the iterations of every training phase",
    "net": "a larger network (hidden layers twice as wide, one more layer)",
    "points": "3× the training points",
    "lbfgs": "an L-BFGS phase after Adam",
    "all": "3× iterations, a larger network, 3× points and an L-BFGS phase together",
    "dataw": "10× the weight of the measured data",
    "lbfgs3": "3× the L-BFGS iterations",
    "invall": "10× the data weight, 3× the L-BFGS iterations and a larger network",
    "windows2": "twice as many time windows",
}
SWITCH_TEXT = {"soft_ic": "IC as a loss term", "exact_ic": "exact IC", "soft_bc": "periodic BC as a loss term",
               "exact_bc": "exact periodicity", "whole": "the whole time interval",
               "windows": "time windows", "rba": "RBA weights", "sa": "SA-PINN weights",
               "causal": "causal training", "msffn": "multi-scale Fourier features"}


def _phases(cfg):
    if cfg.optimizer_scheduler and cfg.scheduler_phases:
        return json.loads(cfg.scheduler_phases)
    ph = [{"optimizer": cfg.optimizer, "iterations": cfg.iterations, "lr": cfg.learning_rate, "decay_type": "none"}]
    if cfg.optimizer2 and cfg.optimizer2 != "none":
        ph.append({"optimizer": cfg.optimizer2, "iterations": cfg.iterations2, "lr": cfg.learning_rate,
                   "decay_type": "none"})
    return ph


def _set_phases(cfg, ph):
    cfg.optimizer_scheduler = True
    cfg.scheduler_phases = json.dumps(ph)


def _has_lbfgs(cfg):
    return any(p.get("optimizer", "").lower().startswith("l") for p in _phases(cfg))


def _set_ic_weights(cfg, n_out_ic):
    n_eq = len([e for e in (cfg.pde_expressions or "").split("|")])
    n_bc = len(json.loads(cfg.custom_bc_json or "[]"))
    ph = _phases(cfg)
    for p in ph:
        w = [v for v in p.get("weights", "").split(",") if v.strip()]
        w = (w + ["1.0"] * (n_eq + n_bc))[:n_eq + n_bc] + [str(IC_WEIGHT)] * n_out_ic
        p["weights"] = ",".join(w)
    _set_phases(cfg, ph)


def apply_switches(cfg, switches):
    """The training methods study's setup switches (benchmarks/study_runner.py), on a copy."""
    cfg = copy.deepcopy(cfg)
    n_out = len(cfg.output_names.split(","))
    for sw in switches:
        if sw == "soft_ic":
            cfg.ic_hard = ",".join(["False"] * n_out)
            cfg.ic_active = ",".join(["True"] + ["False"] * (n_out - 1))
            _set_ic_weights(cfg, 1)
        elif sw == "exact_ic":
            cfg.ic_hard = ",".join(["True"] + ["False"] * (n_out - 1))
            cfg.ic_active = ",".join(["False"] * n_out)
            _set_ic_weights(cfg, 0)
        elif sw == "soft_bc":
            cfg.periodic_inputs = False
        elif sw == "exact_bc":
            cfg.periodic_inputs = True
        elif sw == "whole":
            cfg.time_adaptive = False
            cfg.adapt_method = "None"
        elif sw == "windows":
            cfg.time_adaptive = True
            cfg.adapt_method = "Time Adaptive"
            cfg.ta_transfer_learning = True
        elif sw == "rba":
            cfg.point_weights, cfg.point_weights_lbfgs = "RBA", "keep"
        elif sw == "sa":
            cfg.point_weights, cfg.point_weights_lbfgs, cfg.sa_normalize = "SA-PINN", "keep", False
        elif sw == "causal":
            cfg.causal_training, cfg.causal_anneal = True, True
        elif sw == "msffn":
            cfg.network_type, cfg.msffn_sigmas = "MsFFN", "1, 5"
    if switches:
        if not any(s in switches for s in ("rba", "sa")):
            cfg.point_weights = "None"
        if "causal" not in switches:
            cfg.causal_training = False
        if "msffn" not in switches:
            cfg.network_type = "FNN"
    return cfg


def apply_variant(cfg, variant):
    """One of the tuning study's changes on a copy of cfg, or None when it does not apply."""
    cfg = copy.deepcopy(cfg)
    if variant in ("base", "", None):
        return cfg
    if variant == "windows2":
        if not cfg.time_adaptive or not (cfg.ta_step_groups or "").strip():
            return None
        groups = json.loads(cfg.ta_step_groups)
        for g in groups:
            g["steps"] = int(g["steps"]) * 2
        cfg.ta_step_groups = json.dumps(groups)
        cfg.ta_num_steps = sum(g["steps"] for g in groups)
        return cfg
    if variant in ("dataw", "lbfgs3", "invall"):
        if not (cfg.inverse_obs_files_json or "").strip():
            return None
        if variant in ("dataw", "invall"):
            obs = json.loads(cfg.inverse_obs_files_json)
            for o in obs:
                o["weight"] = float(o.get("weight", 1.0)) * 10
            cfg.inverse_obs_files_json = json.dumps(obs)
        if variant in ("lbfgs3", "invall"):
            ph = _phases(cfg)
            lb = [p for p in ph if p.get("optimizer", "").lower().startswith("l")]
            if lb:
                for p in lb:
                    p["iterations"] = int(p["iterations"]) * 3
            else:
                ph.append(dict(LBFGS_PHASE, iterations=45000))
            _set_phases(cfg, ph)
        if variant == "invall":
            cfg = apply_variant(cfg, "net")
        return cfg
    parts = ["iters", "net", "points", "lbfgs"] if variant == "all" else [variant]
    if variant == "lbfgs" and _has_lbfgs(cfg):
        return None
    for part in parts:
        if part == "iters":
            ph = _phases(cfg)
            for p in ph:
                p["iterations"] = int(p["iterations"]) * 3
            _set_phases(cfg, ph)
            cfg.rar_adam_iters *= 3
            cfg.rar_lbfgs_iters *= 3
        elif part == "net":
            lay = list(cfg.layers)
            hidden = [min(512, int(h) * 2) for h in lay[1:-1]]
            hidden.append(hidden[-1])
            cfg.layers = [lay[0]] + hidden + [lay[-1]]
        elif part == "points":
            cfg.num_domain *= 3
            cfg.num_boundary *= 3
            cfg.num_initial *= 3
        elif part == "lbfgs":
            if not _has_lbfgs(cfg):
                ph = _phases(cfg)
                ph.append(dict(LBFGS_PHASE))
                _set_phases(cfg, ph)
        elif part not in ("base",):
            return None
    return cfg


def tuned_table():
    try:
        with open(_TUNED_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def settings_for(cfg, key, inverse, mode):
    """(config, message): the example `key` (its internal name) as loaded, changed to the chosen Settings.
    The message says what was changed, for the Training Log ("" when nothing was)."""
    from pinnstudio.core import paper_settings as ps
    if mode == DEFAULT:
        v = ps.TEMPLATE_PAPER.get(key)
        if v and not inverse and (key, v) in ps.PAPER:
            out = ps.apply(cfg, v, key)
            return out, f"Default settings: {key} solved as in its paper ({ps.PAPER_LABEL.get(v, v)})."
        return cfg, ""
    t = tuned_table().get(key, {}).get("inverse" if inverse else "forward")
    if not t:
        if key in ps.TEMPLATE_PAPER:
            return cfg, "Tuned settings: the template's own settings (PINNStudio's best for it)."
        return cfg, "Tuned settings: none for this example (it was already accurate) -- its default settings."
    out = apply_switches(cfg, t.get("switches") or [])
    v = t.get("variant") or "base"
    # the paper's method (or its base PINN) can itself be the best run
    out2 = ps.apply(out, v, key) if v == "plain" or v.startswith("paper_") else apply_variant(out, v)
    if out2 is None:
        return cfg, f"Tuned settings: the change '{t.get('variant')}' does not apply here -- default settings kept."
    what = [", ".join(SWITCH_TEXT.get(s, s) for s in t.get("switches") or [])] if t.get("switches") else []
    if t.get("variant") not in (None, "", "base"):
        what.append(VARIANT_TEXT.get(t["variant"], t["variant"]))
    err = t.get("error")
    msg = "Tuned settings: " + ("; ".join(w for w in what if w) or "the example's own settings")
    if err:
        msg += f" (benchmarks: {err})"
    return out2, msg + "."

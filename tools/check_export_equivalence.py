"""Check that the exported script ("Export as DeepXDE Script") trains exactly the same model as the script
Solve runs, for every example.

PINNStudio writes two scripts from one configuration: generate_script() for Solve and generate_clean_script()
for the export. For each example (forward, and inverse where it has one) this loads its Tuned settings,
shortens the training (every Adam phase to --adam steps, every L-BFGS phase to --lbfgs steps, IC pre-training
and RAR rounds cut down the same way), writes both scripts, runs them with the same seed on the same device
and compares the printed loss history line by line, and for inverse problems the identified parameters.

    python tools/check_export_equivalence.py --out equivalence_runs [--names 'a|b'] [--device cpu]

Besides the printed loss history (DeepXDE prints three significant digits), each script runs with a small
import hook (sitecustomize.py, see HOOK) that records every model.train() call at full precision: the loss
history (steps and every loss term) and, after each call, a SHA-256 of the network's weights and the values of
the trained unknowns. "identical" needs the printed history and parameters to match; "bitwise" says whether the
full-precision records match as well. When the two scripts differ, the Solve script is run a second time with
the same seed (the control): if two runs of the same script differ as much, the difference is run-to-run
nondeterminism of the CPU computation, not a difference between the scripts.

Writes <out>/results.json and prints a summary table. Exit code 1 if any example differs or fails.
"""
import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pinnstudio.core import codegen  # noqa: E402
from pinnstudio.core import deepxde_examples as dx  # noqa: E402
from pinnstudio.core import example_catalog as cat  # noqa: E402
from pinnstudio.core import example_settings as es  # noqa: E402
import tuning_study as ts  # noqa: E402

LOSS_LINE = re.compile(r"^\d+\s+\[")
PARAM_LINE = re.compile(r"(?:=== (\S+) final value: ([-\d.e+]+) ===|Final inferred (\S+): ([-\d.e+]+))")


HOOK = r'''
# Recorded by tools/check_export_equivalence.py: every deepxde Model.train() call at full precision.
import atexit, hashlib, importlib.abc, importlib.util, json, os, sys

_REC = []


def _weights_hash(model):
    try:
        import torch
        h = hashlib.sha256()
        for k, v in sorted(model.net.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    except Exception as ex:  # noqa: BLE001
        return f"unavailable: {ex!r}"


def _unknowns(model):
    out = []
    for v in getattr(model, "external_trainable_variables", None) or []:
        try:
            out.append(repr(float(v.detach().cpu().reshape(-1)[0])))
        except Exception:  # noqa: BLE001
            out.append(repr(v))
    return out


def _patch(mod):
    orig = mod.Model.train

    def train(self, *a, **k):
        res = orig(self, *a, **k)
        try:
            lh = self.losshistory
            _REC.append(dict(steps=[int(x) for x in lh.steps],
                             loss=[[repr(float(y)) for y in row] for row in lh.loss_train],
                             weights=_weights_hash(self), unknowns=_unknowns(self)))
        except Exception as ex:  # noqa: BLE001
            _REC.append(dict(error=repr(ex)))
        return res
    mod.Model.train = train


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "deepxde.model":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch(module)
        loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())


@atexit.register
def _dump():
    p = os.environ.get("EQUIV_RECORD")
    if p:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_REC, f)
'''


def base_config(e, inverse):
    if e.source == cat.DEEPXDE:
        ex = dx.BY_NAME[e.key]
        return dx.build_config(ex, ts.REF_ROOT, inverse=inverse and ex.inverse is not None)
    return ts.template_config(e.key, inverse)


def problems():
    """(entry, inverse) for every visible example and each of its modes."""
    out = []
    for e in cat.ENTRIES:
        if e.hidden:
            continue
        inv_only = e.key.startswith("Inverse:") or e.problem == cat.INVERSE
        if inv_only:
            out.append((e, False))
            continue
        out.append((e, False))
        has_inv = (dx.BY_NAME[e.key].inverse is not None) if e.source == cat.DEEPXDE else True   # every template has an inverse version
        if has_inv:
            out.append((e, True))
    return out


def shorten(cfg, n_adam, n_lbfgs, out_dir):
    """The same short training for both scripts; nothing else changes."""
    upd = {}
    if cfg.optimizer_scheduler and (cfg.scheduler_phases or "").strip():
        ph = []
        for p in json.loads(cfg.scheduler_phases):
            p = dict(p)
            p["iterations"] = n_lbfgs if str(p.get("optimizer", "")).lower() in ("lbfgs", "l-bfgs") else n_adam
            ph.append(p)
        upd["scheduler_phases"] = json.dumps(ph)
    for f, v in (("iterations", n_adam), ("lbfgs_maxiter", n_lbfgs), ("ic_pretrain_iterations", min(n_adam, 100)),
                 ("rar_adam_iters", n_adam), ("rar_lbfgs_iters", n_lbfgs), ("rar_cycles", min(int(cfg.rar_cycles), 1)),
                 ("ta_resample_iters", min(int(cfg.ta_resample_iters), n_adam)),
                 ("loss_display_every", max(n_adam // 10, 1)), ("plot_resolution", 40), ("plot_dpi", 60),
                 ("num_timesteps_anim", 3), ("save_dir", out_dir), ("train_device", "auto")):
        if hasattr(cfg, f):
            upd[f] = v
    return dataclasses.replace(cfg, **upd)


def run(script, workdir, timeout, device, record=None, hook_dir=None):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    if hook_dir:
        env["PYTHONPATH"] = hook_dir + os.pathsep + env.get("PYTHONPATH", "")
    if record:
        env["EQUIV_RECORD"] = record
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, os.path.basename(script)], cwd=workdir, env=env, capture_output=True,
                           text=True, timeout=timeout)
        rc, log = p.returncode, p.stdout + "\n--- stderr ---\n" + p.stderr
    except subprocess.TimeoutExpired as ex:
        rc, log = "timeout", (ex.stdout or b"").decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
    return rc, log, time.perf_counter() - t0


def _run_one(path, d, kind, args):
    """Run one script with the recording hook; its printed loss lines, parameters and full-precision record."""
    hook_dir = os.path.join(args.out, "_hook")
    os.makedirs(hook_dir, exist_ok=True)
    hp = os.path.join(hook_dir, "sitecustomize.py")
    if not os.path.exists(hp):
        with open(hp, "w", encoding="utf-8") as f:
            f.write(HOOK)
    rec_path = os.path.join(d, f"{kind}.record.json")
    if os.path.exists(rec_path):
        os.remove(rec_path)
    rc, log, sec = run(path, d, args.timeout, args.device, record=rec_path, hook_dir=hook_dir)
    with open(os.path.join(d, f"{kind}.log"), "w", encoding="utf-8") as f:
        f.write(log)
    try:
        record = json.load(open(rec_path, encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    return dict(rc=rc, seconds=round(sec, 1), record=record,
                loss=[ln.strip() for ln in log.splitlines() if LOSS_LINE.match(ln.strip())],
                params={(m.group(1) or m.group(3)): (m.group(2) or m.group(4)) for m in PARAM_LINE.finditer(log)})


def _param_diffs(a, b):
    """{parameter: relative difference} of the printed parameter values of two runs."""
    out = {}
    for k in set(a) | set(b):
        try:
            x, y = float(a[k]), float(b[k])
            out[k] = abs(x - y) / max(abs(x), abs(y), 1e-300)
        except (KeyError, ValueError):
            out[k] = None
    return out


def compare(e, inverse, args):
    slug = ts._slug(e.key) + ("_inv" if inverse else "")
    d = os.path.join(args.out, slug)
    os.makedirs(d, exist_ok=True)
    try:
        cfg, _msg = es.settings_for(base_config(e, inverse), e.key, inverse, es.TUNED)
    except SystemExit as ex:              # e.g. a template with no inverse version
        return dict(status="skipped", note=f"no such problem: {ex}")
    except Exception as ex:  # noqa: BLE001
        return dict(status="error", note=f"config: {ex!r}")
    if getattr(cfg, "sweep_enabled", False):
        return dict(status="skipped", note="parameter sweep example")
    cfg = shorten(cfg, args.adam, args.lbfgs, os.path.join(d, "runs"))
    res = {}
    for kind, gen in (("solve", codegen.generate_script), ("export", codegen.generate_clean_script)):
        path = os.path.join(d, f"{kind}.py")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(gen(cfg))
        except Exception as ex:  # noqa: BLE001
            return dict(status="error", note=f"{kind} script not generated: {ex!r}")
        res[kind] = _run_one(path, d, kind, args)
    s, x = res["solve"], res["export"]
    out = dict(solve_rc=s["rc"], export_rc=x["rc"], lines=len(s["loss"]), seconds=(s["seconds"], x["seconds"]),
               train_calls=len(s["record"] or []), bitwise=(s["record"] is not None and s["record"] == x["record"]),
               params_solve=s["params"], params_export=x["params"])
    if s["rc"] != 0 or x["rc"] != 0:
        out.update(status="error", note=f"exit codes: solve {s['rc']}, export {x['rc']}")
    elif not s["loss"]:
        out.update(status="error", note="no loss history printed")
    elif s["loss"] == x["loss"] and s["params"] == x["params"]:
        out.update(status="identical")
    if out.get("status") != "error" and not out["bitwise"]:
        # the control: the Solve script once more, same seed; how far apart are two runs of the same script?
        c = _run_one(os.path.join(d, "solve.py"), d, "solve_control", args)
        out["control"] = dict(rc=c["rc"], same_printed_loss=c["loss"] == s["loss"], same_params=c["params"] == s["params"],
                              bitwise=c["record"] is not None and c["record"] == s["record"], params=c["params"])
        out["param_rel_diff"] = _param_diffs(s["params"], x["params"])
        out["control_param_rel_diff"] = _param_diffs(s["params"], c["params"])
    shutil.rmtree(os.path.join(d, "runs"), ignore_errors=True)     # the trained models and plots are not needed
    if out.get("status") in ("identical", "error"):
        return out
    else:
        i = next((k for k, (a, b) in enumerate(zip(s["loss"], x["loss"])) if a != b), min(len(s["loss"]), len(x["loss"])))
        out.update(status="differs", note=(f"first difference at loss line {i}: solve {s['loss'][i] if i < len(s['loss']) else '-'}"
                                           f" | export {x['loss'][i] if i < len(x['loss']) else '-'}"
                                           + (f"; parameters solve {s['params']} export {x['params']}" if s["params"] != x["params"] else "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="", help="internal example names, '|' separates several (default: all)")
    ap.add_argument("--skip", default="", help="internal example names to leave out, '|' separates several")
    ap.add_argument("--adam", type=int, default=200, help="steps for every Adam phase")
    ap.add_argument("--lbfgs", type=int, default=50, help="steps for every L-BFGS phase")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--device", default="cpu", choices=("cpu", "auto"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "results.json")
    results = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    import platform
    meta = dict(python=platform.python_version(), platform=platform.platform(), adam=args.adam, lbfgs=args.lbfgs,
                device=args.device)
    try:
        meta["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    for mod in ("torch", "deepxde", "numpy"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    results["_meta"] = meta
    want = {n for n in args.names.split("|") if n}
    for e, inverse in problems():
        if want and e.key not in want:
            continue
        if e.key in {n for n in args.skip.split("|") if n}:
            continue
        key = f"{e.key} | {'inverse' if inverse else 'forward'}"
        if key in results and results[key].get("status") in ("identical", "skipped") and "bitwise" in results[key]:
            continue
        r = compare(e, inverse, args)
        r.update(name=e.name, source=e.source)
        results[key] = r
        print(f"{r['status']:9s} {key[:70]:70s} {r.get('note', '')[:160]}", flush=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
    rs = [r for k, r in results.items() if k != "_meta"]
    n = {s: sum(1 for r in rs if r["status"] == s) for s in ("identical", "differs", "error", "skipped")}
    n["bitwise"] = sum(1 for r in rs if r.get("bitwise"))
    print("summary:", n)
    sys.exit(1 if n["differs"] or n["error"] else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Error of saved PINN runs against the reference data that replaced the COMSOL files, without retraining.

A forward run uses the reference data only to measure its error after training, so its error against the new
reference follows from its saved network. This script rebuilds each network from its last checkpoint (the one
the run's own Error Analysis used) and computes, per output and reference time, the relative L2 error and the
mean absolute error against both the COMSOL files (reference_data/ at COMSOL_COMMIT) and the current
reference_data/. The COMSOL numbers must reproduce the ones the run logged; the run is only used if they do.

Inverse runs are evaluated too, for the record, but their *measurements* also came from the COMSOL files, so
the benchmark uses inverse runs trained again on the new data (see README.md).

Input: one bundle per run, made on the GPU machine from the run folder by extract_weights.py (the weights of
the last checkpoint in each folder as .npz, and the run's log.txt and model/step configs as .json).

    python benchmarks/reference_regeneration/reeval_runs.py BUNDLE_DIR > convergence/pinn_reeval.json

Needs NumPy only; the networks are evaluated in float64.
"""
import glob
import io
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COMSOL_COMMIT = "1adc3446fd232dca3e8668420e4cea1516624780"
TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
NAMES = {0.0: "0", 0.25: "0.25", 0.5: "0.50", 0.75: "0.75", 1.0: "1.0"}
PROBLEMS = {                      # run-folder name -> reference folder
    "Template_2D_Burgers_Mathias": "2D/burgers_mathias",
    "Template_2D_Poisson_Disk": "2D/poisson_disk",
    "Template_3D_Poisson_Sphere": "3D/poisson_sphere",
    "Poisson_over_an_L_shaped_domain": "2D/poisson_lshape",
}


def _read(rel, old):
    if old:
        out = subprocess.run(["git", "-C", ROOT, "show", f"{COMSOL_COMMIT}:reference_data/{rel}"], check=True,
                             capture_output=True).stdout.decode()
        return io.StringIO(out)
    return open(os.path.join(ROOT, "reference_data", rel))


def reference(folder, old):
    """[(t, points, {output: values})] of a reference folder."""
    if folder.endswith("burgers_mathias"):
        out = []
        for t in TIMES:
            vals, pts = {}, None
            for c in "uv":
                a = np.loadtxt(_read(f"{folder}/t_{NAMES[t]}_{c}.txt", old))
                pts, vals[c] = a[:, :3], a[:, 3]
            out.append((t, pts, vals))
        return out
    a = np.genfromtxt(_read(f"{folder}/solution.txt", old))
    m = np.isfinite(a[:, -1])
    return [(None, a[m, :-1], {"u": a[m, -1]})]


# --- networks ----------------------------------------------------------------------------------------------
def _layers(W, i):
    return [(W[f"c{i}|linears.{k}.weight"].astype(float), W[f"c{i}|linears.{k}.bias"].astype(float))
            for k in range(1000) if f"c{i}|linears.{k}.weight" in W]


def network(W, i, cfg):
    act = {"tanh": np.tanh, "sin": np.sin, "relu": lambda z: np.maximum(z, 0)}[cfg.get("activation", "tanh")]
    lin = _layers(W, i)
    ntype = cfg.get("network_type", "FNN")
    if ntype == "FNN":
        def f(X):
            h = X
            for k, (w, b) in enumerate(lin):
                h = h @ w.T + b
                if k < len(lin) - 1:
                    h = act(h)
            return h
    elif ntype == "MsFFN" and f"c{i}|B_t0" not in W:
        Bs = [W[f"c{i}|B_x{k}"].astype(float) for k in range(100) if f"c{i}|B_x{k}" in W]
        dw, db = W[f"c{i}|dense.weight"].astype(float), W[f"c{i}|dense.bias"].astype(float)

        def f(X):
            ys = []
            for B in Bs:
                p = X @ B
                h = np.concatenate([np.cos(p), np.sin(p)], axis=1)
                for w, b in lin:
                    h = act(h @ w.T + b)
                ys.append(h)
            return np.concatenate(ys, axis=1) @ dw.T + db
    else:
        raise NotImplementedError(f"network type {ntype}")
    if cfg.get("input_transform_enabled") and cfg.get("input_transform_mode", "scale") != "scale":
        raise NotImplementedError("input feature formula")
    in_sc = np.array(cfg.get("input_transform_scale") or [], float) if cfg.get("input_transform_enabled") else None
    in_sh = np.array(cfg.get("input_transform_shift") or [], float) if cfg.get("input_transform_enabled") else None
    names = [n.strip() for n in str(cfg.get("output_names", "u")).split(",")]

    def g(X):
        Z = X * in_sc + in_sh if in_sc is not None and in_sc.size else X
        Y = f(Z)
        if not cfg.get("output_transform_enabled"):
            return Y
        if cfg.get("output_transform_mode", "scale") == "scale":
            return Y * np.array(cfg["output_transform_scale"], float) + np.array(cfg["output_transform_shift"], float)
        exprs = str(cfg.get("output_transform_formula", "")).split("|")
        env = {"np": np, "pi": np.pi, "sin": np.sin, "cos": np.cos, "tanh": np.tanh, "exp": np.exp,
               "sqrt": np.sqrt, "log": np.log, "abs": np.abs}
        coords = ["x", "y", "z"][: X.shape[1] - (1 if cfg.get("_time") else 0)] + (["t"] if cfg.get("_time") else [])
        env.update({c: X[:, k] for k, c in enumerate(coords)})
        env.update({n: Y[:, k] for k, n in enumerate(names)})
        cols = []
        for k, n in enumerate(names):
            e = exprs[k].strip() if k < len(exprs) and exprs[k].strip() else n
            cols.append(eval(e.replace("^", "**"), {"__builtins__": {}}, env) * np.ones(len(X)))
        return np.stack(cols, axis=1)
    return g, names


# --- one run -----------------------------------------------------------------------------------------------
def _log_metrics(log):
    """[(output, t, L2, MeanAbs)] in log order; the output from the preceding 'Loaded ground truth' line."""
    out, cur = [], None
    for ln in log.splitlines():
        m = re.search(r"Loaded ground truth.*?from (\S+)", ln)
        if m:
            f = os.path.basename(m.group(1)).replace(".txt", "")
            f = re.sub(r"^t_[-\d.e]+_?", "", f)
            cur = f if f in ("u", "v") else "u"
            continue
        m = re.search(r"t=([-\d.]+) . L(?:2|₂)=([\d.e+-]+), MeanAbs=([\d.e+-]+)", ln)
        if m and cur:
            out.append((cur, float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def evaluate(bundle, refs):
    meta = json.load(open(bundle + ".json"))
    W = np.load(bundle + ".npz")
    run = meta["run"]
    prob = next((p for p in PROBLEMS if f"/{p}/" in run or f"/{p}_inv/" in run), None)
    folder = PROBLEMS[prob]
    cps = [(i, c) for i, c in enumerate(meta["checkpoints"]) if "error" not in c]
    cfg = {}
    for k, t in meta["texts"].items():
        if k.endswith("model_config.json"):
            cfg = json.loads(t)
    time_dep = folder.endswith("burgers_mathias")
    cfg["_time"] = time_dep
    windows = []
    for i, c in cps:
        m = re.search(r"step_\d+_t([\d.]+)_to_t([\d.]+)", c["path"])
        if m:
            windows.append((float(m.group(1)), float(m.group(2)), i))
    if not windows:                                   # one network; with RAR rounds, the last trained one
        i = max(cps, key=lambda ic: ic[1]["iteration"])[0]
        windows = [(-np.inf, np.inf, i)]
    windows.sort()
    nets = {i: network(W, i, cfg) for _, _, i in windows}
    res = {"run": run, "problem": folder, "inverse": "_inv/" in run, "outputs": {}}
    for which in ("old", "new"):
        for t, pts, vals in refs[which]:
            if time_dep:
                cand = [w for w in windows if w[0] - 1e-9 <= t <= w[1] + 1e-9]
                if not cand:                          # a run stopped before this time (timeout)
                    continue
                i = cand[0][2]                        # at a window boundary: the window that ends there
                X = pts[:, [0, 1]]
                X = np.column_stack([X, np.full(len(X), t)])
            else:
                i = windows[0][2]
                X = pts
            f, names = nets[i]
            P = f(X)
            for k, n in enumerate(names):
                if n not in vals:
                    continue
                e = P[:, k] - vals[n]
                o = res["outputs"].setdefault(n, {"t": [], "old": {"l2": [], "abs": [], "sq": [], "ref": []},
                                                  "new": {"l2": [], "abs": [], "sq": [], "ref": []}})
                if which == "old":
                    o["t"].append(0.0 if t is None else t)
                d = o[which]
                d["l2"].append(float(np.linalg.norm(e) / np.linalg.norm(vals[n])))
                d["abs"].append(float(np.abs(e).mean()))
                d["sq"].append(float((e ** 2).sum()))
                d["ref"].append(float((vals[n] ** 2).sum()))
    for n, o in res["outputs"].items():
        for which in ("old", "new"):
            d = o[which]
            d["all_times"] = float(np.sqrt(sum(d.pop("sq")) / sum(d.pop("ref"))))
    logged = _log_metrics(meta["texts"].get("log.txt", ""))
    dev = []
    for n, t, l2, ab in logged:
        o = res["outputs"].get(n)
        if o and t in o["t"]:
            j = o["t"].index(t)
            dev.append(max(abs(o["old"]["l2"][j] - l2) / l2, abs(o["old"]["abs"][j] - ab) / ab))
    res["log_compared"] = len(dev)
    res["log_max_rel_dev"] = max(dev) if dev else None
    res["reproduces_log"] = bool(dev) and max(dev) < 2e-3
    return res


def main():
    bundles = sorted(p[:-5] for p in glob.glob(os.path.join(sys.argv[1], "*.json")))
    refs = {}
    out = []
    for b in bundles:
        run = json.load(open(b + ".json"))["run"]
        prob = next((p for p in PROBLEMS if f"/{p}/" in run or f"/{p}_inv/" in run), None)
        if prob is None:
            continue
        folder = PROBLEMS[prob]
        if folder not in refs:
            refs[folder] = {"old": reference(folder, True), "new": reference(folder, False)}
        try:
            r = evaluate(b, refs[folder])
        except NotImplementedError as exc:
            r = {"run": run, "problem": folder, "error": str(exc)}
        out.append(r)
        print(f"{run}: {'reproduces the log' if r.get('reproduces_log') else 'DOES NOT reproduce the log'} "
              f"({r.get('log_compared')} values, max rel. deviation {r.get('log_max_rel_dev')})", file=sys.stderr)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()

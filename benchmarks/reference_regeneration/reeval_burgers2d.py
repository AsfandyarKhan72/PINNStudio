"""Relative L2 errors of the saved Burgers 2D (Mathias) PINN models against the COMSOL reference files and
against the regenerated Chebyshev reference (grid.npz), at t = 0, 0.25, 0.5, 0.75, 1.

    python reeval_burgers2d.py NEW_GRID_NPZ COMSOL_DIR RUN_DIR [RUN_DIR ...] > out.json

A run folder is one tools/tuning_study.py run (log.txt, and the saved models of each time window under
time_adaptive_steps/). The network is rebuilt from the checkpoint (plain tanh FNN, no input or output transform,
as these runs use). Needs PyTorch.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
NAMES = {0.0: "0", 0.25: "0.25", 0.5: "0.50", 0.75: "0.75", 1.0: "1.0"}


def load_net(pt):
    sd = torch.load(pt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    W = sorted({int(m.group(1)) for k in sd for m in [re.match(r"linears\.(\d+)\.weight", k)] if m})
    layers = [(sd[f"linears.{i}.weight"].double(), sd[f"linears.{i}.bias"].double()) for i in W]
    extra = [k for k in sd if not k.startswith("linears.")]

    def f(X):
        h = torch.as_tensor(X, dtype=torch.float64)
        for i, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if i < len(layers) - 1:
                h = torch.tanh(h)
        return h.numpy()
    return f, extra


def final_model(folder):
    pts = glob.glob(os.path.join(folder, "*.pt"))
    if not pts:
        return None
    it = lambda p: int(re.findall(r"(\d+)\.pt$", p)[0]) if re.findall(r"(\d+)\.pt$", p) else -1
    return max(pts, key=it)          # the last checkpoint (L-BFGS ends last)


def main():
    new = np.load(sys.argv[1])
    old_dir = sys.argv[2]
    g = np.round(np.linspace(0, 1, 101), 10)
    X, Y = np.meshgrid(g, g)
    out = {}
    for run in sys.argv[3:]:
        steps = sorted(glob.glob(os.path.join(run, "**", "time_adaptive_steps", "step_*"), recursive=True))
        models = []
        for s in steps:
            m = re.search(r"t([\d.]+)_to_t([\d.]+)", s)
            pt = final_model(s)
            if pt:
                models.append((float(m.group(1)), float(m.group(2)), pt))
        if not models:
            sol = [p for p in glob.glob(os.path.join(run, "**", "solution_results"), recursive=True)]
            pt = final_model(sol[0]) if sol else None
            if not pt:
                continue
            models = [(0.0, 1.0, pt)]
        try:
            nets = [(a, b, *load_net(p)) for a, b, p in models]
        except Exception as exc:      # noqa: BLE001
            out[run] = {"error": repr(exc)}
            continue
        res = {"models": [os.path.relpath(p, run) for *_, p in models], "extra_keys": sorted({k for *_, e in nets for k in e})}
        for which in ("first", "last"):          # at a window boundary: the window that ends there, or starts there
            r = {}
            for t in TIMES:
                cand = [n for n in nets if n[0] - 1e-9 <= t <= n[1] + 1e-9]
                if not cand:
                    continue
                a, b, f, _ = cand[0] if which == "first" else cand[-1]
                P = f(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, t)]))
                for i, c in enumerate("uv"):
                    old = np.loadtxt(os.path.join(old_dir, f"t_{NAMES[t]}_{c}.txt"))[:, 3]
                    nw = new[f"{c}_{t}"].ravel()
                    r[f"{c}@{t}"] = {"comsol": float(np.linalg.norm(P[:, i] - old) / np.linalg.norm(old)),
                                     "new": float(np.linalg.norm(P[:, i] - nw) / np.linalg.norm(nw))}
            res[which] = r
        log = os.path.join(run, "log.txt")
        if os.path.exists(log):
            res["log_L2"] = re.findall(r"t=([\d.]+) . L(?:2|₂)=([\d.e+-]+)", open(log, errors="replace").read())
        out[run] = res
        print(run, json.dumps({k: v for k, v in res.get("last", {}).items()}), file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()

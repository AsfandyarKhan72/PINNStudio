#!/usr/bin/env python3
"""Redraw a saved forward run's PINN / reference / error figure against the current reference data, without
retraining.

For a run whose reference data were regenerated after it was trained (the 2D Allen-Cahn of Mattey & Ghosh, Heat
2D/3D, the 2D Allen-Cahn of Wight & Zhao), the figure PINNStudio drew at the time compares with the old
reference. Its network is saved in a bundle (benchmarks/records/corrected_2026-10-31/reeval_bundles/, made by
extract_weights.py), and the corrected records' errors were computed from it (reeval_runs.py). This script
rebuilds the same network with reeval_runs.network, evaluates it at the current reference's points and draws the
figure in the layout of PINNStudio's Error Analysis surface comparison (one row per reference time: PINN,
reference, |error|), so the figure shows exactly the network and the errors the records report. It prints each
time's relative L2 error, which must equal the corrected record.

    python benchmarks/reference_regeneration/redraw_from_bundle.py BUNDLE OUT.png

BUNDLE is the path without .json/.npz. Needs NumPy and Matplotlib; the network is evaluated in float64.
"""
import json
import sys

import numpy as np

import reeval_runs as rr


def main():
    bundle, out = sys.argv[1], sys.argv[2]
    meta = json.load(open(bundle + ".json"))
    run = meta["run"]
    prob = next(p for p in sorted(rr.PROBLEMS, key=len, reverse=True) if f"/{p}/" in run or f"__{p}__" in run)
    folder = rr.PROBLEMS[prob]
    W = np.load(bundle + ".npz")
    cfg = {}
    for k, t in meta["texts"].items():
        if k.endswith("model_config.json"):
            cfg = json.loads(t)
    cfg["_time"] = True
    cps = [(i, c) for i, c in enumerate(meta["checkpoints"]) if "error" not in c]
    if any("step_" in c["path"] and "_to_t" in c["path"] for _, c in cps):
        raise SystemExit("time-window runs are not handled here")
    i = max(cps, key=lambda ic: ic[1]["iteration"])[0]          # the last trained network, as Error Analysis
    f, names = rr.network(W, i, cfg)
    refs = rr.reference(folder, old=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for t, pts, vals in refs:
        name = next(iter(vals))
        X = np.column_stack([pts[:, :2], np.full(len(pts), t)])
        p = f(X)[:, 0]
        r = vals[name]
        rows.append((t, pts, p, r, float(np.linalg.norm(p - r) / np.linalg.norm(r))))
        print(f"t={t:g}  L2={rows[-1][4]:.5g}")
    sci = lambda v: (lambda m, e: f"{m} \\times 10^{{{int(e)}}}")(*f"{v:.4g}".split("e")) if "e" in f"{v:.4g}" else f"{v:.4g}"
    fig, axs = plt.subplots(len(rows), 3, figsize=(14, 3.45 * len(rows)))
    for k, (t, pts, p, r, l2) in enumerate(rows):
        x, y = pts[:, 0], pts[:, 1]
        lo, hi = min(p.min(), r.min()), max(p.max(), r.max())
        err = np.abs(p - r)
        tl = f"{t:g}"
        for j, (z, title, vmin, vmax) in enumerate(((p, f"PINN  t={tl}  $L_2$ = ${sci(l2)}$", lo, hi),
                                                    (r, f"Reference  t={tl}", lo, hi),
                                                    (err, f"|Error|  t={tl}  max = ${sci(err.max())}$", 0, err.max()))):
            ax = axs[k, j]
            c = ax.tricontourf(x, y, z, levels=100, cmap="jet", vmin=vmin, vmax=vmax)
            fig.colorbar(c, ax=ax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    fig.suptitle("PINN vs Reference — 2D Heatmaps", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main()

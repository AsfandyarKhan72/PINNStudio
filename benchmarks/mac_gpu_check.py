"""Mac GPU check: is training on Apple's GPU (MPS) faster than on the CPU,
and does it give the same answer?

Three problems with an exact solution (Heat 1D, Poisson 2D, Heat 2D, from
small to large) are trained the same way on:
  cpu      PyTorch's own thread count
  cpu-4    4 threads (an M1 has 4 performance cores)
  mps      the Mac GPU
Each run is Adam, then L-BFGS, in float32. For each it records the time per
iteration, the final loss, the L2 error against the exact solution, and any
error (an L-BFGS that fails or goes to NaN on MPS shows up here).

Run with PINNStudio's own Python (the installer puts it under `uv tool dir`):

    "$(uv tool dir)/pinnstudio/bin/python" benchmarks/mac_gpu_check.py

--quick halves the iterations. The results are printed as a table and saved
to mac_gpu_check.json next to where you run it.
"""
import argparse
import json
import os
import platform
import time
import traceback

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")   # an op MPS lacks should fail, not silently use the CPU

import numpy as np
import torch
import deepxde as dde

PI = np.pi
A = 0.1   # diffusion coefficient of the heat problems


def heat1d():
    geom = dde.geometry.GeometryXTime(dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 1))

    def pde(x, u):
        return dde.grad.jacobian(u, x, j=1) - A * dde.grad.hessian(u, x, i=0, j=0)

    exact = lambda x: np.exp(-A * PI**2 * x[:, 1:2]) * np.sin(PI * x[:, 0:1])
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_b: on_b)
    ic = dde.icbc.IC(geom, lambda x: np.sin(PI * x[:, 0:1]), lambda _, on_i: on_i)
    data = dde.data.TimePDE(geom, pde, [bc, ic], num_domain=2000, num_boundary=100, num_initial=100)
    xs = np.stack(np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101)), -1).reshape(-1, 2)
    return data, [2] + [64] * 3 + [1], xs, exact


def poisson2d():
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    def pde(x, u):
        lap = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
        return lap + 2 * PI**2 * torch.sin(PI * x[:, 0:1]) * torch.sin(PI * x[:, 1:2])

    exact = lambda x: np.sin(PI * x[:, 0:1]) * np.sin(PI * x[:, 1:2])
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_b: on_b)
    data = dde.data.PDE(geom, pde, [bc], num_domain=20000, num_boundary=800)
    xs = np.stack(np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101)), -1).reshape(-1, 2)
    return data, [2] + [128] * 4 + [1], xs, exact


def heat2d():
    geom = dde.geometry.GeometryXTime(dde.geometry.Rectangle([0, 0], [1, 1]), dde.geometry.TimeDomain(0, 1))

    def pde(x, u):
        u_t = dde.grad.jacobian(u, x, j=2)
        return u_t - A * (dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1))

    exact = lambda x: np.exp(-2 * A * PI**2 * x[:, 2:3]) * np.sin(PI * x[:, 0:1]) * np.sin(PI * x[:, 1:2])
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_b: on_b)
    ic = dde.icbc.IC(geom, lambda x: np.sin(PI * x[:, 0:1]) * np.sin(PI * x[:, 1:2]), lambda _, on_i: on_i)
    data = dde.data.TimePDE(geom, pde, [bc, ic], num_domain=30000, num_boundary=2000, num_initial=1000)
    g = np.linspace(0, 1, 31)
    xs = np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3)
    return data, [3] + [64] * 4 + [1], xs, exact


PROBLEMS = {"Heat 1D (2k points)": heat1d, "Poisson 2D (20k points)": poisson2d, "Heat 2D (30k points)": heat2d}


def _sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def run(name, build, dev, threads, n_adam, n_lbfgs):
    r = {"problem": name, "device": dev if not threads else f"{dev}-{threads}"}
    torch.set_default_device(dev)
    if threads:
        torch.set_num_threads(threads)
    dde.config.set_random_seed(0)
    torch.set_default_device(dev)          # the seed call can reset it in some DeepXDE versions
    try:
        data, layers, xs, exact = build()
        model = dde.Model(data, dde.nn.FNN(layers, "tanh", "Glorot normal"))
        model.compile("adam", lr=1e-3)
        model.train(iterations=50, display_every=10**9, verbose=0)       # warm-up, not timed
        _sync(dev)
        t0 = time.perf_counter()
        lh, _ = model.train(iterations=n_adam, display_every=10**9, verbose=0)
        _sync(dev)
        r["adam_it_per_s"] = n_adam / (time.perf_counter() - t0)
        r["adam_loss"] = float(np.sum(lh.loss_train[-1]))
        r["adam_l2"] = float(dde.metrics.l2_relative_error(exact(xs), model.predict(xs)))
    except Exception as e:
        r["error"] = f"Adam: {type(e).__name__}: {e}"
        r["trace"] = traceback.format_exc(limit=3)
        return r
    try:
        dde.optimizers.set_LBFGS_options(maxiter=n_lbfgs)
        model.compile("L-BFGS")
        s0 = int(model.train_state.step)
        _sync(dev)
        t0 = time.perf_counter()
        lh, ts = model.train(display_every=10**9, verbose=0)
        _sync(dev)
        steps = max(1, int(ts.step) - s0)
        r["lbfgs_s"] = time.perf_counter() - t0
        r["lbfgs_steps"] = steps
        r["lbfgs_loss"] = float(np.sum(lh.loss_train[-1]))
        r["lbfgs_l2"] = float(dde.metrics.l2_relative_error(exact(xs), model.predict(xs)))
        if not np.isfinite(r["lbfgs_loss"]):
            r["error"] = "L-BFGS: loss is NaN/inf"
    except Exception as e:
        r["error"] = f"L-BFGS: {type(e).__name__}: {e}"
        r["trace"] = traceback.format_exc(limit=3)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="half the iterations")
    ap.add_argument("--only", choices=["cpu", "mps"], help="one device only")
    a = ap.parse_args()
    n_adam, n_lbfgs = (1000, 250) if a.quick else (2000, 500)

    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "deepxde": dde.__version__,
            "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
            "cpu_count": os.cpu_count(), "torch_threads_default": torch.get_num_threads(),
            "adam_iterations": n_adam, "lbfgs_maxiter": n_lbfgs}
    print(json.dumps(info, indent=1))
    devices = [("cpu", None), ("cpu", 4), ("mps", None)]
    if a.only:
        devices = [d for d in devices if d[0] == a.only]
    if not info["mps_available"]:
        print("\nThe Mac GPU (MPS) is not available to this PyTorch; only the CPU is timed.")
        devices = [d for d in devices if d[0] != "mps"]
    default_threads = torch.get_num_threads()

    results = []
    for name, build in PROBLEMS.items():
        for dev, threads in devices:
            torch.set_num_threads(threads or default_threads)
            label = dev if not threads else f"{dev}-{threads}"
            print(f"\n{name} on {label} ...", flush=True)
            r = run(name, build, dev, threads, n_adam, n_lbfgs)
            results.append(r)
            print("  " + (r.get("error") or
                          f"Adam {r['adam_it_per_s']:.0f} it/s, L2 {r['adam_l2']:.2e} | "
                          f"L-BFGS {r['lbfgs_steps']} steps in {r['lbfgs_s']:.1f} s, L2 {r['lbfgs_l2']:.2e}"), flush=True)

    print("\n" + "=" * 96)
    print(f"{'Problem':26} {'Device':7} {'Adam it/s':>10} {'Adam L2':>10} {'L-BFGS s':>9} {'steps':>6} {'final L2':>10}  note")
    for r in results:
        if "adam_it_per_s" not in r:
            print(f"{r['problem']:26} {r['device']:7} {'':>10} {'':>10} {'':>9} {'':>6} {'':>10}  {r.get('error', '')}")
            continue
        print(f"{r['problem']:26} {r['device']:7} {r['adam_it_per_s']:10.0f} {r['adam_l2']:10.2e} "
              f"{r.get('lbfgs_s', float('nan')):9.1f} {r.get('lbfgs_steps', 0):6d} {r.get('lbfgs_l2', float('nan')):10.2e}  "
              f"{r.get('error', '')}")
    with open("mac_gpu_check.json", "w") as f:
        json.dump({"info": info, "results": results}, f, indent=1)
    print("\nSaved mac_gpu_check.json -- please send this file (or paste the table above).")


if __name__ == "__main__":
    main()

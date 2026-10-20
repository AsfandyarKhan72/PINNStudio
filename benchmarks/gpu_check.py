"""NVIDIA GPU check (Windows or Linux): is PyTorch using the GPU, and is the
GPU running at full speed while PINNStudio-style training runs?

What it checks
  1. The PyTorch build: a CPU-only build (common on Windows when torch comes
     from plain `pip install torch`) never uses the GPU.
  2. The GPU: driver, memory, the PCIe link, the power limit.
  3. Training speed on the CPU and on the GPU for three problems with an exact
     solution (Heat 1D 2k points, Poisson 2D 20k, Heat 2D 30k), Adam then
     L-BFGS in float32, and the L2 error, so both devices give the same answer.
  4. While the GPU trains: utilization, memory, clock, power and power state
     (P0/P2 = full speed; P5 and up = the card is saving power).
  5. TF32 (faster on RTX, less precise -- PINNStudio leaves it off) and
     float64 (very slow on GeForce/RTX cards) on the Poisson problem.
At the end it lists anything that looks wrong, and saves gpu_check.json.

    python gpu_check.py            # all checks (about 10-25 minutes)
    python gpu_check.py --quick    # half the iterations
    python gpu_check.py --watch    # only record the GPU every second while you
                                   # train in PINNStudio; Ctrl+C to stop
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import threading
import time
import traceback

os.environ.setdefault("DDE_BACKEND", "pytorch")

SMI = shutil.which("nvidia-smi")


def smi_query(field, idx=0):
    if not SMI:
        return None
    try:
        out = subprocess.run([SMI, "-i", str(idx), f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        v = out.stdout.strip()
        return None if (out.returncode or not v or "N/A" in v or "Not Supported" in v) else v
    except Exception:
        return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class GpuSampler(threading.Thread):
    """Reads utilization, memory, SM clock, power and power state every
    half second (or every `period` s) until stop()."""
    FIELDS = ["utilization.gpu", "memory.used", "clocks.sm", "power.draw", "pstate", "temperature.gpu"]

    def __init__(self, idx=0, period=0.5):
        super().__init__(daemon=True)
        self.idx, self.period, self.rows, self._stop = idx, period, [], threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run([SMI, "-i", str(self.idx), "--query-gpu=" + ",".join(self.FIELDS),
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                parts = [p.strip() for p in out.stdout.strip().split(",")]
                if len(parts) == len(self.FIELDS):
                    self.rows.append(dict(zip(self.FIELDS, parts)))
            except Exception:
                pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=15)
        return self.summary()

    def summary(self):
        def col(k):
            return [x for x in (_num(r.get(k)) for r in self.rows) if x is not None]
        s = {"samples": len(self.rows)}
        for k, name in (("utilization.gpu", "util_pct"), ("memory.used", "mem_mib"), ("clocks.sm", "sm_clock_mhz"),
                        ("power.draw", "power_w"), ("temperature.gpu", "temp_c")):
            v = col(k)
            if v:
                s[name + "_avg"] = sum(v) / len(v)
                s[name + "_max"] = max(v)
        ps = [r.get("pstate") for r in self.rows if r.get("pstate")]
        if ps:
            s["pstates"] = sorted(set(ps))
        return s


def watch():
    if not SMI:
        print("nvidia-smi was not found, so the GPU cannot be watched.")
        return
    print("Recording the GPU every second -- start training in PINNStudio now. Ctrl+C to stop.\n")
    print(f"{'time':>6} {'util %':>7} {'mem MiB':>8} {'SM MHz':>7} {'power W':>8} {'state':>6} {'temp C':>7}")
    smp = GpuSampler(period=1.0)
    smp.start()
    t0, shown = time.time(), 0
    try:
        while True:
            time.sleep(1.0)
            for r in smp.rows[shown:]:
                print(f"{time.time() - t0:6.0f} {r['utilization.gpu']:>7} {r['memory.used']:>8} {r['clocks.sm']:>7} "
                      f"{r['power.draw']:>8} {r['pstate']:>6} {r['temperature.gpu']:>7}", flush=True)
            shown = len(smp.rows)
    except KeyboardInterrupt:
        s = smp.stop()
        print("\nSummary:", json.dumps(s, indent=1))


# ── test problems (exact solutions known) ─────────────────────────────────
import numpy as np  # noqa: E402

PI, A = np.pi, 0.1


def problems():
    import deepxde as dde
    import torch

    def heat1d():
        geom = dde.geometry.GeometryXTime(dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 1))
        pde = lambda x, u: dde.grad.jacobian(u, x, j=1) - A * dde.grad.hessian(u, x, i=0, j=0)
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
            return dde.grad.jacobian(u, x, j=2) - A * (dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1))
        exact = lambda x: np.exp(-2 * A * PI**2 * x[:, 2:3]) * np.sin(PI * x[:, 0:1]) * np.sin(PI * x[:, 1:2])
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_b: on_b)
        ic = dde.icbc.IC(geom, lambda x: np.sin(PI * x[:, 0:1]) * np.sin(PI * x[:, 1:2]), lambda _, on_i: on_i)
        data = dde.data.TimePDE(geom, pde, [bc, ic], num_domain=30000, num_boundary=2000, num_initial=1000)
        g = np.linspace(0, 1, 31)
        return data, [3] + [64] * 4 + [1], np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3), exact

    return {"Heat 1D (2k points)": heat1d, "Poisson 2D (20k points)": poisson2d, "Heat 2D (30k points)": heat2d}


def run_case(name, build, dev, n_adam, n_lbfgs, tf32=False, f64=False, sample=False):
    import deepxde as dde
    import torch
    label = dev + ("+TF32" if tf32 else "") + ("+float64" if f64 else "")
    r = {"problem": name, "setup": label}
    torch.set_default_device(dev)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    dde.config.set_default_float("float64" if f64 else "float32")
    dde.config.set_random_seed(0)
    torch.set_default_device(dev)
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    smp = None
    try:
        data, layers, xs, exact = build()
        model = dde.Model(data, dde.nn.FNN(layers, "tanh", "Glorot normal"))
        model.compile("adam", lr=1e-3)
        model.train(iterations=50, display_every=10**9, verbose=0)          # warm-up
        if sample and SMI:
            smp = GpuSampler()
            smp.start()
        sync()
        t0 = time.perf_counter()
        lh, _ = model.train(iterations=n_adam, display_every=10**9, verbose=0)
        sync()
        r["adam_it_per_s"] = n_adam / (time.perf_counter() - t0)
        r["adam_l2"] = float(dde.metrics.l2_relative_error(exact(xs), model.predict(xs)))
        if n_lbfgs:
            dde.optimizers.set_LBFGS_options(maxiter=n_lbfgs)
            model.compile("L-BFGS")
            s0 = int(model.train_state.step)
            sync()
            t0 = time.perf_counter()
            lh, ts = model.train(display_every=10**9, verbose=0)
            sync()
            r["lbfgs_s"] = time.perf_counter() - t0
            r["lbfgs_steps"] = max(1, int(ts.step) - s0)
            r["final_l2"] = float(dde.metrics.l2_relative_error(exact(xs), model.predict(xs)))
            if not np.isfinite(float(np.sum(lh.loss_train[-1]))):
                r["error"] = "L-BFGS: loss is NaN/inf"
        if dev == "cuda":
            r["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        r["trace"] = traceback.format_exc(limit=3)
    finally:
        if smp:
            r["gpu"] = smp.stop()
        dde.config.set_default_float("float32")
        if dev == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="half the iterations")
    ap.add_argument("--watch", action="store_true", help="only record the GPU while you train in PINNStudio")
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU runs (the slowest part)")
    ap.add_argument("--iters", type=int, help=argparse.SUPPRESS)        # smoke tests: Adam iterations
    a = ap.parse_args()
    if a.watch:
        return watch()

    import torch
    import deepxde as dde
    n_adam, n_lbfgs = (1000, 250) if a.quick else (2000, 500)
    if a.iters:
        n_adam, n_lbfgs = a.iters, max(1, a.iters // 4)
    warn = []

    # ── 1-2: the install and the GPU ──────────────────────────────────────
    info = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(), "deepxde": dde.__version__,
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "nvidia_smi": bool(SMI)}
    if torch.cuda.is_available():
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "compute_capability": f"{p.major}.{p.minor}", "vram_gib": round(p.total_memory / 2**30, 1)})
    if SMI:
        for f in ("driver_version", "memory.total", "power.limit", "power.max_limit", "clocks.max.sm",
                  "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max",
                  "driver_model.current"):
            info["smi_" + f] = smi_query(f)
    print(json.dumps(info, indent=1), flush=True)

    if not torch.cuda.is_available():
        if torch.version.cuda is None and SMI:
            warn.append("PyTorch is a CPU-only build, but this PC has an NVIDIA GPU: PyTorch can't use it. "
                        "Reinstall torch with CUDA (the PINNStudio installer does this; with pip use the "
                        "--index-url from pytorch.org, e.g. https://download.pytorch.org/whl/cu128).")
        elif SMI:
            warn.append("PyTorch has CUDA but cannot reach the GPU: update the NVIDIA driver (it must be at "
                        f"least as new as CUDA {torch.version.cuda} needs) and restart.")
        else:
            warn.append("No NVIDIA GPU or driver found: training runs on the CPU.")
    gw, gm = _num(info.get("smi_pcie.link.width.current")), _num(info.get("smi_pcie.link.width.max"))
    if gw and gm and gw < gm and gw < 8:
        warn.append(f"The GPU's PCIe link runs at x{gw:.0f} of x{gm:.0f} (an external GPU, or a slot that "
                    "shares lanes). Reported at idle it can be lower; check under load if training is slow.")

    # ── 3-4: CPU vs GPU, with the GPU recorded while it trains ────────────
    results = []
    probs = problems()
    setups = ([] if a.no_cpu else [("cpu", False, False)]) + ([("cuda", False, False)] if torch.cuda.is_available() else [])
    for name, build in probs.items():
        for dev, tf32, f64 in setups:
            print(f"\n{name} on {dev} ...", flush=True)
            r = run_case(name, build, dev, n_adam, n_lbfgs, tf32, f64, sample=(dev == "cuda"))
            results.append(r)
            print("  " + _line(r), flush=True)

    # ── 5: TF32 and float64 on the GPU (Poisson 2D, Adam only) ────────────
    if torch.cuda.is_available():
        build = probs["Poisson 2D (20k points)"]
        for tf32, f64 in ((True, False), (False, True)):
            print(f"\nPoisson 2D on cuda {'+TF32' if tf32 else '+float64'} (Adam only) ...", flush=True)
            r = run_case("Poisson 2D (20k points)", build, "cuda", n_adam // 2, 0, tf32, f64)
            results.append(r)
            print("  " + _line(r), flush=True)

    # ── what looks wrong ──────────────────────────────────────────────────
    by = {(r["problem"], r["setup"]): r for r in results}
    for name in probs:
        g, c = by.get((name, "cuda")), by.get((name, "cpu"))
        if g and "error" in g:
            warn.append(f"{name}: the GPU run failed: {g['error']}")
            continue
        if not g:
            continue
        s = g.get("gpu", {})
        big = "1D" not in name
        if c and "adam_it_per_s" in c and big and g["adam_it_per_s"] < 1.5 * c["adam_it_per_s"]:
            warn.append(f"{name}: the GPU is only {g['adam_it_per_s'] / c['adam_it_per_s']:.1f}x the CPU on a large problem.")
        if big and s.get("util_pct_avg") is not None and s["util_pct_avg"] < 50:
            warn.append(f"{name}: GPU utilization averaged {s['util_pct_avg']:.0f}% while training a large problem "
                        "(another program on the GPU, or the CPU/Python can't feed it).")
        mx = _num(info.get("smi_clocks.max.sm"))
        if big and mx and s.get("sm_clock_mhz_avg") and s["sm_clock_mhz_avg"] < 0.6 * mx:
            warn.append(f"{name}: the GPU clock averaged {s['sm_clock_mhz_avg']:.0f} of {mx:.0f} MHz -- a power-saving mode "
                        "(laptop on battery, Windows 'Best power efficiency', NVIDIA 'Optimal power'), or overheating "
                        f"(max {s.get('temp_c_max', 0):.0f} C).")
        if big and s.get("pstates") and not any(p in ("P0", "P1", "P2") for p in s["pstates"]):
            warn.append(f"{name}: the GPU never reached a full-speed power state (saw {', '.join(s['pstates'])}).")
        if c and "final_l2" in c and "final_l2" in g and g["final_l2"] > 3 * c["final_l2"] + 1e-3:
            warn.append(f"{name}: the GPU's final L2 error ({g['final_l2']:.2e}) is much worse than the CPU's ({c['final_l2']:.2e}).")
    t32, f64, base = (by.get(("Poisson 2D (20k points)", k)) for k in ("cuda+TF32", "cuda+float64", "cuda"))
    if f64 and base and "adam_it_per_s" in f64 and "adam_it_per_s" in base:
        info["float64_slowdown"] = base["adam_it_per_s"] / f64["adam_it_per_s"]
    if t32 and base and "adam_it_per_s" in t32 and "adam_it_per_s" in base:
        info["tf32_speedup"] = t32["adam_it_per_s"] / base["adam_it_per_s"]

    print("\n" + "=" * 110)
    print(f"{'Problem':25} {'Setup':14} {'Adam it/s':>9} {'Adam L2':>9} {'L-BFGS s':>8} {'final L2':>9} "
          f"{'GPU util':>9} {'SM MHz':>7} {'power W':>8} {'state':>6} {'peak MiB':>8}")
    for r in results:
        s = r.get("gpu", {})
        print(f"{r['problem']:25} {r['setup']:14} {_f(r, 'adam_it_per_s', '9.0f')} {_f(r, 'adam_l2', '9.2e')} "
              f"{_f(r, 'lbfgs_s', '8.1f')} {_f(r, 'final_l2', '9.2e')} {_f(s, 'util_pct_avg', '8.0f')}% "
              f"{_f(s, 'sm_clock_mhz_avg', '7.0f')} {_f(s, 'power_w_avg', '8.0f')} {','.join(s.get('pstates', [])):>6} "
              f"{_f(r, 'peak_mem_mib', '8.0f')}  {r.get('error', '')}")
    if "tf32_speedup" in info:
        print(f"\nTF32 on: {info['tf32_speedup']:.2f}x the speed (PINNStudio keeps it off for accuracy).")
    if "float64_slowdown" in info:
        print(f"float64: {info['float64_slowdown']:.1f}x slower than float32 on this GPU.")
    print("\nThings to look at:" if warn else "\nNothing looks wrong.")
    for w in warn:
        print("  - " + w)
    with open("gpu_check.json", "w") as f:
        json.dump({"info": info, "results": results, "warnings": warn}, f, indent=1)
    print("\nSaved gpu_check.json -- please send this file (or paste everything from the table down).")


def _f(d, k, fmt):
    v = d.get(k)
    w = int(fmt.split(".")[0]) if fmt[0].isdigit() else 8
    return format(v, fmt) if isinstance(v, (int, float)) else " " * w


def _line(r):
    if "error" in r and "adam_it_per_s" not in r:
        return r["error"]
    s = r.get("gpu", {})
    g = (f" | GPU {s['util_pct_avg']:.0f}% busy, {s.get('sm_clock_mhz_avg', 0):.0f} MHz, "
         f"{s.get('power_w_avg', 0):.0f} W, {','.join(s.get('pstates', []))}" if s.get("util_pct_avg") is not None else "")
    lb = f" | L-BFGS {r['lbfgs_s']:.1f} s, L2 {r['final_l2']:.2e}" if "lbfgs_s" in r else ""
    return f"Adam {r['adam_it_per_s']:.0f} it/s, L2 {r['adam_l2']:.2e}{lb}{g}" + (f"  [{r['error']}]" if "error" in r else "")


if __name__ == "__main__":
    main()

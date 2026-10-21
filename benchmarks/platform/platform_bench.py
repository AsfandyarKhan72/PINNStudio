#!/usr/bin/env python3
"""Speed checks for one computer, to compare platforms (Windows, Linux, macOS).

Adds what benchmarks/gpu_check.py and benchmarks/timing_study.py don't measure:

    python platform_bench.py threads OUT.json [--counts 2,4,8,all] [--repeat 3]
        CPU training speed (Adam iterations a second) of gpu_check's three
        problems for each number of CPU threads, each count in a fresh process
        (OMP_NUM_THREADS and torch.set_num_threads), plus PINNStudio's
        Automatic rule.
    python platform_bench.py sizes OUT.json [--repeat 2]
        Poisson 2D with 500 to 64,000 points, CPU (Automatic threads) against
        the GPU: where the GPU starts to pay off.
    python platform_bench.py startup OUT.json [--repeat 5]
        Time to start Python and import torch, deepxde and matplotlib in a new
        process, as every PINNStudio run does.
    python platform_bench.py output OUT.json [--repeat 2]
        Heat 1D on the GPU with DeepXDE printing every 1000/100/10 steps, read
        through a pipe the way PINNStudio's window reads a run.
    python platform_bench.py compile OUT.json
        Whether torch.compile works on this computer.

Each command adds its results to OUT.json (a list of dicts) and prints a table.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
os.environ.setdefault("DDE_BACKEND", "pytorch")


def auto_threads():
    """PINNStudio's Automatic rule for CPU training (codegen.py)."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def save(out, rows):
    old = []
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            old = json.load(f)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(old + rows, f, indent=1)


def child(args, env=None, timeout=3600):
    """Run this file again in a new Python process; return its last JSON line."""
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([sys.executable, os.path.abspath(__file__)] + args, env=e, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"error": (p.stderr or p.stdout)[-1500:]}


# -- one case in this process ------------------------------------------------
def _case(a):
    sys.path.insert(0, BENCH)
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    import gpu_check
    probs = gpu_check.problems()
    if a.problem.startswith("Poisson 2D n="):
        n = int(a.problem.split("=")[1])
        build = _poisson(n)
    else:
        build = probs[a.problem]
    r = gpu_check.run_case(a.problem, build, a.device, a.adam, 0)
    r["torch_threads"] = torch.get_num_threads()
    print(json.dumps(r), flush=True)


def _poisson(n):
    def build():
        import deepxde as dde
        import numpy as np
        import torch
        pi = np.pi
        geom = dde.geometry.Rectangle([0, 0], [1, 1])

        def pde(x, u):
            lap = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
            return lap + 2 * pi**2 * torch.sin(pi * x[:, 0:1]) * torch.sin(pi * x[:, 1:2])
        exact = lambda x: np.sin(pi * x[:, 0:1]) * np.sin(pi * x[:, 1:2])
        bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_b: on_b)
        data = dde.data.PDE(geom, pde, [bc], num_domain=n, num_boundary=max(40, n // 25))
        xs = np.stack(np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101)), -1).reshape(-1, 2)
        return data, [2] + [64] * 4 + [1], xs, exact
    return build


# -- commands ----------------------------------------------------------------
PROBLEMS = {"Heat 1D (2k points)": 1500, "Poisson 2D (20k points)": 60, "Heat 2D (30k points)": 60}


def cmd_threads(a):
    n_cpu = os.cpu_count() or 1
    counts = []
    for c in a.counts.split(","):
        c = c.strip()
        counts.append(n_cpu if c == "all" else int(c))
    counts = sorted(set(counts + [auto_threads()]))
    rows = []
    for prob, adam in PROBLEMS.items():
        if a.only and a.only.lower() not in prob.lower():
            continue
        for n in counts:
            speeds = []
            for rep in range(a.repeat):
                r = child(["_case", prob, "cpu", str(adam), "--threads", str(n)],
                          env={"OMP_NUM_THREADS": str(n), "CUDA_VISIBLE_DEVICES": ""})
                speeds.append(r.get("adam_it_per_s"))
                if "error" in r:
                    print(f"  {prob} threads={n}: {r['error'][-300:]}")
            ok = [s for s in speeds if s]
            row = dict(test="threads", problem=prob, threads=n, auto=(n == auto_threads()), adam=adam,
                       it_per_s=speeds, best=max(ok) if ok else None)
            rows.append(row)
            print(f"{prob:25s} threads {n:3d}{' (Automatic)' if row['auto'] else '            '}  "
                  f"best {row['best'] or 0:8.1f} it/s   all {[round(s or 0, 1) for s in speeds]}", flush=True)
    save(a.out, rows)


def cmd_sizes(a):
    rows = []
    for n in (500, 2000, 8000, 32000, 64000):
        for dev in ("cpu", "cuda"):
            adam = 2000 if dev == "cuda" else max(40, int(400 * 2000 / n))
            speeds = []
            for rep in range(a.repeat):
                env = {"CUDA_VISIBLE_DEVICES": ""} if dev == "cpu" else {}
                r = child(["_case", f"Poisson 2D n={n}", dev, str(adam)] +
                          (["--threads", str(auto_threads())] if dev == "cpu" else []), env=env)
                speeds.append(r.get("adam_it_per_s"))
                if "error" in r:
                    print(f"  n={n} {dev}: {r['error'][-300:]}")
            ok = [s for s in speeds if s]
            rows.append(dict(test="sizes", points=n, device=dev, adam=adam, it_per_s=speeds, best=max(ok) if ok else None))
            print(f"Poisson 2D {n:6d} points  {dev:4s}  best {rows[-1]['best'] or 0:8.1f} it/s", flush=True)
    save(a.out, rows)


STARTUP = {
    "python only": "pass",
    "import torch": "import torch",
    "torch + CUDA init": "import torch; torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')",
    "torch + deepxde": "import torch, deepxde",
    "torch + deepxde + matplotlib": "import torch, deepxde, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot",
}


def cmd_startup(a):
    rows = []
    env = dict(os.environ, DDE_BACKEND="pytorch", PYTHONUTF8="1")
    for name, code in STARTUP.items():
        ts = []
        for rep in range(a.repeat):
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            ts.append(time.perf_counter() - t0)
            if p.returncode:
                raise SystemExit(f"{name} failed:\n{p.stderr[-1000:]}")
        rows.append(dict(test="startup", what=name, seconds=ts, first=ts[0], median=statistics.median(ts)))
        print(f"{name:32s} first {ts[0]:6.2f} s   median {statistics.median(ts):6.2f} s   all {[round(t, 2) for t in ts]}",
              flush=True)
    # where the import time goes (one run, -X importtime, top-level packages)
    p = subprocess.run([sys.executable, "-X", "importtime", "-c", "import torch, deepxde"], env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    top = {}
    for line in p.stderr.splitlines():
        if not line.startswith("import time:") or "|" not in line:
            continue
        parts = [x.strip() for x in line[len("import time:"):].split("|")]
        try:
            cum = int(parts[1])
        except ValueError:
            continue
        mod = parts[2].strip()
        if "." not in mod and not line.split("|")[2].startswith("  "):
            top[mod] = max(top.get(mod, 0), cum)
    best = sorted(top.items(), key=lambda kv: -kv[1])[:10]
    rows.append(dict(test="importtime", top_us=best))
    print("largest imports:", ", ".join(f"{m} {us / 1e6:.2f} s" for m, us in best))
    save(a.out, rows)


OUTPUT_SCRIPT = r'''
import os, sys, time
os.environ["DDE_BACKEND"] = "pytorch"
sys.path.insert(0, {bench!r})
import torch, deepxde as dde, gpu_check
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(dev)
dde.config.set_random_seed(0)
data, layers, xs, exact = gpu_check.problems()["Heat 1D (2k points)"]()
model = dde.Model(data, dde.nn.FNN(layers, "tanh", "Glorot normal"))
model.compile("adam", lr=1e-3)
model.train(iterations=50, display_every=10**9, verbose=0)
t0 = time.perf_counter()
model.train(iterations={iters}, display_every={every})
if dev == "cuda": torch.cuda.synchronize()
print("TRAIN_SECONDS", time.perf_counter() - t0, flush=True)
'''


def cmd_output(a):
    import tempfile
    rows = []
    iters = 3000
    d = tempfile.mkdtemp(prefix="pbench_out_")
    for every in (1000, 100, 10):
        path = os.path.join(d, f"run_{every}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(OUTPUT_SCRIPT.format(bench=BENCH, iters=iters, every=every))
        ts = []
        for rep in range(a.repeat):
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            p = subprocess.Popen([sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", env=env,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            lines, t = 0, None
            for line in p.stdout:          # read line by line, as pinnstudio/core/runner.py does
                lines += 1
                if line.startswith("TRAIN_SECONDS"):
                    t = float(line.split()[1])
            p.wait()
            ts.append(t)
        ok = [t for t in ts if t]
        rows.append(dict(test="output", display_every=every, iters=iters, lines=lines, seconds=ts,
                         best=min(ok) if ok else None, it_per_s=(iters / min(ok)) if ok else None))
        print(f"display every {every:5d}: {lines:5d} lines  best {min(ok) if ok else 0:6.2f} s  "
              f"({rows[-1]['it_per_s'] or 0:7.0f} it/s)", flush=True)
    save(a.out, rows)


def cmd_compile(a):
    import torch
    r = dict(test="compile", torch=torch.__version__)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        f = torch.compile(lambda x: torch.tanh(x @ x.T).sum())
        x = torch.randn(64, 64, device=dev)
        t0 = time.perf_counter()
        f(x)
        r["ok"] = True
        r["first_call_s"] = time.perf_counter() - t0
    except Exception as e:
        r["ok"] = False
        r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    print(json.dumps(r))
    save(a.out, [r])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("threads", "sizes", "startup", "output", "compile"):
        s = sub.add_parser(name)
        s.add_argument("out")
        s.add_argument("--repeat", type=int, default=3 if name == "threads" else (5 if name == "startup" else 2))
        s.add_argument("--counts", default="1,2,4,6,8,12,16,all")
        s.add_argument("--only", default="")
    c = sub.add_parser("_case")
    c.add_argument("problem")
    c.add_argument("device")
    c.add_argument("adam", type=int)
    c.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "_case":
        return _case(a)
    print(f"{platform.platform()}  Python {platform.python_version()}  {os.cpu_count()} logical CPUs, "
          f"Automatic threads = {auto_threads()}", flush=True)
    {"threads": cmd_threads, "sizes": cmd_sizes, "startup": cmd_startup, "output": cmd_output,
     "compile": cmd_compile}[a.cmd](a)


if __name__ == "__main__":
    main()

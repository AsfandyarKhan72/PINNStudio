#!/usr/bin/env python3
"""
Performance study for the paper (Section 5, computational performance): three parts, one results file each.

  overhead     Table 2 (Section 5.1). Five problems with their full Tuned settings, each solved three ways from the same
               configuration: the exported standalone script run with python ("export"), the internal Solve
               script run with python and no GUI ("solve"), and Solve in the PINNStudio GUI with the live loss plot
               and log on ("gui"). Three seeds; the order of the three modes rotates with the seed. Run it in a real
               desktop session (DISPLAY set); --offscreen only for a quick check.
  throughput   Supplementary Table S3 (Section 5.2). The same five problems on the CPU (8 threads) and on the GPU: a short Adam warm-up and
               then a fixed number of Adam steps, timed separately (DeepXDE's own per-call timing), same
               network, points and float precision on both devices.
  threads      Supplementary Table S4 (Section 5.2). Adam throughput of three fixed workloads (benchmarks/gpu_check.py: Heat 1D 2k points,
               Poisson 2D 20k, Heat 2D 30k) on the CPU with 1, 2, 4, 8, 12, 16, 24 and 32 threads, each thread
               count in a fresh process with OMP_NUM_THREADS and MKL_NUM_THREADS set; the thread count PyTorch
               actually used is recorded.
  tables       Writes the three tables (Markdown) and a CSV of every run to benchmarks/performance/.

    python benchmarks/perf_study.py overhead   runs/perf/overhead.jsonl [--seeds 1,2,3] [--only NAME] [--offscreen]
    python benchmarks/perf_study.py throughput runs/perf/throughput.jsonl [--repeat 3]
    python benchmarks/perf_study.py threads    runs/perf/threads.jsonl [--repeat 3]
    python benchmarks/perf_study.py tables     runs/perf

Every row also records the commit, Python, PyTorch, CUDA, DeepXDE, GPU, driver and CPU. Nothing in the generated
scripts is changed to make them faster; only the seed, the save folder and (for throughput) the training
schedule are set.
"""
import argparse
import copy
import dataclasses
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# label, internal example name (catalogue key), inverse
PROBLEMS = [
    ("Lorenz ODE, inverse", "Inverse: Lorenz system", False),
    ("Wave 1D, multi-scale Fourier features", "Wave propagation – space-time multi-scale Fourier features", False),
    ("Allen-Cahn 2D (Mattey & Ghosh)", "2D Allen-Cahn (Mattey & Ghosh)", False),
    ("Navier-Stokes 2D, Kovasznay flow", "Kovasznay flow", False),
    ("Allen-Cahn 3D (Wight & Zhao)", "3D Allen-Cahn (Wight & Zhao)", False),
]
MODES = ["export", "solve", "gui"]
THREADS = [1, 2, 4, 8, 12, 16, 24, 32]

TRAIN_RE = re.compile(r"'train' took ([\d.]+) s")
STEP_RE = re.compile(r"^\s*(\d+)\s+\[", re.M)
GT_RE = re.compile(r"Loaded ground truth.*?from (\S+)")
L2_RE = re.compile(r"t=([-\d.]+) (?:—|--|.) L(?:2|₂)=([\d.e+-]+)")
PARAM_RE = re.compile(r"(?:=== (\S+) final value: ([-\d.e+]+) ===|Final inferred (\S+): ([-\d.e+]+))")
SMI = shutil.which("nvidia-smi")


# ---------------------------------------------------------------------------------------------------------- environment
def environment():
    info = dict(python=platform.python_version(), platform=platform.platform(), cpu=platform.processor() or "")
    try:
        with open("/proc/cpuinfo") as f:
            m = re.search(r"model name\s*:\s*(.+)", f.read())
            info["cpu"] = m.group(1).strip() if m else info["cpu"]
    except OSError:
        pass
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # noqa: BLE001
        pass
    try:
        import deepxde
        info["deepxde"] = deepxde.__version__
    except Exception:  # noqa: BLE001
        pass
    if SMI:
        q = subprocess.run([SMI, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True)
        info["driver"] = q.stdout.strip() or None
    q = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True)
    info["commit"] = q.stdout.strip() or None
    q = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True)
    info["dirty"] = bool(q.stdout.strip())
    return info


class MemSampler(threading.Thread):
    """Peak GPU memory in use (MiB), read with nvidia-smi from outside the training process, so the
    training script itself is not changed."""
    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period, self.peak, self.base, self._halt = period, None, None, threading.Event()

    def _read(self):
        try:
            q = subprocess.run([SMI, "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            return float(q.stdout.strip())
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        self.base = self._read()
        while not self._halt.is_set():
            v = self._read()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        if self.peak is None:
            return None
        return round(self.peak - (self.base or 0.0), 1)


# ------------------------------------------------------------------------------------------------------------ log parsing
def parse_log(log, phases):
    """Training times (DeepXDE's own "'train' took" lines), split by optimizer when the calls line up with the
    phases, steps, the main output's L2 at the last reference time, and identified parameters."""
    took = [float(x) for x in TRAIN_RE.findall(log)]
    r = dict(train=round(sum(took), 3), train_calls=took)
    opt = [str(p.get("optimizer", "adam")).lower() for p in phases]
    if took and opt and len(took) % len(opt) == 0:
        by = {}
        for i, t in enumerate(took):
            k = "lbfgs" if "lbfgs" in opt[i % len(opt)].replace("-", "") else "adam"
            by[k] = by.get(k, 0.0) + t
        r.update(adam=round(by.get("adam", 0.0), 3), lbfgs=round(by.get("lbfgs", 0.0), 3))
    steps = [int(x) for x in STEP_RE.findall(log)]
    r["steps"] = max(steps) if steps else None
    outs, cur = {}, None
    for ln in log.splitlines():
        if ln.startswith("[Sweep"):
            continue
        m = GT_RE.search(ln)
        if m:
            f = os.path.basename(m.group(1)).replace(".txt", "")
            f = re.sub(r"^t_[-\d.e]+_?", "", f)
            f = re.sub(r"_?t_[-\d.e]+$", "", f)
            cur = f or "u"
            continue
        m = L2_RE.search(ln)
        if m:
            t = float(m.group(1))
            if cur is None or (cur.startswith("out") and t in outs.get(cur, {})):
                cur = f"out{len(outs)}"     # the exported script prints no file names: outputs in order
            outs.setdefault(cur, {})[t] = float(m.group(2))
    r["l2"] = {k: v[max(v)] for k, v in outs.items() if v}
    r["params"] = {(m.group(1) or m.group(3)): float(m.group(2) or m.group(4)) for m in PARAM_RE.finditer(log)}
    return r


def phases_of(cfg):
    if cfg.optimizer_scheduler and (cfg.scheduler_phases or "").strip():
        return json.loads(cfg.scheduler_phases)
    return [{"optimizer": "adam"}]


# --------------------------------------------------------------------------------------------------------- the window
def window(offscreen):
    no_screen = (sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
                 and not os.environ.get("WAYLAND_DISPLAY"))
    if offscreen or no_screen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
    app = QApplication.instance() or QApplication(["pinnstudio"])
    QMessageBox.exec = lambda self: 0
    QDialog.exec = lambda self: QDialog.DialogCode.Accepted.value
    for n in ("question", "warning", "information"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    app.processEvents()
    return app, w


def load_tuned(app, w, key, inverse):
    """The example as a user gets it: Examples list, Settings: Tuned."""
    from pinnstudio.core import example_catalog as cat
    from pinnstudio.core import example_settings as es
    e = next(x for x in cat.ENTRIES if x.key == key)
    category = "Inverse" if (inverse or e.key.startswith("Inverse:") or e.problem == cat.INVERSE) else "Forward"
    w.quick_examples_settings_combo.setCurrentText(es.TUNED)
    w.quick_examples_category_combo.setCurrentText(category)
    app.processEvents()
    i = w.quick_examples_combo.findText(e.name)
    assert i >= 0, f"no example {e.name!r} in {category}"
    w.quick_examples_combo.setCurrentIndex(-1)
    app.processEvents()
    w.quick_examples_combo.setCurrentIndex(i)
    for _ in range(30):
        app.processEvents()
    return w._build_config()


def run_script(text, workdir, device="auto", threads=None, sample=True):
    path = os.path.join(workdir, "run.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", DDE_BACKEND="pytorch")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if threads:
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(threads)
    mem = MemSampler() if (sample and SMI and device != "cpu") else None
    if mem:
        mem.start()
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, path], cwd=workdir, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    total = time.perf_counter() - t0
    log = p.stdout + "\n" + p.stderr
    with open(os.path.join(workdir, "log.txt"), "w", encoding="utf-8") as f:
        f.write(log)
    return dict(rc=p.returncode, total=round(total, 2), gpu_mem_mib=mem.stop() if mem else None), log


def run_window(app, w, cfg):
    """Solve in the window with the live loss plot and log on, timed from the click until the window has the
    results (Solve enabled again)."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    if hasattr(w, "live_loss_cb"):
        w.live_loss_cb.setChecked(True)
    real_build = w._build_config
    w._build_config = lambda: copy.deepcopy(cfg)
    mem = MemSampler() if SMI else None
    if mem:
        mem.start()
    t0 = time.perf_counter()
    try:
        w._on_solve()
    finally:
        w._build_config = real_build
    while not w.solve_btn.isEnabled():
        app.processEvents()
        time.sleep(0.01)
    total = time.perf_counter() - t0
    log = w.log_box.toPlainText()
    return dict(rc=0 if "Training complete" in log else 1, total=round(total, 2),
                gpu_mem_mib=mem.stop() if mem else None), log


def _done(path):
    if not os.path.exists(path):
        return []
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def _append(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------------------------------------------------- overhead
def overhead(a):
    from pinnstudio.core.codegen import generate_clean_script, generate_script
    seeds = [int(s) for s in a.seeds.split(",")]
    app, w = window(a.offscreen)
    env = environment()
    env["qt_platform"] = app.platformName()
    done = {(r["problem"], r["mode"], r["seed"]) for r in _done(a.out)}
    base = tempfile.mkdtemp(prefix="pinnstudio_perf_")
    cfg_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    for label, key, inverse in PROBLEMS:
        if a.only and a.only.lower() not in label.lower():
            continue
        cfg0 = load_tuned(app, w, key, inverse)
        with open(os.path.join(cfg_dir, re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") + ".json"), "w") as f:
            json.dump(dataclasses.asdict(cfg0), f, indent=1, default=str)
        for k, seed in enumerate(seeds):
            order = MODES[k % 3:] + MODES[:k % 3]          # rotate the order with the seed
            for mode in order:
                if (label, mode, seed) in done:
                    continue
                d = os.path.join(base, f"{re.sub(r'[^A-Za-z0-9]+', '_', label)}_{mode}_s{seed}")
                os.makedirs(d, exist_ok=True)
                cfg = dataclasses.replace(copy.deepcopy(cfg0), random_seed=seed, use_random_seed=True, save_dir=d)
                if a.scale != 1.0:            # a quick check of the script only, never for the table
                    ph = [dict(p, iterations=max(1, int(int(p.get("iterations", 0) or 0) * a.scale))) for p in phases_of(cfg)]
                    cfg = dataclasses.replace(cfg, scheduler_phases=json.dumps(ph),
                                              iterations=max(1, int(cfg.iterations * a.scale)))
                if mode == "export":
                    r, log = run_script(generate_clean_script(cfg), d)
                elif mode == "solve":
                    r, log = run_script(generate_script(cfg), d)
                else:
                    r, log = run_window(app, w, cfg)
                    with open(os.path.join(d, "log.txt"), "w", encoding="utf-8") as f:
                        f.write(log)
                r.update(parse_log(log, phases_of(cfg)))
                r.update(problem=label, key=key, mode=mode, seed=seed, order=order.index(mode), env=env, workdir=d,
                         scale=a.scale)
                _append(a.out, r)
                print(f"{label:40s} {mode:6s} seed {seed}: train {r['train']:9.2f} s  total {r['total']:9.2f} s  "
                      f"steps {r['steps']}  L2 {r['l2']}  rc={r['rc']}", flush=True)
    w.close()


# ----------------------------------------------------------------------------------------------------------- throughput
def throughput(a):
    """Adam only, same schedule on both devices: phase 1 = warm-up, phase 2 = the timed steps."""
    from pinnstudio.core.codegen import generate_script
    app, w = window(True)
    env = environment()
    done = {(r["problem"], r["device"], r["rep"]) for r in _done(a.out)}
    base = tempfile.mkdtemp(prefix="pinnstudio_tp_")
    steps = dict(a.steps)
    for label, key, inverse in PROBLEMS:
        if a.only and a.only.lower() not in label.lower():
            continue
        cfg0 = load_tuned(app, w, key, inverse)
        lr = next((float(p.get("lr", 1e-3)) for p in phases_of(cfg0) if "adam" in str(p.get("optimizer", "adam")).lower()), 1e-3)
        n = steps.get(label, a.adam)
        ph = [{"optimizer": "adam", "iterations": a.warmup, "lr": lr}, {"optimizer": "adam", "iterations": n, "lr": lr}]
        # one training run: no time windows, no RAR rounds, no IC pre-training, no plots (the workload is the
        # network, the points and the residual, the same on both devices)
        upd = dict(optimizer_scheduler=True, scheduler_phases=json.dumps(ph), time_adaptive=False,
                   adapt_method="None", ic_pretrain=False, random_seed=0, use_random_seed=True)
        cfg1 = dataclasses.replace(copy.deepcopy(cfg0), **{k: v for k, v in upd.items() if hasattr(cfg0, k)})
        for rep in range(a.repeat):
            for device in (["gpu", "cpu"] if rep % 2 == 0 else ["cpu", "gpu"]):
                if (label, device, rep) in done:
                    continue
                d = os.path.join(base, f"{re.sub(r'[^A-Za-z0-9]+', '_', label)}_{device}_{rep}")
                os.makedirs(d, exist_ok=True)
                cfg = dataclasses.replace(cfg1, save_dir=d, cpu_threads=a.cpu_threads if device == "cpu" else 0)
                r, log = run_script(generate_script(cfg), d, device=device,
                                    threads=a.cpu_threads if device == "cpu" else None)
                took = [float(x) for x in TRAIN_RE.findall(log)]
                m = re.search(r"torch threads: (\d+)", log)
                r.update(problem=label, device=device, rep=rep, warmup=a.warmup, adam_steps=n,
                         warmup_s=took[0] if len(took) > 0 else None, timed_s=took[1] if len(took) > 1 else None,
                         threads=a.cpu_threads if device == "cpu" else None, env=env, workdir=d)
                r["s_per_1000"] = round(1000 * r["timed_s"] / n, 3) if r["timed_s"] else None
                _append(a.out, r)
                print(f"{label:40s} {device}  rep {rep}: {r['s_per_1000']} s per 1000 Adam steps  rc={r['rc']}", flush=True)
    w.close()


# -------------------------------------------------------------------------------------------------------------- threads
def _thread_worker(a):
    """Runs inside a fresh process with OMP/MKL set: one gpu_check workload on the CPU."""
    import torch
    torch.set_num_threads(a.n)
    import gpu_check as gc
    probs = gc.problems()
    r = gc.run_case(a.problem, probs[a.problem], "cpu", a.adam, 0)
    r["torch_threads"] = torch.get_num_threads()
    print("RESULT " + json.dumps(r), flush=True)


def threads(a):
    env = environment()
    done = {(r["problem"], r["threads"], r["rep"]) for r in _done(a.out)}
    for rep in range(a.repeat):
        for problem, n_adam in (("Heat 1D (2k points)", 2000), ("Poisson 2D (20k points)", 300), ("Heat 2D (30k points)", 300)):
            for n in (THREADS if rep % 2 == 0 else THREADS[::-1]):
                if (problem, n, rep) in done:
                    continue
                e = dict(os.environ, OMP_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n), CUDA_VISIBLE_DEVICES="-1")
                p = subprocess.run([sys.executable, __file__, "_thread_worker", "--n", str(n), "--problem", problem,
                                    "--adam", str(n_adam)], env=e, capture_output=True, text=True)
                m = re.search(r"^RESULT (.+)$", p.stdout, re.M)
                r = json.loads(m.group(1)) if m else {"error": (p.stderr or p.stdout)[-800:]}
                r.update(problem=problem, threads=n, rep=rep, adam=n_adam, env=env)
                _append(a.out, r)
                print(f"{problem:26s} {n:3d} threads (torch {r.get('torch_threads')}), rep {rep}: "
                      f"{r.get('adam_it_per_s', float('nan')):.1f} it/s", flush=True)


# --------------------------------------------------------------------------------------------------------------- tables
def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _rng(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (None, None)


def tables(a):
    out_dir = os.path.join(ROOT, "benchmarks", "performance")
    os.makedirs(out_dir, exist_ok=True)
    md = []
    ov = _done(os.path.join(a.folder, "overhead.jsonl"))
    if ov:
        md += ["### Table 2. Training time in the three execution modes: exported standalone script, internal Solve script and PINNStudio GUI (RTX 4090; each problem as loaded from the Examples list, see the settings in configs/)", "",
               "| Problem | Exported script (s) | Solve script (s) | PINNStudio GUI (s) | GUI vs. Solve (%) | Steps (export / solve / GUI) | Main-output L₂ (export / solve / GUI) |",
               "|---|---|---|---|---|---|---|"]
        for label, _, _ in PROBLEMS:
            rs = [r for r in ov if r["problem"] == label and r["rc"] == 0]
            if not rs:
                continue
            t = {m: [r["train"] for r in rs if r["mode"] == m] for m in MODES}
            med = {m: _med(t[m]) for m in MODES}
            cell = lambda m: ("–" if med[m] is None else f"{med[m]:.1f} ({_rng(t[m])[0]:.1f}–{_rng(t[m])[1]:.1f})")
            ovh = (f"{100 * (med['gui'] - med['solve']) / med['solve']:+.1f}" if med["gui"] and med["solve"] else "–")
            stp = " / ".join(str(_med([r["steps"] for r in rs if r["mode"] == m])) for m in MODES)
            l2 = " / ".join((lambda v: f"{v:.2e}" if v else "–")(_med([next(iter(r["l2"].values()), None) for r in rs if r["mode"] == m and r["l2"]])) for m in MODES)
            md.append(f"| {label} | {cell('export')} | {cell('solve')} | {cell('gui')} | {ovh} | {stp} | {l2} |")
        md += ["", "*Median of three seeds (range in brackets) of the training duration DeepXDE reports, summed over all "
               "optimization phases. This is not the total wall-clock time: building the data and network before "
               "training and the plots and error analysis after it are not included. GUI vs. Solve = 100 × (GUI − "
               "Solve script) / Solve script; a negative value is within the run-to-run variation, not a speed-up "
               "caused by the GUI. The same settings in all three execution modes of a problem; the settings each "
               "problem ran with are saved in configs/."
               " The L₂ errors are medians over the seeds; on the GPU, two runs of the same script with the same seed can end at slightly different errors, because some GPU operations do not add numbers in a fixed order (Allen-Cahn 3D: 4.04e-4 against 4.28e-4). The CPU check in the Export check section, where runs are repeatable, is the one that compares the two scripts.*", ""]
    tp = _done(os.path.join(a.folder, "throughput.jsonl"))
    if tp:
        md += ["### Supplementary Table S3. CPU versus GPU Adam training throughput (CPU: 8 threads)", "",
               "| Problem | CPU, s per 1,000 Adam steps | GPU, s per 1,000 Adam steps | GPU speedup |", "|---|---|---|---|"]
        for label, _, _ in PROBLEMS:
            c = _med([r["s_per_1000"] for r in tp if r["problem"] == label and r["device"] == "cpu" and r["rc"] == 0])
            g = _med([r["s_per_1000"] for r in tp if r["problem"] == label and r["device"] == "gpu" and r["rc"] == 0])
            if c is None and g is None:
                continue
            sp = (f"{c / g:.1f}×" if c and g and c >= g else (f"CPU {g / c:.1f}× faster" if c and g else "–"))
            md.append(f"| {label} | {c if c is not None else '–'} | {g if g is not None else '–'} | {sp} |")
        md += ["", "*Median of the repeats; a separately timed warm-up comes first.*", ""]
    th = _done(os.path.join(a.folder, "threads.jsonl"))
    if th:
        probs = list(dict.fromkeys(r["problem"] for r in th))
        md += ["### Supplementary Table S4. CPU-thread scaling of Adam throughput (iterations per second)", "",
               "| Threads | " + " | ".join(probs) + " |", "|---" * (len(probs) + 1) + "|"]
        for n in THREADS:
            row = [str(n)]
            for p in probs:
                v = _med([r.get("adam_it_per_s") for r in th if r["problem"] == p and r["threads"] == n])
                row.append(f"{v:.1f}" if v else "–")
            md.append("| " + " | ".join(row) + " |")
        md += ["", "*Median of the repeats; each thread count in a fresh process.*", ""]
    rows = ov + tp + th
    if rows:
        e = rows[0].get("env", {})
        md += [f"*Environment: {e.get('gpu')} (driver {e.get('driver')}), {e.get('cpu')}; Python {e.get('python')}, "
               f"PyTorch {e.get('torch')}, CUDA {e.get('cuda')}, DeepXDE {e.get('deepxde')}; commit {e.get('commit')}.*", ""]
    with open(os.path.join(out_dir, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    import csv
    keys = sorted({k for r in rows for k in r if k not in ("env", "train_calls")})
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    print("wrote", os.path.join(out_dir, "tables.md"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("overhead")
    p.add_argument("out")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--only", default="")
    p.add_argument("--offscreen", action="store_true")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every phase's steps (a quick check only)")
    p = sub.add_parser("throughput")
    p.add_argument("out")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--adam", type=int, default=2000)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--only", default="")
    p.set_defaults(steps={"Allen-Cahn 3D (Wight & Zhao)": 300, "Allen-Cahn 2D (Mattey & Ghosh)": 500})
    p = sub.add_parser("threads")
    p.add_argument("out")
    p.add_argument("--repeat", type=int, default=3)
    p = sub.add_parser("_thread_worker")
    p.add_argument("--n", type=int)
    p.add_argument("--problem")
    p.add_argument("--adam", type=int)
    p = sub.add_parser("tables")
    p.add_argument("folder")
    a = ap.parse_args()
    {"overhead": overhead, "throughput": throughput, "threads": threads, "_thread_worker": _thread_worker,
     "tables": tables}[a.cmd](a)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

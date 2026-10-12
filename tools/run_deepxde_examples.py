#!/usr/bin/env python3
"""Train DeepXDE's examples (pinnstudio/core/deepxde_examples.py) outside
the GUI and summarise their Error Analysis results.

    python tools/run_deepxde_examples.py --out runs/            # full settings
    python tools/run_deepxde_examples.py --smoke --out runs/    # a few hundred iterations each
    python tools/run_deepxde_examples.py --only "Burgers" --export

Each example runs the script the GUI's Solve button runs (or, with
--export, the "Export DeepXDE Script" one) in <out>/<example>/, with its
console output saved to log.txt there. The summary lists the exit code,
the run time, the L2 relative errors printed by Error Analysis and, for
inverse problems, the inferred parameters.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from pinnstudio.core import deepxde_examples as dx  # noqa: E402
from pinnstudio.core.codegen import generate_clean_script, generate_script  # noqa: E402

REF_ROOT = os.path.join(_REPO, "reference_data")


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _smoke(cfg, n=300):
    """Same problem, a few hundred iterations: checks that it runs."""
    cfg.iterations = min(cfg.iterations, n)
    cfg.iterations2 = min(cfg.iterations2, n // 3)
    if cfg.optimizer_scheduler:
        phases = json.loads(cfg.scheduler_phases)
        for p in phases:
            p["iterations"] = min(int(p["iterations"]), n)
        cfg.scheduler_phases = json.dumps(phases)
    cfg.rar_cycles = min(cfg.rar_cycles, 2)
    cfg.rar_adam_iters = min(cfg.rar_adam_iters, n // 3)
    cfg.rar_lbfgs_iters = min(cfg.rar_lbfgs_iters, n // 3)
    cfg.rar_candidates = min(cfg.rar_candidates, 5000)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="run examples whose name contains this text ('|' separates several)")
    ap.add_argument("--skip", default="", help="leave out examples whose name contains this text ('|' separates several)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--export", action="store_true", help="run the exported script instead of the live one")
    ap.add_argument("--timeout", type=float, default=7200)
    args = ap.parse_args()

    wanted = [s for s in args.only.split("|") if s]
    skipped = [s for s in args.skip.split("|") if s]
    examples = [e for e in dx.EXAMPLES if (not wanted or any(s in e.name for s in wanted))
                and not any(s in e.name for s in skipped)]
    gen = generate_clean_script if args.export else generate_script
    rows = []
    for ex in examples:
        d = os.path.abspath(os.path.join(args.out, _slug(ex.name)))
        os.makedirs(d, exist_ok=True)
        cfg = dx.build_config(ex, REF_ROOT, keep={"save_dir": d})
        if args.smoke:
            _smoke(cfg)
        with open(os.path.join(d, "run.py"), "w") as fh:
            fh.write(gen(cfg))
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True,
                               timeout=args.timeout)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = "timeout", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), ""
        dt = time.time() - t0
        with open(os.path.join(d, "log.txt"), "w") as fh:
            fh.write(out + "\n--- stderr ---\n" + err)
        l2 = re.findall(r"L2=([0-9.eE+-]+)", out)
        true = {v["name"]: v.get("true") for v in json.loads(cfg.inverse_variables_json or "[]")}
        params = [f"{n}={v} (true {true.get(n)})" for n, v in re.findall(r"=== (\w+) final value: (\S+) ===", out)]
        tail = ""
        if rc != 0:
            tail = " | ".join(line for line in (err or out).strip().splitlines()[-3:])
        rows.append((ex.name, rc, dt, l2, tail))
        print(f"{'OK ' if rc == 0 else 'ERR'} {ex.name:55s} {dt:7.0f}s  L2: {' '.join(l2[-8:])}  "
              f"{'  '.join(params)}  {tail[:400]}",
              flush=True)
    bad = [r for r in rows if r[1] != 0]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} ran without error")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

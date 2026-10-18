#!/usr/bin/env python3
"""Which line of Solve's script makes it slower than the plain script?

Builds the plain script and Solve's script for an example (as
timing_study.py does), then variants of Solve's script with one setting
taken out, and times each a few times on the GPU (training only).

    python benchmarks/timing_ab.py OUT.json [--only NAME] [--repeat 3]
"""
import argparse
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_study as ts  # noqa: E402

VARIANTS = {
    "solve": [],
    "solve, TF32 lines out": [r"^\s*torch\.backends\.(cuda\.matmul|cudnn)\.allow_tf32\s*=.*$"],
    "solve, memory limit out": [r"^\s*torch\.cuda\.set_per_process_memory_fraction\(.*$"],
    "solve, all torch.backends lines out": [r"^\s*torch\.backends\.\S+\s*=.*$",
                                            r"^\s*torch\.cuda\.set_per_process_memory_fraction\(.*$"],
}


def strip(text, patterns):
    for p in patterns:
        text, n = re.subn(p, lambda m: re.match(r"\s*", m.group(0)).group(0) + "pass  # (A/B: taken out)",
                          text, flags=re.M)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    from pinnstudio.core.codegen import generate_clean_script, generate_script
    app, w = ts._window()
    res = []
    base = tempfile.mkdtemp(prefix="pinnstudio_ab_")
    for name, example, iters in ts.PROBLEMS:
        if a.only and a.only.lower() not in name.lower():
            continue
        ts.load(app, w, example)
        for rep in range(a.repeat):
            runs = {"script": generate_clean_script}
            runs.update({v: (lambda c, pats=pats: strip(generate_script(c), pats)) for v, pats in VARIANTS.items()})
            for v, gen in runs.items():
                d = os.path.join(base, f"{len(res):03d}")
                os.makedirs(d)
                cfg = ts.config_for(w, iters, d)
                r = ts.run_script(gen(cfg), "gpu", d)
                r.update(problem=name, variant=v, rep=rep)
                res.append(r)
                print(f"{name:30s} {v:38s} rep {rep}: train {r['train']:7.2f} s  rc={r['rc']}", flush=True)
                with open(a.out, "w") as f:
                    json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

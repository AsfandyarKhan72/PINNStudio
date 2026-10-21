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


def _plain_pde(solve_text, clean_text):
    """Solve's script, with the plain script's pde() in place of _pde_standard."""
    m = re.search(r"^def pde\(x, y\):\n(?:(?:    .*|)\n)+", clean_text, flags=re.M)
    if not m:
        return solve_text
    block = m.group(0).replace("def pde(", "def _plain_pde(")
    return solve_text.replace("if not _IS_FECR:\n    pde = _pde_standard\n",
                              "if not _IS_FECR:\n    pde = _pde_standard\n" + block + "\npde = _plain_pde  # (A/B)\n", 1)


def _no_cuda_setup(solve_text, clean_text):
    return solve_text.replace("if torch.cuda.is_available():\n    torch.cuda.init()",
                              "if False:  # (A/B: CUDA setup taken out)\n    torch.cuda.init()", 1)


def _no_warmup(solve_text, clean_text):
    return re.sub(r"^(\s*)_dummy = torch\.zeros\(.*?\n(?:\1.*\n)*?\1del _dummy, _loss\n",
                  lambda m: m.group(1) + "pass  # (A/B: warm-up taken out)\n", solve_text, count=1, flags=re.M)


def _no_init_set_device(solve_text, clean_text):
    return re.sub(r"^(\s*)torch\.cuda\.(init\(\)|set_device\(.*\))\s*$",
                  lambda m: m.group(1) + "pass  # (A/B: taken out)", solve_text, flags=re.M)


def _no_empty_cache(solve_text, clean_text):
    return re.sub(r"^(\s*)torch\.cuda\.empty_cache\(\)\s*$",
                  lambda m: m.group(1) + "pass  # (A/B: taken out)", solve_text, flags=re.M)


TEXT_VARIANTS = {
    "solve, warm-up out": _no_warmup,
    "solve, cuda.init/set_device out": _no_init_set_device,
    "solve, empty_cache out": _no_empty_cache,
    "solve, plain script's pde()": _plain_pde,
    "solve, CUDA setup block out": _no_cuda_setup,
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
    ap.add_argument("--variants", default="", help="names to run, separated by ; (default: all)")
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
            runs.update({v: (lambda c, fn=fn: fn(generate_script(c), generate_clean_script(c)))
                         for v, fn in TEXT_VARIANTS.items()})
            if a.variants:
                keep = {x.strip() for x in a.variants.split(";")}
                runs = {k: g for k, g in runs.items() if k in keep}
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

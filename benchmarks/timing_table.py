#!/usr/bin/env python3
"""The README tables from a timing_study.py result file.

    python benchmarks/timing_table.py timing_v2.json

Each time is the best of the repeats (the least disturbed run), as seconds
of training per 1,000 iterations (Adam and L-BFGS together), so the CPU
runs (fewer iterations) and the GPU runs can be compared.
"""
import collections
import json
import sys

MODES = ["script", "solve", "window", "window-quiet"]


def best(results):
    b = collections.defaultdict(dict)
    for r in results:
        if r.get("rc") != 0 or not r.get("train"):
            continue
        k = (r["problem"], r["device"])
        cur = b[k].get(r["mode"])
        if cur is None or per_k(r) < per_k(cur):
            b[k][r["mode"]] = r
    return b


def per_k(r):
    n = r.get("steps") or sum(r["iters"].values())     # the steps taken (L-BFGS may stop early)
    return 1000.0 * r["train"] / n


def main():
    res = json.load(open(sys.argv[1]))
    b = best(res)
    problems = list(dict.fromkeys(r["problem"] for r in res))
    print("| Problem | Device | Plain script | Solve's script | PINNStudio window | window, live plot off |")
    print("|---|---|---|---|---|---|")
    for p in problems:
        for dev in ("gpu", "cpu"):
            m = b.get((p, dev))
            if not m or "script" not in m:
                continue
            base = per_k(m["script"])
            cells = []
            for mode in MODES:
                if mode not in m:
                    cells.append("–")
                    continue
                v = per_k(m[mode])
                cells.append(f"{v:.2f} s" if mode == "script" else f"{v:.2f} s ({100 * (v / base - 1):+.0f}%)")
            print(f"| {p} | {dev.upper()} | " + " | ".join(cells) + " |")
    print()
    print("| Problem | GPU vs CPU, plain script (per 1,000 iterations) | start-up and saving, whole run minus training |")
    print("|---|---|---|")
    for p in problems:
        g, c = b.get((p, "gpu"), {}).get("script"), b.get((p, "cpu"), {}).get("script")
        if g and c:
            ratio = per_k(c) / per_k(g)
            other = g["total"] - g["train"]
            faster = f"GPU {ratio:.1f}× faster" if ratio >= 1 else f"CPU {1 / ratio:.1f}× faster"
            print(f"| {p} | {per_k(g):.2f} s on the GPU, {per_k(c):.2f} s on the CPU: {faster} | {other:.1f} s |")


if __name__ == "__main__":
    main()

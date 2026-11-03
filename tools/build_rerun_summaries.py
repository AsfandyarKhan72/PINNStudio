#!/usr/bin/env python3
"""Summary records (primary and all-times errors) of benchmark runs, rebuilt from the runs' own error files.

The benchmark pages judge a forward run by its main output's relative L2 error at the last reference time, and a
Cahn-Hilliard run also by the chemical potential's (tools/make_benchmarks_readme.py: _judged). Those values come from
a primary file (--primary) and the error over all reference times from a spacetime file (--spacetime). This script
writes both for a set of run folders, from the files each run's Error Analysis wrote:

    <out>/<run folder>/<variant>_s<seed>/<timestamp>/error_analysis/error_metrics_<output>.txt   (t, L2_relative, Mean_abs_error)
    <out>/<run folder>/<variant>_s<seed>/<timestamp>/error_analysis/error_spacetime_<output>.txt (all times together)

and checks every value against two independent copies written by the same run: the L2 lines of the run's log.txt
(4 significant figures) and the `l2` list of the run's row in results.jsonl (tools/tuning_study.py, 5 significant
figures). A value that does not appear in both copies stops the script. Nothing is computed from rounded tables.

The main output is the first of MAIN that the run has (as in make_benchmarks_readme.py); the all-times value is the
main output's. A run that has no error files (it timed out or failed) is written with "l2": null, which tells
make_benchmarks_readme.py to drop any older entry for that run instead of using it.

    python tools/build_rerun_summaries.py OUT_DIR [OUT_DIR ...] --primary P.json --spacetime S.json --manifest M.json
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

MAIN = ("u", "c", "phi", "h", "U", "solution")
LOG_L2 = re.compile(r"t=([-\d.]+) (?:—|--|.) L(?:2|₂)=([\d.e+-]+)(?:, MeanAbs=([\d.e+-]+))?")
LOG_ALL = re.compile(r"All (\d+) reference times together: L(?:2|₂)=([\d.e+-]+)")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def close(a, b, sig):
    """a (full precision) printed to `sig` significant figures equals b"""
    return float(f"{a:.{sig - 1}e}") == float(f"{b:.{sig - 1}e}") or abs(a - b) <= 0.5 * 10 ** (1 - sig) * abs(a) * 1.0001


def run_folder(out, folder, variant, seed):
    d = os.path.join(out, folder, f"{variant}_s{seed}")
    ea = sorted(glob.glob(os.path.join(d, "*", "error_analysis")))
    return d, ea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outs", nargs="+", help="tools/tuning_study.py --out folders (each with results.jsonl)")
    ap.add_argument("--primary", required=True)
    ap.add_argument("--spacetime", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    prim, st, man, problems = {}, {}, {"runs": [], "inputs": {}}, []
    for out in a.outs:
        res = os.path.join(out, "results.jsonl")
        man["inputs"][os.path.basename(os.path.dirname(os.path.abspath(out))) + "/results.jsonl"] = sha(res)
        for line in open(res, encoding="utf-8"):
            r = json.loads(line)
            folder = slug(r["example"]) + ("_inv" if r["inverse"] else "")
            d, ea = run_folder(out, folder, r["variant"], r["seed"])
            key = f"{folder}/{r['variant']}_s{r['seed']}"
            entry = {"seed": str(r["seed"])}
            rec = {"run": key, "example": r["example"], "variant": r["variant"], "seed": r["seed"], "rc": r["rc"]}
            if r["rc"] != 0 or not ea:
                entry.update(l2=None, note=f"no error files (rc={r['rc']}): any older entry for this run is dropped")
                prim.setdefault(folder, {}).setdefault(r["variant"], []).append(entry)
                rec["status"] = "no result"
                man["runs"].append(rec)
                continue
            if len(ea) != 1:
                sys.exit(f"{key}: {len(ea)} error_analysis folders, expected one")
            ea = ea[0]
            log = open(os.path.join(d, "log.txt"), encoding="utf-8", errors="replace").read()
            logged = [(float(t), float(v)) for t, v, _ in LOG_L2.findall(log)]
            logged_all = [float(v) for _, v in LOG_ALL.findall(log)]
            fields, outs, absd, times, allt, files = [], {}, {}, {}, {}, {}
            for f in sorted(glob.glob(os.path.join(ea, "error_metrics_*.txt"))):
                name = os.path.basename(f)[len("error_metrics_"):-4]
                rows = list(csv.DictReader(open(f, encoding="utf-8")))
                ts = [float(x["t"]) for x in rows]
                l2 = [float(x["L2_relative"]) for x in rows]
                ab = [float(x["Mean_abs_error"]) for x in rows]
                sp = os.path.join(ea, f"error_spacetime_{name}.txt")
                al = float(list(csv.DictReader(open(sp, encoding="utf-8")))[0]["L2_relative_all_times"])
                # cross-checks: every value is in the log (4 s.f.) and in results.jsonl (5 s.f.)
                for t, v in zip(ts, l2):
                    if not any(abs(t - lt) < 1e-9 and close(v, lv, 5) for lt, lv in logged):
                        sys.exit(f"{key} {name} t={t}: {v} not in log.txt")
                    if not any(close(v, x, 5) for x in r["l2"]):
                        sys.exit(f"{key} {name} t={t}: {v} not in results.jsonl l2")
                if not any(close(al, x, 5) for x in logged_all) or not any(close(al, x, 5) for x in r["l2"]):
                    sys.exit(f"{key} {name}: all-times {al} not in log.txt / results.jsonl")
                fields.append(name)
                outs[name], absd[name], times[name], allt[name] = l2[-1], ab[-1], ts, al
                files[os.path.relpath(f, out)] = sha(f)
                files[os.path.relpath(sp, out)] = sha(sp)
            main_out = next((m for m in MAIN if m in outs), fields[0])
            entry.update(out=main_out, t=times[main_out][-1], l2=outs[main_out], params=r.get("params") or {},
                         outs=outs, abs=absd)
            prim.setdefault(folder, {}).setdefault(r["variant"], []).append(entry)
            st[key] = allt[main_out]
            rec.update(status="ok", main=main_out, fields=fields, times=times[main_out], final_time=outs,
                       all_times=allt, files=files, log_sha256=sha(os.path.join(d, "log.txt")),
                       checked="every value found in log.txt (4 s.f.) and results.jsonl (5 s.f.)")
            man["runs"].append(rec)
    json.dump(prim, open(a.primary, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(st, open(a.spacetime, "w", encoding="utf-8"), indent=1)
    json.dump(man, open(a.manifest, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{sum(1 for x in man['runs'] if x['status'] == 'ok')} runs written, "
          f"{sum(1 for x in man['runs'] if x['status'] != 'ok')} without result")


if __name__ == "__main__":
    main()

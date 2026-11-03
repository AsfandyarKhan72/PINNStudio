#!/usr/bin/env python3
"""Benchmark result files with the errors measured against the regenerated reference data.

The benchmark pages are made by tools/make_tuning_page.py and tools/make_benchmarks_readme.py from the study
results: the tools/tuning_study.py result files (*.jsonl, one row per run), primary.json (the main output's error
per run, from the runs' logs) and spacetime.json (the error over all reference times). This script writes copies
of those files in which, for every run whose problem uses replaced reference data:

  - a forward run's errors are the ones reeval_runs.py measured against the new reference (its training did not
    use the reference data). Each value is replaced in place, in the order the run logged its errors, so a row
    keeps its structure. A run that reeval_runs.py could not reproduce exactly against COMSOL (the Mathias et al.
    method run, 0.5% off) gets its logged value plus the change reeval_runs.py measured;
  - an inverse run whose measurements changed is left out: it is replaced by the run trained again on the new
    data (the --new result files, e.g. done/a10_refregen/out/results.jsonl, and their primary.json entries).
    The ball's measurements did not change (identical values), so its inverse runs stay, with re-measured errors.

    python benchmarks/reference_regeneration/apply_reeval.py REEVAL_JSON OUT_DIR --jsonl A.jsonl,B.jsonl,...
        --primary primary.json [--spacetime S1.json,S2.json] [--new-jsonl N.jsonl] [--new-primary NP.json]

Writes OUT_DIR/<same file names> and OUT_DIR/changes.md (every value that changed, old and new).
"""
import argparse
import json
import os
import re

RETRAINED_INVERSE = {"Template_2D_Burgers_Mathias_inv", "Template_2D_Poisson_Disk_inv"}
# second round (Allen-Cahn of Mattey & Ghosh and of Wight & Zhao, Heat 2D and 3D): run on the first round's output
# with --retrained ROUND2, so the first round's retrained runs are kept
ROUND2 = "Template_2D_Allen_Cahn_Mattey_Ghosh_inv,Template_2D_Heat_inv,Template_3D_Heat_inv,Template_2D_Allen_Cahn_Wight_Zhao_inv"
DELTA_TOL = 2e-3


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def log_sequence(log):
    """The L2 values in a log, in order, as keys: ('t', output, t) or ('all', output)."""
    seq, cur = [], None
    for ln in log.splitlines():
        m = re.search(r"Loaded ground truth.*?from (\S+)", ln)
        if m:
            f = os.path.basename(m.group(1)).replace(".txt", "")
            f = re.sub(r"^t_[-\d.e]+_?", "", f)
            cur = f if f in ("u", "v") else "u"
            continue
        for m in re.finditer(r"L(?:2|₂)=([0-9.eE+-]+)", ln):
            t = re.search(r"t=([-\d.]+) .", ln)
            seq.append((("t", cur, float(t.group(1))) if t else ("all", cur), float(m.group(1))))
    return seq


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reeval")
    ap.add_argument("out")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--primary", required=True)
    ap.add_argument("--spacetime", default="")
    ap.add_argument("--new-jsonl", default="")
    ap.add_argument("--new-primary", default="")
    ap.add_argument("--bundles", required=True, help="the bundle folder of reeval_runs.py (for the run logs)")
    ap.add_argument("--retrained", default="", help="inverse run folders trained again ('|'- or ','-separated; "
                    "ROUND2 for the second round); default: the first round's")
    a = ap.parse_args()
    global RETRAINED_INVERSE
    if a.retrained:
        RETRAINED_INVERSE = set((ROUND2 if a.retrained == "ROUND2" else a.retrained).replace("|", ",").split(","))
    os.makedirs(a.out, exist_ok=True)
    changes = ["| file | run | value | old | new |", "|---|---|---|---|---|"]

    runs = {}
    for r in json.load(open(a.reeval)):
        if "outputs" not in r or not r["outputs"]:
            continue
        ex, rv = r["run"].split("/out/")[1].split("/")[:2]
        variant, seed = rv.rsplit("_s", 1)
        r["delta"] = not r.get("reproduces_log")
        if r["delta"] and (r.get("log_max_rel_dev") or 1) > 1e-2:
            continue                                   # not usable at all
        runs[(ex, variant, seed)] = r

    def new_value(r, key, logged):
        """The new value of one logged error (key from log_sequence)."""
        outs = r["outputs"]
        # one-output problems (h of Allen-Cahn of Mattey & Ghosh, phi of Wight & Zhao): their logs say u
        o = outs[key[1]] if key[1] in outs else (outs["u"] if "u" in outs else next(iter(outs.values())) if len(outs) == 1 else outs["u"])
        if key[0] == "all":
            old, new = o["old"]["all_times"], o["new"]["all_times"]
        else:
            j = min(range(len(o["t"])), key=lambda k: abs(o["t"][k] - key[2]))
            old, new = o["old"]["l2"][j], o["new"]["l2"][j]
        return logged + (new - old) if r["delta"] else new

    logs = {}
    for f in os.listdir(a.bundles):
        if f.endswith(".json"):
            m = json.load(open(os.path.join(a.bundles, f)))
            ex, rv = m["run"].split("/out/")[1].split("/")[:2]
            variant, seed = rv.rsplit("_s", 1)
            logs[(ex, variant, seed)] = m["texts"].get("log.txt", "")

    # result rows
    for path in [p for p in a.jsonl.split(",") if p]:
        rows = []
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            key = (slug(row["example"]) + ("_inv" if row["inverse"] else ""), row["variant"], str(row["seed"]))
            if key[0] in RETRAINED_INVERSE:
                changes.append(f"| {os.path.basename(path)} | {'/'.join(key)} | row | kept | left out (trained again) |")
                continue
            if key in runs and row.get("l2"):
                seq = log_sequence(logs.get(key, ""))
                if [round(v, 12) for _, v in seq] != [round(v, 12) for v in row["l2"]]:
                    raise SystemExit(f"{path}: {key}: the row's errors are not the log's")
                new = [new_value(runs[key], k, v) for k, v in seq]
                for (k, v), n in zip(seq, new):
                    if f"{v:.4e}" != f"{n:.4e}":
                        changes.append(f"| {os.path.basename(path)} | {'/'.join(key)} | L2 {k[0]} {k[1]} "
                                       f"{k[2] if k[0] == 't' else ''} | {v:.4e} | {n:.4e} |")
                row["l2"] = [float(f"{n:.5g}") for n in new]
                row["reference"] = "regenerated"
            rows.append(row)
        with open(os.path.join(a.out, os.path.basename(path)), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # primary.json
    P = json.load(open(a.primary))
    for ex in list(P):
        if ex in RETRAINED_INVERSE:
            del P[ex]
            changes.append(f"| primary.json | {ex} | all runs | kept | left out (trained again) |")
            continue
        for variant, recs in P[ex].items():
            for rec in recs:
                r = runs.get((ex, variant, str(rec["seed"])))
                if not r:
                    continue
                t_last = rec["t"]
                for o_name in list(rec["outs"]):
                    outs = r["outputs"]
                    o = outs[o_name] if o_name in outs else (outs["u"] if "u" in outs or len(outs) != 1
                                                             else next(iter(outs.values())))
                    j = min(range(len(o["t"])), key=lambda k: abs(o["t"][k] - t_last))
                    old_l2, new_l2 = o["old"]["l2"][j], o["new"]["l2"][j]
                    val = rec["outs"][o_name] + (new_l2 - old_l2) if r["delta"] else new_l2
                    changes.append(f"| primary.json | {ex}/{variant}_s{rec['seed']} | L2 {o_name} at t = {t_last} | "
                                   f"{rec['outs'][o_name]:.4e} | {val:.4e} |")
                    rec["outs"][o_name] = float(f"{val:.5g}")
                    if o_name == rec["out"]:
                        rec["l2"] = rec["outs"][o_name]
                    if o_name in rec.get("abs", {}):
                        old_a, new_a = o["old"]["abs"][j], o["new"]["abs"][j]
                        rec["abs"][o_name] = float(f"{(rec['abs'][o_name] + new_a - old_a) if r['delta'] else new_a:.7g}")
                rec["reference"] = "regenerated"
    if a.new_primary:
        for ex, vs in json.load(open(a.new_primary)).items():
            for variant, recs in vs.items():
                P.setdefault(ex, {}).setdefault(variant, []).extend(recs)
    json.dump(P, open(os.path.join(a.out, "primary.json"), "w"), indent=1)

    # spacetime
    for path in [p for p in a.spacetime.split(",") if p]:
        S = json.load(open(path))
        for k in list(S):
            ex, rv = k.split("/")
            variant, seed = rv.rsplit("_s", 1)
            if ex in RETRAINED_INVERSE:
                del S[k]
                continue
            r = runs.get((ex, variant, seed))
            if r and ("u" in r["outputs"] or len(r["outputs"]) == 1):
                o = r["outputs"]["u"] if "u" in r["outputs"] else next(iter(r["outputs"].values()))
                val = S[k] + o["new"]["all_times"] - o["old"]["all_times"] if r["delta"] else o["new"]["all_times"]
                changes.append(f"| {os.path.basename(path)} | {k} | L2 over all reference times | {S[k]:.4e} | {val:.4e} |")
                S[k] = val
        json.dump(S, open(os.path.join(a.out, os.path.basename(path)), "w"), indent=1)

    if a.new_jsonl:
        for path in a.new_jsonl.split(","):
            with open(os.path.join(a.out, os.path.basename(path)), "w", encoding="utf-8") as fh:
                fh.write(open(path, encoding="utf-8").read())
    open(os.path.join(a.out, "changes.md"), "w").write("\n".join(changes) + "\n")
    print(len(changes) - 2, "changes ->", a.out)


if __name__ == "__main__":
    main()

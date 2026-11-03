"""Builds the corrected Figure 7 data (fig7_corrected.json) from records in this repository; nothing is retrained,
interpolated or invented, and the inputs are not modified.

  python benchmarks/figures/fig7_corrected/build_fig7_corrected.py

Inputs (all seed 2026, the Figure 7 protocol of benchmarks/plans/fig7_v2.json):
  - benchmarks/figures/fig7_v2.json: the 52 runs behind the approved figure, with panel (c), 2D Allen-Cahn of
    Mattey & Ghosh, already re-scored against the regenerated reference (ee6d9dd;
    benchmarks/reference_regeneration/convergence/f7_ac2dmg_reeval.json). Panel (a) and the runs below that are not
    replaced are taken from it unchanged.
  - panel (b), 1D Cahn-Hilliard of Wight & Zhao, windowed runs (windows, bcpinn, windowsEq, bcpinnEq): rerun 1
    (time-window loss-weight fix, audit/v1.6.0 79edd5d), per-time error files written by the runs themselves,
    benchmarks/records/reruns_2026-11-01/rerun1_ch1d_per_time/f7v2_ch1d_<run>/error_metrics_{u,mu}.txt.
  - panel (d), 2D Cahn-Hilliard of Mattey & Ghosh, the same four windowed runs: rerun 3,
    benchmarks/records/reruns_2026-11-01/rerun3_per_time_<run>.json.
Window 1 of every replaced run (t <= 0.25 in (b), t <= 0.2 in (d)) is unaffected by the fix and is checked to agree
with the old values (to the 5 significant figures of the old logs); the times and fields must match exactly.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RR = os.path.join(ROOT, "benchmarks", "records", "reruns_2026-11-01")
WINDOWED = ("windows", "bcpinn", "windowsEq", "bcpinnEq")


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def main(out=os.path.join(HERE, "fig7_corrected.json")):
    src = os.path.join(ROOT, "benchmarks", "figures", "fig7_v2.json")
    d = json.load(open(src, encoding="utf-8"))
    assert len(d) == 52, len(d)
    for k in d:
        d[k].setdefault("source", rel(src) + " (unchanged)")
    # (b) 1D Cahn-Hilliard (Wight & Zhao): rerun 1, the runs' own error_metrics files
    for k in WINDOWED:
        r = d[f"f7v2_ch1d_{k}"]
        rd = os.path.join(RR, "rerun1_ch1d_per_time", f"f7v2_ch1d_{k}")
        for f in ("u", "mu"):
            rows = list(csv.DictReader(open(os.path.join(rd, f"error_metrics_{f}.txt"), encoding="utf-8")))
            new = [[float(x["t"]), float(x["L2_relative"])] for x in rows]
            old = r["l2"][f]
            assert [t for t, _ in new] == [t for t, _ in old], (k, f)
            for (t, a), (_, b) in zip(old, new):
                if t <= 0.25:                                     # window 1: not affected by the fix
                    assert abs(a - b) / b < 1e-4, ("window 1 differs", k, f, t, a, b)
            r["l2"][f] = new
        res = json.load(open(os.path.join(rd, "result.json"), encoding="utf-8"))
        assert res["rc"] == 0
        r["seconds"] = res["seconds"]
        r["source"] = rel(rd) + "/error_metrics_{u,mu}.txt (rerun 1, job zzzzzp1_ch_f7_1d, audit/v1.6.0 79edd5d)"
    # (d) 2D Cahn-Hilliard (Mattey & Ghosh): rerun 3
    for k in WINDOWED:
        r = d[f"f7v2_ch2dmg_{k}"]
        p = os.path.join(RR, f"rerun3_per_time_{k}.json")
        pt = json.load(open(p, encoding="utf-8"))
        assert pt["rc"] == 0
        for f in ("h", "mu"):
            old, new = r["l2"][f], pt["l2"][f]
            assert [t for t, _ in new] == [t for t, _ in old], (k, f)
            for (t, a), (_, b) in zip(old, new):
                if t <= 0.2:
                    assert abs(a - b) / b < 1e-4, ("window 1 differs", k, f, t, a, b)
        r["l2"] = {"h": pt["l2"]["h"], "mu": pt["l2"]["mu"]}
        r["seconds"] = pt["seconds"]
        r["source"] = rel(p) + " (rerun 3)"
    json.dump(d, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", rel(out))


if __name__ == "__main__":
    main(*sys.argv[1:])

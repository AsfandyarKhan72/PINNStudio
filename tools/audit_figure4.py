#!/usr/bin/env python3
"""Audit of Figure 4's numbers: for each of the paper's examples, where the starting and tuned values come from,
how many seeds each rests on, which variant was chosen and by which rule, and the summary statistics, computed
from the same inputs and code as tools/make_benchmarks_readme.py (same arguments).

    python tools/audit_figure4.py RESULTS.jsonl ... --primary primary.json --papers ... --spacetime ...
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_benchmarks_readme as mb  # noqa: E402

mtp = mb.mtp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--primary", default="")
    ap.add_argument("--papers", default="")
    ap.add_argument("--spacetime", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    for slug, vs in json.load(open(a.primary, encoding="utf-8")).items():
        for variant, runs in vs.items():
            for r in runs:
                mtp.PRIMARY[(slug, variant, str(r["seed"]))] = r["l2"]
                if r.get("outs"):
                    mb.OUTS[(slug, variant, str(r["seed"]))] = dict(r["outs"])
    rows = mtp._load(a.results)
    summ = mtp.summarize(rows)
    papers = mtp._load([x for x in a.papers.split(",") if x]) if a.papers else []
    st = {}
    for f in [x for x in a.spacetime.split(",") if x]:
        st.update(json.load(open(f, encoding="utf-8")))
    mb.SPACETIME.update(st)
    try:
        readme = json.load(open(mtp.README_RESULTS, encoding="utf-8"))
    except OSError:
        readme = {}
    recs = mb.build(summ, papers, st, readme)
    names = mb._paper_names()
    by = {r["entry"].name: r for r in recs}
    P = [by[n] for n in names if n in by]
    out = []
    print(f"{'example':<44} {'start':>9} {'start src':<22} {'tuned':>9} {'n':>2} {'variant':<14} | "
          f"{'inv err':>9} {'n':>2} {'variant':<8} {'inv start src'}")
    for r in P:
        e = r["entry"]
        pm = [(v, r["paper_runs"][v]) for _, _, v in (r["paper"] or []) if v in (r.get("paper_runs") or {})]
        if r["deepxde"]:
            start, ssrc, sn = r.get("f_ref"), "DeepXDE settings", (r["f_plain_s"] or {}).get("n_base")
        elif pm:
            v, q = min(pm, key=lambda x: x[1]["l2"])
            start, ssrc, sn = q["l2"], f"paper method ({v})", q.get("n")
        else:
            start, ssrc, sn = None, "none (template)", None
        if r.get("f_best_paper"):
            q = r["paper_runs"][r["f_best_paper"]]
            tv, tn = r["f_best_paper"], q.get("n", 1)
        elif r.get("f_best_s"):
            fs = r["f_best_s"]
            tv, tn = fs["chosen"] + (" (best setup)" if r.get("f_best_is_setup") else ""), len(fs["variants"][fs["chosen"]])
        else:
            tv, tn = ("README" if r.get("f_best") is not None else "–"), (1 if r.get("f_best") is not None else 0)
        ip = r.get("i_s")
        ierr = mb._perr(r.get("i_best_par")) if r.get("i_best_par") else None
        iv = ip["chosen"] if ip else ("README" if r.get("i_src") == "README" else "–")
        inn = len(ip["variants"][ip["chosen"]]) if ip else (1 if r.get("i_src") == "README" else 0)
        rec = dict(example=e.name, source=e.source, start=start, start_source=ssrc, start_seeds=sn,
                   tuned=r.get("f_best"), tuned_variant=tv, tuned_seeds=tn,
                   inv_err=ierr, inv_variant=iv, inv_seeds=inn, inv_src=r.get("i_src"),
                   field_unknown=bool(r.get("i_field")))
        out.append(rec)
        f = lambda x: "–" if x is None else f"{x:.3g}"
        print(f"{e.name[:44]:<44} {f(start):>9} {ssrc[:22]:<22} {f(r.get('f_best')):>9} {tn:>2} {tv[:14]:<14} | "
              f"{f(ierr):>9} {inn:>2} {iv[:8]:<8} {r.get('i_src')}")
    s = mb._stats(P)
    sa = mb._stats(recs)
    print("\nthese 25:", json.dumps(s))
    print(f"all {len(recs)}:", json.dumps(sa))
    if a.json:
        json.dump(dict(rows=out, stats25=s, stats_all=sa, n_all=len(recs)), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

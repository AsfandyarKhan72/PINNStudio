"""Build the Figure 4 records, Supplementary Table S2 and the per-seed tables from recorded run summaries:
summary statistics, per-row detail, Table S2 rows, per-seed CSVs and the Figure 4 selection (pickled for drawing).

    cd <repo>; python tools/build_benchmark_records.py RESULTS... --primary P --papers A,B --spacetime S1,S2 --out DIR
"""
import argparse, json, os, pickle, statistics as st, sys, time
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import make_benchmarks_readme as mb  # noqa: E402
mtp = mb.mtp

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+"); ap.add_argument("--primary"); ap.add_argument("--papers")
ap.add_argument("--spacetime"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
mb.load_primary(a.primary)
summ = mtp.summarize(mtp._load(a.results))
papers = mtp._load([x for x in a.papers.split(",") if x])
stt = {}
for f in [x for x in a.spacetime.split(",") if x]:
    stt.update(json.load(open(f)))
mb.SPACETIME.update(stt)
try:
    readme = json.load(open(mtp.README_RESULTS))
except OSError:
    readme = {}
recs = mb.build(summ, papers, stt, readme)
by = {r["entry"].name: r for r in recs}
P = [by[n] for n in mb._paper_names() if n in by]
print("built", len(recs), "records in", round(time.time() - t0), "s", flush=True)

mb.inverse_per_seed_csv(recs, os.path.join(a.out, "inverse_per_seed.csv"))
mb.forward_per_seed_csv(recs, os.path.join(a.out, "forward_per_seed.csv"))
mb.table_s2_tex(recs, os.path.join(a.out, "table_s2.tex"))
pickle.dump(P, open(os.path.join(a.out, "fig4_rows.pkl"), "wb"))


def per_seed_worst(r):
    ip = r.get("i_s")
    if not ip:
        return None, None
    w = []
    for x in ip["variants"][ip["chosen"]]:
        e = [abs(x["params"][k] - t) / abs(t) for k, t in (x.get("true") or {}).items()
             if t not in (None, 0) and k in x.get("params", {})]
        if e:
            w.append(max(e))
    return (st.median(w) if w else None), [round(100 * v, 4) for v in w]


def detail(r):
    pm = [r["paper_runs"][v]["l2"] for _, _, v in (r["paper"] or []) if v in (r.get("paper_runs") or {})]
    s0 = r.get("f_ref") if r["deepxde"] else (min(pm) if pm else None)
    fs = r.get("f_best_s")
    seeds, fvars = None, {}
    if r.get("f_best_paper"):
        q = r["paper_runs"][r["f_best_paper"]]; tv, tn = r["f_best_paper"], q.get("n", 1)
    elif fs:
        tv = fs["chosen"]; tn = len(fs["variants"][tv])
        seeds = sorted(round(mtp._maxl2(x), 7) for x in fs["variants"][tv] if mtp._maxl2(x) is not None)
        for v, rs in fs["variants"].items():
            e = [mtp._maxl2(x) for x in rs if mtp._maxl2(x) is not None]
            fvars[v] = dict(n=len(e), median=st.median(e) if e else None, l2=sorted(e),
                            s=st.median([x.get("seconds") or 0 for x in rs]) if rs else None)
    else:
        tv, tn = None, 0
    ip = r.get("i_s")
    ivars = {}
    if ip:
        for v, rs in ip["variants"].items():
            w = []
            for x in rs:
                e = [abs(x["params"][k] - t) / abs(t) for k, t in (x.get("true") or {}).items()
                     if t not in (None, 0) and k in x.get("params", {})]
                if e:
                    w.append(max(e))
            ivars[v] = dict(n=len(rs), median_worst=st.median(w) if w else None, worst=[round(100 * q, 4) for q in w],
                            s=st.median([x.get("seconds") or 0 for x in rs]) if rs else None)
    psw, psl = per_seed_worst(r)
    return dict(name=r["entry"].name, start=s0, tuned=r.get("f_best"), variant=tv, n=tn, seeds_l2=seeds,
                fwd_variants=fvars, gain=(s0 / r["f_best"]) if (s0 and r.get("f_best")) else None,
                inv=mb._perr(r.get("i_best_par")) if r.get("i_best_par") else None,
                inv_variant=ip["chosen"] if ip else None, inv_n=len(ip["variants"][ip["chosen"]]) if ip else 0,
                inv_params=r.get("i_best_par"), inv_perseed_worst_median=psw, inv_perseed_worst_pct=psl,
                inv_variants=ivars, f_best_paper=r.get("f_best_paper"))


out = dict(stats25=mb._stats(P), stats_all=mb._stats(recs), rows=[detail(r) for r in P],
           all_rows=[detail(r) for r in recs])
json.dump(out, open(os.path.join(a.out, "stats.json"), "w"), indent=1, default=str)
print(json.dumps(out["stats25"]))
print("done in", round(time.time() - t0), "s")

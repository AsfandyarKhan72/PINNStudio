"""The benchmark selection, rebuilt from the committed records with the final-build inputs
(benchmarks/records/reruns_2026-11-01/final_build), gives the shipped Tuned presets and the statistics quoted
for the paper. A record missing or edited by hand shows up here."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "benchmarks", "records")
FB = os.path.join(REC, "reruns_2026-11-01", "final_build")
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _args(name):
    return [os.path.join(REC, l.strip()) for l in open(os.path.join(FB, name), encoding="utf-8") if l.strip()]


@pytest.fixture(scope="module")
def recs():
    import make_benchmarks_readme as mb
    mtp = mb.mtp
    mb.load_primary(",".join(_args("primary_arg.txt")))
    summ = mtp.summarize(mtp._load(_args("results_args.txt")))
    papers = mtp._load(_args("papers_arg.txt"))
    st = {}
    for f in _args("spacetime_arg.txt"):
        st.update(json.load(open(f, encoding="utf-8")))
    mb.SPACETIME.update(st)
    try:
        readme = json.load(open(mtp.README_RESULTS, encoding="utf-8"))
    except OSError:
        readme = {}
    return mb, mb.build(summ, papers, st, readme)


def test_ch1d_mattey_ghosh_selects_windows2(recs):
    mb, rs = recs
    r = next(x for x in rs if x["entry"].key == "1D Cahn-Hilliard (Mattey & Ghosh)")
    assert r["f_best_paper"] is None                      # the paper's method (bc-PINN) is not the best run
    fs = r["f_best_s"]
    assert fs["chosen"] == "windows2"
    judged = {v: mb.mtp._med(rows, mb._judged) for v, rows in fs["variants"].items()}
    assert judged["windows2"] == pytest.approx(0.06220309, rel=1e-6)      # max(h, mu): mu decides
    assert min(judged, key=judged.get) == "windows2"
    assert r["paper_runs"]["paper_mg"]["judged"] > 10 * judged["windows2"]


def test_tuned_settings_file_matches_the_selection(recs):
    """pinnstudio/resources/tuned_settings.json is what write_tuned() writes from these records."""
    mb, rs = recs
    shipped = json.load(open(mb.TUNED_JSON, encoding="utf-8"))
    path = mb.TUNED_JSON
    try:
        mb.TUNED_JSON = os.path.join(os.environ.get("PYTEST_TMP", "/tmp"), "tuned_check.json")
        mb.write_tuned(rs)
        built = json.load(open(mb.TUNED_JSON, encoding="utf-8"))
    finally:
        mb.TUNED_JSON = path
    assert built == shipped


def test_paper_statistics(recs):
    mb, rs = recs
    names = mb._paper_names()
    by = {r["entry"].name: r for r in rs}
    s = mb._stats([by[n] for n in names if n in by])
    assert (s["n_fwd"], s["n_inv"]) == (22, 20)
    assert s["med_fwd"] == pytest.approx(5.1637e-4, rel=1e-4)
    assert (s["below_1e2"], s["below_1e3"], s["n_better"], s["n_start"]) == (20, 13, 12, 20)
    assert s["med_perr"] == pytest.approx(1.045e-3, rel=1e-6) and s["below_1pct"] == 19
    a = mb._stats(rs)
    assert (a["n_fwd"], a["n_inv"]) == (53, 32)
    assert a["med_fwd"] == pytest.approx(1.1493e-3, rel=1e-4)
    assert a["med_perr"] == pytest.approx(1.315e-3, rel=1e-6) and a["below_1pct"] == 30

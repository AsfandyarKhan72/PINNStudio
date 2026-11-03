"""The controlled comparison of benchmarks/training_methods.md (paper Figure 7, setups 1-10) must state the values
of the corrected Figure 7 records, benchmarks/figures/fig7_corrected/fig7_corrected.json: the error of the main
output at t = 1 (two significant figures, in %) and the training time (minutes)."""
import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "benchmarks", "training_methods.md")
DATA = os.path.join(ROOT, "benchmarks", "figures", "fig7_corrected", "fig7_corrected.json")
PROBLEM = {"1D Allen-Cahn": "ac1d", "1D Cahn-Hilliard": "ch1d", "2D Allen-Cahn": "ac2dmg", "2D Cahn-Hilliard": "ch2dmg"}
TABLES = {  # first header cell after "Problem" -> run names of the columns
    "Plain PINN": ["plain", "exact", "rba", "sa", "windows", "bcpinn", "windowsEq", "bcpinnEq"],
    "+ SA-PINN (4)": ["sa", "causal", "causalmsffn", "rar", "rarfull", "growing"],
}
CELL = re.compile(r"\**([\d.]+)%\**\s*‡?\s*\((\d+) min\)")


def _final(run):
    l2 = run["l2"]
    l2 = ast.literal_eval(l2) if isinstance(l2, str) else l2
    out = "u" if "u" in l2 else "h"
    t, v = l2[out][-1]
    assert t == 1.0
    return 100 * v


def _tables():
    lines = open(PAGE, encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("## The controlled comparison"))
    found = {}
    for i in range(start, len(lines)):
        if lines[i].startswith("## ") and i > start:
            break
        cells = [c.strip() for c in lines[i].strip("|").split("|")]
        if cells and cells[0] == "Problem" and len(cells) > 1 and cells[1] in TABLES and cells[1] not in found:
            rows, j = {}, i + 2
            while j < len(lines) and lines[j].startswith("|"):
                c = [x.strip() for x in lines[j].strip("|").split("|")]
                rows[c[0]] = c[1:]
                j += 1
            found[cells[1]] = rows
    return found


def test_controlled_comparison_matches_corrected_records():
    data = json.load(open(DATA, encoding="utf-8"))
    tables = _tables()
    assert set(tables) == set(TABLES)
    n = 0
    for head, runs in TABLES.items():
        assert set(tables[head]) == set(PROBLEM), tables[head].keys()
        for prob, cells in tables[head].items():
            for run, cell in zip(runs, cells):
                m = CELL.search(cell)
                assert m, (prob, run, cell)
                rec = data[f"f7v2_{PROBLEM[prob]}_{run}"]
                val = _final(rec)
                assert float(m.group(1)) == float(f"{val:.2g}"), (prob, run, cell, val)
                assert int(m.group(2)) == round(rec["seconds"] / 60), (prob, run, cell, rec["seconds"])
                n += 1
    assert n == 4 * 8 + 4 * 6


def test_rerun_marks_are_the_rerun_records():
    """A value marked ‡ (rerun after the time-window fix) is a rerun record, and every rerun record is marked."""
    data = json.load(open(DATA, encoding="utf-8"))
    tables = _tables()
    for prob, cells in tables["Plain PINN"].items():
        for run, cell in zip(TABLES["Plain PINN"], cells):
            rerun = "rerun" in data[f"f7v2_{PROBLEM[prob]}_{run}"]["source"]
            assert ("‡" in cell) == rerun, (prob, run, cell)

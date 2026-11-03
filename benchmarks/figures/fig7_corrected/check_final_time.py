"""Checks the final-time (t = 1) values of fig7_corrected.json against the independent benchmark records and prints a
table (written to check_final_time.txt):

  - panel (b) windowed runs: the rerun-1 table of benchmarks/records/reruns_2026-11-01/README.md (u, 3 s.f.) and the
    runs' result.json (5 s.f.);
  - panel (d) windowed runs: the rerun-3 table of the same README (h and mu, 4 / 3 s.f.);
  - panel (c), all 13 runs: the re-scoring record benchmarks/reference_regeneration/convergence/f7_ac2dmg_reeval.json
    (full precision, "new" = regenerated reference);
  - panels (a), and the unreplaced runs of (b) and (d): unchanged from benchmarks/figures/fig7_v2.json.

  python benchmarks/figures/fig7_corrected/check_final_time.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RR = os.path.join(ROOT, "benchmarks", "records", "reruns_2026-11-01")
d = json.load(open(os.path.join(HERE, "fig7_corrected.json"), encoding="utf-8"))
v2 = json.load(open(os.path.join(ROOT, "benchmarks", "figures", "fig7_v2.json"), encoding="utf-8"))
rows, bad = [], []


def t1(rec, f):
    return dict(map(tuple, rec["l2"][f]))[1.0]


def chk(key, field, ref, src, rtol):
    v = t1(d[key], field)
    ok = abs(v - ref) <= rtol * abs(ref)
    rows.append(f"{key:24s} {field:3s} t=1  figure {v:.6g}  record {ref:.6g}  ({src})  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append(key)


# (b) rerun 1: README table (u at t = 1, 3 s.f.) and result.json
readme_b = {"windows": 3.94e-3, "bcpinn": 4.54e-3, "windowsEq": 4.81e-3, "bcpinnEq": 9.87e-3}
for k, v in readme_b.items():
    chk(f"f7v2_ch1d_{k}", "u", v, "reruns README, rerun 1", 6e-3)
    res = json.load(open(os.path.join(RR, "rerun1_ch1d_per_time", f"f7v2_ch1d_{k}", "result.json"), encoding="utf-8"))
    for f in ("u", "mu"):
        chk(f"f7v2_ch1d_{k}", f, dict(map(tuple, res["l2"][f]))[1.0], "rerun-1 result.json", 1e-4)
# (d) rerun 3: README table (h, mu at t = 1)
readme_d = {"windows": (1.331e-2, 9.74e-2), "bcpinn": (1.971e-2, 0.277), "windowsEq": (5.31e-2, 0.718),
            "bcpinnEq": (0.128, 1.088)}
for k, (h, mu) in readme_d.items():
    chk(f"f7v2_ch2dmg_{k}", "h", h, "reruns README, rerun 3", 6e-3)
    chk(f"f7v2_ch2dmg_{k}", "mu", mu, "reruns README, rerun 3", 6e-3)
# (c) re-scoring record (regenerated reference), all 13 runs
ree = json.load(open(os.path.join(ROOT, "benchmarks", "reference_regeneration", "convergence", "f7_ac2dmg_reeval.json"),
                     encoding="utf-8"))
for e in ree:
    run = e["run"].rstrip("/").split("/")[-1].split("_s")[0]
    key = f"f7v2_ac2dmg_{run}"
    if key in d:
        chk(key, "u", e["outputs"]["h"]["new"]["l2"][-1], "f7_ac2dmg_reeval.json (new reference)", 1e-3)
# unchanged runs equal fig7_v2.json
same = [k for k in d if d[k]["l2"] == v2[k]["l2"]]
changed = sorted(set(d) - set(same))
rows.append("")
rows.append(f"runs identical to benchmarks/figures/fig7_v2.json: {len(same)} of {len(d)}")
rows.append("runs replaced from the rerun records: " + ", ".join(changed))
rows.append("")
rows.append("RESULT: " + ("all final-time values agree with the records" if not bad else f"MISMATCH in {sorted(set(bad))}"))
txt = "\n".join(rows) + "\n"
open(os.path.join(HERE, "check_final_time.txt"), "w", encoding="utf-8").write(txt)
print(txt)
raise SystemExit(1 if bad else 0)

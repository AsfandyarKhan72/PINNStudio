# Reproducing the paper

The paper describing PINNStudio is in preparation. Its tables and figures are
produced from the benchmark records with the scripts below; each record states the code version it ran on.
The records in the repository are the authoritative values: some Cahn–Hilliard runs were repeated with corrected
code and some reference data were regenerated after the manuscript's first draft (`benchmarks/records/`, with the
three record sets and what changed between them).

| Result | Runs | Tables and figures |
|---|---|---|
| Example accuracy (forward and inverse) | `tools/tuning_study.py` (reference runs, tuning study) | `tools/make_tuning_page.py`, `tools/make_benchmarks_readme.py` |
| Published methods (base PINN vs. the paper's method vs. PINNStudio) | `tools/tuning_study.py --variants plain,paper_*` | [`benchmarks/paper_comparison.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/paper_comparison.md) |
| Controlled comparison of training methods | `python benchmarks/study_runner.py benchmarks/plans/fig7_v2.json OUT --jobs 1`, then the same with `fig7_v2b.json` | `tools/make_methods_figure.py` |
| Example figures | the examples' Tuned runs | `tools/make_paper_examples.py` |
| Export equivalence | `tools/check_export_equivalence.py` | [Export equivalence](export-equivalence.md) |
| Performance | `benchmarks/perf_study.py` | [`benchmarks/performance.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/performance.md) |

## Steps

1. Install PINNStudio from source ([Installation](../getting-started/installation.md)),
   at the commit the result was produced with.
2. Run the study script; each writes one folder per run with the generated script, its log and a `result.json`.
3. Run the table or figure script on those folders.

GPU runs reproduce the recorded values up to the run-to-run variation described in
[Seeds, precision and hardware](environment.md); CPU runs with the same seed are repeatable on the same machine.

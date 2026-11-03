# Final build: corrected records with the Cahn–Hilliard reruns

These argument files define the benchmark build used from v1.6.0 on: the corrected record set (`../../corrected_2026-10-31/`), with the rerun rows taking precedence.

**Three ways a rerun row takes precedence:**
- **Run lists:** the build tools keep the first row they see for each (example, mode, variant, seed), so the rerun files are listed first.
- **Primary and all-times files:** a later entry replaces an earlier one, so the rerun summaries are listed last.
- **Timed-out reruns:** in the primary file, an entry with `"l2": null` removes the older value for that run. This covers rerun 2's `all` variant, which timed out after 14,400 s.

| File | Contents |
|---|---|
| `results_args.txt` | tuning-study and best-setup run lists: `results_reruns.jsonl`, then the corrected set's list |
| `papers_arg.txt` | runs of the papers' methods: `papers_reruns.jsonl`, then the corrected set's list |
| `primary_arg.txt` | the corrected `primary.json`, then `../primary_reruns.json` |
| `spacetime_arg.txt` | the corrected all-times files, then `../spacetime_reruns.json` |
| `results_reruns.jsonl`, `papers_reruns.jsonl` | the lines of `../rerun2_ch1d_mg_results.jsonl`, `../rerun4_ch2d_results.jsonl` and `../rerun5_ch1d_wz_results.jsonl`, copied unchanged and split by kind: best-setup rows go to results, rows of the papers' methods go to papers |

`../primary_reruns.json` and `../spacetime_reruns.json` were rebuilt from the runs' own error files by `tools/build_rerun_summaries.py`. The sources are `../run_outputs/`, and `../summaries_manifest.json` lists every value, file hash and check.

## Rebuilding

Run from the repository root:

```bash
B=benchmarks/records; F=$B/reruns_2026-11-01/final_build
python tools/build_rerun_summaries.py $B/reruns_2026-11-01/run_outputs/{p2_ch_tune_1d,p4_ch_tune_2d,p5_ch1d_wz_paper}/out \
    --primary /tmp/p.json --spacetime /tmp/s.json --manifest /tmp/m.json   # identical to the committed summaries
python tools/build_benchmark_records.py $(sed "s#^#$B/#" $F/results_args.txt) \
    --primary "$(sed "s#^#$B/#" $F/primary_arg.txt | paste -sd,)" \
    --papers "$(sed "s#^#$B/#" $F/papers_arg.txt | paste -sd,)" \
    --spacetime "$(sed "s#^#$B/#" $F/spacetime_arg.txt | paste -sd,)" --out /tmp/records_check
diff /tmp/records_check/table_s2.tex benchmarks/tables/table_s2.tex
```

## Guard against stale or missing summaries

`make_benchmarks_readme.py` stops with `MissingSummary` in any of these cases:
- a finished run of a problem covered by the primary records has no entry of its own;
- a Cahn–Hilliard run's entry lacks μ;
- the entry's values are not among the errors that run itself printed, which happens when the entry comes from an earlier run of the same variant and seed.

Without the guard, such a run would be judged by a different metric: the worst error it printed, or the main output alone. Building with the rerun rows but without `primary_reruns.json` stops at the first stale entry, CH 1D W&Z `paper_wz` seed 2026.

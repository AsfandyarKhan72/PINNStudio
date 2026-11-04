# Cahn-Hilliard reruns, 2026-10-31 to 2026-11-01 (third record set)

Approved reruns of the Cahn-Hilliard runs affected by two defects found in the v1.6.0 audit. These records
**supersede** the affected rows of `corrected_2026-10-31/` (and `original_2026-10-30/`) for the next paper
version; the older records are kept unchanged as history. Nothing here has been applied to the manuscript.

| Defect (fixed on `audit/v1.6.0`) | Effect on the recorded runs | Reruns |
|---|---|---|
| Time windows: the configured loss weights were dropped from window 2 on ("configured weights have N entries but N+1 loss terms are active ... using uniform weights"), i.e. the IC weight was lost | every windowed CH run with non-uniform weights | 1, 2, 3, 4a, 4b |
| Resampling rounds inside a window: the model saved before the resampled training was the one evaluated | 1D CH (Wight & Zhao) paper_wz runs | 5 |

Code: reruns 1-4 cloned `audit/v1.6.0` (commits 79edd5d, 5dad9b5, 96b0fed, 77889c0 as printed by each job); rerun
5 `rerun/ch1d-wz-resampled-model` (45490a4 = 5d7c821 + only the resampled-model fix 63de1b4). Settings,
seeds and points as in the original records. GPU: one RTX 4090. Full logs and run folders stay on the GPU
machine (`.gpu_runner/done/zzzzzp1_ch_f7_1d` ... `p5_ch1d_wz_paper`).

## Files

| File | Contents | SHA-256 (first 16) |
|---|---|---|
| `rerun1_fig7_ch1d_mg_per_time.json` | rerun 1, Figure 7(b): error over all times per output, from the run logs (the runs are the 1D Cahn-Hilliard example of **Wight & Zhao**; the file name's "mg" is a mislabel) | `ebdef08e3fc662db` |
| `rerun2_ch1d_mg_results.jsonl` | rerun 2, 1D CH (Mattey & Ghosh) paper_mg and best-setup variants, `tools/tuning_study.py` records | `835f22a0e2060657` |
| `rerun3_fig7_ch2d_mg_per_time.json` | rerun 3, Figure 7(d): error over all times, from the run logs | `a86a3aa8095f7dae` |
| `rerun3_per_time_bcpinn.json` | rerun 3: per-time L2 of h and mu | `7f052d95a1ece5c7` |
| `rerun3_per_time_bcpinnEq.json` | rerun 3: per-time L2 of h and mu | `547968ba5997717b` |
| `rerun3_per_time_windows.json` | rerun 3: per-time L2 of h and mu | `899cd5a6884c4e2d` |
| `rerun3_per_time_windowsEq.json` | rerun 3: per-time L2 of h and mu | `a94b479fe8677c10` |
| `rerun4_ch2d_results.jsonl` | reruns 4a (2D CH Wight & Zhao paper_wz, seeds 2026/1/2) and 4b (2D CH Mattey & Ghosh paper_mg, seed 2026) | `9207bff93d1ce0e1` |
| `rerun5_ch1d_wz_results.jsonl` | rerun 5, 1D CH (Wight & Zhao) paper_wz, seeds 2026/1/2 | `e8d413cc1bd2b9c1` |
| `rerun5_old_records.jsonl` | the old records rerun 5 replaces (from `a0d_papers`, `a11_seeds4_papers`) | `fba7c448c622d440` |

## Results: final-time error (t = 1) and error over all reference times, kept apart

Relative L2. "Final time" is what Figure 4, Figure 7 and Table S2 report; "all times" is the error over every
reference time together (the benchmark pages' second number). The two can move in opposite directions, so any
statement in the paper must say which one it uses.

### Figure 7(b), 1D Cahn-Hilliard (Wight & Zhao), u at t = 1 (rerun 1)
| Run | old | new |
|---|---|---|
| windows | 4.03e-3 | 3.94e-3 |
| bcpinn | 3.96e-3 | 4.54e-3 |
| windowsEq | 4.44e-3 | 4.81e-3 |
| bcpinnEq | 4.15e-3 | 9.87e-3 |

All times, new: windows u 1.24e-2 / mu 2.69e-2; bcpinn u 1.24e-2 / mu 2.83e-2; windowsEq u 5.87e-2 / mu 6.73e-2;
bcpinnEq u 5.95e-2 / mu 8.28e-2. Window 1 identical to the old runs.

### Table S2 / Figure 4, 1D Cahn-Hilliard (Mattey & Ghosh), seed 2026, t = 1 (rerun 2)
| Variant | h old | h new | mu old | mu new |
|---|---|---|---|---|
| paper_mg | 5.66e-3 | 7.52e-3 | 4.62e-2 | 1.07 |
| base | 3.90e-3 | 4.37e-3 | 0.383 | 0.124 |
| iters | 6.06e-2 | 5.51e-3 | 1.87 | 0.238 |
| net | 0.985 | 3.18e-3 | 63.8 | 0.375 |
| windows2 | 1.47e-2 | 6.26e-3 | 1.39 | 6.22e-2 |
| all | 0.989 | timed out (14,400 s) | 39.4 | - |

paper_mg h over all times: 5.5e-3 -> 6.85e-3. By the selection rule (worse of h and mu at t = 1) the chosen
variant would move from paper_mg to windows2.

### Figure 7, 2D Cahn-Hilliard (Mattey & Ghosh), t = 1 (rerun 3)
| Run | h old | h new | mu old | mu new |
|---|---|---|---|---|
| windows | 1.343e-2 | 1.331e-2 | 4.06e-2 | 9.74e-2 |
| bcpinn | 1.311e-2 | 1.971e-2 | 5.28e-2 | 0.277 |
| windowsEq | 5.56e-2 | 5.31e-2 | 0.136 | 0.718 |
| bcpinnEq | 5.37e-2 | 0.128 | 0.161 | 1.088 |

### Figure 4 paper-method row, 2D Cahn-Hilliard (Wight & Zhao), paper_wz, t = 1 (rerun 4a)
| Seed | phi old | phi new | mu old | mu new |
|---|---|---|---|---|
| 2026 | 0.244 | 0.248 | 0.606 | 0.471 |
| 1 | 0.217 | 0.235 | 0.526 | 0.322 |
| 2 | 0.199 | 0.218 | 0.429 | 0.286 |
| median | 0.217 | 0.235 | 0.526 | 0.322 |

### Figure 4 / Table S2 paper-method row, 2D Cahn-Hilliard (Mattey & Ghosh), paper_mg, seed 2026, t = 1 (rerun 4b)
h 2.04e-2 -> 2.42e-2; mu 0.258 -> 0.421; run time 27,707 s -> 30,133 s.

### Figure 4 / Table S2 paper-method row, 1D Cahn-Hilliard (Wight & Zhao), paper_wz (rerun 5)
| Seed | u(t=1) old | new | mu(t=1) old | new | u all times old | new | mu all times old | new |
|---|---|---|---|---|---|---|---|---|
| 2026 | 8.55e-3 | 2.39e-2 | 2.06e-2 | 3.44e-2 | 5.25e-2 | 4.82e-2 | 4.30e-2 | 6.49e-2 |
| 1 | 6.46e-3 | 1.06e-2 | 2.06e-2 | 1.89e-2 | 4.75e-2 | 3.65e-2 | 4.20e-2 | 5.69e-2 |
| 2 | 8.09e-3 | 7.08e-3 | 2.21e-2 | 2.58e-2 | 5.50e-2 | 3.84e-2 | 4.81e-2 | 5.99e-2 |
| median | 8.09e-3 | 1.06e-2 | 2.06e-2 | 2.58e-2 | 5.25e-2 | 3.84e-2 | 4.30e-2 | 5.99e-2 |

Final-time judged error (worse of u and mu at t = 1) 2.06e-2 -> 2.58e-2 (+25 %), while u over all times improves
(5.25e-2 -> 3.84e-2, -27 %): the two error measures disagree in direction for this row.

## Effect on the reported values

As assessed when the reruns were made; the final build ([`final_build/`](final_build/README.md)) has since
applied them, and the current tables and pages come from it.

- **Figure 7 (b), (d)**: the windowed curves (windows, bcpinn, windowsEq, bcpinnEq) change from window 2 on;
  final-time values above. The text's range for the windowed final-time errors needs the new values.
- **Figure 4 paper-method rows** (Figure 4 plots the main output): 1D CH W&Z u 8.09e-3 -> 1.06e-2 (median of three
  seeds; 2.06e-2 -> 2.58e-2 is the worse of u and mu, used only for selection); 2D CH W&Z phi 0.217 -> 0.235. The
  Mattey & Ghosh Cahn-Hilliard problems are not Figure 4 rows: their paper-method values (2D h 2.04e-2 -> 2.42e-2;
  1D h 5.66e-3 -> 7.52e-3, mu 4.6e-2 -> 1.07) appear on the benchmark pages only.
- **Table S2**: one row changes, 1D CH M&G, whose selected configuration changes (paper_mg -> windows2) under the
  stated selection rule (h 6.3e-3, mu 6.2e-2). The 2D CH M&G row (selected: iters) does not change.
- **Aggregates**: the forward median (5.2e-4 over 22 benchmarks) and the inverse median change only if a rerun row
  is among the selected configurations they summarize. Rebuilt with the final build: the forward median of the
  22 Figure 4 benchmarks is 5.16 × 10⁻⁴ and the median worst parameter error of the 20 inverse benchmarks 0.1045 %,
  both unchanged (see [`../VERIFICATION.md`](../VERIFICATION.md)).
- **Conclusions**: PINNStudio's own best setups for the 2D problems and for 1D Wight & Zhao were not affected (they
  do not use the windowed weights or resampling rounds); the 1D Mattey & Ghosh best-setup candidates were (they are
  windowed with an IC weight of 100), which is why they were rerun in rerun 2, so "exact constraints + SA-PINN give the lowest final-time errors" stands; the
  paper-method rows remain far behind them. Statements must keep final-time and all-times errors apart.

## Added 2026-11-03: complete per-time records and rebuilt summaries

| Path | Contents |
|---|---|
| `rerun1_ch1d_per_time/` | rerun 1: the four windowed runs' own per-time u and mu errors, result/meta files and logs (`SOURCE.md`) |
| `run_outputs/` | reruns 2, 4 and 5: each run's error files, log and model configuration, results.jsonl and job scripts |
| `primary_reruns.json`, `spacetime_reruns.json`, `summaries_manifest.json` | the summaries the build needs (main output and every output at the last reference time; the main output's error over all times), rebuilt by `tools/build_rerun_summaries.py` from `run_outputs/` and checked against each run's log and results.jsonl |
| `final_build/` | argument files of the final build (corrected set with these reruns taking precedence) |

Note on the all-times column: for the Wight & Zhao Cahn-Hilliard paper-method and plain runs, the earlier
`spacetime*.json` files of the original and corrected sets hold the chemical potential's error over all times (the
first output in those logs), not the main output's. `spacetime_reruns.json` holds the main output's, so for the rerun
rows the column now shows u (1D) or phi (2D). The plain runs of those two problems were not rerun and keep the older
value; the records are left unchanged.

## Added 2026-11-03: figures of the rerun runs

The benchmark pages show surface and line comparisons of the runs their tables report. For the rerun runs the
committed figures were still those of the earlier runs (2026-10-25), and the selected CH 1D Mattey & Ghosh run
(`windows2`, best setup) had none. They are replaced by the figures the reruns wrote themselves
(`error_analysis/{surface,line}_comparison_<output>.png` of seed 2026), scaled to 1400 px wide and saved with a
256-colour palette like the other figures in `benchmarks/figures/tuning/`:

| Figures | Run they come from (workstation `.gpu_runner/done/`) |
|---|---|
| `Template_1D_Cahn_Hilliard/paper_wz_*` | `p5_ch1d_wz_paper/out/Template_1D_Cahn_Hilliard/paper_wz_s2026/1D_Cahn-Hilliard__2026-11-01_17-18-11` (rerun 5) |
| `Template_1D_Cahn_Hilliard_Mattey_Ghosh/paper_mg_*` | `p2_ch_tune_1d/out/Template_1D_Cahn_Hilliard_Mattey_Ghosh/paper_mg_s2026/1D_Cahn-Hilliard_Mattey__Ghosh__2026-10-31_14-31-34` (rerun 2) |
| `Template_1D_Cahn_Hilliard_Mattey_Ghosh_best_setup/windows2_*` (new) | `p2_ch_tune_1d/out/Template_1D_Cahn_Hilliard_Mattey_Ghosh_best_setup/windows2_s2026/1D_Cahn-Hilliard_Mattey__Ghosh__2026-10-31_16-55-23` (rerun 2) |
| `Template_2D_Cahn_Hilliard_Wight_Zhao/paper_wz_*` | `p4_ch_tune_2d/out/Template_2D_Cahn_Hilliard_Wight_Zhao/paper_wz_s2026/2D_Cahn-Hilliard_Wight__Zhao__2026-11-01_04-05-01` (rerun 4) |
| `Template_2D_Cahn_Hilliard_Mattey_Ghosh/paper_mg_*` | `p4_ch_tune_2d/out/Template_2D_Cahn_Hilliard_Mattey_Ghosh/paper_mg_s2026/2D_Cahn-Hilliard_Mattey__Ghosh__2026-11-01_08-55-45` (rerun 4) |

These are the same run folders whose error files, logs and `model_config.json` are in `run_outputs/`. Rerun 2's
`all` run (best setup + every change) timed out after 14,400 s and has no result; it is no longer listed on the
benchmark pages (the earlier `all` row, 0.99, came from a run with the time-window weight defect).

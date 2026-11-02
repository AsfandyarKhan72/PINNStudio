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
| `rerun1_fig7_ch1d_mg_per_time.json` | rerun 1, Figure 7(b): error over all times per output, from the run logs | `ebdef08e3fc662db` |
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

### Figure 7, 1D Cahn-Hilliard (Mattey & Ghosh), u at t = 1 (rerun 1)
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

## What changes for the paper (to decide; the manuscript is not edited)

- **Figure 7 (b), (d)**: the windowed curves (windows, bcpinn, windowsEq, bcpinnEq) change from window 2 on;
  final-time values above. The text's range for the windowed final-time errors needs the new values.
- **Figure 4 paper-method rows**: 1D CH W&Z 2.06e-2 -> 2.58e-2; 2D CH W&Z 0.217 -> 0.235 (phi); 2D CH M&G
  (paper_mg) h 2.04e-2 -> 2.42e-2; 1D CH M&G paper_mg h 5.66e-3 -> 7.52e-3 with mu 4.6e-2 -> 1.07.
- **Table S2**: the same rows; for 1D CH M&G the selected configuration changes (paper_mg -> windows2) under the
  stated selection rule.
- **Aggregates**: the forward median (5.2e-4 over 22 benchmarks) and the inverse median change only if a rerun row
  is among the selected configurations they summarize; this is settled by rebuilding the records with these files
  (not done yet).
- **Conclusions**: PINNStudio's own best setups for these problems were not affected (they do not use the windowed
  weights or resampling rounds), so "exact constraints + SA-PINN give the lowest final-time errors" stands; the
  paper-method rows remain far behind them. Statements must keep final-time and all-times errors apart.

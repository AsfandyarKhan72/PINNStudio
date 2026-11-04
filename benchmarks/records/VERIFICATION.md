# Verification of the benchmark build records

Rebuilt independently on 2026-11-01 from the files in this folder with `tools/build_benchmark_records.py`,
each with the code of the commit that produced it, and compared byte for byte with the committed tables.

| Build | Code | Primary file | `table_s2.tex` | `forward_per_seed.csv` | `inverse_per_seed.csv` |
|---|---|---|---|---|---|
| original_2026-10-30 | `af321c1` | `primary_final.json` | identical | identical | identical |
| corrected_2026-10-31 | `a2c5292` | `primary.json` | identical | identical | identical |

Both result sets reproduce the tables committed with them exactly. (With the original folder's older `primary.json`
the original build differs in 16 rows of `forward_per_seed.csv`, the seed-1/seed-2 runs of the papers'
methods, which that file lacks; Table S2 is the same either way.)

## Differences between the two builds

The two builds use the same runs; the corrected build re-scores the forward runs of Allen-Cahn 2D (Wight & Zhao,
Mattey & Ghosh) and Heat 2D/3D against the regenerated reference data and replaces the affected inverse runs with
runs retrained on the regenerated data (`corrected_2026-10-31/changes.md` lists every value and its reason).

**Supplementary Table S2** (median relative L2 error of the selected configuration at the last reference time):

| Row | Original | Corrected |
|---|---|---|
| Allen-Cahn 2D – Wight & Zhao, u | 1.6 × 10⁻³ | 5.9 × 10⁻⁵ |
| Allen-Cahn 2D – Mattey & Ghosh, u | 3.6 × 10⁻⁴ | 1.3 × 10⁻⁴ |
| Heat 2D, u | 5.7 × 10⁻³ | 6.1 × 10⁻³ |

All other Table S2 rows are unchanged.

**Summary statistics** (`stats.json` of the build; the 25 Figure 4 examples, then all 64 examples):

| Statistic | Original | Corrected |
|---|---|---|
| Figure 4: median improvement over the starting setup, all examples (`med_gain_all`) | 5.22× | 7.91× |
| Figure 4: median worst relative parameter error, inverse (`med_perr`) | 0.126 % | 0.105 % |
| All 64: median forward error (`med_fwd`) | 1.30 × 10⁻³ | 1.15 × 10⁻³ |
| All 64: examples below 10⁻³ (`below_1e3`) | 24 | 25 |
| All 64: median improvement where improved (`med_gain`) | 14.8× | 15.7× |
| All 64: median worst parameter error (`med_perr`) | 0.127 % | 0.132 % |
| All 64: inverse examples below 1 % (`below_1pct`) | 29 | 30 |

Unchanged: the number of examples, the Figure 4 median forward error (5.16 × 10⁻⁴), the counts below 10⁻² and
10⁻³ in Figure 4 (20, 13), the number of starting setups improved (12 of 20), the Figure 4 median gain where
improved (20.1×), and the Figure 4 inverse count below 1 % (19 of 20).

Per-seed rows that differ (main output's relative L2 at the last reference time, forward; estimate and worst
relative parameter error, inverse):

| Example | Variant | Selected | Starting | Seed | Original | Corrected |
|---|---|---|---|---|---|---|
| Allen-Cahn 2D – Mattey & Ghosh | all |  |  | 2026 | 0.00034948 | 0.00019468 |
| Allen-Cahn 2D – Mattey & Ghosh | base |  |  | 1 | — | 0.00020729 |
| Allen-Cahn 2D – Mattey & Ghosh | base |  |  | 2 | — | 0.00021893 |
| Allen-Cahn 2D – Mattey & Ghosh | base |  |  | 2026 | — | 0.00021226 |
| Allen-Cahn 2D – Mattey & Ghosh | base | yes |  | 1 | 0.00035557 | — |
| Allen-Cahn 2D – Mattey & Ghosh | base | yes |  | 2 | 0.0003624 | — |
| Allen-Cahn 2D – Mattey & Ghosh | base | yes |  | 2026 | 0.0003634 | — |
| Allen-Cahn 2D – Mattey & Ghosh | iters |  |  | 2026 | 0.00031729 | — |
| Allen-Cahn 2D – Mattey & Ghosh | iters | yes |  | 1 | — | 0.00012998 |
| Allen-Cahn 2D – Mattey & Ghosh | iters | yes |  | 2 | — | 0.00013423 |
| Allen-Cahn 2D – Mattey & Ghosh | iters | yes |  | 2026 | — | 0.00012529 |
| Allen-Cahn 2D – Mattey & Ghosh | net |  |  | 2026 | 0.00059048 | 0.00051211 |
| Allen-Cahn 2D – Mattey & Ghosh | paper_mg |  | yes | 1 | 0.0028744 | 0.0028648 |
| Allen-Cahn 2D – Mattey & Ghosh | paper_mg |  | yes | 2 | 0.0021523 | 0.002149 |
| Allen-Cahn 2D – Mattey & Ghosh | paper_mg |  | yes | 2026 | 0.0025381 | 0.0025421 |
| Allen-Cahn 2D – Mattey & Ghosh | plain |  |  | 2026 | 0.027556 | 0.027567 |
| Allen-Cahn 2D – Wight & Zhao | base | yes |  | 2026 | 0.0015528 | 5.8662e-05 |
| Allen-Cahn 2D – Wight & Zhao | iters |  |  | 2026 | 0.0015567 | 5.201e-05 |
| Allen-Cahn 2D – Wight & Zhao | paper_wz |  | yes | 2026 | 0.0028408 | 0.002379 |
| Allen-Cahn 2D – Wight & Zhao | windows2 |  |  | 2026 | 0.0015488 | 6.3616e-05 |
| Heat 2D | base | yes |  | 1 | 0.0056727 | 0.0061469 |
| Heat 2D | base | yes |  | 2 | 0.025696 | 0.025588 |
| Heat 2D | base | yes |  | 2026 | 0.0038729 | 0.0042914 |
| Heat 3D | base | yes |  | 2026 | 0.0026027 | 0.00261 |

24 forward rows differ

| Example | Variant | Selected | Seed | Parameter | Original estimate, worst error | Corrected |
|---|---|---|---|---|---|---|
| Allen-Cahn 2D – Mattey & Ghosh | base |  | 2026 | c1_sq | ('0.000105791', '0.05791') | ('0.000107239', '0.07239') |
| Allen-Cahn 2D – Mattey & Ghosh | base |  | 2026 | c2 | ('1.00367', '0.05791') | ('1.00559', '0.07239') |
| Allen-Cahn 2D – Mattey & Ghosh | dataw |  | 1 | c1_sq | ('0.000106572', '0.06572') | ('0.000103597', '0.03597') |
| Allen-Cahn 2D – Mattey & Ghosh | dataw |  | 1 | c2 | ('1.00366', '0.06572') | ('1.00203', '0.03597') |
| Allen-Cahn 2D – Mattey & Ghosh | dataw |  | 2 | c1_sq | ('0.000103547', '0.03547') | ('0.000107107', '0.07107') |
| Allen-Cahn 2D – Mattey & Ghosh | dataw |  | 2 | c2 | ('1.00065', '0.03547') | ('1.00498', '0.07107') |
| Allen-Cahn 2D – Mattey & Ghosh | dataw |  | 2026 | c1_sq | ('0.000102452', '0.02452') | ('0.000106591', '0.06591') |
| Allen-Cahn 2D – Mattey & Ghosh | dataw |  | 2026 | c2 | ('1.00025', '0.02452') | ('1.00498', '0.06591') |
| Allen-Cahn 2D – Mattey & Ghosh | invall |  | 2026 | c1_sq | ('0.000107025', '0.07025') | ('0.000109768', '0.09768') |
| Allen-Cahn 2D – Mattey & Ghosh | invall |  | 2026 | c2 | ('1.00387', '0.07025') | ('1.00764', '0.09768') |
| Allen-Cahn 2D – Mattey & Ghosh | setup | yes | 1 | c1_sq | ('0.0001008', '0.010572') | ('0.000103879', '0.03879') |
| Allen-Cahn 2D – Mattey & Ghosh | setup | yes | 1 | c2 | ('0.989428', '0.010572') | ('1.00192', '0.03879') |
| Allen-Cahn 2D – Mattey & Ghosh | setup | yes | 2 | c1_sq | ('0.000101499', '0.01499') | ('0.000101907', '0.01907') |
| Allen-Cahn 2D – Mattey & Ghosh | setup | yes | 2 | c2 | ('0.999581', '0.01499') | ('1.00021', '0.01907') |
| Allen-Cahn 2D – Mattey & Ghosh | setup | yes | 2026 | c1_sq | ('0.000104321', '0.04321') | ('0.000106056', '0.06056') |
| Allen-Cahn 2D – Mattey & Ghosh | setup | yes | 2026 | c2 | ('1.00173', '0.04321') | ('1.00355', '0.06056') |
| Allen-Cahn 2D – Wight & Zhao | base |  | 2026 | diff_coeff | ('0.00598505', '0.045572') | ('0.00620447', '0.007931') |
| Allen-Cahn 2D – Wight & Zhao | base |  | 2026 | lambda_ | ('9.54428', '0.045572') | ('9.92069', '0.007931') |
| Allen-Cahn 2D – Wight & Zhao | dataw |  | 2026 | diff_coeff | ('0.00211525', '0.683061') | ('0.00210999', '0.686216') |
| Allen-Cahn 2D – Wight & Zhao | dataw |  | 2026 | lambda_ | ('3.16939', '0.683061') | ('3.13784', '0.686216') |
| Allen-Cahn 2D – Wight & Zhao | invall | yes | 2026 | diff_coeff | ('0.00615365', '0.017235') | ('0.00628417', '0.0054672') |
| Allen-Cahn 2D – Wight & Zhao | invall | yes | 2026 | lambda_ | ('9.82765', '0.017235') | ('10.0536', '0.0054672') |
| Allen-Cahn 2D – Wight & Zhao | setup |  | 2026 | diff_coeff | ('0.00611763', '0.023357') | ('0.0037128', '0.515581') |
| Allen-Cahn 2D – Wight & Zhao | setup |  | 2026 | lambda_ | ('9.76643', '0.023357') | ('4.84419', '0.515581') |
| Allen-Cahn 2D – Wight & Zhao | times |  | 2026 | diff_coeff | ('0.00273061', '0.575741') | ('0.00283064', '0.560703') |
| Allen-Cahn 2D – Wight & Zhao | times |  | 2026 | lambda_ | ('4.24259', '0.575741') | ('4.39297', '0.560703') |
| Heat 2D | base |  | 1 | alpha | — | ('0.398788', '0.00303') |
| Heat 2D | base |  | 2 | alpha | — | ('0.406354', '0.015885') |
| Heat 2D | base |  | 2026 | alpha | — | ('0.400273', '0.0006825') |
| Heat 2D | base | yes | 1 | alpha | ('0.39915', '0.002125') | — |
| Heat 2D | base | yes | 2 | alpha | ('0.406798', '0.016995') | — |
| Heat 2D | base | yes | 2026 | alpha | ('0.399893', '0.0002675') | — |
| Heat 2D | dataw |  | 2026 | alpha | ('0.401262', '0.003155') | ('0.401007', '0.0025175') |
| Heat 2D | invall |  | 2026 | alpha | ('0.402561', '0.0064025') | ('0.39833', '0.004175') |
| Heat 2D | times |  | 1 | alpha | ('0.401998', '0.004995') | — |
| Heat 2D | times |  | 2 | alpha | ('0.406973', '0.0174325') | — |
| Heat 2D | times |  | 2026 | alpha | ('0.399697', '0.0007575') | — |
| Heat 2D | times | yes | 1 | alpha | — | ('0.39956', '0.0011') |
| Heat 2D | times | yes | 2 | alpha | — | ('0.405931', '0.0148275') |
| Heat 2D | times | yes | 2026 | alpha | — | ('0.399692', '0.00077') |
| Heat 3D | base |  | 2026 | alpha | — | ('0.398656', '0.00336') |
| Heat 3D | base | yes | 2026 | alpha | ('0.400303', '0.0007575') | — |
| Heat 3D | dataw |  | 2026 | alpha | ('0.398051', '0.0048725') | — |
| Heat 3D | dataw | yes | 2026 | alpha | — | ('0.39911', '0.002225') |
| Heat 3D | invall |  | 2026 | alpha | ('0.396365', '0.0090875') | ('0.358028', '0.10493') |

45 inverse rows differ

## Final build (corrected records with the 2026-11-01 reruns)

Rebuilt independently on 2026-11-03 from `reruns_2026-11-01/final_build/` (the corrected set with the rerun rows
taking precedence) with `tools/build_benchmark_records.py` at `main` 78bf1a9 (Python 3.13, CPU), and compared byte
for byte with the current tables:

| Build | Code | Primary files | `table_s2.tex` | `forward_per_seed.csv` | `inverse_per_seed.csv` |
|---|---|---|---|---|---|
| final (`final_build/`) | `78bf1a9` | `corrected_2026-10-31/primary.json`, then `reruns_2026-11-01/primary_reruns.json` | identical | identical | identical |

Summary statistics of this build (`stats.json`; the 25 Figure 4 examples), with the corrected build for comparison:

| Statistic | Corrected | Final |
|---|---|---|
| Forward problems; median forward error (`med_fwd`) | 22; 5.16 × 10⁻⁴ | 22; 5.16 × 10⁻⁴ |
| Forward errors below 10⁻² / 10⁻³ | 20 / 13 | 20 / 13 |
| Starting setups improved by more than 1.25× (`n_better` of `n_start`) | 12 of 20 | 12 of 20 |
| Median gain where improved (`med_gain`) | 20.1× | 21.4× |
| Median gain over all 20 (`med_gain_all`) | 7.91× | 7.91× |
| Inverse problems; median worst parameter error (`med_perr`) | 20; 0.1045 % | 20; 0.1045 % |
| Inverse problems below 1 % | 19 | 19 |

The only Figure 4 statistic the reruns change is the median gain where improved (20.1× → 21.4×). The reruns of the
papers' methods raised two starting values while the tuned values stayed the same: Cahn-Hilliard 1D – Wight & Zhao
(u 8.09 × 10⁻³ → 1.06 × 10⁻², gain 15.7× → 20.5×) and Cahn-Hilliard 2D – Wight & Zhao (φ 0.217 → 0.235, gain
20.6× → 22.3×); see `reruns_2026-11-01/README.md`. The corrected build's statistics were recomputed the same way on
2026-11-03 to fill the left column.

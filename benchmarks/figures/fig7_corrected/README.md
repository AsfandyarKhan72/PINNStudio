# Figure 7, corrected data (2026-11-03)

This folder holds the error-versus-time data behind a corrected Figure 7 of the PINNStudio paper, together with the scripts that build, check and draw it. The figure compares training methods on four phase-field problems.

**Not changed by this folder:**
- The approved data file, `../fig7_v2.json`.
- The historical record sets in `../../records/`.
- The manuscript's figure.

Nothing was retrained for this folder: every value comes from runs already recorded in the repository.

## What changed and where each value comes from

The protocol is the same for all four panels: seed **2026**, the runs of `benchmarks/plans/fig7_v2.json` (13 configurations per panel, 52 runs), and relative L2 error at each reference time.

| Panel | Problem (example) | Fields | Reference times | Runs taken from | Changed? |
|---|---|---|---|---|---|
| (a) | Allen–Cahn 1D – Wight & Zhao | u | 0, 0.25, 0.5, 0.75, 1 | `fig7_v2.json` | no (13 of 13 identical) |
| (b) | Cahn–Hilliard 1D – Wight & Zhao | u (plotted), μ | 0, 0.25, 0.5, 0.75, 1 | The 4 windowed runs (`windows`, `bcpinn`, `windowsEq`, `bcpinnEq`) come from **rerun 1**; the other 9 from `fig7_v2.json` | 4 windowed runs, from t = 0.5 on |
| (c) | Allen–Cahn 2D – Mattey & Ghosh | u (h) | 0, 0.25, 0.5, 0.75, 1 | `fig7_v2.json`, which already holds the re-scoring against the regenerated reference (ee6d9dd) | already in `fig7_v2.json` (all 13 re-scored) |
| (d) | Cahn–Hilliard 2D – Mattey & Ghosh | h (plotted), μ | 0, 0.2, 0.4, 0.6, 0.8, 1 | The 4 windowed runs come from **rerun 3**; the other 9 from `fig7_v2.json` | 4 windowed runs, from t = 0.4 on |

**Rerun 1** is the 1D Cahn–Hilliard time-window loss-weight fix:
- **Code:** commit `79edd5d` on `audit/v1.6.0`.
- **Job:** workstation job `zzzzzp1_ch_f7_1d`, exit 0, 3,161 s, one RTX 4090.
- **Data:** each run wrote its own per-time error files. They are copied unchanged to `../../records/reruns_2026-11-01/rerun1_ch1d_per_time/f7v2_ch1d_<run>/`, together with `result.json`, `meta.json` (the full run settings, seed 2026) and the training log. The job script and its log are in `job/`.
- **Generated scripts:** the job's `run.py` files were not copied (474 kB each, regenerated from `meta.json` by `benchmarks/study_runner.py`). Their SHA-256 values:

| Run | SHA-256 of `run.py` |
|---|---|
| windows | `c528da61b4e831669dabe04ca057d432c528c4d6cf1957ca1529d0607aa6f6a7` |
| bcpinn | `531db6ed8fc614dc9e32b748256032984567b65b10a7d93a531486da66fe757f` |
| windowsEq | `16f663b506a22ac7f02fad0e1b4a2261b8b64a4e985aa2bbf733c348cbc4960d` |
| bcpinnEq | `8e822a2b4c60113a156a2c27070e5afa4f9bb0ed088685f74472be986c20cc86` |

**Rerun 3** is the same fix for the 2D Mattey & Ghosh problem:
- **Code:** commit `96b0fed`.
- **Job:** workstation job `zzzzzp3_ch_f7_2d`, exit 0.
- **Data:** `../../records/reruns_2026-11-01/rerun3_per_time_<run>.json`, already in the repository.

**Window 1 is unaffected.** In both reruns the defect touched only windows 2–4. `build_fig7_corrected.py` checks that window 1 of every replaced run equals the old values, to the 5 significant figures of the old logs.

**Panel (b) is Wight & Zhao.** The rerun README and the file name `rerun1_fig7_ch1d_mg_per_time.json` call it "Mattey & Ghosh". The runs are the "Cahn-Hilliard 1D – Wight & Zhao" example, as `meta.json` and `fig7_v2.json` record.

## Files

| File | What it is |
|---|---|
| `fig7_corrected.json` | The corrected data. Same format as `fig7_v2.json`: per run, the error at each reference time for every field, plus `source`. SHA-256 (first 16 hex digits): `b00c02bf4f4f6417`. |
| `build_fig7_corrected.py` | Builds `fig7_corrected.json` from the records listed above. |
| `check_final_time.py`, `check_final_time.txt` | Compares every replaced or re-scored final-time value with an independent record (see below) and saves the result. |
| `make_methods_figure_070b284.py` | The approved figure tool, unchanged: `tools/make_methods_figure.py` at `070b284`, the version that drew the manuscript's Figure 7. SHA-256 `653d006acbf9ce027082335b7fa0c03ce5f0b936a0a47c6ced5e57fbf039404c`. |
| `draw_fig7.py` | Draws the figure with that tool from a data file. It sets nothing but the data path and output folder. |
| `paper_figure7_corrected.pdf`, `.png` | The corrected figure: same size (395.5 × 332.0 pt), layout, fonts, colours, markers and legend as the approved one. |

**Input SHA-256 values** (first 16 hex digits):

| Input | SHA-256 (first 16) |
|---|---|
| `fig7_v2.json` | `858dfa86dffdc102` |
| `rerun3_per_time_bcpinn.json` | `7f052d95a1ece5c7` |
| `rerun3_per_time_bcpinnEq.json` | `547968ba5997717b` |
| `rerun3_per_time_windows.json` | `899cd5a6884c4e2d` |
| `rerun3_per_time_windowsEq.json` | `a94b479fe8677c10` |
| `f7_ac2dmg_reeval.json` | `1128e8f2a2118699` |

## Reproducing

```bash
python benchmarks/figures/fig7_corrected/build_fig7_corrected.py     # -> fig7_corrected.json
python benchmarks/figures/fig7_corrected/check_final_time.py         # -> check_final_time.txt, exit 0 when all agree
python benchmarks/figures/fig7_corrected/draw_fig7.py                # -> paper_figure7_corrected.pdf / .png
```

**Checks done before committing.** Drawing `fig7_v2.json` at `5d9dac0` with `draw_fig7.py` reproduces the manuscript's approved `fig7.pdf` with zero pixel difference at 150 dpi; the only difference between the two figures is therefore the data.

## Final-time check (`check_final_time.txt`)

**Panel (b), rerun 1:**
- u at t = 1 matches the rerun-1 table in `../../records/reruns_2026-11-01/README.md`, at 3 significant figures:
  - windows 3.94e-3
  - bcpinn 4.54e-3
  - windowsEq 4.81e-3
  - bcpinnEq 9.87e-3
- u and μ at t = 1 also match each run's `result.json` at 5 significant figures.

**Panel (d), rerun 3:** h and μ at t = 1 match the rerun-3 table in the same README:

| Run | h at t = 1 | μ at t = 1 |
|---|---|---|
| windows | 1.331e-2 | 9.74e-2 |
| bcpinn | 1.971e-2 | 0.277 |
| windowsEq | 5.31e-2 | 0.718 |
| bcpinnEq | 0.128 | 1.088 |

**Panel (c):** all 13 runs match the re-scoring record `../../reference_regeneration/convergence/f7_ac2dmg_reeval.json` (regenerated reference, full precision). Exact + SA-PINN, for example, is 1.1434e-4 at t = 1.

**The other 44 runs** are identical to `fig7_v2.json`.

**Result:** all final-time values agree with the records.

## Axis limits

The tool sets each panel's lower y-limit to the power of ten at or below the smallest plotted error, leaving out the exact-initial-condition points at t = 0. With the corrected data, panel (c)'s lower limit is therefore 10⁻⁵ (Exact + SA-PINN is 7.55e-5 at t = 0.25), where it was 10⁻⁴. The other panels are unchanged.

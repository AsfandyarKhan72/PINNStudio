# Execution-mode performance study on PINNStudio v1.6.0 (PySide6)

The five-problem execution-mode study (manuscript Section 5.1, Table 2) repeated on the final v1.6.0 code, with the
PySide6 interface. The earlier run of the same study, on the PyQt6 code of 2026-10-26 (`594d295`), is kept unchanged
in [`../performance/`](../performance/) as the historical record.

## What was run

- **Problems:** the five problems of the earlier study: Lorenz ODE (inverse), Wave 1D with multi-scale Fourier
  features, Allen–Cahn 2D (Mattey & Ghosh), Navier–Stokes 2D (Kovasznay flow) and Allen–Cahn 3D (Wight & Zhao).
- **Settings:** exactly the configurations the earlier study saved (`../performance/configs/`), loaded with
  `perf_study.py overhead --configs`. The configurations used are in `configs/`. They differ from the earlier ones
  only in the reference-data paths and in one field added to the configuration since, `ta_backward_seed`, which
  keeps its default (−1, not used by these problems). Lorenz, Wave 1D, Kovasznay and Allen–Cahn 3D use the
  examples' Default settings; Allen–Cahn 2D uses the saved best setup (exact IC and periodicity, SA-PINN,
  20,000 Adam + 10,000 L-BFGS).
- **Modes:** the exported standalone script, the internal Solve script and the PINNStudio window (GUI), on the
  logged-in desktop (Qt platform xcb).
- **Seeds:** 1, 2 and 3, with the order of the three modes rotated per seed, as in the earlier study.
- **Measurement:** unchanged. The reported time is the training time DeepXDE reports, summed over all optimization
  phases; the table gives the median over the three seeds with the range. The wall-clock totals of each run
  are in `runs.csv` (`total`).
- **Code and environment:** commit `6dfab1c` (branch `perf/v1.6.0-pyside6`; PINNStudio 1.6.0, imported from the
  cloned tree). The same machine and Python environment as the earlier study: NVIDIA GeForce RTX 4090 (driver
  550.144.03), Intel Core i9-13900KF, Python 3.11.5, PyTorch 2.6.0+cu124, CUDA 12.4, DeepXDE 1.15.0, NumPy 2.4.6,
  PySide6 6.11.3. Run on 2026-11-03, 12:30–18:14 MT, all 45 runs with exit code 0.

## Files

| File | Contents |
|---|---|
| `tables.md` | Table 2, written by `perf_study.py tables` from `overhead.jsonl` |
| `runs.csv` | every run: problem, mode, seed, order, training time per phase, total, steps, errors, parameters |
| `overhead.jsonl` | the raw records, one line per run, with the environment of each |
| `configs/` | the configuration each problem ran with |
| `run/` | the job script, its log, commit, environment, GPU, CPU and Qt platform |

Rebuild the table from the records: `python benchmarks/perf_study.py tables benchmarks/performance_v160 --out-dir /tmp/t`.

## Result (Table 2)

| Problem | Exported script (s) | Solve script (s) | PINNStudio GUI (s) | GUI vs. Solve | Steps |
|---|---|---|---|---|---|
| Lorenz ODE, inverse | 71.2 | 73.1 | 73.7 | +0.7 % | 22,670 |
| Wave 1D, MsFFN | 71.3 | 70.7 | 70.9 | +0.2 % | 10,000 |
| Allen–Cahn 2D (Mattey & Ghosh) | 253.8 | 255.4 | 255.5 | +0.0 % | 30,000 |
| Navier–Stokes 2D, Kovasznay | 190.7 | 194.9 | 190.3 | −2.4 % | 34,579 |
| Allen–Cahn 3D (Wight & Zhao) | 1582.1 | 1605.2 | 1612.9 | +0.5 % | 20,000 |

Medians of seeds 1, 2, 3; steps of seed 1 (Lorenz and Kovasznay stop at a different step per seed: 22,670 / 23,413 /
22,424 and 34,579 / 34,546 / 35,970). The GUI is within −2.4 % to +0.7 % of the Solve script, and within −0.5 % to
+3.5 % of the exported script.

**Numerical agreement.** For every one of the 15 problem–seed pairs, the three modes took the same number of
optimization steps and ended at the same errors and learned parameters, to all printed digits.

## Comparison with the earlier study (`../performance/`)

| Problem | Earlier (export / Solve / GUI, s) | Now (s) | Change |
|---|---|---|---|
| Lorenz ODE, inverse | 71.5 / 76.7 / 72.8 | 71.2 / 73.1 / 73.7 | −0.5 / −4.7 / +1.2 % |
| Wave 1D, MsFFN | 75.5 / 74.7 / 75.0 | 71.3 / 70.7 / 70.9 | −5.6 / −5.3 / −5.5 % |
| Allen–Cahn 2D (Mattey & Ghosh) | 266.3 / 265.7 / 259.4 | 253.8 / 255.4 / 255.5 | −4.7 / −3.9 / −1.5 % |
| Navier–Stokes 2D, Kovasznay | 204.0 / 207.4 / 204.4 | 190.7 / 194.9 / 190.3 | −6.5 / −6.0 / −6.9 % |
| Allen–Cahn 3D (Wight & Zhao) | 1563.2 / 1615.3 / 1553.7 | 1582.1 / 1605.2 / 1612.9 | +1.2 / −0.6 / +3.8 % |

- **The finding is the same:** the GUI and the Solve script train within a few percent of the exported script.
  The earlier GUI-vs-Solve range was −5.0 % to +0.5 %; it is now −2.4 % to +0.7 %.
- **Same computation, run for run.** For Lorenz, Wave 1D and Kovasznay, every run took the same number of steps and
  ended at exactly the same errors (and, for Lorenz, parameters) as the run of the same mode and seed in the
  earlier study. For Allen–Cahn 3D, the Solve and GUI runs match the earlier ones exactly.
- **Allen–Cahn 3D, exported script.** Earlier, the exported script ended at slightly different errors from Solve and
  the GUI (e.g. 4.15 × 10⁻⁴ against 4.61 × 10⁻⁴ for seed 1), which the earlier table put down to GPU run-to-run
  variation. Now all three modes match for every seed. So the earlier difference was between the exported script
  of that code version and Solve's, not run-to-run variation; the final export check (`../export_check/`) found the
  two scripts of every example bitwise identical on the CPU.
- **Allen–Cahn 2D errors.** The errors are now 2.07 × 10⁻⁴ (median) against 3.56 × 10⁻⁴ earlier, with the same
  settings and steps. The reference solution of this problem was regenerated after the earlier study
  (`reference_data/2D/allen_cahn_mattey`, commit `5dd1cb2`, 2026-10-30; see `../records/`), and the errors are
  measured against the current reference.
- **Times.** Wave 1D, Allen–Cahn 2D and Kovasznay train 4–7 % faster in all three modes, the plain exported script
  included, while Lorenz and Allen–Cahn 3D are within −1 % to +4 %. Since the shift is the same in all three modes
  and the arithmetic is the same (identical steps and errors), it is not an effect of the interface. A likely cause
  is the state of the machine: the earlier overhead runs followed the GPU throughput and CPU-thread studies in the
  same job (GPU at 46 °C at the start of that job, 35 °C at the start of this one), but this was not tested
  separately.

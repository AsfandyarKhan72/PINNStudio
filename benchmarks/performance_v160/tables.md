### Table 2. Training time in the three execution modes: exported standalone script, internal Solve script and PINNStudio GUI (RTX 4090; each problem as loaded from the Examples list, see the settings in configs/)

| Problem | Exported script (s) | Solve script (s) | PINNStudio GUI (s) | GUI vs. Solve (%) | Steps (export / solve / GUI) | Main-output L₂ (export / solve / GUI) |
|---|---|---|---|---|---|---|
| Lorenz ODE, inverse | 71.2 (65.7–72.3) | 73.1 (65.2–76.8) | 73.7 (67.8–74.3) | +0.7 | 22670 / 22670 / 22670 | 1.10e-04 / 1.10e-04 / 1.10e-04 |
| Wave 1D, multi-scale Fourier features | 71.3 (71.1–72.0) | 70.7 (70.5–72.4) | 70.9 (70.5–71.1) | +0.2 | 10000 / 10000 / 10000 | 3.04e-01 / 3.04e-01 / 3.04e-01 |
| Allen-Cahn 2D (Mattey & Ghosh) | 253.8 (253.4–257.5) | 255.4 (253.2–255.5) | 255.5 (252.6–257.0) | +0.0 | 30000 / 30000 / 30000 | 2.07e-04 / 2.07e-04 / 2.07e-04 |
| Navier-Stokes 2D, Kovasznay flow | 190.7 (189.8–201.8) | 194.9 (192.0–202.4) | 190.3 (190.1–204.8) | -2.4 | 34579 / 34579 / 34579 | 1.97e-04 / 1.97e-04 / 1.97e-04 |
| Allen-Cahn 3D (Wight & Zhao) | 1582.1 (1580.3–1644.1) | 1605.2 (1579.3–1640.3) | 1612.9 (1583.9–1635.1) | +0.5 | 20000 / 20000 / 20000 | 4.28e-04 / 4.28e-04 / 4.28e-04 |

*Median of three seeds (range in brackets) of the training duration DeepXDE reports, summed over all optimization phases. This is not the total wall-clock time: building the data and network before training and the plots and error analysis after it are not included. GUI vs. Solve = 100 × (GUI − Solve script) / Solve script; a negative value is within the run-to-run variation, not a speed-up caused by the GUI. The same settings in all three execution modes of a problem; the settings each problem ran with are saved in configs/. The L₂ errors are medians over the seeds. For each of the 15 problem-seed pairs, the three modes took the same number of optimization steps and ended at the same errors (and learned parameters) to all printed digits.*

*Environment: NVIDIA GeForce RTX 4090 (driver 550.144.03), 13th Gen Intel(R) Core(TM) i9-13900KF; Python 3.11.5, PyTorch 2.6.0+cu124, CUDA 12.4, DeepXDE 1.15.0; commit 6dfab1cdd3e63c665115521316afcca7a24a66f2.*

### Table 2. Training time in the three execution modes: exported standalone script, internal Solve script and PINNStudio GUI (RTX 4090; each problem as loaded from the Examples list, see the settings in configs/)

| Problem | Exported script (s) | Solve script (s) | PINNStudio GUI (s) | GUI vs. Solve (%) | Steps (export / solve / GUI) | Main-output L₂ (export / solve / GUI) |
|---|---|---|---|---|---|---|
| Lorenz ODE, inverse | 71.5 (68.5–75.5) | 76.7 (76.2–77.0) | 72.8 (71.6–79.3) | -5.0 | 22670 / 22670 / 22670 | 1.10e-04 / 1.10e-04 / 1.10e-04 |
| Wave 1D, multi-scale Fourier features | 75.5 (75.5–75.9) | 74.7 (73.0–75.4) | 75.0 (74.4–75.7) | +0.5 | 10000 / 10000 / 10000 | 3.04e-01 / 3.04e-01 / 3.04e-01 |
| Allen-Cahn 2D (Mattey & Ghosh) | 266.3 (264.9–266.9) | 265.7 (259.9–269.3) | 259.4 (259.4–262.9) | -2.4 | 30000 / 30000 / 30000 | 3.56e-04 / 3.56e-04 / 3.56e-04 |
| Navier-Stokes 2D, Kovasznay flow | 204.0 (203.1–217.4) | 207.4 (202.6–218.3) | 204.4 (203.0–217.6) | -1.5 | 34579 / 34579 / 34579 | 1.97e-04 / 1.97e-04 / 1.97e-04 |
| Allen-Cahn 3D (Wight & Zhao) | 1563.2 (1551.1–1615.0) | 1615.3 (1558.8–1644.6) | 1553.7 (1547.7–1556.8) | -3.8 | 20000 / 20000 / 20000 | 4.04e-04 / 4.28e-04 / 4.28e-04 |

*Median of three seeds (range in brackets) of the training duration DeepXDE reports, summed over all optimization phases. This is not the total wall-clock time: building the data and network before training and the plots and error analysis after it are not included. GUI vs. Solve = 100 × (GUI − Solve script) / Solve script; a negative value is within the run-to-run variation, not a speed-up caused by the GUI. The same settings in all three execution modes of a problem; the settings each problem ran with are saved in configs/. The L₂ errors are medians over the seeds; on the GPU, two runs of the same script with the same seed can end at slightly different errors, because some GPU operations do not add numbers in a fixed order (Allen-Cahn 3D: 4.04e-4 against 4.28e-4). The CPU check in the Export check section, where runs are repeatable, is the one that compares the two scripts.*

### Supplementary Table S3. CPU versus GPU Adam training throughput (CPU: 8 threads)

| Problem | CPU, s per 1,000 Adam steps | GPU, s per 1,000 Adam steps | GPU speedup |
|---|---|---|---|
| Lorenz ODE, inverse | 1.785 | 2.956 | CPU 1.7× faster |
| Wave 1D, multi-scale Fourier features | 2.425 | 0.814 | 3.0× |
| Allen-Cahn 2D (Mattey & Ghosh) | 73.398 | 7.136 | 10.3× |
| Navier-Stokes 2D, Kovasznay flow | 22.525 | 5.464 | 4.1× |
| Allen-Cahn 3D (Wight & Zhao) | 526.154 | 28.461 | 18.5× |

*Median of the repeats; a separately timed warm-up comes first.*

### Supplementary Table S4. CPU-thread scaling of Adam throughput (iterations per second)

| Threads | Heat 1D (2k points) | Poisson 2D (20k points) | Heat 2D (30k points) |
|---|---|---|---|
| 1 | 122.7 | 2.0 | 3.1 |
| 2 | 188.9 | 3.3 | 4.9 |
| 4 | 311.1 | 5.3 | 7.5 |
| 8 | 329.6 | 7.2 | 10.0 |
| 12 | 240.9 | 6.4 | 8.8 |
| 16 | 174.0 | 6.7 | 9.2 |
| 24 | 216.2 | 7.0 | 9.7 |
| 32 | 2.0 | 0.6 | 0.5 |

*Median of the repeats; each thread count in a fresh process.*

*Environment: NVIDIA GeForce RTX 4090 (driver 550.144.03), 13th Gen Intel(R) Core(TM) i9-13900KF; Python 3.11.5, PyTorch 2.6.0+cu124, CUDA 12.4, DeepXDE 1.15.0; commit 594d2957a4c2934718b70305c9226c615b1f46bc.*

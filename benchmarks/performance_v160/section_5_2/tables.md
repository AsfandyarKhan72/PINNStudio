### Supplementary Table S3. CPU versus GPU Adam training throughput (CPU: 8 threads)

| Problem | CPU, s per 1,000 Adam steps | GPU, s per 1,000 Adam steps | GPU speedup |
|---|---|---|---|
| Lorenz ODE, inverse | 1.851 | 2.666 | CPU 1.4× faster |
| Wave 1D, multi-scale Fourier features | 2.433 | 0.816 | 3.0× |
| Allen-Cahn 2D (Mattey & Ghosh) | 76.248 | 7.69 | 9.9× |
| Navier-Stokes 2D, Kovasznay flow | 23.844 | 5.489 | 4.3× |
| Allen-Cahn 3D (Wight & Zhao) | 546.477 | 28.564 | 19.1× |

*Median of the repeats; a separately timed warm-up comes first.*

### Supplementary Table S4. CPU-thread scaling of Adam throughput (iterations per second)

| Threads | Heat 1D (2k points) | Poisson 2D (20k points) | Heat 2D (30k points) |
|---|---|---|---|
| 1 | 118.2 | 1.9 | 3.2 |
| 2 | 194.4 | 3.3 | 5.1 |
| 4 | 256.2 | 5.2 | 7.8 |
| 8 | 321.8 | 7.5 | 10.5 |
| 12 | 233.9 | 5.8 | 9.0 |
| 16 | 188.7 | 6.6 | 9.7 |
| 24 | 159.8 | 7.0 | 9.9 |
| 32 | 1.2 | 0.5 | 0.5 |

*Median of the repeats; each thread count in a fresh process.*

*Environment: NVIDIA GeForce RTX 4090 (driver 550.144.03), 13th Gen Intel(R) Core(TM) i9-13900KF; Python 3.11.5, PyTorch 2.6.0+cu124, CUDA 12.4, DeepXDE 1.15.0; commit 239d925bcba1168c0fcd7f4ff6afb95475cbf1b7.*

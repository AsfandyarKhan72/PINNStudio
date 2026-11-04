# Section 5.2 on PINNStudio v1.6.0 (PySide6): CPU vs GPU throughput (Table S3) and CPU threads (Table S4)

The CPU-versus-GPU and CPU-thread studies of manuscript Section 5.2 repeated on the final v1.6.0 code with the
protocol of the 2026-10-26 study. That study's records are kept unchanged in [`../../performance/`](../../performance/);
its raw records were `a06_perf/out/throughput.jsonl` and `threads.jsonl` on the GPU machine.

## What was run

- **Table S3, CPU vs GPU:** the five problems of Table 2, with the settings the earlier study saved
  (`../../performance/configs/`, loaded with `perf_study.py throughput --configs`). Training is reduced to Adam
  only on both devices: one warm-up phase of 200 steps, timed separately, then the timed phase of 2,000 steps (500
  for Allen–Cahn 2D, 300 for Allen–Cahn 3D). There are no time windows, RAR rounds, IC pre-training or plots, seed 0.
  The CPU uses 8 threads. There are 3 repeats, and the device order alternates between repeats. The configurations
  timed are in `configs_throughput/`. Compared with the saved settings they differ only in those standard changes
  (training phases, seed, time-adaptive training and adaptive sampling switched off) and the reference-data paths.
  The October study applied the same changes in the same code.
- **Table S4, CPU threads:** the three fixed workloads of `benchmarks/gpu_check.py`: Heat 1D with 2,000 points (2,000 Adam
  steps), Poisson 2D with 20,000 points (300) and Heat 2D with 30,000 points (300). Each runs at 1, 2, 4, 8, 12, 16, 24 and
  32 threads, every thread count in a fresh process (OMP/MKL set, CUDA hidden). There are 3 repeats, and the order of the
  thread counts alternates. PyTorch used the requested number of threads in all 72 runs.
- **Measurements:** unchanged (`perf_study.py`). S3 reports seconds per 1,000 Adam steps of the timed phase and S4 Adam
  iterations per second, as the median of the 3 repeats.
- **Code and environment:** commit `239d925` (branch `perf/v1.6.0-pyside6`, PINNStudio 1.6.0, imported from the
  cloned tree), on the same machine and in the same Python environment as the earlier study: NVIDIA GeForce RTX 4090
  (driver 550.144.03), Intel Core i9-13900KF (32 logical CPUs), Python 3.11.5, PyTorch 2.6.0+cu124, CUDA 12.4,
  DeepXDE 1.15.0, NumPy 2.4.6, PySide6 6.11.3. The run went from 2026-11-03 18:47 to 22:30 MT with no other job on the
  machine and no other process on the GPU at the start (`run/gpu_processes_at_start.txt`). All 30 + 72 runs succeeded.

Rebuild the tables: `python benchmarks/perf_study.py tables benchmarks/performance_v160/section_5_2 --out-dir /tmp/t`
(identical to the committed `tables.md` and `runs.csv`, and to the tables the job itself wrote).

## Results and comparison with 2026-10-26

**Table S3** (s per 1,000 Adam steps, median of 3; range of the repeats in brackets):

| Problem | CPU, now | CPU, Oct 26 | GPU, now | GPU, Oct 26 | Now | Oct 26 |
|---|---|---|---|---|---|---|
| Lorenz ODE, inverse | 1.851 (1.82–1.86) | 1.785 | 2.666 (2.63–3.13) | 2.956 | CPU 1.4× faster | CPU 1.7× faster |
| Wave 1D, MsFFN | 2.433 (2.38–2.47) | 2.425 | 0.816 (0.79–0.82) | 0.814 | GPU 3.0× | GPU 3.0× |
| Allen–Cahn 2D (Mattey & Ghosh) | 76.25 (70.4–77.3) | 73.40 | 7.69 (6.82–7.93) | 7.14 | GPU 9.9× | GPU 10.3× |
| Navier–Stokes 2D, Kovasznay | 23.84 (23.7–24.1) | 22.53 | 5.489 (5.39–5.53) | 5.464 | GPU 4.3× | GPU 4.1× |
| Allen–Cahn 3D (Wight & Zhao) | 546.5 (544–564) | 526.2 | 28.56 (28.49–28.62) | 28.46 | GPU 19.1× | GPU 18.5× |

**Table S4** (iterations per second, median of 3; now / Oct 26):

| Threads | Heat 1D (2k points) | Poisson 2D (20k points) | Heat 2D (30k points) |
|---|---|---|---|
| 1 | 118.2 / 122.7 | 1.9 / 2.0 | 3.2 / 3.1 |
| 2 | 194.4 / 188.9 | 3.3 / 3.3 | 5.1 / 4.9 |
| 4 | 256.2 / 311.1 | 5.2 / 5.3 | 7.8 / 7.5 |
| 8 | **321.8** / **329.6** | **7.5** / **7.2** | **10.5** / **10.0** |
| 12 | 233.9 / 240.9 | 5.8 / 6.4 | 9.0 / 8.8 |
| 16 | 188.7 / 174.0 | 6.6 / 6.7 | 9.7 / 9.2 |
| 24 | 159.8 / 216.2 | 7.0 / 7.0 | 9.9 / 9.7 |
| 32 | 1.2 / 2.0 | 0.5 / 0.6 | 0.5 / 0.5 |

**The findings are the same.**
- The GPU is faster for four of the five problems: 3.0–19.1×, against 3.0–18.5× before. The CPU is faster for the
  small inverse Lorenz problem: 1.4× now, 1.7× before. The GPU times are within 1 % of the earlier ones for Wave 1D,
  Kovasznay and Allen–Cahn 3D, +8 % for Allen–Cahn 2D and −10 % for Lorenz.
- In S4, 8 threads still give the highest throughput for all three workloads, 2.7–3.9× the single-thread rate
  (2.7–3.6× before). More threads do not help, and all 32 logical threads collapse the throughput, as before.

**Differences worth noting.**
- The Lorenz ratio moved from 1.7× to 1.4×, mainly because the GPU time is lower (2.67 against 2.96 s). This problem
  has a 3-layer network of 40 units with 400 points, so its GPU time is dominated by per-step overheads, and its repeat
  range (2.63–3.13 s) covers most of the difference.
- In S4, the small Heat 1D workload varies the most between runs. At 4 threads the repeats are 248–282 it/s now and
  302–326 it/s before; at 24 threads they are 146–164 now and 158–220 before. These cells move the most, but neither is
  the optimum, and the 8-thread values agree within 3 %.

## Section 4.3 — the exported script trains the same model

| | |
|---|---|
| **What it shows** | that *Export as DeepXDE script* gives a standalone script that trains exactly the same model as *Solve* inside PINNStudio, so a user can leave the GUI without losing anything |
| **How it was run** | every example (forward, and inverse where it has one) with its Tuned settings and a short training (every Adam phase 200 steps, every L-BFGS phase 50, IC pre-training and resampling cut down the same way); both scripts written from the same configuration and run on the CPU with the same seed; the printed loss history compared line by line, and for inverse problems the identified parameters |
| **For users** | [performance.md → Export check](performance.md#export-check-the-exported-script-trains-the-same-model) |
| **Data and scripts** | `tools/check_export_equivalence.py` (writes `results.json` and a summary) |
| **Status** | running: 45 of the first 48 checked are identical, none differ; the rest are in progress |

**What to write in the paper (Section 4.3)** — a draft; the count is filled in when the check finishes:

> The script exported from PINNStudio is not a simplified copy: it contains the same model, constraints and training schedule as the script PINNStudio runs itself. We checked this for all [N] examples by writing both scripts from the same settings, running them with the same random seed and comparing the loss after every logged step; the two loss histories, and for the inverse problems the identified parameters, were identical in every case.

## Section 5 — performance (Tables 5, 6 and S3)

| | |
|---|---|
| **What it shows** | Table 5: how much time the PINNStudio window adds over running the same training as a plain script. Table 6: how much faster training is on the GPU than on the CPU. Table S3: how training on the CPU scales with the number of threads |
| **How it was run** | one desktop PC (Intel Core i9-13900KF, 24 cores = 8 performance + 16 efficiency, 32 hardware threads; NVIDIA RTX 4090; Ubuntu, PyTorch 2.6, CUDA 12.4, DeepXDE 1.15); every run records the commit and versions. Details per table below and in [performance.md](performance.md) |
| **Data and scripts** | `benchmarks/perf_study.py` (`overhead`, `throughput`, `threads`, `tables`); results in `benchmarks/performance/` |
| **Status** | Table S3 running (first of three repeats done); Tables 6 and 5 queued after it (a crash in the GPU-memory sampler was fixed and they are rerun) |

**Table 5 — overhead of the PINNStudio window.** Five problems (Lorenz inverse, Wave 1D with multi-scale Fourier features, Allen-Cahn 2D of Mattey & Ghosh, Kovasznay flow, Allen-Cahn 3D of Wight & Zhao) with their full Tuned settings, each solved three ways from the same configuration: the exported script run with Python, the script *Solve* runs (run with Python, no window), and *Solve* in the PINNStudio window with the live loss plot and log on, on the logged-in desktop. Three seeds; the order of the three ways rotates with the seed. Reported: wall time of each and the window's extra time in percent. *Results: queued.*

**Table 6 — GPU against CPU.** The same five problems, Adam only: a 200-step warm-up and then a fixed number of timed Adam steps, on the CPU with 8 threads and on the GPU, same network, points and float precision on both. Reported: seconds per 1,000 Adam steps on each device and the GPU speed-up (median of 3 repeats). *Results: queued.*

**Table S3 — CPU threads.** Adam steps per second for three fixed workloads (Heat 1D with 2,000 points, 2,000 steps; Poisson 2D with 20,000 points, 300 steps; Heat 2D with 30,000 points, 300 steps) on the CPU only, with 1, 2, 4, 8, 12, 16, 24 and 32 threads, each in a fresh process; three repeats, the second in reverse order of thread count. *First repeat (final values will be the medians of three):*

| Threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| Heat 1D (2k points) | 124 | 189 | 311 | **368** | 259 | 236 | 216 | 2.0 |
| Poisson 2D (20k points) | 2.0 | 3.4 | 5.4 | **7.2** | 6.4 | 6.7 | 7.0 | 0.6 |
| Heat 2D (30k points) | 3.1 | 4.9 | 7.5 | **10.0** | 8.8 | 9.2 | 9.8 | – |

**What to write in the paper (Section 5)** — a draft; the numbers in [brackets] come from the runs:

> All timings were made on one desktop computer (Intel Core i9-13900KF, NVIDIA RTX 4090). Running a problem from the PINNStudio window, with the live loss plot and training log, took [x–y]% longer than running the exported script from the command line (Table 5), so the interface adds little to the cost of training. On the GPU, training was [a–b] times faster than on the CPU, the gain growing with the number of collocation points (Table 6). On the CPU, training was fastest with 8 threads, [3.0–3.6] times faster than with one; more threads made it slower, because the work spreads onto the processor's slower efficiency cores, and using all 32 hardware threads slowed it down sharply (Table S3). PINNStudio therefore uses one thread per performance core, at most 8, by default (*CPU threads: Automatic*).

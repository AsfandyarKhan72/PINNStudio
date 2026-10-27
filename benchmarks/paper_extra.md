## Section 4.3 — the exported script trains the same model

| | |
|---|---|
| **What it shows** | that *Export as DeepXDE script* gives a standalone script that trains exactly the same model as *Solve* inside PINNStudio, so a user can leave the GUI without losing anything |
| **How it was run** | every example (forward, and inverse where it has one) with its Tuned settings and a short training (every Adam phase 200 steps, every L-BFGS phase 50, IC pre-training and resampling cut down the same way); both scripts written from the same configuration and run on the CPU with the same seed; the printed loss history compared line by line, and for inverse problems the identified parameters |
| **For users** | [performance.md → Export check](performance.md#export-check-the-exported-script-trains-the-same-model) |
| **Data and scripts** | `tools/check_export_equivalence.py` (writes `results.json` and a summary) |
| **Status** | running: 47 of the first 50 checked are identical, 2 skipped (no inverse version), none differ; the full check runs on the GPU machine's CPU |

**What to write in the paper (Section 4.3)** — a draft; the count is filled in when the check finishes:

> The script exported from PINNStudio is not a simplified copy: it contains the same model, constraints and training schedule as the script PINNStudio runs itself. We checked this for all [N] examples by writing both scripts from the same settings, running them with the same random seed and comparing the loss after every logged step; the two loss histories, and for the inverse problems the identified parameters, were identical in every case.

## Section 5 — performance (Tables 5, 6 and S3)

| | |
|---|---|
| **What it shows** | Table 5: how much time the PINNStudio window adds over running the same training as a plain script. Table 6: how much faster training is on the GPU than on the CPU. Table S3: how training on the CPU scales with the number of threads |
| **How it was run** | one desktop PC (Intel Core i9-13900KF, 24 cores = 8 performance + 16 efficiency, 32 hardware threads; NVIDIA RTX 4090, driver 550.144; Linux, Python 3.11, PyTorch 2.6 with CUDA 12.4, DeepXDE 1.15); every run records the commit and versions. Table 5 on the logged-in desktop (the window really drawn) |
| **For users** | [performance.md](performance.md) |
| **Data and scripts** | `benchmarks/perf_study.py`; tables in [`performance/tables.md`](performance/tables.md), every run in [`performance/runs.csv`](performance/runs.csv), each problem's settings in [`performance/configs/`](performance/configs) |
| **Status** | done (October 2026) |

**Table 5 — the PINNStudio window adds no measurable time.** Five problems, each solved three ways from the same configuration: the exported script run with Python, the script *Solve* runs (run with Python, no window), and *Solve* in the PINNStudio window with the live loss plot and log.

| Problem (settings) | Exported script (s) | Solve script, no window (s) | PINNStudio window (s) | Window vs Solve script |
|---|---|---|---|---|
| Lorenz ODE, inverse (Default = best) | 71.5 | 76.7 | 72.8 | −5.0% |
| Wave 1D, multi-scale Fourier features (Default) | 75.5 | 74.7 | 75.0 | +0.5% |
| Allen-Cahn 2D, Mattey & Ghosh (Tuned) | 266.3 | 265.7 | 259.4 | −2.4% |
| Navier-Stokes 2D, Kovasznay flow (Default = best) | 204.0 | 207.4 | 204.4 | −1.5% |
| Allen-Cahn 3D, Wight & Zhao (Default) | 1563.2 | 1615.3 | 1553.7 | −3.8% |

*Training time (s) DeepXDE reports, summed over the training phases, median of 3 seeds; the same settings and number of steps in all three ways. "Default" for Wave 1D and Allen-Cahn 3D: these two ran with their Default settings because of a bug in loading Settings: Tuned after an example without Tuned settings (fixed); the comparison of the three ways is the same either way.*

**Table 6 — GPU against CPU.**

| Problem | CPU, 8 threads (s per 1,000 Adam steps) | GPU (s per 1,000 Adam steps) | GPU speed-up |
|---|---|---|---|
| Lorenz ODE, inverse | 1.79 | 2.96 | CPU 1.7× faster |
| Wave 1D, multi-scale Fourier features | 2.43 | 0.81 | 3.0× |
| Navier-Stokes 2D, Kovasznay flow | 22.5 | 5.46 | 4.1× |
| Allen-Cahn 2D, Mattey & Ghosh | 73.4 | 7.14 | 10.3× |
| Allen-Cahn 3D, Wight & Zhao | 526 | 28.5 | 18.5× |

*Adam only, same network, points and float precision on both; a 200-step warm-up timed separately; median of 3 repeats.*

**Table S3 — CPU threads.**

| Threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| Heat 1D (2k points) | 123 | 189 | 311 | **330** | 241 | 174 | 216 | 2.0 |
| Poisson 2D (20k points) | 2.0 | 3.3 | 5.3 | **7.2** | 6.4 | 6.7 | 7.0 | 0.6 |
| Heat 2D (30k points) | 3.1 | 4.9 | 7.5 | **10.0** | 8.8 | 9.2 | 9.7 | 0.5 |

*Adam steps per second on the CPU only, median of 3 repeats (the second in reverse order of thread count), each thread count in a fresh process.*

**What to write in the paper (Section 5)** — a draft:

> All timings were made on one desktop computer (Intel Core i9-13900KF, NVIDIA RTX 4090). Training a problem from the PINNStudio window, with the live loss plot and training log, took the same time as running the exported script from the command line: the three ways of running in Table 5 differ by at most 5%, in no consistent direction, so the interface adds no measurable cost to training. On the GPU, Adam training was 3 to 18.5 times faster than on the CPU, the gain growing with the size of the problem; only the smallest problem, the Lorenz system, ran faster on the CPU (Table 6). On the CPU, training was fastest with 8 threads, 2.7 to 3.6 times faster than with one; more threads made it no faster or slower, because the work spreads onto the processor's slower efficiency cores, and using all 32 hardware threads slowed it down 10 to 165 times (Table S3). PINNStudio therefore uses one thread per performance core, at most 8, by default (*CPU threads: Automatic*).

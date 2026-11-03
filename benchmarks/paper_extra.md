## Section 4.6 — consistency of generated scripts

| | |
|---|---|
| **What it shows** | that *Export as DeepXDE script* gives a standalone script that trains exactly the same model as *Solve* inside PINNStudio, so a user can leave the GUI without losing anything |
| **How it was run** | every example (forward, and inverse where it has one) with its Tuned settings and a short training (every Adam phase 200 steps, every L-BFGS phase 50, IC pre-training and resampling cut down the same way); both scripts written from the same configuration and run on the CPU with the same seed; the printed loss history compared line by line, and for inverse problems the identified parameters |
| **For users** | [performance.md → Export check](performance.md#export-check-the-exported-script-trains-the-same-model) |
| **Data and scripts** | `tools/check_export_equivalence.py` (writes `results.json` and a summary) |
| **Status** | done on the GPU machine's CPU, every example: 91 checked, 84 identical, 3 skipped (a parameter sweep and two templates without an inverse version), and in 4 inverse problems (Linear ODE system: learned w = 1.81166 against 1.81165; Lorenz: the learned C's; the linear elastic plate: μ = 0.819561 against 0.81956; 2D Heat: α = 0.591128 against 0.591129) the loss history is identical and only the last printed digit of a parameter differs, as two runs of the same script on several CPU threads can. The check found two bugs, both fixed and checked again: the short export left out IC-guided learning and backward compatibility (1D Cahn-Hilliard of Mattey & Ghosh), and with time windows it started the first window from different random weights than Solve (2D Allen-Cahn of Wight & Zhao, 2D Burgers of Mathias et al.). The Poisson 3D cut-out example was added after this check |

**What to write in the paper (Section 4.6)** — a draft:

> The script exported from PINNStudio is not a simplified copy: it contains the same model, constraints and training schedule as the internal Solve script that PINNStudio runs itself. We checked this for 88 problems, every example in forward mode and, where it has one, in inverse mode, by writing both scripts from the same settings, running them with the same random seed and comparing the loss after every logged step; the two loss histories were identical in every case, and the identified parameters of the inverse problems agreed to at least five significant digits. Two differences the check found along the way were fixed.

## Section 5 — computational performance (Table 2, Supplementary Tables S3 and S4)

| | |
|---|---|
| **What it shows** | Section 5.1, Table 2: the training time in the three execution modes (exported standalone script, internal Solve script, PINNStudio GUI). Section 5.2, Supplementary Table S3: CPU versus GPU Adam training throughput; Supplementary Table S4: how training on the CPU scales with the number of threads |
| **How it was run** | one desktop PC (Intel Core i9-13900KF, 24 cores = 8 performance + 16 efficiency, 32 hardware threads; NVIDIA RTX 4090, driver 550.144; Linux, Python 3.11, PyTorch 2.6 with CUDA 12.4, DeepXDE 1.15); every run records the commit and versions. Table 2 on the logged-in desktop (the GUI really drawn) |
| **For users** | [performance.md](performance.md) |
| **Data and scripts** | `benchmarks/perf_study.py`; tables in [`performance/tables.md`](performance/tables.md), every run in [`performance/runs.csv`](performance/runs.csv), each problem's settings in [`performance/configs/`](performance/configs) |
| **Status** | done (October 2026) |

**Table 2 — training time across the three execution modes.** Five problems, each solved three ways from the same configuration: the exported standalone script run with Python, the internal Solve script run with Python and no GUI, and *Solve* in the PINNStudio GUI with the live loss plot and log.

| Problem (settings) | Exported script (s) | Solve script (s) | PINNStudio GUI (s) | GUI vs. Solve (%) |
|---|---|---|---|---|
| Lorenz ODE, inverse (Default = best) | 71.5 | 76.7 | 72.8 | −5.0 |
| Wave 1D, multi-scale Fourier features (Default) | 75.5 | 74.7 | 75.0 | +0.5 |
| Allen-Cahn 2D, Mattey & Ghosh (Tuned) | 266.3 | 265.7 | 259.4 | −2.4 |
| Navier-Stokes 2D, Kovasznay flow (Default = best) | 204.0 | 207.4 | 204.4 | −1.5 |
| Allen-Cahn 3D, Wight & Zhao (Default) | 1563.2 | 1615.3 | 1553.7 | −3.8 |

*Training duration (s) reported by DeepXDE, summed over the optimization phases, median of 3 seeds. It is not the total wall-clock time: building the data and network before training and the plots and error analysis after it are not included. GUI vs. Solve = 100 × (GUI − Solve script) / Solve script; the negative values are within the run-to-run variation (see the ranges in [`performance/tables.md`](performance/tables.md)), not a speed-up caused by the GUI. The same settings and number of optimizer steps in all three modes. "Default" for Wave 1D and Allen-Cahn 3D: these two ran with their Default settings because of a bug in loading Settings: Tuned after an example without Tuned settings (fixed); the comparison of the three modes is the same either way.*

**Supplementary Table S3 — CPU versus GPU Adam training throughput.**

| Problem | CPU, 8 threads (s per 1,000 Adam steps) | GPU (s per 1,000 Adam steps) | GPU speed-up |
|---|---|---|---|
| Lorenz ODE, inverse | 1.79 | 2.96 | CPU 1.7× faster |
| Wave 1D, multi-scale Fourier features | 2.43 | 0.81 | 3.0× |
| Navier-Stokes 2D, Kovasznay flow | 22.5 | 5.46 | 4.1× |
| Allen-Cahn 2D, Mattey & Ghosh | 73.4 | 7.14 | 10.3× |
| Allen-Cahn 3D, Wight & Zhao | 526 | 28.5 | 18.5× |

*Adam only, same network, points and float precision on both; a 200-step warm-up timed separately; median of 3 repeats.*

**Supplementary Table S4 — CPU-thread scaling.**

| Threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| Heat 1D (2k points) | 123 | 189 | 311 | **330** | 241 | 174 | 216 | 2.0 |
| Poisson 2D (20k points) | 2.0 | 3.3 | 5.3 | **7.2** | 6.4 | 6.7 | 7.0 | 0.6 |
| Heat 2D (30k points) | 3.1 | 4.9 | 7.5 | **10.0** | 8.8 | 9.2 | 9.7 | 0.5 |

*Adam steps per second on the CPU only, median of 3 repeats (the second in reverse order of thread count), each thread count in a fresh process.*

**What to write in the paper (Section 5)** — a draft:

> All timings were made on one desktop computer (Intel Core i9-13900KF, NVIDIA RTX 4090). The time we report is the training duration reported by DeepXDE, summed over the optimization phases; it leaves out building the data and network and the post-processing, so it is not the total wall-clock time. Table 2 compares the three execution modes for five problems: the exported standalone script, the internal Solve script run on its own, and the PINNStudio GUI with the live loss plot and training log. The three modes ran the same number of optimizer steps and their times agree within about 7%, and the GUI runs were between 5.0% below and 0.5% above the Solve script. The negative differences are within the run-to-run variation and do not mean the GUI makes training faster; under the tested conditions we saw no systematic increase in the recorded training time for GUI execution.
>
> On the GPU, Adam training was 3 to 18.5 times faster than on the CPU, and the gain grew with the size of the problem; only the smallest problem, the Lorenz system, ran faster on the CPU (Supplementary Table S3). On the CPU, training was fastest with 8 threads, 2.7 to 3.7 times faster than with one. More threads made it no faster or slower, because the work spreads onto the processor's slower efficiency cores, and using all 32 hardware threads slowed it down 12 to 165 times (Supplementary Table S4). PINNStudio therefore uses half the logical processors, at most 8, by default (*CPU threads: Automatic*), which on this processor is one thread per performance core.

# Performance and checks

*How fast PINNStudio trains, what the window costs, how to choose CPU threads, and the check that an exported script trains the same model. The tables are filled in as the runs finish; the paper's versions are on [paper.md](paper.md).*

**The machine.** One desktop PC: Intel Core i9-13900KF (24 cores: 8 performance and 16 efficiency, 32 hardware threads), NVIDIA RTX 4090, Ubuntu, Python 3.11, PyTorch 2.6 with CUDA 12.4, DeepXDE 1.15. Every run records the PINNStudio commit and these versions. Script: [`perf_study.py`](perf_study.py).

## Does the PINNStudio window slow training down?

Five problems (Lorenz inverse, Wave 1D with multi-scale Fourier features, Allen-Cahn 2D of Mattey & Ghosh, Kovasznay flow, Allen-Cahn 3D of Wight & Zhao) are solved with their full *Tuned* settings in three ways from the same configuration:

- **Exported script**: *Export as DeepXDE script*, run with Python from a terminal.
- **Solve script**: the script PINNStudio's *Solve* runs, run with Python without the window.
- **PINNStudio window**: *Solve* in the window, with the live loss plot and training log on.

Each with three seeds; the order of the three rotates with the seed so that none always runs first.

*Results: queued.*

## How much faster is the GPU?

The same five problems, Adam only: 200 warm-up steps, then a fixed number of timed Adam steps, on the CPU (8 threads) and on the GPU, with the same network, points and float precision. Reported: seconds per 1,000 Adam steps and the GPU speed-up, median of 3 repeats.

*Results: queued.*

## How many CPU threads?

Adam steps per second on the CPU only, for three fixed workloads (Heat 1D with 2,000 points, Poisson 2D with 20,000 points, Heat 2D with 30,000 points), with 1 to 32 threads, each thread count in a fresh process. Three repeats; the second runs the thread counts in reverse order, so that a slow-down caused by the order would show.

First repeat (the final table gives the median of three):

| Threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| Heat 1D (2k points) | 124 | 189 | 311 | **368** | 259 | 236 | 216 | 2.0 |
| Poisson 2D (20k points) | 2.0 | 3.4 | 5.4 | **7.2** | 6.4 | 6.7 | 7.0 | 0.6 |
| Heat 2D (30k points) | 3.1 | 4.9 | 7.5 | **10.0** | 8.8 | 9.2 | 9.8 | – |

**What this means for you.** On a processor with performance and efficiency cores, use about one thread per performance core. Past that the work spreads onto the slower cores and every step waits for them, and using every hardware thread can make training many times slower. PINNStudio's *GPU / Hardware settings → CPU threads: Automatic* does this for you (one thread per two logical cores, at most 8); set a number there only if you have measured something better on your machine. The error is the same with any thread count; only the speed changes.

## Export check: the exported script trains the same model

PINNStudio writes two scripts from one configuration: the one *Solve* runs, and the one *Export as DeepXDE script* saves. For every example (forward, and inverse where it has one) the check loads its *Tuned* settings, shortens the training (every Adam phase to 200 steps, every L-BFGS phase to 50), writes both scripts, runs them on the CPU with the same seed, and compares the printed loss history line by line and, for inverse problems, the identified parameters.

```
python tools/check_export_equivalence.py --out equivalence_runs [--names 'a|b']
```

*Status: running. Of the first 48 problems checked, 45 are identical and 2 are skipped (two templates that have no inverse version); none differ. One is being rerun with a longer time limit.*

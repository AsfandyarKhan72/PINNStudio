# Performance and checks

*How fast PINNStudio trains, what the window costs, how to choose CPU threads, and the check that an exported script trains the same model. The paper's versions of these tables are on [paper.md](paper.md).*

**The machine.** One desktop PC: Intel Core i9-13900KF (24 cores: 8 performance and 16 efficiency, 32 hardware threads), NVIDIA RTX 4090, Ubuntu, Python 3.11, PyTorch 2.6 with CUDA 12.4, DeepXDE 1.15. Every run records the PINNStudio commit and these versions. Script: [`perf_study.py`](perf_study.py).

## Does the PINNStudio window slow training down?

Five problems (Lorenz inverse, Wave 1D with multi-scale Fourier features, Allen-Cahn 2D of Mattey & Ghosh, Kovasznay flow, Allen-Cahn 3D of Wight & Zhao) are solved in three ways from the same configuration, with the settings given in the table:

- **Exported script**: *Export as DeepXDE script*, run with Python from a terminal.
- **Solve script**: the script PINNStudio's *Solve* runs, run with Python without the window.
- **PINNStudio window**: *Solve* in the window, with the live loss plot and training log on.

Each with three seeds; the order of the three rotates with the seed so that none always runs first.

| Problem (settings) | Exported script (s) | Solve script, no window (s) | PINNStudio window (s) | Window vs Solve script |
|---|---|---|---|---|
| Lorenz ODE, inverse (Default = best) | 71.5 | 76.7 | 72.8 | −5.0% |
| Wave 1D, multi-scale Fourier features (Default) | 75.5 | 74.7 | 75.0 | +0.5% |
| Allen-Cahn 2D, Mattey & Ghosh (Tuned) | 266.3 | 265.7 | 259.4 | −2.4% |
| Navier-Stokes 2D, Kovasznay flow (Default = best) | 204.0 | 207.4 | 204.4 | −1.5% |
| Allen-Cahn 3D, Wight & Zhao (Default) | 1563.2 | 1615.3 | 1553.7 | −3.8% |

*Training time (s) DeepXDE reports, summed over the training phases, median of 3 seeds; the same settings and number of steps in all three ways. "Default" for Wave 1D and Allen-Cahn 3D: these two ran with their Default settings because of a bug in loading Settings: Tuned after an example without Tuned settings (fixed); the comparison of the three ways is the same either way.*

**What this means for you.** Use the window: watching training live costs nothing. The window was within 5% of the same script run without it, and slightly faster in four of the five problems; differences this small also appear between two runs of the same script.

## How much faster is the GPU?

The same five problems, Adam only: 200 warm-up steps, then a fixed number of timed Adam steps, on the CPU (8 threads) and on the GPU, with the same network, points and float precision. Reported: seconds per 1,000 Adam steps and the GPU speed-up, median of 3 repeats.

| Problem | CPU, 8 threads (s per 1,000 Adam steps) | GPU (s per 1,000 Adam steps) | GPU speed-up |
|---|---|---|---|
| Lorenz ODE, inverse | 1.79 | 2.96 | CPU 1.7× faster |
| Wave 1D, multi-scale Fourier features | 2.43 | 0.81 | 3.0× |
| Navier-Stokes 2D, Kovasznay flow | 22.5 | 5.46 | 4.1× |
| Allen-Cahn 2D, Mattey & Ghosh | 73.4 | 7.14 | 10.3× |
| Allen-Cahn 3D, Wight & Zhao | 526 | 28.5 | 18.5× |

*Adam only, same network, points and float precision on both; a 200-step warm-up timed separately; median of 3 repeats.*

**What this means for you.** A GPU pays off as soon as a problem has a few thousand points or more: about 3× for a 1D problem, 4–10× in 2D and close to 20× in 3D. For a very small problem (an ODE with a few hundred points) the CPU is faster, because starting work on the GPU costs more than it saves.

## How many CPU threads?

Adam steps per second on the CPU only, for three fixed workloads (Heat 1D with 2,000 points, Poisson 2D with 20,000 points, Heat 2D with 30,000 points), with 1 to 32 threads, each thread count in a fresh process. Three repeats; the second runs the thread counts in reverse order, so that a slow-down caused by the order would show.

| Threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| Heat 1D (2k points) | 123 | 189 | 311 | **330** | 241 | 174 | 216 | 2.0 |
| Poisson 2D (20k points) | 2.0 | 3.3 | 5.3 | **7.2** | 6.4 | 6.7 | 7.0 | 0.6 |
| Heat 2D (30k points) | 3.1 | 4.9 | 7.5 | **10.0** | 8.8 | 9.2 | 9.7 | 0.5 |

*Adam steps per second on the CPU only, median of 3 repeats (the second in reverse order of thread count), each thread count in a fresh process.*

**What this means for you.** On a processor with performance and efficiency cores, use about one thread per performance core. Past that the work spreads onto the slower cores and every step waits for them, and using every hardware thread can make training many times slower. PINNStudio's *Settings → GPU / Hardware → CPU threads: Automatic* does this for you (one thread per two logical cores, at most 8); set a number there only if you have measured something better on your machine. The error is the same with any thread count; only the speed changes.

## Export check: the exported script trains the same model

PINNStudio writes two scripts from one configuration: the one *Solve* runs, and the one *Export as DeepXDE script* saves. For every example (forward, and inverse where it has one) the check loads its *Tuned* settings, shortens the training (every Adam phase to 200 steps, every L-BFGS phase to 50), writes both scripts, runs them on the CPU with the same seed, and compares the printed loss history line by line and, for inverse problems, the identified parameters.

```
python tools/check_export_equivalence.py --out equivalence_runs [--names 'a|b']
```

*Status: done on the GPU machine's CPU, every example: 91 checked, 81 identical, 3 skipped (a parameter sweep and two templates without an inverse version). In four inverse problems (Linear ODE system, Lorenz, the linear elastic plate, 2D Heat) the loss history is identical and a learned parameter differs only in its sixth significant digit, as two runs of the same script on several CPU threads can. The third, 1D Cahn-Hilliard of Mattey & Ghosh, showed a real difference: the export left out IC-guided learning and backward compatibility. This is fixed, and that example is being checked again. Two examples with time windows (2D Allen-Cahn of Wight & Zhao, 2D Burgers of Mathias et al.) start from different random weights in the two scripts, because Solve's script builds one more network before the windows; a fix is being decided.*

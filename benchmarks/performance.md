# Performance and checks

*How fast PINNStudio trains, whether training from the GUI takes longer, how to choose CPU threads, and the check that an exported script trains the same model.*

**The machine.** One desktop PC: Intel Core i9-13900KF (24 cores: 8 performance and 16 efficiency, 32 hardware threads), NVIDIA RTX 4090 (driver 550.144.03), Ubuntu, Python 3.11.5, PyTorch 2.6.0 with CUDA 12.4, DeepXDE 1.15.0, PySide6 6.11.3. Script: [`perf_study.py`](perf_study.py).

**The measurements below are those of PINNStudio v1.6.0** (PySide6), made on 2026-11-03 with the settings and protocol of the first study: records, configurations, raw measurements, environment and rebuild commands in [`performance_v160/`](performance_v160/README.md) (Table 2) and [`performance_v160/section_5_2/`](performance_v160/section_5_2/README.md) (the CPU/GPU and thread tables), each with a comparison with the first study. The first study (2026-10-26, the PyQt6 code before v1.6.0) is kept unchanged in [`performance/`](performance/tables.md) as the historical record; its findings are the same.

## Does the PINNStudio GUI slow training down?

*Paper: Table 2.* Five problems (Lorenz inverse, Wave 1D with multi-scale Fourier features, Allen-Cahn 2D of Mattey & Ghosh, Kovasznay flow, Allen-Cahn 3D of Wight & Zhao) are solved in three ways from the same configuration, with the settings given in the table:

- **Exported standalone script**: *Export as DeepXDE script*, run with Python from a terminal.
- **Internal Solve script**: the script PINNStudio's *Solve* runs, run with Python on its own, without the GUI.
- **PINNStudio GUI**: *Solve* in the GUI, with the live loss plot and training log on.

Each with three seeds; the order of the three rotates with the seed so that none always runs first.

| Problem (settings) | Exported script (s) | Solve script (s) | PINNStudio GUI (s) | GUI vs. Solve (%) |
|---|---|---|---|---|
| Lorenz ODE, inverse (Default = best) | 71.2 | 73.1 | 73.7 | +0.7 |
| Wave 1D, multi-scale Fourier features (Default) | 71.3 | 70.7 | 70.9 | +0.2 |
| Allen-Cahn 2D, Mattey & Ghosh (fixed best setup: exact IC and periodicity, SA-PINN, 20,000 Adam + 10,000 L-BFGS) | 253.8 | 255.4 | 255.5 | 0.0 |
| Navier-Stokes 2D, Kovasznay flow (Default = best) | 190.7 | 194.9 | 190.3 | −2.4 |
| Allen-Cahn 3D, Wight & Zhao (Default) | 1582.1 | 1605.2 | 1612.9 | +0.5 |

*Training duration (s) reported by DeepXDE, summed over the optimization phases, median of seeds 1, 2 and 3. It is not the total wall-clock time: building the data and network before training and the plots and error analysis after it are not included. GUI vs. Solve = 100 × (GUI − Solve script) / Solve script; the negative value is within the run-to-run variation (ranges in [`performance_v160/tables.md`](performance_v160/tables.md)), not a speed-up caused by the GUI. The same settings in all three modes; for every problem and seed the three modes took the same number of optimizer steps and ended at the same errors and parameters. "Default" for Wave 1D and Allen-Cahn 3D: these two ran with their Default settings because of a bug in loading Settings: Tuned after an example without Tuned settings (fixed); the comparison of the three modes is the same either way.*

**What this means for you.** Use the GUI: watching training live did not make it slower. The GUI runs were within −2.4% to +0.7% of the same Solve script run on its own, and differences this small also appear between two runs of the same script, so there was no systematic increase in the training time DeepXDE reports from the GUI in these tests.

## How much faster is the GPU?

*Paper: Supplementary Table S3.* The same five problems, Adam only: 200 warm-up steps, then a fixed number of timed Adam steps (2,000; 500 for Allen-Cahn 2D, 300 for Allen-Cahn 3D), on the CPU (8 threads) and on the GPU, with the same network, points and float precision. Reported: seconds per 1,000 Adam steps and the GPU speed-up, median of 3 repeats.

| Problem | CPU, 8 threads (s per 1,000 Adam steps) | GPU (s per 1,000 Adam steps) | GPU speed-up |
|---|---|---|---|
| Lorenz ODE, inverse | 1.85 | 2.67 | CPU 1.4× faster |
| Wave 1D, multi-scale Fourier features | 2.43 | 0.82 | 3.0× |
| Navier-Stokes 2D, Kovasznay flow | 23.8 | 5.49 | 4.3× |
| Allen-Cahn 2D, Mattey & Ghosh | 76.2 | 7.69 | 9.9× |
| Allen-Cahn 3D, Wight & Zhao | 546 | 28.6 | 19.1× |

*Adam only, same network, points and float precision on both; a 200-step warm-up timed separately; median of 3 repeats.*

**What this means for you.** On this machine, the GPU was faster for every tested problem except the smallest: about 3× for the 1D wave problem, 4–10× for the 2D problems and about 19× for the 3D problem. For the Lorenz ODE (a few hundred points and a small network) the CPU was faster, because starting work on the GPU costs more than it saves. These speed-ups belong to these five workloads, this CPU and this GPU: they are not a general threshold. Where a GPU starts to pay off depends on the network size, the number of points, the derivatives in the equations and the hardware, so for a new problem compare a short run on each device (*Settings → GPU / Hardware*).

## How many CPU threads?

*Paper: Supplementary Table S4.* Adam steps per second on the CPU only, for three fixed workloads (Heat 1D with 2,000 points, Poisson 2D with 20,000 points, Heat 2D with 30,000 points), with 1 to 32 threads, each thread count in a fresh process. Three repeats; the second runs the thread counts in reverse order, so that a slow-down caused by the order would show.

| Threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| Heat 1D (2k points) | 118 | 194 | 256 | **322** | 234 | 189 | 160 | 1.2 |
| Poisson 2D (20k points) | 1.9 | 3.3 | 5.2 | **7.5** | 5.8 | 6.6 | 7.0 | 0.5 |
| Heat 2D (30k points) | 3.2 | 5.1 | 7.8 | **10.5** | 9.0 | 9.7 | 9.9 | 0.5 |

*Adam steps per second on the CPU only, median of 3 repeats (the second in reverse order of thread count), each thread count in a fresh process.*

**What this means for you.** On a processor with performance and efficiency cores, use about one thread per performance core. Past that the work spreads onto the slower cores and every step waits for them, and using every hardware thread can make training many times slower. PINNStudio's *Settings → GPU / Hardware → CPU threads: Automatic* does this for you (one thread per two logical cores, at most 8); set a number there only if you have measured something better on your machine.

**The thread count can change the result slightly.** These workloads are short timing runs (seed 0, 2,000 Adam steps for Heat 1D and 300 for the others, not trained to convergence), and their records also hold the error reached (`adam_l2` in [`threads.jsonl`](performance_v160/section_5_2/threads.jsonl)). For one thread count, the error is identical in all three repeats and in the first study too. Between thread counts it differs: Poisson 2D 0.1140–0.1141, Heat 2D 0.2215–0.2243, and Heat 1D 0.0074–0.0117 except 0.0552 at 12 threads. The most likely cause, consistent with these records: on the CPU, the number of threads decides how sums inside the network's operations are split, so the rounding differs and the training follows a slightly different path. That effect is usually small but can be larger early in training, as in the Heat 1D run at 12 threads. It is the same kind of difference as between CPU and GPU, not an error, but it means a result is exactly repeatable only with the same thread count (and device).

## Apple GPU (MPS) on a Mac

*Not in the paper.* Measured on an Apple M1 MacBook Air with [`mac_gpu_check.py`](mac_gpu_check.py) (first recorded in the README in commit 28b7087, 2026-10-20; kept here since the README's timing section was retired). Three problems with an exact solution (Heat 1D, Poisson 2D with 20,000 points, Heat 2D with 30,000 points), Adam then L-BFGS in float32, on the CPU and on MPS:

- **Heat 1D:** MPS ran at half the speed of the CPU (73 against 136 Adam iterations a second), and L-BFGS was 7 times slower on MPS.
- **Poisson 2D and Heat 2D:** about the same on both, 3 to 4 Adam iterations a second.
- **Accuracy:** the same on both. MPS has no float64.

For comparison, an RTX 4090 ran the same 2D problems at 139 to 162 iterations a second ([`gpu_check.py`](gpu_check.py), which also checks an NVIDIA GPU's setup on Windows or Linux).

**What this means for you.** PINNStudio trains on the CPU on a Mac. A Mac with more GPU cores (an M-series Pro or Max) may do better on large problems; `mac_gpu_check.py` measures it on your machine.

The Windows measurements (a laptop with a Core i9-8950HK and an RTX 2080 Max-Q: start-up time, Windows Defender, power mode, when the GPU pays off) are in [`platform/windows_speed.md`](platform/windows_speed.md).

## Export check: the exported script trains the same model

PINNStudio writes two scripts from one configuration: the one *Solve* runs, and the one *Export as DeepXDE script* saves. For every example (forward, and inverse where it has one) the check loads its *Tuned* settings, shortens the training (every Adam phase to 200 steps, every L-BFGS phase to 50), writes both scripts, runs them on the CPU with the same seed, and compares the printed loss history line by line and, for inverse problems, the identified parameters.

```
python tools/check_export_equivalence.py --out equivalence_runs [--names 'a|b']
```

*Status: the latest run (v1.6.0 code, `main` 2ea54c3, CPU) checked 93 configurations: 89 bitwise identical (88 comparing two independently generated scripts, 1 identical by construction), 0 different, 0 failed; the other 4 do not apply to the comparison (three inverse modes that do not exist and the parameter-sweep example). Records: [`export_check/`](export_check/README.md). An earlier run (2026-10-28) compared printed values and found four inverse problems differing in the sixth significant digit of a printed parameter; with full-precision records they are identical (the two scripts printed different numbers of digits). The check found two bugs, both fixed and checked again: the export left out IC-guided learning and backward compatibility (1D Cahn-Hilliard of Mattey & Ghosh), and with time windows it started the first window from different random weights than Solve (2D Allen-Cahn of Wight & Zhao, 2D Burgers of Mathias et al.).*

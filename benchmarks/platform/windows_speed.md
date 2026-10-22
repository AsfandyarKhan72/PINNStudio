# PINNStudio speed on Windows

Measured on one Windows laptop, October 2026, with PINNStudio from the `cleanup` branch installed the way
the one-command installer installs it (uv, Python 3.12, `--torch-backend auto`), in a test folder of its
own. Short runs, best of 2–3 repeats unless noted. The Linux numbers to compare with are in the README
("Speed: PINNStudio and a plain script"): RTX 4090 and Core i9-13900KF.

## The laptop

| | |
|---|---|
| Computer | Alienware m17 (2018) |
| Windows | Windows 11 Enterprise, build 26200, PowerShell 5.1 |
| CPU | Intel Core i9-8950HK, 6 cores / 12 threads (no efficiency cores), 2.9 GHz base |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 2080 with Max-Q Design, 8 GB, Turing (compute 7.5), 90 W limit, PCIe 3.0 |
| Driver | 546.29 (CUDA 12.3), WDDM mode, display on the Intel GPU (Optimus) |
| Disks | C: NVMe SSD (PINNStudio and its environment), D: hard disk |
| Power | plugged in; Balanced plan with power mode "Best performance" |
| Other | home folder `C:\Users\CMD LAB` (with a space); long paths off; Defender real-time protection on |
| Installed by the installer | Python 3.12.15, torch 2.9.0+cu129, DeepXDE 1.15.0, NumPy 2.4.6, PyQt6 6.6.1 |

**The installer picks the CUDA 12.9 build of PyTorch for a driver that reports CUDA 12.3**, and it works:
newer CUDA 12.x builds run on any 12.x driver (minor-version compatibility). Installing from a local copy
of `cleanup` took 7.5 minutes, most of it downloading torch (2.9 GB).

**This laptop's CPU runs hot.** Under training load the CPU sensor read 96 °C and the CPU stayed at its
2.9 GHz base clock (no turbo, "% Processor Performance" about 92 %). While the GPU trained, other programs
used about 3.5 of the 12 threads: Windows Search (indexing the new environment), the WMI host (polled by
Dell's Alienware software), AnyDesk, Task Manager. The absolute numbers below are therefore low for this
GPU; comparisons made on this laptop (threads, builds, settings) are still fair.

## GPU (benchmarks/gpu_check.py --quick)

Adam iterations a second, float32, 1,000 Adam + 250 L-BFGS, with the GPU recorded while it trained:

| Problem | Adam it/s | GPU busy | GPU clock | Power | L-BFGS (250 steps) | final L2 |
|---|---|---|---|---|---|---|
| Heat 1D (2k points) | 59 | 30 % | 470 MHz | 22 W | 43 s | 2.4e-4 |
| Poisson 2D (20k points) | 26 | 43 % | 1240 MHz | 59 W | 107 s | 6.7e-4 |
| Heat 2D (30k points) | 28 | 38 % | 790 MHz | 53 W | 131 s | 1.8e-3 |

The RTX 4090 runs Poisson 2D and Heat 2D at 139–162 it/s. Here the GPU is busy less than half the time
and never reaches its full clock: it waits for the CPU, which runs the Python side of every step and
launches the GPU's kernels (each launch costs more under Windows' WDDM driver model than on Linux). On this
laptop even the 2D problems are limited by the CPU, not by the GPU.

| Setting (Poisson 2D, Adam) | it/s | |
|---|---|---|
| float32 (PINNStudio's default) | 26 | |
| TF32 on | 28 | 1.07×: Turing has no TF32 hardware |
| float64 | 4 | 6× slower |

`torch.compile` does not work out of the box on Windows: it fails with `TritonMissing` (Triton has no
Windows build). Not worth pursuing for PINNStudio.

## CPU threads (benchmarks/platform/platform_bench.py threads)

Each count in a fresh process (`OMP_NUM_THREADS` and `torch.set_num_threads`), best of 3. PINNStudio's
Automatic rule, min(8, logical CPUs / 2), gives 6 here.

| Threads | Heat 1D (2k points) | Poisson 2D (20k points) | Heat 2D (30k points) |
|---|---|---|---|
| 1 | 49 | 1.0 | |
| 2 | 66 | | |
| 3 | 61 | | |
| 4 | **69** | | |
| **6 (Automatic)** | 62 | | |
| 8 | 65 | | |
| 10 | 59 | | |
| 12 (every thread) | 42 | | |

## Start-up of a run

Time to start a new Python process and import what a PINNStudio script imports (median of 5, the laptop
otherwise idle). Linux: about 4 s.

| | Defender as installed | Defender exclusion for the environment folder |
|---|---|---|
| `python -c pass` | 0.07 s | 0.05 s |
| `import torch` | 4.2 s | 2.4 s |
| `import torch` + start CUDA | 4.2 s | 3.2 s |
| `import torch, deepxde` | 7.9 s | 6.7 s |
| `import torch, deepxde, matplotlib.pyplot` | 8.7 s | 6.8 s |
| installing torch + DeepXDE from uv's cache into a new environment | 15.3 s | 7.5 s |
| first import in that brand-new environment | 16.3 s | 17.9 s |

Under load (the first measurement, while Windows Search was indexing) `import torch, deepxde` took 9.9 s.
DeepXDE's own import takes about 4 s on top of torch's.

## Output while training

Heat 1D on the GPU, 3,000 Adam steps, the script's output read line by line through a pipe as
`pinnstudio/core/runner.py` reads it:

| DeepXDE prints every | lines | it/s |
|---|---|---|
| 1000 steps | 25 | 53 |
| 100 steps | 52 | 55 |
| 10 steps | 321 | 52 |

Printing and reading the log costs nothing measurable.

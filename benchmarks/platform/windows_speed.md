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
| 1 | 49 | 1.0 | 1.3 |
| 2 | 66 | 1.3 | 1.7 |
| 3 | 61 | 1.4 | |
| 4 | **69** | **1.5** | **1.9** |
| **6 (Automatic)** | 62 | **1.5** | **1.9** |
| 8 | 65 | **1.5** | 1.8 |
| 10 | 59 | **1.5** | |
| 12 (every thread) | 42 | 1.3 | 1.1 |

(Heat 2D: best of 2. The 1-thread Heat 2D run is from the first sweep, the others from a second one.)

**The Automatic rule picks well on this CPU:** 6 threads is the fastest or within 10 % of it for every
size, and every thread (12) is 15–40 % slower. On this 6-core CPU without efficiency cores, the slowdown
from using every thread is far smaller than on the i9-13900KF (10–28×), but it is still there.

## GPU or CPU: where the GPU starts to pay off (platform_bench.py sizes)

Poisson 2D, a 4×64 network, Adam iterations a second, best of 2 (CPU with Automatic threads):

| Points | CPU | GPU | |
|---|---|---|---|
| 500 | 47 | 39 | CPU 1.2× faster |
| 2,000 | 37 | 47 | GPU 1.3× |
| 8,000 | 11 | 43 | GPU 3.8× |
| 32,000 | 2.1 | 34 | GPU 16× |
| 64,000 | 1.3 | 23 | GPU 18× |

The crossover is around 1,000 points. Up to 8,000 points the GPU's speed hardly changes (39–47 it/s): each
step costs the same fixed time to launch, whatever the size, and only from about 30,000 points does the
GPU's own work start to show. The same holds on Linux, with a much lower fixed cost (README: the 4090
runs these sizes at 139–162 it/s).

## PINNStudio against a plain script (benchmarks/timing_study.py)

The README's five examples, run three ways: the plain exported script, the script Solve runs (with
`python`, no window), and Solve in the PINNStudio window with the live loss plot. 2,000 Adam + 400 L-BFGS
steps on the GPU, 400 + 80 on the CPU, best of 2. Training time per 1,000 steps (the sum of DeepXDE's
"'train' took" lines over the steps actually taken):

| Example | GPU: script | Solve's script | window | CPU: script | Solve's script | window |
|---|---|---|---|---|---|---|
| Heat 1D | 22.8 s | 22.9 s | 22.8 s | 12.0 s | 12.4 s | 12.1 s |
| Burgers 1D | 25.2 s | 24.8 s | 24.7 s | 16.9 s | 17.4 s | 17.1 s |
| Navier-Stokes 2D (Kovasznay) | 53.6 s | 53.9 s | 53.2 s | 74.4 s | 73.5 s | 72.9 s |
| Allen-Cahn 1D (causal, MsFFN) | 69.6 s | 72.6 s | 72.5 s | 564 s | 554 s | 525 s |
| Diffusion 1D, inverse | 19.6 s | 19.2 s | 19.4 s | 6.8 s | 6.8 s | 6.7 s |

**PINNStudio trains as fast as the plain script on Windows too**, on the GPU and on the CPU; the
differences are within the spread between repeats. A run also spends 10–25 s outside training (start-up,
plots, saving): Heat 1D 68 s in all on the GPU for 55 s of training.

(The CPU columns were measured with `CUDA_VISIBLE_DEVICES="-1"`: with the study's `""` the CPU runs either
crashed or trained on the GPU, see the bug below.)

The same examples against the Linux workstation (plain script, per 1,000 steps):

| Example | Laptop GPU | Laptop CPU | Linux RTX 4090 | Linux i9-13900KF CPU |
|---|---|---|---|---|
| Heat 1D | 22.8 s | **12.0 s** | 2.10 s | 2.13 s |
| Burgers 1D | 25.2 s | **16.9 s** | 2.68 s | 3.62 s |
| Navier-Stokes 2D | **53.6 s** | 74.4 s | 5.78 s | 22.8 s |
| Allen-Cahn 1D | **69.6 s** | 564 s | 7.21 s | 90 s |
| Diffusion 1D, inverse | 19.6 s | **6.8 s** | 2.50 s | 1.14 s |

The laptop's GPU is about 10× slower than the 4090 on every example. Its CPU is 4–6× slower than the
i9 on the 1D examples and Allen-Cahn, 3× on Navier-Stokes. **On this laptop three of the five examples
train faster on the CPU**, the 1D ones with a small network. PINNStudio's default, the GPU when there is
one, is the slower choice for them here.

## The fixed cost of a step (platform_bench.py launch)

A tiny GPU operation (adding 1 to 16 numbers) and one training step of a 3×64 network on 64 points with
second derivatives and Adam, best of 3:

| | per tiny operation | per training step | steps a second |
|---|---|---|---|
| CPU (6 threads) | 9.2 µs | 5.4 ms | 187 |
| GPU | 29.2 µs | 11.1 ms | 90 |

Every GPU step pays this fixed cost before the GPU does any real work; for a small 1D problem it is most
of the step, which is why the GPU is flat at 40–60 it/s up to about 8,000 points and loses to the CPU
below about 1,000. (Linux numbers for the same test are requested in STATUS.md.)

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

Where the time goes (`python -X importtime -c "import torch, deepxde"`, one run, 7.5 s in all):

| Module | Time | |
|---|---|---|
| torch | 2.7 s | |
| deepxde | 4.7 s | |
| &nbsp;&nbsp;scikit-learn (imported by `deepxde/utils/external.py`) | 2.0 s | only used by DeepXDE's `standardize()` helper |
| &nbsp;&nbsp;`deepxde.backend.pytorch.tensor` | 1.2 s | starts CUDA (needed on the GPU anyway) |
| &nbsp;&nbsp;matplotlib.pyplot | 0.5 s | PINNStudio's scripts plot anyway |
| &nbsp;&nbsp;scipy.spatial.distance | 0.5 s | |

About 2 s of every run's start-up here is scikit-learn, which PINNStudio's scripts don't use.

## Output while training

Heat 1D on the GPU, 3,000 Adam steps, the script's output read line by line through a pipe as
`pinnstudio/core/runner.py` reads it:

| DeepXDE prints every | lines | it/s |
|---|---|---|
| 1000 steps | 25 | 53 |
| 100 steps | 52 | 55 |
| 10 steps | 321 | 52 |

Printing and reading the log costs nothing measurable.

## Bug found: "Train on: CPU" on Windows with an NVIDIA GPU

PINNStudio hides the GPU for *Train on: CPU* with `os.environ["CUDA_VISIBLE_DEVICES"] = ""`. On Windows an
empty variable counts as unset for CUDA, so the GPU stays visible, while PyTorch's Python side sees 0
devices:

| `CUDA_VISIBLE_DEVICES` | `cuda.is_available()` | `device_count()` | `get_device_properties(0)` |
|---|---|---|---|
| not set | True | 1 | ok |
| `""` | True | 0 | AssertionError: Invalid device id |
| `"-1"` | False | 0 | RuntimeError: no CUDA GPUs |

With *Train on: CPU*, Solve's script stops with "Invalid device id" before training, and the exported
script trains on the GPU. With `"-1"` both run on the CPU. Proposed in STATUS.md (a one-line change in
`codegen.py`, plus the tests and `timing_study.py` that use the same empty value).

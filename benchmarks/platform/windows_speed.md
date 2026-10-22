# PINNStudio speed on Windows

Measured on one Windows laptop, October 2026, with PINNStudio from the `cleanup` branch installed the way
the one-command installer installs it (uv, Python 3.12, `--torch-backend auto`), in a test folder of its
own. Short runs, best of 2–3 repeats unless noted. The Linux numbers to compare with are in the README
("Speed: PINNStudio and a plain script"): RTX 4090 and Core i9-13900KF.

## Summary

- **PINNStudio trains as fast as a plain DeepXDE script on Windows**, on the GPU and the CPU (Solve's
  script and the window within the spread of a plain script). Printing the log costs nothing.
- **Bug: *Train on: CPU* does not work on Windows with an NVIDIA GPU.** Solve's script crashes ("Invalid
  device id") and the exported script trains on the GPU. Cause: `CUDA_VISIBLE_DEVICES=""`; `"-1"` fixes it.
- **Every GPU run prints two warnings in the Training Log on Windows** (`expandable_segments not supported
  on this platform`, `PYTORCH_CUDA_ALLOC_CONF is deprecated`), from `codegen.py` line 14.
- **Start-up costs 7–9 s a run** (Linux about 4 s): torch 3–4 s, DeepXDE 4–5 s, of which 2 s is
  scikit-learn, which PINNStudio doesn't use. Windows Defender adds 1.5–2 s; an exclusion removes it.
- **The first start after installing takes 17–40 s**; installing with uv's `--compile-bytecode` makes it
  8.5 s, for 20–55 s more at install time.
- **The installer's PyTorch choice is right** (cu129 for driver 546.29). cu126 is the same speed; cu130 can't
  use this driver and would fall back to the CPU with only a warning.
- **CPU threads: the Automatic rule (6 here) is right**; every thread is 15–40 % slower.
- **TF32 off and float32 are right**; float64 is 6× slower on this GPU, `torch.compile` doesn't work on
  Windows (no Triton).
- **On this laptop the GPU only pays off above about 1,000 points**: each GPU step has a fixed cost of
  11 ms (CPU: 5 ms), so the 1D examples train 1.5–3× faster on the CPU. This laptop's CPU is also
  thermally limited (96 °C, no turbo) and busy with background programs, which slows the GPU too.

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

## PyTorch builds (same torch 2.9.0, different CUDA versions)

Each build in a fresh environment on D: (a hard disk, so start-up there is only compared between builds),
installed with `uv pip install --torch-backend <build>`. GPU: `gpu_check.py --no-cpu --iters 400`, Adam
it/s. Launch: `platform_bench.py launch`, best of 3.

| Build | GPU found | Heat 1D | Poisson 2D | Heat 2D | GPU step (launch test) | CPU step | `import torch, deepxde` |
|---|---|---|---|---|---|---|---|
| cu126 | yes | 62 | 30 | 30 | 12.2 ms | 5.5 ms | 8.0 s |
| **cu129** (the installer's choice) | yes | 54 | 29 | 30 | 14.5 ms | 6.2 ms | 9.2 s |
| cu130 | **no** | – | – | – | – | 5.4 ms | 8.1 s |
| cpu | – | – | – | – | – | 4.9 ms | 6.3 s |

- cu126 and cu129 are the same speed within this laptop's spread (the installer's own cu129 environment on
  C: measured 11.1 ms a GPU step). Nothing to gain from pinning another CUDA build.
- **cu130 cannot use this driver** (546.29, CUDA 12.3): `torch.cuda.is_available()` is False with only a
  UserWarning ("cudaGetDeviceCount() returned cudaErrorNotSupported"), so training would quietly run on
  the CPU. The installer's `--torch-backend auto` correctly chose cu129 instead; a plain `pip install
  torch` that picks a CUDA 13 build would hit this. `gpu_check.py` says so clearly.
- The CPU-only build starts 2–3 s faster (no CUDA libraries to load) and its CPU step is about 10 %
  faster. That is what the installer installs on a computer without an NVIDIA GPU.
- TF32 was 0.93× and 1.08× (no TF32 on Turing); float64 6.2–6.4× slower, as before.

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

## The first start after installing

uv installs packages without compiling them to bytecode, so the first start compiles every module (and
Windows Defender scans each new file once). Fresh environments on C: from uv's cache (no download), the
time to `import torch, deepxde, matplotlib.pyplot` (`jobs/010b_bytecode`):

| Install | install | first start | second | third |
|---|---|---|---|---|
| `uv pip install` (as the installer does now) | 6.7–10.5 s | **17.0–17.6 s** | 7.5–8.5 s | 7.7–8.9 s |
| `uv pip install --compile-bytecode` | 27.8–65.2 s | **8.4–8.8 s** | 7.3–7.5 s | 7.2–7.6 s |

The first time an environment was used after a reboot-cold cache, the first start took 37 s (and 34–41 s
on the hard disk D:). With `--compile-bytecode` the first Solve after installing or updating starts like
every later one.

## Warnings in every GPU run on Windows

`codegen.py` sets `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"`.
On Windows with torch 2.9 every GPU run then prints, into the Training Log:

    UserWarning: expandable_segments not supported on this platform
    Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead

The first is Windows only (expandable segments are a Linux feature); the second comes with torch 2.9 on
every platform. The TF32 deprecation warning torch 2.9 has does not appear in PINNStudio's scripts.

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

## Windows checks (benchmarks/platform/win_checks.py and the job runner)

| Check | Result |
|---|---|
| Python environment, TEMP, home and results all in `…\Ünï Ölçü 用户 é\…` | Solve, Error Analysis and the exported script run and save normally |
| Results folder of 210 characters, long paths off (Windows' default) | works; the longest file written is 259 characters |
| Results folder of 227 characters, long paths off | fails mid-run: `FileNotFoundError: [WinError 206] The filename or extension is too long`, then "❌ Error during training"; **now** (7ac37a6) a folder over 199 characters is refused before Solve with a clear message |
| Stop while training | the training process is gone in 0.1 s, Solve enabled again |
| Closing the window while training (answering Yes) | the training process is gone in 0.1 s |
| Ctrl+C in the terminal PINNStudio was started from | the window keeps running and prints a KeyboardInterrupt traceback in the terminal; the run goes on |
| PINNStudio's process ended outright while training (terminal closed, Task Manager, crash) | **before:** the training process kept running on its own; **now** (`pyexe.end_with_pinnstudio`, a job object with kill-on-close): it ends with PINNStudio |
| `pinnstudio --create-shortcut` / `--remove-shortcut` | Start menu and desktop icons made and removed; the desktop icon opens PINNStudio without a console window |
| The installer over PINNStudio 1.4.0 (uv tool, torch 2.5.1+cu124, NumPy 2.5.4) | replaced by `cleanup` with torch 2.9.0+cu129 and NumPy 2.4.6 in 60 s (bytecode compiled: 9,442 files in 23 s); settings and results kept; first start 7.9 s |
| The installer with an old pip copy (1.4.0) first on the PATH | installs, then says "Another PINNStudio (1.4.0) is installed, and typing pinnstudio starts it", with the command to remove it |
| Train on: CPU with the fix, Solve's script | runs on the CPU; the log now says "Training on the CPU (Settings -> GPU / Hardware -> Train on: CPU)" |
| TF32 off through `fp32_precision` (torch 2.9) | accepted; "TF32: False", no warning |
| The tip for small 1D problems on the GPU | shown |

The home folder here is `C:\Users\CMD LAB` (a space) throughout, and nothing failed because of it.

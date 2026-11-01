# Seeds, precision and hardware

## Random seed

**Settings › Random Seed...** sets whether runs are seeded and with which seed (on by default, seed 2026). With the seed
on, both Solve's script and the exported script call `dde.config.set_random_seed(seed)` before any random numbers are
drawn: the network initialization, the training points, Fourier feature matrices and anchor points. With the seed off,
every run draws new values.

!!! info "New in v1.6.0 (in development)"
    bc-PINN's stored points now follow the run's seed too (they used a fixed seed of 1234 in v1.5.0). The benchmark
    configurations keep 1234 explicitly, the value their recorded runs used.

A seed makes a run repeatable on the same hardware and software. It does not make runs bit-identical across machines:
GPU kernels are not run in deterministic mode (`cudnn.deterministic` is left off for speed), and different PyTorch,
CUDA or CPU instruction sets round differently. The [export-equivalence check](export-equivalence.md) therefore compares
Solve and the exported script on the same machine, on the CPU.

## Precision

Training runs in float32 by default. The precision is one setting, shown in two places: **Settings › Float
Precision...** and **Precision:** on the **Training** page, below the training phases; changing one changes the
other. float64 runs the whole training in double precision.

!!! info "New in v1.6.0 (in development)"
    In v1.5.0 the two were separate controls, and float64 in either ran the whole training in float64. The
    development version also writes the precision into the training log and `model_config.json`, and
    **Restore & visualize** rebuilds the model in that precision.

## Hardware

The benchmark runs were made on one NVIDIA RTX 4090 under Linux, unless a record says otherwise. PINNStudio trains on the
GPU when PyTorch finds one, and otherwise on the CPU (**Settings › GPU / Hardware...**: device, GPU index, memory
fraction, CPU threads). On a Mac it always trains on the CPU.

## Training time

Reported times are the training times recorded by the benchmark scripts. They depend on the GPU, the driver, the PyTorch and CUDA versions and other load on the machine, and are meant for
comparison within one study.

## Training speed

The speed measurements are recorded in [`benchmarks/performance.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/performance.md) (one desktop
PC: Intel Core i9-13900KF with 8 performance and 16 efficiency cores, NVIDIA RTX 4090, Ubuntu, Python 3.11,
PyTorch 2.6 with CUDA 12.4, DeepXDE 1.15; script `benchmarks/perf_study.py`) and, for Windows, in
[`benchmarks/platform/windows_speed.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/platform/windows_speed.md). What they show:

- **The window adds no measurable time.** **Solve** writes a DeepXDE script and runs it in a Python process of its
  own; the window only reads what the script prints. Five problems, each trained from the same settings as the
  exported script run from a terminal, as Solve's script run without the window, and with **Solve** in the window,
  took the same time within 5 %, in no consistent direction, as two runs of the same script do. Starting Python
  and importing PyTorch and DeepXDE adds a few seconds to every run (about 4 s on that PC), either way.
- **GPU or CPU.** On that PC the GPU was 3 times faster than the CPU for a 1D problem with a few thousand points,
  4 to 10 times in 2D and about 18 times in 3D. A very small problem (an ODE with a few hundred points) was faster
  on the CPU, because starting work on the GPU costs more than it saves: **Settings › GPU / Hardware...** ›
  **Train on:** **CPU**.
- **CPU threads.** On a processor with performance and efficiency cores, about one thread per performance core is
  fastest; past that the slow cores hold every step back, and using every hardware thread was many times slower.
  **CPU threads:** **Automatic** uses half the logical processors, at most 8. A number set there is used as it
  is; with **Automatic**, the environment variable `OMP_NUM_THREADS`, when set, decides instead. The error is the
  same with any thread count.
- **Mac.** PINNStudio trains on the CPU on a Mac. On an M1 MacBook Air, Apple's GPU (MPS) was half the speed of
  the CPU on a small 1D problem (and L-BFGS several times slower), about the same on 2D problems with 20,000 and
  30,000 points, and gave the same accuracy; it also has no float64. A Mac with more GPU cores may do better on
  large problems: `benchmarks/mac_gpu_check.py` measures it on your machine (see the Mac section of
  `performance.md`).
- **Windows.** On a Windows laptop (Core i9-8950HK, RTX 2080 Max-Q) PINNStudio trained as fast as the plain
  script, and the installer's PyTorch build and the automatic CPU threads were the fastest choices. Starting a run
  took longer than on Linux (7 to 9 s, most of it importing PyTorch and DeepXDE), and the GPU only paid off above
  about 1,000 points. [Training is slow](../help/troubleshooting.md#training-is-slow) has the practical tips
  (Windows Defender, power mode).

These are measurements on particular machines, not guarantees; on other hardware the cross-over between CPU and GPU
and the best thread count can differ. `benchmarks/gpu_check.py` checks an NVIDIA GPU's set-up and speed on Windows
or Linux.

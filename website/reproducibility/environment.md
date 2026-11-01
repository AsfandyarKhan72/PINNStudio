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

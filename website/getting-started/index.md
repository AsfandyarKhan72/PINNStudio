# Getting started

This section takes you from a computer without PINNStudio to a trained model that you have saved, reopened and
exported as a script. It assumes no experience with DeepXDE or PyTorch. Some familiarity with the differential
equation you want to solve helps, because PINNStudio asks you to write it down.

## Who PINNStudio is for

PINNStudio is a desktop application for researchers, engineers and students who want to set up, train and evaluate
physics-informed neural networks (PINNs) without writing the training code by hand. You describe the problem in the
window: equations, domain, initial and boundary conditions, network and training. PINNStudio writes a
[DeepXDE](https://github.com/lululxvi/deepxde) script from these settings, runs it and shows the results. The same
script can be exported and run without the window.

It is useful when you want to

- try a PINN on an ODE or PDE without writing a new script for each problem,
- reproduce one of the built-in examples (DeepXDE's documented examples and published PINN benchmarks) and then
  change it,
- estimate unknown coefficients of an equation from measured data (inverse problems),
- compare training settings and methods on the same problem, with the error against reference data.

PINNStudio helps you set up, train and evaluate a PINN. It does not guarantee that a PINN converges, or that its
result is accurate or physically valid, for an arbitrary problem. Check results against reference data or another
solver whenever you can (see [Error Analysis](../guide/error-analysis.md)).

## The path through this section

1. **[Installation](installation.md).** Install the current release with one command or with `pip`, or install
   the development version from source. The page also covers CPU-only and NVIDIA GPU installations of PyTorch.
2. **[First run](first-run.md).** Start PINNStudio, load an example from the **Examples** list, press **Solve**
   and follow the run in the **Training Log** and the **Results** tab.
3. **[Save, reopen and export](save-and-export.md).** Save the problem as a `.pinn.json` file, open it again later,
   and export the standalone Python script.
4. **[Reporting a problem](reporting-bugs.md).** What to include in a GitHub issue when something does not work.

After this section, the [User guide](../guide/index.md) explains every page of the model tree, and the
[Tutorials](../tutorials/index.md) work through complete problems, from a first ODE to inverse problems and
adaptive training. If something goes wrong on the way, see [Troubleshooting](../help/troubleshooting.md).

!!! info "Versions"
    The current release on PyPI is **v1.5.0**. This documentation describes the development version that will
    become **v1.6.0**. Features and fixes that exist only in the development version are marked
    *New in v1.6.0 (in development)*. The [Installation](installation.md) page explains how to install either.

## Demo video

A [demo video on YouTube](https://youtu.be/5ILxuutvEZw) walks through the window: loading an example, setting up a
PDE of your own, sketching a domain and its boundary conditions, training with the live loss, Error Analysis, the
run history and parameter sweeps. Some labels in the video may differ from the version these pages describe.

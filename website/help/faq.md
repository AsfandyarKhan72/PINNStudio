# FAQ

## General

**Do I need to know Python, DeepXDE or PyTorch?**
No. Problems are set up in the window, and equations are typed in a readable form such as
`du_t + u*du_x - 0.01/pi*du_xx`. Python helps if you want to change the [exported script](../guide/export.md) or
[contribute](../developer/contributing.md).

**What kinds of problems can PINNStudio solve?**
Forward and inverse problems for ODEs and for PDEs in one, two and three space dimensions, steady or
time-dependent, with several outputs and coupled equations, on intervals, rectangles, disks, polygons, cuboids,
spheres and shapes combined from them. See the [Feature reference](../reference/index.md) and the
[Examples](../examples/index.md).

**Does PINNStudio guarantee an accurate solution?**
No. PINNStudio helps set up, train and evaluate a PINN; it does not guarantee convergence, physical validity or
accuracy for an arbitrary problem. Check results against reference data or another method whenever possible
([Error Analysis](../guide/error-analysis.md), [Scope and limitations](../theory/limitations.md)).

**How many examples are there, and where do they come from?**
64: DeepXDE's documented examples, benchmark problems from published papers (Raissi et al., Wight & Zhao,
Mattey & Ghosh, Mathias et al.) and PINNStudio's own. 54 of them are in the **Forward** category of the
**Examples** list and on the **Start** tab; the other 10 learn unknowns from data and are only in the **Inverse**
category, which also offers inverse versions of many forward examples. See the [Examples](../examples/index.md).

**What is the difference between Default and Tuned settings?**
*Default* loads an example as its source solves it: a DeepXDE example with DeepXDE's settings, an example from a
paper with the paper's method. *Tuned* loads the settings that gave the lowest error in PINNStudio's benchmarks (and of
those about as accurate, the fastest). The list is new in v1.6.0 (in development). See
[First run](../getting-started/first-run.md#load-an-example).

**Can I solve my own equation, not only the examples?**
Yes. Start from **NEW** on the Start tab (or **Problem › New**), or load an example and change it. The
[Tutorials](../tutorials/index.md) show complete set-ups.

**Does PINNStudio handle physical units?**
No. All values are plain numbers. Write the equations in consistent units, or better in nondimensional form, so that
the terms of the residuals have comparable size; this also helps training.

## Installation and hardware

**Which systems and Python versions are supported?**
Linux, Windows and macOS with Python 3.11 to 3.14; the tested versions are listed under
[Supported systems](../getting-started/installation.md#supported-systems).

**Do I need a GPU?**
No. Every problem trains on the CPU. An NVIDIA GPU is faster for larger 2D and 3D problems and inverse runs; small
problems are often as fast or faster on the CPU (**Settings › GPU / Hardware...** › **Train on**).

**Does PINNStudio use the GPU of my Mac?**
No. On a Mac it trains on the CPU: in the project's tests on an M1, Apple's GPU (MPS) was not faster for PINNs, and it
has no float64.

**Can I use TensorFlow or JAX?**
No. PINNStudio uses DeepXDE with the PyTorch backend only, and its scripts set `DDE_BACKEND` to `pytorch`.

**Does PINNStudio need an internet connection?**
Only for installing and updating. When it starts, it checks PyPI in the background for a newer release and, if there
is one, shows a banner with the command that updates your copy; without a connection the check fails silently.

**How do I update PINNStudio?**
Run the one-command installer again, or `pip install --upgrade pinnstudio` in the environment you installed it in;
in the development version (v1.6.0), the update banner shows the right command for your installation. For a source installation, `git pull` and run the
source installer again. See [Installation](../getting-started/installation.md).

**Which version do I have?**
See [Reporting a problem](../getting-started/reporting-bugs.md#what-to-include-in-a-bug-report): PINNStudio has no
*About* dialog, but `pip show pinnstudio` and every saved `.pinn.json` file give the version.

## Working with problems and results

**Where are the results saved?**
Each run gets its own folder `<name>__<date>_<time>` inside the **Save to** folder of the **Results** page, by default
`PINNStudio_Results` in your home folder. The folder you chose last is remembered. See
[Results and plots](../guide/results.md).

**What is the difference between Save and Export script?**
**Save** writes a `.pinn.json` problem file that PINNStudio opens again; **Export script** writes a standalone
Python script that trains the same model without the window. See
[Save, reopen and export](../getting-started/save-and-export.md).

**Can I run a problem without the window, for example on a cluster?**
Yes: export the script and run it with a Python that has DeepXDE, PyTorch, NumPy, SciPy and Matplotlib. Adjust the
paths in the script to the other computer. See [Generated scripts](../reference/scripts.md).

**Can I look at a trained model again without retraining?**
Yes, under **Restore & visualize**: choose the model file of a run, and PINNStudio finds its configuration and plots
the solution again, with the error analysis if it is set up. See [Restore & visualize](../guide/restore.md).

**I trained without reference data. Can I compute the error afterwards?**
Yes. Set up the reference data (or an exact-solution formula) in **Error Analysis** on the **Results** page, then load
the trained model under **Restore & visualize**. **Run Error Analysis** in the dialog by itself only sets up the error
analysis for the next run.

**Are runs reproducible?**
By default every run uses the random seed 2026 (**Settings › Random Seed...**, **Fix random seed for reproducible
runs**), which fixes the network's initial weights and the sampled points. Results on the same computer and device
are then repeatable up to the nondeterminism of floating-point computation (mostly on GPUs); results on another
device or with other library versions can differ slightly. See
[Seeds, precision and hardware](../reproducibility/environment.md).

**Why is float32 the default precision?**
It is faster. float64 (**Settings › Float Precision...**) is recommended for L-BFGS, which can stop early in float32.

**A small loss, but the solution looks wrong. Why?**
A PINN can settle on a wrong solution with a small loss, especially for sharp fronts, phase fields and long time
ranges. Judge a result by its error against reference data, and see
[Training does not converge](troubleshooting.md#training-does-not-converge-or-converges-to-a-wrong-solution).

## Citing and licensing

**How do I cite PINNStudio?**
See [Citing PINNStudio](../about/cite.md). Please also cite DeepXDE and the methods and examples you used
([Scientific references](../about/references.md)).

**Under which license is PINNStudio distributed?**
See [License](../about/license.md). The licenses of the libraries PINNStudio uses are listed under
[Dependencies and attribution](../about/dependencies.md).

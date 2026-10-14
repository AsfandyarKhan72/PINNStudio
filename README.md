# PINNStudio

*A no-code scientific computing environment for solving forward and inverse problems with physics-informed neural networks — from setting up the equations to training, error analysis, and visualization.*

<p align="center">
  <a href="https://pypi.org/project/pinnstudio/"><img src="https://img.shields.io/pypi/v/pinnstudio.svg" alt="PyPI version"></a>
  <a href="https://pepy.tech/project/pinnstudio"><img src="https://static.pepy.tech/badge/pinnstudio" alt="PyPI downloads"></a>
  <a href="https://pypi.org/project/pinnstudio/"><img src="https://img.shields.io/pypi/pyversions/pinnstudio.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/AsfandyarKhan72/PINNStudio.svg" alt="License: MIT"></a>
  <a href="https://github.com/AsfandyarKhan72/PINNStudio/stargazers"><img src="https://img.shields.io/github/stars/AsfandyarKhan72/PINNStudio?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/AsfandyarKhan72/PINNStudio/commits/main"><img src="https://img.shields.io/github/commit-activity/m/AsfandyarKhan72/PINNStudio" alt="Commit activity"></a>
</p>

<p align="center">
  <em>One PDE, three dimensions — the same panel drives all of them, solved and visualized entirely within PINNStudio.</em>
</p>

<table>
<tr>
<td width="33%" align="center">
<img src="assets/Images/Top_GIFs/1D_Burger.gif" alt="PINNStudio — 1D Burgers PINN solution animated over time" width="100%">
<sub><strong>1D</strong> — <code>(x, t)</code> — <a href="https://doi.org/10.1016/j.jcp.2018.10.045">Burgers</a></sub>
</td>
<td width="33%" align="center">
<img src="assets/Images/Top_GIFs/2D_Allen_Cahn_Wight.gif" alt="PINNStudio — 2D Allen-Cahn (Wight & Zhao) PINN solution animated over time" width="100%">
<sub><strong>2D</strong> — <code>(x, y, t)</code> — <a href="https://doi.org/10.4208/cicp.OA-2020-0086">Allen-Cahn (Wight & Zhao)</a></sub>
</td>
<td width="33%" align="center">
<img src="assets/Images/Top_GIFs/3D_Heat.gif" alt="PINNStudio — 3D Heat PINN solution animated over time" width="100%">
<sub><strong>3D</strong> — <code>(x, y, z, t)</code> — Heat</sub>
</td>
</tr>
</table>

<p align="center">
  <strong><a href="https://asfandyarkhan72.github.io/PINNStudio/">Website</a></strong> &nbsp;·&nbsp;
  <strong><a href="#installation">Install</a></strong> &nbsp;·&nbsp;
  <strong><a href="#quick-start">Quick Start</a></strong> &nbsp;·&nbsp;
  <strong><a href="#examples">Examples</a></strong> &nbsp;·&nbsp;
  <strong><a href="#solving-your-own-problem">Your Own Problem</a></strong> &nbsp;·&nbsp;
  <strong><a href="https://github.com/AsfandyarKhan72/PINNStudio/discussions">Discussions</a></strong> &nbsp;·&nbsp;
  <strong><a href="#citation">Citation</a></strong>
</p>

<details>
<summary><strong>Table of contents</strong></summary>

- [Overview](#overview)
- [Demo Video](#demo-video)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Features](#features)
- [Solving Your Own Problem](#solving-your-own-problem)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [References](#references)
- [Acknowledgment](#acknowledgment)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

</details>

---

## Overview

**PINNStudio** is a scientific computing environment for solving differential equations with physics-informed neural networks (PINNs) and related scientific machine learning methods, without writing any code. You describe the problem — the equations, the domain, the boundary and initial conditions, the network and how it should be trained — and PINNStudio trains the model, tracks it while it runs, and gives back the solution, its error against reference data, plots and animations, and the trained model ready to reload.

Setting up a PINN by hand means writing a new script for every problem: the residual of the equations through automatic differentiation, the sampling of collocation points, the weighting of the loss terms, the optimizer schedule, and then the plotting and error analysis afterwards. Each of these is easy to get subtly wrong. PINNStudio puts all of it into one place, so researchers and students in science and engineering — including those new to scientific computing or machine learning — can spend their time on the physics of their problem rather than on the code around it.

**What it can solve**

- **Forward and inverse problems:** solve a known equation, or learn unknown coefficients or whole unknown fields from measured data.
- **ODEs and PDEs in 1D, 2D and 3D**, steady or time-dependent, including coupled multi-output systems, integro-differential and fractional equations, typed in a readable form such as `du_t + u*du_x - 0.01/pi*du_xx`.
- **Many kinds of domains** — intervals, rectangles, disks, polygons, cuboids, spheres, and custom shapes combined from them (e.g. a domain with a hole) — with Dirichlet, Neumann, Robin, periodic and point-data conditions, or conditions imposed exactly through the network output.

**How it trains**

- **Multi-phase training** (e.g. Adam followed by L-BFGS) with learning-rate schedules and loss weights per phase.
- **Adaptive methods:** residual-based adaptive refinement (RAR), which adds points where the error is largest; time-adaptive training, which marches through time in steps, each starting from the previous one (transfer learning); and initial-condition-guided pre-training.
- **Point resampling, early stopping, checkpointing and time budgets**, and live monitors of any output or derivative during training.
- **Parameter sweeps** over almost any setting — one at a time, all combinations, paired values, or Bayesian search — with a summary of how the error changes with each parameter.

**What you get back**

- A live training log, loss curves, and error analysis against reference data or an exact formula (relative $L_2$, MSE, maximum error).
- Line, surface and animated solution plots, solution data export, and a saved model that can be reloaded and visualized again without retraining.
- A standalone Python script for every problem, so the work can be inspected, shared and reproduced outside PINNStudio.

**Ready-to-run examples.** PINNStudio comes with 60 worked examples, from simple ODEs to the Navier–Stokes equations in 3D, including published PINN benchmarks (Raissi et al., Wight & Zhao, Mattey & Ghosh, and others) and DeepXDE's examples. Each one loads in one click, and 58 of them come with an exact solution or reference data to compare against — see [Examples](#examples).

PINNStudio is under active development, and we are extending it with newer scientific machine learning and PINN methods, aimed at solving harder problems more accurately and faster.

## Demo Video

[![PINNStudio demo - setting up a PDE](https://img.youtube.com/vi/lxSJTJvU45Q/maxresdefault.jpg)](https://youtu.be/lxSJTJvU45Q)

*Click to watch a full walkthrough of the PDE setup panel on YouTube.*

## Installation

**Quickest way to try PINNStudio** (any operating system):

```bash
pip install pinnstudio
pinnstudio
```

<details>
<summary><strong>Command not found after <code>pip install</code>?</strong></summary>

- **<code>pip: command not found</code>?** Use <code>pip3</code> instead - many systems (macOS especially) only ship <code>pip3</code>, not a plain <code>pip</code>.
- **<code>pinnstudio: command not found</code> even though the install said it succeeded?** pip installed it into a per-user folder that is not on your shell's PATH yet. The install output actually tells you the exact folder, in a line like <code>WARNING: The script pinnstudio is installed in '.../bin' which is not on PATH.</code> Add that folder to your PATH:

```bash
echo 'export PATH="<folder from the warning above>:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

(Use <code>~/.bashrc</code> instead of <code>~/.zshrc</code> if your shell is bash.) Then <code>pinnstudio</code> should launch directly. This never comes up with <code>install.sh</code>/<code>install.bat</code>, since those use a virtual environment where the command is always found automatically.

- **The GPU fix command PINNStudio suggests fails with a <code>flit_core</code> or "Could not find a version that satisfies" error?** Your pip is too old to resolve PyTorch's package index correctly. Upgrade it first, then retry:

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

</details>

**Have an NVIDIA GPU? Use the install script instead.** It creates its own environment and checks that the installed PyTorch can use your GPU, installing a compatible build if the default one cannot (for example with an older driver) — which plain `pip install` does not do.

<details>
<summary><strong>Before you start: open a terminal and check git and Python 3.9+</strong></summary>

- **Windows:** click the Start menu, type `PowerShell`, and open **Windows PowerShell**.
- **macOS:** press `Cmd + Space` to open Spotlight, type `Terminal`, and press Enter (or find it under Applications -> Utilities -> Terminal).
- **Linux:** open your terminal application (commonly `Ctrl + Alt + T`, or search "Terminal" in your application menu).

Then check git and Python by pasting these one at a time:

```bash
git --version
python3 --version
```

(On Windows, use `python --version` instead of `python3 --version`.)

If either command isn't recognized:

- **git missing?** Install it from [git-scm.com/downloads](https://git-scm.com/downloads). Default options are fine. On macOS, running `git --version` for the first time may itself prompt you to install Apple's Command Line Tools — accept and let it finish, then try again.
- **Python missing, or older than 3.9?** Install it from [python.org/downloads](https://www.python.org/downloads/). **On Windows, check "Add python.exe to PATH"** on the installer's first screen — this is the single most common thing people miss.

After installing either one, close your terminal window completely and open a new one before continuing, so the change takes effect.

</details>

```bash
git clone https://github.com/AsfandyarKhan72/PINNStudio.git
cd PINNStudio
```

**macOS / Linux:**
```bash
bash install.sh
./venv/bin/pinnstudio
```

**Windows:**
```
install.bat
.\venv\Scripts\pinnstudio.exe
```

Everything is installed into a virtual environment inside the `PINNStudio` folder — nothing system-wide, and deleting the folder removes it completely. The install needs an internet connection and takes a few minutes.

<details>
<summary><strong>Running PINNStudio again</strong></summary>

You only need to install once. After that, launch PINNStudio again anytime with:

**macOS / Linux**, from inside the `PINNStudio` folder:
```bash
./venv/bin/pinnstudio
```

**Windows**, from inside the `PINNStudio` folder:
```
.\venv\Scripts\pinnstudio.exe
```

That's it - no need to reinstall or recreate the virtual environment.

</details>

<details>
<summary><strong>What gets installed</strong></summary>

`install.sh` / `install.bat` set up an isolated Python virtual environment and install:

- [DeepXDE](https://github.com/lululxvi/deepxde) (PyTorch backend)
- PyTorch
- PyQt6
- NumPy (below 2.5 -- DeepXDE 1.15's Polygon/Triangle geometry breaks on NumPy 2.5+)
- Matplotlib
- Pandas

A CUDA-capable GPU is optional but recommended for larger 2D/3D problems and inverse runs.

</details>

**Something not working?** Open an issue on GitHub with the exact command you ran and the full error message — see [Contributing](#contributing).

## Quick Start

Your first result takes about a minute:

1. Start PINNStudio (`pinnstudio`, or `./venv/bin/pinnstudio` after the install script) and maximize the window — the left panel holds the problem setup.
2. In **Quick Examples**, keep the category on **Forward** and pick **Heat 1D – Dirichlet**. Every setting — equation, domain, conditions, network, training — fills in.
3. Press **Solve**. The Training Log shows the progress, and the loss curve and solution appear when the run finishes.
4. At the end of the run, the Training Log reports the relative $L_2$ error against the exact solution, and the comparison plots are saved with the results (**Error Analysis** sets which references and metrics are used).

From there:
- Pick the same example from the **Inverse** list to learn the unknown diffusivity from measured data instead.
- Browse the other examples below.
- Follow [Solving Your Own Problem](#solving-your-own-problem) to set up an equation of your own.

## Examples

PINNStudio comes with 60 ready-to-run examples, all listed in the table below and loaded from *Quick Examples*. Its **Category** is **Forward** (every example, solving for the field) or **Inverse** (examples that learn unknown parameters or fields from data), and each list is ordered like the table, under ODE / 1D / 2D / 3D headings. Every name follows one style, *Equation Dimension – detail*. The table's **Source** column says where each example comes from:

- **PINNStudio templates** (10): our own problems; most reproduce a published PINN result, and the Source column names and links that paper (the heat and the disk/sphere Poisson templates are our own and show "—"). Each one works in both **Forward** and **Inverse** mode — picked from the Inverse list, it loads in Inverse mode with its measured-data file — and all of them ship with reference data in [`reference_data/`](reference_data), so Error Analysis is ready as soon as you load one.
- **DeepXDE examples** (40 forward, 10 inverse): the **DeepXDE** link in the Source column opens each one's script in DeepXDE's repository. These are the PINN examples from DeepXDE's documentation ([forward](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward.html), [inverse](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse.html)), set up with each script's own equation, geometry, boundary/initial conditions, network, optimizers, iteration counts and loss weights, so pressing **Solve** trains what the DeepXDE script trains. Anything that differs from the script is printed in the Training Log when the example loads, and noted in its entry below.

Loading an example fills in every setting, and you can change any of them before pressing **Solve**; if you do, the Training Log lists what you changed. Your save folder, GPU, random-seed and plot-style settings are kept when you load one.

**Error Analysis reference:** where the problem has an exact solution, Error Analysis compares against it directly (*Exact solution*). Otherwise it uses reference files computed independently of the PINN (*Generated data*): PINNStudio's own reference solutions in [`reference_data/`](reference_data), or data computed by [`tools/make_deepxde_reference_data.py`](tools/make_deepxde_reference_data.py) (no data is copied from DeepXDE's repository). Examples marked "—" have no bundled reference; you can add your own files in the Error Analysis dialog.

Every result shown below comes from one training run with the example's settings on an RTX 4090. Errors are relative $L_2$ errors from Error Analysis, and they vary a little from run to run.

The table is ordered by dimension (ODE, 1D, 2D, 3D), then steady before time-dependent and forward before inverse; examples of the same problem sit next to each other.

| # | Example | Dimension | Regime | Problem | Geometry | Error Analysis reference | Source |
|---|---|---|---|---|---|---|---|
| | **ODE (t only)** | | | | | | |
| 1 | [Integro-differential ODE](#dx-integro-differential-equation) | ODE | Time-dependent | Forward | — (t only) | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/ide.py) |
| 2 | [Integro-differential ODE – Volterra kernel](#dx-volterra-integro-differential-equation) | ODE | Time-dependent | Forward | — (t only) | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py) |
| 3 | [Linear ODE system](#dx-ode-system) | ODE | Time-dependent | Forward | — (t only) | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/ode_system.py) |
| 4 | [Lotka-Volterra ODE](#dx-lotka-volterra-equation) | ODE | Time-dependent | Forward | — (t only) | Generated data | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Lotka_Volterra.py) |
| 5 | [Second-order ODE](#dx-second-order-ode) | ODE | Time-dependent | Forward | — (t only) | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/ode_2nd.py) |
| 6 | [Lorenz ODE – unknown C₁, C₂, C₃](#dx-inverse-lorenz-system) | ODE | Time-dependent | Inverse | — (t only) | Generated data | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/Lorenz_inverse.py) |
| 7 | [Lorenz ODE – unknown C₁, C₂, C₃, exogenous input](#dx-inverse-lorenz-system-with-exogenous-input) | ODE | Time-dependent | Inverse | — (t only) | Generated data | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/Lorenz_inverse_forced.py) |
| | **1D** | | | | | | |
| 8 | [Euler beam 1D](#dx-euler-beam) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Euler_beam.py) |
| 9 | [Fractional Poisson 1D](#dx-fractional-poisson-equation-1d) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_1d.py) |
| 10 | [Poisson 1D – Dirichlet](#dx-poisson-1d-dirichlet) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Dirichlet_1d.py) |
| 11 | [Poisson 1D – Dirichlet/Neumann](#dx-poisson-1d-dirichlet-neumann) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Neumann_1d.py) |
| 12 | [Poisson 1D – Dirichlet/Periodic](#dx-poisson-1d-dirichlet-periodic) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_periodic_1d.py) |
| 13 | [Poisson 1D – Dirichlet/point-set operator](#dx-poisson-1d-dirichlet-pointsetoperator) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_PointSetOperator_1d.py) |
| 14 | [Poisson 1D – Dirichlet/Robin](#dx-poisson-1d-dirichlet-robin) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Robin_1d.py) |
| 15 | [Poisson 1D – hard BCs](#dx-poisson-1d-hard-boundary-conditions) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Dirichlet_1d_exactBC.py) |
| 16 | [Poisson 1D – multi-scale Fourier features](#dx-poisson-1d-multi-scale-fourier-features) | 1D | Steady | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_multiscale_1d.py) |
| 17 | [Brinkman-Forchheimer 1D – unknown ν_e, K](#dx-inverse-brinkman-forchheimer-model) | 1D | Steady | Inverse | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/brinkman_forchheimer.py) |
| 18 | [Diffusion-reaction 1D – unknown rate field](#dx-inverse-diffusion-reaction-with-space-dependent-rate) | 1D | Steady | Inverse | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/diffusion_reaction_rate.py) |
| 19 | [Fractional Poisson 1D – unknown order α](#dx-inverse-fractional-poisson-equation-1d) | 1D | Steady | Inverse | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/fractional_Poisson_1d_inverse.py) |
| 20 | [Poisson 1D – unknown forcing field](#dx-inverse-poisson-with-unknown-forcing-field) | 1D | Steady | Inverse | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/elliptic_inverse_field.py) |
| 21 | [Poisson 1D – unknown forcing field (batches)](#dx-inverse-poisson-with-unknown-forcing-field-batches) | 1D | Steady | Inverse | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/elliptic_inverse_field_batch.py) |
| 22 | [Allen-Cahn 1D – Wight & Zhao](#1d-allen-cahn) | 1D | Time-dependent | Forward / Inverse | Interval | Generated data | [Wight & Zhao (2021)](https://doi.org/10.4208/cicp.OA-2020-0086) |
| 23 | [Allen-Cahn 1D](#dx-allen-cahn-equation) | 1D | Time-dependent | Forward | Interval | Generated data | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Allen_Cahn.py) |
| 24 | [Black-Scholes 1D – American put](#dx-american-put-option) | 1D | Time-dependent | Forward | Interval | — | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/american_put.py) |
| 25 | [Black-Scholes 1D – European call](#dx-black-scholes-equation-call-option) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/black_scholes_call.py) |
| 26 | [Burgers 1D – Raissi et al.](#1d-burgers) | 1D | Time-dependent | Forward / Inverse | Interval | Generated data | [Raissi et al. (2019)](https://doi.org/10.1016/j.jcp.2018.10.045) |
| 27 | [Burgers 1D – RAR](#dx-burgers-equation-with-rar) | 1D | Time-dependent | Forward | Interval | Generated data | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Burgers_RAR.py) |
| 28 | [Diffusion 1D](#dx-diffusion-equation) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d.py) |
| 29 | [Diffusion 1D – hard IC and BCs](#dx-diffusion-equation-with-hard-ic-and-bc) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d_exactBC.py) |
| 30 | [Diffusion 1D – resampling](#dx-diffusion-equation-with-resampling) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d_resample.py) |
| 31 | [Diffusion-reaction 1D](#dx-diffusion-reaction-equation) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_reaction.py) |
| 32 | [Fractional diffusion 1D](#dx-fractional-diffusion-equation) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_diffusion_1d.py) |
| 33 | [Heat 1D – Dirichlet](#dx-heat-equation) | 1D | Time-dependent | Forward / Inverse | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/heat.py) |
| 34 | [Heat 1D – resampling](#dx-heat-equation-with-resampling) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/heat_resample.py) |
| 35 | [Klein-Gordon 1D](#dx-klein-gordon-equation) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Klein_Gordon.py) |
| 36 | [Schrödinger 1D](#dx-schr-dinger-equation) | 1D | Time-dependent | Forward / Inverse | Interval | Generated data | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Schrodinger.ipynb) |
| 37 | [Wave 1D – space-time multi-scale Fourier features](#dx-wave-propagation-space-time-multi-scale-fourier-features) | 1D | Time-dependent | Forward | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/wave_1d.py) |
| 38 | [Diffusion 1D – unknown coefficient](#dx-inverse-diffusion-equation) | 1D | Time-dependent | Inverse | Interval | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/diffusion_1d_inverse.py) |
| 39 | [Diffusion-reaction 1D – unknown k_f, D](#dx-inverse-diffusion-reaction-system) | 1D | Time-dependent | Inverse | Interval | — | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/reaction_inverse.py) |
| | **2D** | | | | | | |
| 40 | [Fractional Poisson 2D – disk](#dx-fractional-poisson-equation-2d) | 2D | Steady | Forward | Disk | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_2d.py) |
| 41 | [Helmholtz 2D – Dirichlet](#dx-helmholtz-2d-dirichlet) | 2D | Steady | Forward | Rectangle | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d.py) |
| 42 | [Helmholtz 2D – hyperparameter search](#dx-helmholtz-2d-hyperparameter-optimization) | 2D | Steady | Forward | Rectangle | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d_HPO.py) |
| 43 | [Helmholtz 2D – sound-hard scattering](#dx-helmholtz-sound-hard-scattering-absorbing-bc) | 2D | Steady | Forward | Rectangle − Disk | Generated data | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Sound_hard_ABC_2d.py) |
| 44 | [Helmholtz 2D – square with hole](#dx-helmholtz-2d-with-a-hole) | 2D | Steady | Forward | Rectangle − Disk | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Neumann_2d_hole.py) |
| 45 | [Laplace 2D – disk](#dx-laplace-equation-on-a-disk) | 2D | Steady | Forward | Rectangle in (r, θ) | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Laplace_disk.py) |
| 46 | [Linear elasticity 2D – plate](#dx-linear-elastostatic-plate) | 2D | Steady | Forward | Rectangle | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/elasticity_plate.py) |
| 47 | [Linear elasticity 2D – plate, soft BCs](#dx-linear-elastostatic-plate-soft-bcs) | 2D | Steady | Forward | Rectangle | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/elasticity_plate.py) |
| 48 | [Navier-Stokes 2D – Kovasznay flow](#dx-kovasznay-flow) | 2D | Steady | Forward | Rectangle | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Kovasznay_flow.py) |
| 49 | [Poisson 2D – disk](#2d-poisson-disk) | 2D | Steady | Forward / Inverse | Disk | Generated data | — |
| 50 | [Poisson 2D – L-shape](#dx-poisson-over-an-l-shaped-domain) | 2D | Steady | Forward | Polygon (L-shape) | Generated data | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Lshape.py) |
| 51 | [Poisson 2D – L-shape, Lu et al.](#2d-poisson-l-shape) | 2D | Steady | Forward / Inverse | Polygon (L-shape) | Generated data | [Lu et al. (2021)](https://doi.org/10.1137/19M1274067) |
| 52 | [Fractional Poisson 2D – unknown order α](#dx-inverse-fractional-poisson-equation-2d) | 2D | Steady | Inverse | Disk | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/fractional_Poisson_2d_inverse.py) |
| 53 | [Allen-Cahn 2D – Mattey & Ghosh](#2d-allen-cahn-mattey--ghosh) | 2D | Time-dependent | Forward / Inverse | Rectangle | Generated data | [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474) |
| 54 | [Allen-Cahn 2D – Wight & Zhao](#2d-allen-cahn-wight--zhao) | 2D | Time-dependent | Forward / Inverse | Rectangle | Generated data | [Wight & Zhao (2021)](https://doi.org/10.4208/cicp.OA-2020-0086) |
| 55 | [Burgers 2D – Mathias et al.](#2d-burgers-mathias) | 2D | Time-dependent | Forward / Inverse | Rectangle | Generated data | [Mathias et al. (2022)](https://arxiv.org/abs/2301.07824) |
| 56 | [Heat 2D](#2d-heat) | 2D | Time-dependent | Forward / Inverse | Rectangle | Generated data | — |
| | **3D** | | | | | | |
| 57 | [Fractional Poisson 3D – sphere](#dx-fractional-poisson-equation-3d) | 3D | Steady | Forward | Sphere | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_3d.py) |
| 58 | [Poisson 3D – sphere](#3d-poisson-sphere) | 3D | Steady | Forward / Inverse | Sphere | Generated data | — |
| 59 | [Heat 3D](#3d-heat) | 3D | Time-dependent | Forward / Inverse | Cuboid | Generated data | — |
| 60 | [Navier-Stokes 3D – Beltrami flow](#dx-beltrami-flow) | 3D | Time-dependent | Forward | Cuboid | Exact solution | [DeepXDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Beltrami_flow.py) |

The "unknown field" inverse examples learn the field as an extra network output from measurements given as a Point Set boundary condition, so they run in Forward mode, as in DeepXDE. To train the DeepXDE examples from the command line and print their Error Analysis results:

```bash
python tools/run_deepxde_examples.py --out deepxde_runs            # full settings
python tools/run_deepxde_examples.py --smoke --out deepxde_runs    # a few hundred iterations each
```


### ODE examples

*Click an example to expand it.*

<details id="dx-integro-differential-equation">
<summary><strong>Integro-differential ODE</strong></summary>

DeepXDE example: [`ide.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/ide.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/ide.html))

$$\Large \frac{du}{dt} + \int_0^t u(s)\ ds = 2\pi\cos(2\pi t) + \frac{\sin^2(\pi t)}{\pi}, \qquad t \in [0, 1]$$

- **Problem dimension:** ODE (t) — `du_t + integral(u) - (2*pi*cos(2*pi*t) + sin(pi*t)**2/pi)`
- **Initial condition:** $u(0) = 0$
- **Integral:** 16-point Gauss-Legendre quadrature of the network (*∫ points* 16, DeepXDE's `quad_deg`)
- **Exact solution:** $u = \sin(2\pi t)$
- **Network:** FNN [2, 20, 20, 20, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 16 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Integro-differential ODE**

**Result (RTX 4090):** $L_2$ = 1.5 × 10⁻⁴

*PINN vs reference along a line*

![Integro-differential ODE — PINN vs reference along a line](assets/Images/DeepXDE/Integro_differential_equation/line_comparison.png)

</details>

<details id="dx-volterra-integro-differential-equation">
<summary><strong>Integro-differential ODE – Volterra kernel</strong></summary>

DeepXDE example: [`Volterra_IDE.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py)

$$\Large \frac{du}{dt} + u(t) = \int_0^t e^{s - t}\ u(s)\ ds, \qquad t \in [0, 5]$$

- **Problem dimension:** ODE (t) — `du_t + u - integral(exp(s - t)*u)`, the kernel and $u$ written in the integration variable `s`
- **Initial condition:** $u(0) = 1$
- **Integral:** 20-point Gauss-Legendre quadrature; 10 uniformly spaced collocation points; L-BFGS only, as in DeepXDE's script
- **Exact solution:** $u = e^{-t}\cosh t$
- **Network:** FNN [2, 20, 20, 20, 1], tanh, Glorot uniform
- **Training:** L-BFGS 15,000; 10 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Integro-differential ODE – Volterra kernel**

**Result (RTX 4090):** $L_2$ = 2.5 × 10⁻³

*PINN vs reference along a line*

![Integro-differential ODE – Volterra kernel — PINN vs reference along a line](assets/Images/DeepXDE/Volterra_integro_differential_equation/line_comparison.png)

</details>

<details id="dx-ode-system">
<summary><strong>Linear ODE system</strong></summary>

DeepXDE example: [`ode_system.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/ode_system.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/ode.system.html))

$$\Large \frac{dy_1}{dt} = y_2, \qquad \frac{dy_2}{dt} = -y_1, \qquad t \in [0, 10]$$

- **Problem dimension:** ODE (t) — derivatives written `dy1_t`, `dy2_t`
- **Initial conditions:** $y_1(0) = 0$, $y_2(0) = 1$
- **Exact solution:** $y_1 = \sin t$, $y_2 = \cos t$
- **Network:** FNN [2, 50, 50, 50, 2], tanh, Glorot uniform
- **Training:** Adam 20,000; 35 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Linear ODE system**

**Result (RTX 4090):** $L_2$ = 3.6 × 10⁻³ ($y_1$), 3.1 × 10⁻³ ($y_2$)

*y1 — PINN vs reference along a line*

![Linear ODE system — PINN vs reference along a line (y1)](assets/Images/DeepXDE/ODE_system/line_comparison_y1.png)

*y2 — PINN vs reference along a line*

![Linear ODE system — PINN vs reference along a line (y2)](assets/Images/DeepXDE/ODE_system/line_comparison_y2.png)

</details>

<details id="dx-lotka-volterra-equation">
<summary><strong>Lotka-Volterra ODE</strong></summary>

DeepXDE example: [`Lotka_Volterra.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Lotka_Volterra.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Lotka.Volterra.html))

$$\Large \frac{dr}{dt} = \frac{R}{U}\left(2Ur - 0.04U^2 rp\right), \qquad \frac{dp}{dt} = \frac{R}{U}\left(0.02U^2 rp - 1.06Up\right), \qquad t \in [0, 1]$$

- **Problem dimension:** ODE (t); populations scaled by $U = 200$ and time by $R = 20$ (constants `ub`, `rb`)
- **Initial conditions:** $r(0) = 100/U$, $p(0) = 15/U$, imposed exactly by the output transform $\hat r = r\tanh t + 100/U$, $\hat p = p\tanh t + 15/U$
- **Input features:** $(t, \sin t, \sin 2t, \ldots, \sin 6t)$ through an input-transform formula
- **Reference:** SciPy `solve_ivp` solution of the same system
- **Constants:** `ub = 200`, `rb = 20`
- **Network:** FNN [2, 64, 64, 64, 64, 64, 64, 2], tanh, Glorot normal
- **Training:** Adam 50,000 → L-BFGS 15,000; 3,000 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Lotka-Volterra ODE**

**Result (RTX 4090):** $L_2$ = 2.1 × 10⁻³ ($r$), 1.4 × 10⁻³ ($p$)

*r — PINN vs reference along a line*

![Lotka-Volterra ODE — PINN vs reference along a line (r)](assets/Images/DeepXDE/Lotka_Volterra_equation/line_comparison_r.png)

*p — PINN vs reference along a line*

![Lotka-Volterra ODE — PINN vs reference along a line (p)](assets/Images/DeepXDE/Lotka_Volterra_equation/line_comparison_p.png)

</details>

<details id="dx-second-order-ode">
<summary><strong>Second-order ODE</strong></summary>

DeepXDE example: [`ode_2nd.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/ode_2nd.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/ode.2nd.html))

$$\Large \frac{d^2u}{dt^2} - 10\frac{du}{dt} + 9u = 5t, \qquad t \in [0, 0.25]$$

- **Problem dimension:** ODE (t) — `du_tt - 10*du_t + 9*u - 5*t`
- **Initial conditions:** $u(0) = -1$ (initial condition); $u'(0) = 2$ as an Operator BC `du_t - 2` applied at the initial time
- **Loss weights:** 0.01 for the equation, 1 for both initial conditions, as in DeepXDE's script
- **Exact solution:** $u = \frac{50}{81} + \frac{5}{9}t - 2e^{t} + \frac{31}{81}e^{9t}$
- **Network:** FNN [2, 50, 50, 50, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 16 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Second-order ODE**

**Result (RTX 4090):** $L_2$ = 3.2 × 10⁻⁵

*PINN vs reference along a line*

![Second-order ODE — PINN vs reference along a line](assets/Images/DeepXDE/Second_order_ODE/line_comparison.png)

</details>

<details id="dx-inverse-lorenz-system">
<summary><strong>Lorenz ODE – unknown C₁, C₂, C₃</strong></summary>

DeepXDE example: [`Lorenz_inverse.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/Lorenz_inverse.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/Lorenz.inverse.html))

$$\Large \frac{dy_1}{dt} = C_1(y_2 - y_1), \qquad \frac{dy_2}{dt} = y_1(C_2 - y_3) - y_2, \qquad \frac{dy_3}{dt} = y_1y_2 - C_3y_3, \qquad t \in [0, 3]$$

- **Problem dimension:** ODE (t); initial conditions $y(0) = (-8, 7, 27)$
- **Unknowns:** $C_1, C_2, C_3$ (true $10, 15, 8/3$), all starting from 1
- **Data:** $y_1, y_2, y_3$ measured at 50 times (also used as anchor points), from SciPy's `solve_ivp`
- **Network:** FNN [2, 40, 40, 40, 3], tanh, Glorot uniform
- **Training:** Adam 20,000 → L-BFGS 15,000; 400 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Lorenz ODE – unknown C₁, C₂, C₃**

**Result (RTX 4090):** $C_1$ = 9.9990, $C_2$ = 14.9997, $C_3$ = 2.66675 (true 10, 15, 8/3); $L_2$ = 1.2 × 10⁻⁴ ($y_1$), 1.2 × 10⁻⁴ ($y_2$), 6.7 × 10⁻⁵ ($y_3$)

*y1 — PINN vs reference along a line*

![Lorenz ODE – unknown C₁, C₂, C₃ — PINN vs reference along a line (y1)](assets/Images/DeepXDE/Inverse_Lorenz_system/line_comparison_y1.png)

*y3 — PINN vs reference along a line*

![Lorenz ODE – unknown C₁, C₂, C₃ — PINN vs reference along a line (y3)](assets/Images/DeepXDE/Inverse_Lorenz_system/line_comparison_y3.png)

*y1 — inferred parameters during training*

![Lorenz ODE – unknown C₁, C₂, C₃ — inferred parameters during training (y1)](assets/Images/DeepXDE/Inverse_Lorenz_system/param_plot.png)

</details>

<details id="dx-inverse-lorenz-system-with-exogenous-input">
<summary><strong>Lorenz ODE – unknown C₁, C₂, C₃, exogenous input</strong></summary>

DeepXDE example: [`Lorenz_inverse_forced.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/Lorenz_inverse_forced.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/Lorenz.inverse.forced.html))

$$\Large \frac{dy_1}{dt} = C_1(y_2 - y_1), \qquad \frac{dy_2}{dt} = y_1(C_2 - y_3) - y_2, \qquad \frac{dy_3}{dt} = y_1y_2 - C_3y_3 + 10\sin(2\pi t)$$

- **Problem dimension:** ODE (t), $t \in [0, 3]$; initial conditions $y(0) = (-8, 7, 27)$
- **Exogenous input:** $f(t) = 10\sin(2\pi t)$, written straight into the equation
- **Unknowns:** $C_1, C_2, C_3$ (true $10, 15, 8/3$), all starting from 1
- **Data:** $y_1, y_2, y_3$ measured at 200 times, from SciPy's `solve_ivp`
- **Network:** FNN [2, 40, 40, 40, 3], tanh, Glorot uniform
- **Training:** Adam 60,000; 400 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Lorenz ODE – unknown C₁, C₂, C₃, exogenous input**

**Result (RTX 4090):** $C_1$ = 9.987, $C_2$ = 14.998, $C_3$ = 2.6669 (true 10, 15, 8/3); $L_2$ = 7.3 × 10⁻⁴ ($y_1$), 6.6 × 10⁻⁴ ($y_2$), 2.9 × 10⁻⁴ ($y_3$)

*y1 — PINN vs reference along a line*

![Lorenz ODE – unknown C₁, C₂, C₃, exogenous input — PINN vs reference along a line (y1)](assets/Images/DeepXDE/Inverse_Lorenz_system_with_exogenous_input/line_comparison_y1.png)

*y3 — PINN vs reference along a line*

![Lorenz ODE – unknown C₁, C₂, C₃, exogenous input — PINN vs reference along a line (y3)](assets/Images/DeepXDE/Inverse_Lorenz_system_with_exogenous_input/line_comparison_y3.png)

*y1 — inferred parameters during training*

![Lorenz ODE – unknown C₁, C₂, C₃, exogenous input — inferred parameters during training (y1)](assets/Images/DeepXDE/Inverse_Lorenz_system_with_exogenous_input/param_plot.png)

</details>

### 1D examples

*Click an example to expand it.*

<details id="dx-euler-beam">
<summary><strong>Euler beam 1D</strong></summary>

DeepXDE example: [`Euler_beam.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Euler_beam.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/eulerbeam.html))

$$\Large \frac{d^4u}{dx^4} = -1, \qquad x \in [0, 1]$$

- **Boundary conditions:** $u(0) = 0$, $u'(0) = 0$ (clamped); $u''(1) = 0$, $u'''(1) = 0$ (free) — the last two as Operator BCs
- **Exact solution:** $u = -\frac{x^4}{24} + \frac{x^3}{6} - \frac{x^2}{4}$
- **Network:** FNN [1, 20, 20, 20, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 10 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Euler beam 1D**

**Result (RTX 4090):** $L_2$ = 8.4 × 10⁻³

*PINN vs reference along a line*

![Euler beam 1D — PINN vs reference along a line](assets/Images/DeepXDE/Euler_beam/line_comparison.png)

</details>

<details id="dx-fractional-poisson-equation-1d">
<summary><strong>Fractional Poisson 1D</strong></summary>

DeepXDE example: [`fractional_Poisson_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_1d.py)

$$\Large D_{0+}^{\alpha}u + D_{1-}^{\alpha}u = f(x), \qquad x \in [0, 1],\ \alpha = 1.5$$

- **Equation in the panel:** `frac(u) - (...)`, with $f$ written out with `gamma(...)` as in DeepXDE's script; constant `alpha`
- **Fractional settings:** α `alpha`, resolution 101, static mesh (DeepXDE's FPDE, Grünwald-Letnikov)
- **Boundary conditions:** $u(0) = u(1) = 0$, imposed exactly by the output transform $\hat u = x(1-x)N$
- **Exact solution:** $u = x^3(1-x)^3$
- **Constants:** `alpha = 1.5`
- **Network:** FNN [1, 20, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 10,000; 101 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Fractional Poisson 1D**

**Result (RTX 4090):** $L_2$ = 3.7 × 10⁻²

*PINN vs reference along a line*

![Fractional Poisson 1D — PINN vs reference along a line](assets/Images/DeepXDE/Fractional_Poisson_equation_1D/line_comparison.png)

</details>

<details id="dx-poisson-1d-dirichlet">
<summary><strong>Poisson 1D – Dirichlet</strong></summary>

DeepXDE example: [`Poisson_Dirichlet_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Dirichlet_1d.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.dirichlet.html))

$$\Large -\frac{d^2u}{dx^2} = \pi^2\sin(\pi x), \qquad x \in [-1, 1]$$

- **Boundary conditions:** Dirichlet, $u(\pm 1) = 0$ — in the panel: Where `True` (whole boundary), Value `0`
- **Exact solution:** $u = \sin(\pi x)$
- **Network:** FNN [1, 50, 50, 50, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 16 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Poisson 1D – Dirichlet**

**Result (RTX 4090):** $L_2$ = 3.7 × 10⁻⁴

*PINN vs reference along a line*

![Poisson 1D – Dirichlet — PINN vs reference along a line](assets/Images/DeepXDE/Poisson_1D_Dirichlet/line_comparison.png)

</details>

<details id="dx-poisson-1d-dirichlet-neumann">
<summary><strong>Poisson 1D – Dirichlet/Neumann</strong></summary>

DeepXDE example: [`Poisson_Neumann_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Neumann_1d.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.neumanndirichlet.html))

$$\Large \frac{d^2u}{dx^2} = 2, \qquad x \in [-1, 1]$$

- **Boundary conditions:** Dirichlet $u(-1) = 0$ (Where `left`, Value `0`); Neumann $u'(1) = 4$ (Where `right`, Value `4` — at the right end the outward normal derivative is $du/dx$)
- **Exact solution:** $u = (x + 1)^2$
- **Network:** FNN [1, 50, 50, 50, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 16 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Poisson 1D – Dirichlet/Neumann**

**Result (RTX 4090):** $L_2$ = 2.3 × 10⁻⁴

*PINN vs reference along a line*

![Poisson 1D – Dirichlet/Neumann — PINN vs reference along a line](assets/Images/DeepXDE/Poisson_1D_Dirichlet_Neumann/line_comparison.png)

</details>

<details id="dx-poisson-1d-dirichlet-periodic">
<summary><strong>Poisson 1D – Dirichlet/Periodic</strong></summary>

DeepXDE example: [`Poisson_periodic_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_periodic_1d.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.dirichletperiodic.html))

$$\Large -\frac{d^2u}{dx^2} = \pi^2\sin(\pi x), \qquad x \in [-1, 1]$$

- **Boundary conditions:** Dirichlet $u(-1) = 0$ (Where `left`); periodic $u(-1) = u(1)$ (Where `right`)
- **Exact solution:** $u = \sin(\pi x)$
- **Network:** FNN [1, 50, 50, 50, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 16 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Poisson 1D – Dirichlet/Periodic**

**Result (RTX 4090):** $L_2$ = 1.2 × 10⁻²

*PINN vs reference along a line*

![Poisson 1D – Dirichlet/Periodic — PINN vs reference along a line](assets/Images/DeepXDE/Poisson_1D_Dirichlet_Periodic/line_comparison.png)

</details>

<details id="dx-poisson-1d-dirichlet-pointsetoperator">
<summary><strong>Poisson 1D – Dirichlet/point-set operator</strong></summary>

DeepXDE example: [`Poisson_PointSetOperator_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_PointSetOperator_1d.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.pointsetoperator.html))

$$\Large \frac{d^2u}{dx^2} = 2, \qquad x \in [-1, 1]$$

- **Boundary conditions:** Dirichlet $u(-1) = 0$ (Where `left`); point-set operator condition $u'(1) = 4$ — a Point Set Operator row with the point typed in (Points `1`, Function `du_x`, Target value `4`; no file)
- **Exact solution:** $u = (x + 1)^2$
- **Network:** FNN [1, 50, 50, 50, 1], tanh, Glorot uniform
- **Training:** Adam 20,000; 16 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Poisson 1D – Dirichlet/point-set operator**

**Result (RTX 4090):** $L_2$ = 2.6 × 10⁻⁵

*PINN vs reference along a line*

![Poisson 1D – Dirichlet/point-set operator — PINN vs reference along a line](assets/Images/DeepXDE/Poisson_1D_Dirichlet_PointSetOperator/line_comparison.png)

</details>

<details id="dx-poisson-1d-dirichlet-robin">
<summary><strong>Poisson 1D – Dirichlet/Robin</strong></summary>

DeepXDE example: [`Poisson_Robin_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Robin_1d.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.dirichletrobin.html))

$$\Large \frac{d^2u}{dx^2} = 2, \qquad x \in [-1, 1]$$

- **Boundary conditions:** Dirichlet $u(-1) = 0$ (Where `left`); Robin $u'(1) = u(1)$ (Where `right`, Value `u`)
- **Exact solution:** $u = (x + 1)^2$
- **Network:** FNN [1, 50, 50, 50, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 16 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Poisson 1D – Dirichlet/Robin**

**Result (RTX 4090):** $L_2$ = 2.4 × 10⁻⁴

*PINN vs reference along a line*

![Poisson 1D – Dirichlet/Robin — PINN vs reference along a line](assets/Images/DeepXDE/Poisson_1D_Dirichlet_Robin/line_comparison.png)

</details>

<details id="dx-poisson-1d-hard-boundary-conditions">
<summary><strong>Poisson 1D – hard BCs</strong></summary>

DeepXDE example: [`Poisson_Dirichlet_1d_exactBC.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Dirichlet_1d_exactBC.py)

$$\Large -\frac{d^2u}{dx^2} = \sum_{k=1}^{4} k\sin(kx) + 8\sin(8x), \qquad x \in [0, \pi]$$

- **Boundary conditions:** $u(0) = 0$, $u(\pi) = \pi$, imposed exactly by the output transform $\hat u = x(\pi - x)\ N(x) + x$
- **Exact solution:** $u = x + \sum_{k=1}^{4}\frac{\sin(kx)}{k} + \frac{\sin(8x)}{8}$
- **Learning rate:** inverse-time decay, $10^{-4}/(1 + 0.3\ n/1000)$ at iteration $n$
- **Network:** FNN [1, 50, 50, 50, 1], tanh, Glorot uniform
- **Training:** Adam 30,000; 64 domain points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Poisson 1D – hard BCs**

**Result (RTX 4090):** $L_2$ = 2.2 × 10⁻⁶

*PINN vs reference along a line*

![Poisson 1D – hard BCs — PINN vs reference along a line](assets/Images/DeepXDE/Poisson_1D_hard_boundary_conditions/line_comparison.png)

</details>

<details id="dx-poisson-1d-multi-scale-fourier-features">
<summary><strong>Poisson 1D – multi-scale Fourier features</strong></summary>

DeepXDE example: [`Poisson_multiscale_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_multiscale_1d.py)

$$\Large \frac{d^2u}{dx^2} = -(\pi A)^2\sin(\pi A x) - 0.1(\pi B)^2\sin(\pi B x), \qquad x \in [0, 1],\ A = 2,\ B = 50$$

- **Boundary conditions:** $u(0) = u(1) = 0$
- **Exact solution:** $u = \sin(2\pi x) + 0.1\sin(50\pi x)$ — a slow wave plus a fast, small one
- **Network:** a multi-scale Fourier feature network (MsFFN): the input goes through random Fourier features with scales $\sigma = 1$ and $10$, each through the same hidden layers, then combined
- **Points:** 1,280 pseudo-random collocation points, redrawn every iteration; learning rate $10^{-3}/(1 + 0.9\ n/2000)$
- **Constants:** `A = 2`, `B = 50`
- **Network:** MsFFN [1, 100, 100, 100, 1], Fourier scales σ = 1, 10, tanh, Glorot uniform
- **Training:** Adam 20,000; 1,280 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Poisson 1D – multi-scale Fourier features**

A plain network of the same size learns the slow wave and largely misses the fast one; the Fourier features at scale 10 are what let the PINN represent the sin(50πx) part.

**Result (RTX 4090):** $L_2$ = 3.9 × 10⁻³

*PINN vs reference along a line*

![Poisson 1D – multi-scale Fourier features — PINN vs reference along a line](assets/Images/DeepXDE/Poisson_1D_multi_scale_Fourier_features/line_comparison.png)

</details>

<details id="dx-inverse-brinkman-forchheimer-model">
<summary><strong>Brinkman-Forchheimer 1D – unknown ν_e, K</strong></summary>

DeepXDE example: [`brinkman_forchheimer.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/brinkman_forchheimer.py)

$$\Large -\frac{\nu_e}{\epsilon}\frac{d^2u}{dx^2} + \frac{\nu u}{K} = g, \qquad x \in [0, 1],\ \epsilon = 0.4,\ \nu = 10^{-3},\ g = 1$$

- **Unknowns:** effective viscosity $\nu_e$ and permeability $K$ (both $10^{-3}$, initial guesses 0.1)
- **Data:** 5 measurements of $u$; $u = 0$ at both walls through the output transform $\hat u = x(1-x)N$
- **Exact solution:** $u = 1 - \cosh(20(x - 0.5))/\cosh(10)$
- **Constants:** `g = 1`, `v = 1e-3`, `e = 0.4`
- **Network:** FNN [1, 20, 20, 20, 1], tanh, Glorot uniform
- **Training:** Adam 30,000; 100 domain points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Brinkman-Forchheimer 1D – unknown ν_e, K**

DeepXDE's own script, run unchanged twice on the same GPU, ends with $\nu_e$ = 1.31–1.35 × 10⁻³, $K$ = 9.84 × 10⁻⁴ and an $L_2$ error of 1.8 × 10⁻²–2.0 × 10⁻² for $u$ — the same picture: $K$ is recovered closely, $\nu_e$ only to within about 30–50% from five measurements.

**Result (RTX 4090):** $\nu_e$ = 1.49 × 10⁻³, $K$ = 9.85 × 10⁻⁴ (both true 1 × 10⁻³); $L_2$ of $u$ = 2.9 × 10⁻²

*PINN vs reference along a line*

![Brinkman-Forchheimer 1D – unknown ν_e, K — PINN vs reference along a line](assets/Images/DeepXDE/Inverse_Brinkman_Forchheimer_model/line_comparison.png)

*Inferred parameters during training*

![Brinkman-Forchheimer 1D – unknown ν_e, K — inferred parameters during training](assets/Images/DeepXDE/Inverse_Brinkman_Forchheimer_model/param_plot.png)

</details>

<details id="dx-inverse-diffusion-reaction-with-space-dependent-rate">
<summary><strong>Diffusion-reaction 1D – unknown rate field</strong></summary>

DeepXDE example: [`diffusion_reaction_rate.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/diffusion_reaction_rate.py)

$$\Large \lambda\frac{d^2u}{dx^2} - k(x)\ u = \sin(2\pi x), \qquad \lambda = 0.01,\ x \in [0, 1]$$

- **Unknown:** the reaction rate $k(x)$ (true $0.1 + e^{-0.5(x - 0.5)^2/0.15^2}$), learned as a second network output
- **Data:** 100 measurements of $u$ (computed with SciPy's `solve_bvp`, as DeepXDE does), $u = 0$ at both ends
- **Constants:** `l = 0.01`
- **Network:** PFNN, a separate [20, 20] subnetwork for each of the 2 outputs, tanh, Glorot uniform
- **Training:** Adam 20,000; 50 domain, 8 boundary points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Diffusion-reaction 1D – unknown rate field**

**Result (RTX 4090):** $L_2$ = 4.0 × 10⁻² ($k$)

*u — PINN vs reference along a line*

![Diffusion-reaction 1D – unknown rate field — PINN vs reference along a line (u)](assets/Images/DeepXDE/Inverse_diffusion_reaction_with_space_dependent_rate/line_comparison.png)

*u — PINN solution*

![Diffusion-reaction 1D – unknown rate field — PINN solution (u)](assets/Images/DeepXDE/Inverse_diffusion_reaction_with_space_dependent_rate/solution_plot.png)

</details>

<details id="dx-inverse-fractional-poisson-equation-1d">
<summary><strong>Fractional Poisson 1D – unknown order α</strong></summary>

DeepXDE example: [`fractional_Poisson_1d_inverse.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/fractional_Poisson_1d_inverse.py)

$$\Large \frac{D_{-1+}^{\alpha}u + D_{1-}^{\alpha}u}{2\cos(\alpha\pi/2)} = \Gamma(\alpha_0 + 2)\ x, \qquad x \in [-1, 1]$$

- **Unknown:** the order $\alpha$ (true $\alpha_0 = 1.8$, start 1.5) — an inverse variable named `alpha`, also the fractional α
- **Data:** 20 measurements of $u = x(1 - x^2)^{\alpha_0/2}$, loss weight 100
- **Fractional settings:** 100 points per unit length, dynamic mesh (rebuilt every step as α changes)
- **Constants:** `a0 = 1.8`
- **Network:** FNN [1, 20, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 10,000; 20 domain points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Fractional Poisson 1D – unknown order α**

**Result (RTX 4090):** $\alpha$ = 1.792 (true 1.8); $L_2$ of $u$ = 1.3 × 10⁻²

*PINN vs reference along a line*

![Fractional Poisson 1D – unknown order α — PINN vs reference along a line](assets/Images/DeepXDE/Inverse_fractional_Poisson_equation_1D/line_comparison.png)

*Inferred parameters during training*

![Fractional Poisson 1D – unknown order α — inferred parameters during training](assets/Images/DeepXDE/Inverse_fractional_Poisson_equation_1D/param_plot.png)

</details>

<details id="dx-inverse-poisson-with-unknown-forcing-field">
<summary><strong>Poisson 1D – unknown forcing field</strong></summary>

DeepXDE example: [`elliptic_inverse_field.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/elliptic_inverse_field.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/elliptic.inverse.field.html))

$$\Large -\frac{d^2u}{dx^2} + q(x) = 0, \qquad x \in [-1, 1]$$

- **Unknown:** the field $q(x)$ (true $-\pi^2\sin(\pi x)$), learned as a second network output
- **Data:** 100 measurements of $u$, given as a Point Set BC (and as anchor points); $u = \sin(\pi x)$ on the boundary
- **Mode:** Forward — there are no unknown constants, only an unknown field, as in DeepXDE
- **Network:** PFNN, a separate [20, 20, 20] subnetwork for each of the 2 outputs, tanh, Glorot uniform
- **Training:** Adam 20,000; 200 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Poisson 1D – unknown forcing field**

**Result (RTX 4090):** $L_2$ = 8.3 × 10⁻⁴ ($u$), 4.3 × 10⁻² ($q$)

*u — PINN vs reference along a line*

![Poisson 1D – unknown forcing field — PINN vs reference along a line (u)](assets/Images/DeepXDE/Inverse_Poisson_with_unknown_forcing_field/line_comparison_u.png)

*q — PINN vs reference along a line*

![Poisson 1D – unknown forcing field — PINN vs reference along a line (q)](assets/Images/DeepXDE/Inverse_Poisson_with_unknown_forcing_field/line_comparison_q.png)

</details>

<details id="dx-inverse-poisson-with-unknown-forcing-field-batches">
<summary><strong>Poisson 1D – unknown forcing field (batches)</strong></summary>

DeepXDE example: [`elliptic_inverse_field_batch.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/elliptic_inverse_field_batch.py)

$$\Large -\frac{d^2u}{dx^2} + q(x) = 0, \qquad x \in [-1, 1]$$

- **Unknown:** the field $q(x)$ (true $-\pi^2\sin(\pi x)$)
- **Data:** 10,000 measurements of $u$, used 100 at a time (Point Set BC with batch size 100, redrawn every iteration)
- **Network:** PFNN, a separate [20, 20, 20] subnetwork for each of the 2 outputs, tanh, Glorot uniform
- **Training:** Adam 20,000; 200 domain, 2 boundary points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Poisson 1D – unknown forcing field (batches)**

**Result (RTX 4090):** $L_2$ = 7.6 × 10⁻³ ($u$), 1.3 × 10⁻¹ ($q$)

*q — PINN vs reference along a line*

![Poisson 1D – unknown forcing field (batches) — PINN vs reference along a line (q)](assets/Images/DeepXDE/Inverse_Poisson_with_unknown_forcing_field_batches/line_comparison_q.png)

</details>

<details id="1d-allen-cahn">
<summary><strong>Allen-Cahn 1D – Wight & Zhao</strong></summary>

Benchmark problem after Wight & Zhao (2021) — see [References](#references).

$$\Large \frac{\partial u}{\partial t} = \gamma_1\frac{\partial^2 u}{\partial x^2} - \gamma_2 u^3 + \gamma_2 u, \qquad x \in [-1, 1],\ t \in [0, 1]$$

where $\gamma_1 = 0.0001$ and $\gamma_2 = 5$.

- **Initial condition:** $u(x, 0) = x^2\cos(\pi x)$
- **Boundary conditions:** Periodic
- **Geometry:** Interval
- **Time-Adaptive default:** on — $t \in [0,1]$ split into 4 steps of 0.25, L-BFGS transfer learning between steps
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Allen-Cahn 1D – Wight & Zhao**

**Result (RTX 4090):** $L_2$ = 1.5 × 10⁻⁴ ($t = 0$), 4.5 × 10⁻⁴ ($t = 0.25$), 1.0 × 10⁻³ ($t = 0.5$), 1.8 × 10⁻³ ($t = 0.75$), 4.8 × 10⁻³ ($t = 1$), against a Fourier pseudo-spectral solution (ETDRK4) in [`reference_data/1D/allen_cahn`](reference_data/1D/allen_cahn) (made by [`tools/make_exact_reference_data.py`](tools/make_exact_reference_data.py)). Inverse mode: $\gamma_1$ = 1.29 × 10⁻⁴ (true 1 × 10⁻⁴), $\gamma_2$ = 5.005 (true 5).

![1D Allen-Cahn — PINN vs Ground Truth, Error Analysis surface comparison](assets/Images/1D-AllenCahn/surface_comparison.png)

![1D Allen-Cahn — line comparison against reference data](assets/Images/1D-AllenCahn/line_comparison.png)

*Inverse mode* — same template, switched to **Inverse**, jointly recovering both $\gamma_1$ (diffusion) and $\gamma_2$ (reaction) as unknown parameters; the live convergence animation below shows each estimate settling onto its true value during training:

![1D Allen-Cahn inverse parameter convergence animation](assets/Images/1D-AllenCahn/param_convergence_animation.gif)

</details>

<details id="dx-allen-cahn-equation">
<summary><strong>Allen-Cahn 1D</strong></summary>

DeepXDE example: [`Allen_Cahn.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Allen_Cahn.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html))

$$\Large \frac{\partial u}{\partial t} = 0.001\frac{\partial^2 u}{\partial x^2} + 5(u - u^3), \qquad x \in [-1, 1],\ t \in [0, 1]$$

- **Initial and boundary conditions:** $u(x, 0) = x^2\cos(\pi x)$ and $u(\pm 1, t) = -1$, imposed exactly by $\hat u = x^2\cos(\pi x) + t(1 - x^2)N$
- **Reference:** computed by `tools/make_deepxde_reference_data.py` (agrees with DeepXDE's `Allen_Cahn.mat` to $5\times 10^{-4}$)
- **Constants:** `d = 0.001`
- **Network:** FNN [2, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 40,000 → L-BFGS 15,000; 8,000 domain, 400 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Allen-Cahn 1D**

**Result (RTX 4090):** $L_2$ = 9.5 × 10⁻⁸ ($t = 0$), 1.9 × 10⁻³ ($t = 0.25$), 5.2 × 10⁻³ ($t = 0.5$), 1.3 × 10⁻² ($t = 0.75$), 2.9 × 10⁻² ($t = 1$)

*u — PINN solution over time*

![Allen-Cahn 1D — PINN solution over time (u)](assets/Images/DeepXDE/Allen_Cahn_equation/animation.gif)

*PINN vs reference, error analysis*

![Allen-Cahn 1D — PINN vs reference, error analysis](assets/Images/DeepXDE/Allen_Cahn_equation/surface_comparison.png)

</details>

<details id="dx-american-put-option">
<summary><strong>Black-Scholes 1D – American put</strong></summary>

DeepXDE example: [`american_put.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/american_put.py)

$$\Large \min\left(-\mathcal{L}V,\ V - \max(K - S, 0)\right) = 0, \qquad \mathcal{L}V = -\frac{\partial V}{\partial \tau} + \frac{\sigma^2}{2}S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV$$

- **Domain:** $S \in [0, 4]$, $\tau \in [0, 1]$; strike $K = 1$, $r = 0.05$, $\sigma = 0.2$
- **Equations:** three residuals for one output (DeepXDE's Fischer-Burmeister form of the complementarity problem)
- **Extra points:** 1000 anchors in $S \in [0.5, 1.5]$ around the exercise boundary; point resampling every 1000 iterations
- **Reference:** none bundled (DeepXDE compares against a binomial-tree price) — add your own in Error Analysis
- **Constants:** `r = 0.05`, `sigma = 0.2`, `K = 1`
- **Network:** FNN [2, 128, 128, 128, 128, 1], tanh, Glorot normal
- **Training:** Adam 12,000 → L-BFGS 15,000; 6,000 domain, 1,500 boundary, 1,500 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Black-Scholes 1D – American put**

**Result (RTX 4090):** trained without errors; no bundled reference to compare against

*u — PINN solution over time*

![Black-Scholes 1D – American put — PINN solution over time (u)](assets/Images/DeepXDE/American_put_option/animation.gif)

</details>

<details id="dx-black-scholes-equation-call-option">
<summary><strong>Black-Scholes 1D – European call</strong></summary>

DeepXDE example: [`black_scholes_call.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/black_scholes_call.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/black_scholes.html))

$$\Large \frac{\partial V}{\partial \tau} = \frac{\sigma^2}{2}S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV, \qquad S \in [0, 150],\ \tau \in [0, 1]$$

- **Parameters:** strike $K = 50$, rate $r = 0.05$, volatility $\sigma = 0.2$; $x$ is the stock price $S$ and $t$ the time to maturity $\tau$
- **Initial condition:** payoff $V(S, 0) = \max(S - K, 0)$; **boundary conditions:** $V(0, \tau) = 0$, $V(150, \tau) = 150 - Ke^{-r\tau}$
- **Exact solution:** the Black-Scholes formula (uses `norm_cdf` in the reference formula)
- **Constants:** `K = 50`, `r = 0.05`, `sigma = 0.2`
- **Network:** FNN [2, 28, 28, 28, 28, 1], tanh, Glorot normal
- **Training:** Adam 20,000 → L-BFGS 15,000; 2,047 domain, 63 boundary, 127 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Black-Scholes 1D – European call**

**Result (RTX 4090):** $L_2$ = 5.7 × 10⁻⁴ ($t = 0.25$), 3.2 × 10⁻⁴ ($t = 0.5$), 4.3 × 10⁻⁴ ($t = 1$)

*u — PINN solution over time*

![Black-Scholes 1D – European call — PINN solution over time (u)](assets/Images/DeepXDE/Black_Scholes_equation_call_option/animation.gif)

*PINN vs reference, error analysis*

![Black-Scholes 1D – European call — PINN vs reference, error analysis](assets/Images/DeepXDE/Black_Scholes_equation_call_option/surface_comparison.png)

</details>

<details id="1d-burgers">
<summary><strong>Burgers 1D – Raissi et al.</strong></summary>

Exact equation, initial and boundary conditions as in Raissi, Perdikaris & Karniadakis (2019) — see [References](#references).

$$\Large \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \frac{0.01}{\pi}\frac{\partial^2 u}{\partial x^2}, \qquad x \in [-1, 1],\ t \in [0, 1]$$

- **Initial condition:** $u(x, 0) = -\sin(\pi x)$
- **Boundary conditions:** Dirichlet, $u = 0$ at both ends
- **Geometry:** Interval
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Burgers 1D – Raissi et al.**

A shock forms near $x = 0$ as $t \to 1$; this template uses a larger fixed collocation count to resolve it rather than adaptive refinement.

**Result (RTX 4090):** $L_2$ = 6.0 × 10⁻⁵ ($t = 0$), 3.7 × 10⁻⁴ ($t = 0.25$), 7.2 × 10⁻⁴ ($t = 0.5$), 1.2 × 10⁻³ ($t = 0.75$), 1.3 × 10⁻³ ($t = 1$), against the exact (Cole–Hopf) solution in [`reference_data/1D/burgers`](reference_data/1D/burgers). Inverse mode: $\lambda_1$ = 0.997 (true 1), $\lambda_2$ = 0.01001 (true 0.01; the template writes the viscous term as $\frac{\lambda_2}{\pi}\frac{\partial^2 u}{\partial x^2}$, so $\lambda_2 = 0.01$ is the paper's $0.01/\pi$).

![1D Burgers — PINN vs Ground Truth, Error Analysis surface comparison](assets/Images/1D-Burgers/surface_comparison.png)

![1D Burgers — line comparison against reference data](assets/Images/1D-Burgers/line_comparison.png)

*Restore & Visualize* — the trained model restored from its saved checkpoint and re-animated without retraining:

![1D Burgers — restored solution animation](assets/Images/1D-Burgers/restored_animation.gif)

*Inverse mode* — same template, switched to **Inverse**, jointly recovering both coefficients of Raissi, Perdikaris & Karniadakis (2019)'s data-driven-discovery form of this equation, Appendix B, eq. B.1 — see [References](#references):

$$\Large \frac{\partial u}{\partial t} + \lambda_1 u\frac{\partial u}{\partial x} - \lambda_2\frac{\partial^2 u}{\partial x^2} = 0$$

with true values $\lambda_1 = 1.0$ and $\lambda_2 = 0.01/\pi$. $\lambda_1$'s initial guess starts at $0.1$ rather than $1.0$, since $1.0$ is already its true value. The live convergence animation below shows both estimates settling onto their true values during training:

![1D Burgers inverse parameter convergence animation](assets/Images/1D-Burgers/param_convergence_animation.gif)

</details>

<details id="dx-burgers-equation-with-rar">
<summary><strong>Burgers 1D – RAR</strong></summary>

DeepXDE example: [`Burgers_RAR.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Burgers_RAR.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.rar.html))

$$\Large \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \frac{0.01}{\pi}\frac{\partial^2 u}{\partial x^2}, \qquad x \in [-1, 1],\ t \in [0, 1]$$

- **Initial / boundary conditions:** $u(x, 0) = -\sin(\pi x)$, $u(\pm 1, t) = 0$
- **Time range / reference:** $t \in [0, 1]$, compared with the exact solution at $t = 0, 0.25, 0.5, 0.75, 1$
- **Residual-based adaptive refinement:** 3 rounds, each adding the 100 points with the largest PDE residual (out of 100,000 random candidates) and retraining (Adam 10,000 → L-BFGS 15,000)
- **Network:** FNN [2, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 10,000 → L-BFGS 15,000 per RAR round; 2,500 domain, 100 boundary, 160 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Burgers 1D – RAR**

**Why not DeepXDE's one-point refinement.** DeepXDE's script adds 1 point per round until the mean residual is below 0.005. Here, after the first training the mean residual is already about 3–4 × 10⁻³, so that loop would add a single point and stop, which shows nothing. This example therefore turns the tolerance off and runs 3 rounds of 100 points instead. The collocation-point figure below shows where they go: onto the shock at $x = 0$ once it has formed ($t \gtrsim 0.3$), the region of largest error. The $L_2$ error at $t = 1$, where the shock is steepest, falls from 2.6 × 10⁻² after round 1 to 4.0 × 10⁻³ after round 2 and 8.4 × 10⁻⁴ after round 3.

**Result (RTX 4090):** $L_2$ = 3.1 × 10⁻³ ($t = 0$), 4.6 × 10⁻³ ($t = 0.25$), 1.0 × 10⁻³ ($t = 0.5$), 1.3 × 10⁻³ ($t = 0.75$), 8.4 × 10⁻⁴ ($t = 1$)

*u — PINN solution over time*

![Burgers 1D – RAR — PINN solution over time (u)](assets/Images/DeepXDE/Burgers_equation_with_RAR/animation.gif)

*PINN vs reference, error analysis*

![Burgers 1D – RAR — PINN vs reference, error analysis](assets/Images/DeepXDE/Burgers_equation_with_RAR/surface_comparison.png)

*Collocation points after the last RAR round: original points (gray), points added in earlier rounds (orange) and in the last round (red)*

![Burgers 1D – RAR — collocation points after the last RAR round: original points (gray), points added in earlier rounds (orange) and in the last round (red)](assets/Images/DeepXDE/Burgers_equation_with_RAR/rar_collocation_points.png)

</details>

<details id="dx-diffusion-equation">
<summary><strong>Diffusion 1D</strong></summary>

DeepXDE example: [`diffusion_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.1d.html))

$$\Large \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - e^{-t}\left(\sin(\pi x) - \pi^2\sin(\pi x)\right), \qquad x \in [-1, 1],\ t \in [0, 1]$$

- **Initial condition:** $u(x, 0) = \sin(\pi x)$; **boundary condition:** $u = e^{-t}\sin(\pi x)$
- **Exact solution:** $u = e^{-t}\sin(\pi x)$
- **Network:** FNN [2, 32, 32, 32, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 40 domain, 20 boundary, 10 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Diffusion 1D**

**Result (RTX 4090):** $L_2$ = 4.8 × 10⁻³ ($t = 0$), 9.0 × 10⁻³ ($t = 0.25$), 6.4 × 10⁻³ ($t = 0.5$), 5.8 × 10⁻³ ($t = 0.75$), 3.0 × 10⁻² ($t = 1$)

*u — PINN solution over time*

![Diffusion 1D — PINN solution over time (u)](assets/Images/DeepXDE/Diffusion_equation/animation.gif)

*PINN vs reference, error analysis*

![Diffusion 1D — PINN vs reference, error analysis](assets/Images/DeepXDE/Diffusion_equation/surface_comparison.png)

</details>

<details id="dx-diffusion-equation-with-hard-ic-and-bc">
<summary><strong>Diffusion 1D – hard IC and BCs</strong></summary>

DeepXDE example: [`diffusion_1d_exactBC.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d_exactBC.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.1d.exactBC.html))

$$\Large \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - e^{-t}\left(\sin(\pi x) - \pi^2\sin(\pi x)\right), \qquad x \in [-1, 1],\ t \in [0, 1]$$

- **Initial and boundary conditions:** imposed exactly by $\hat u = t(1 - x^2)N + \sin(\pi x)$ — no IC/BC loss terms
- **Exact solution:** $u = e^{-t}\sin(\pi x)$
- **Network:** FNN [2, 32, 32, 32, 1], tanh, Glorot uniform
- **Training:** Adam 10,000; 40 domain points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Diffusion 1D – hard IC and BCs**

**Result (RTX 4090):** $L_2$ = 9.4 × 10⁻⁸ ($t = 0$), 1.6 × 10⁻⁴ ($t = 0.25$), 4.6 × 10⁻⁴ ($t = 0.5$), 1.3 × 10⁻³ ($t = 0.75$), 2.8 × 10⁻³ ($t = 1$)

*u — PINN solution over time*

![Diffusion 1D – hard IC and BCs — PINN solution over time (u)](assets/Images/DeepXDE/Diffusion_equation_with_hard_IC_and_BC/animation.gif)

*PINN vs reference, error analysis*

![Diffusion 1D – hard IC and BCs — PINN vs reference, error analysis](assets/Images/DeepXDE/Diffusion_equation_with_hard_IC_and_BC/surface_comparison.png)

</details>

<details id="dx-diffusion-equation-with-resampling">
<summary><strong>Diffusion 1D – resampling</strong></summary>

DeepXDE example: [`diffusion_1d_resample.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d_resample.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.1d.resample.html))

$$\Large \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - e^{-t}\left(\sin(\pi x) - \pi^2\sin(\pi x)\right), \qquad x \in [-1, 1],\ t \in [0, 1]$$

- **Initial / boundary conditions:** $u(x, 0) = \sin(\pi x)$, $u = e^{-t}\sin(\pi x)$ on the boundary
- **Point resampling:** pseudo-random collocation points redrawn every 100 iterations
- **Exact solution:** $u = e^{-t}\sin(\pi x)$
- **Network:** FNN [2, 32, 32, 32, 1], tanh, Glorot uniform
- **Training:** Adam 2,000; 40 domain, 20 boundary, 10 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Diffusion 1D – resampling**

**Result (RTX 4090):** $L_2$ = 1.8 × 10⁻² ($t = 0$), 2.5 × 10⁻² ($t = 0.25$), 1.5 × 10⁻² ($t = 0.5$), 1.4 × 10⁻² ($t = 0.75$), 4.3 × 10⁻² ($t = 1$)

*u — PINN solution over time*

![Diffusion 1D – resampling — PINN solution over time (u)](assets/Images/DeepXDE/Diffusion_equation_with_resampling/animation.gif)

*PINN vs reference, error analysis*

![Diffusion 1D – resampling — PINN vs reference, error analysis](assets/Images/DeepXDE/Diffusion_equation_with_resampling/surface_comparison.png)

</details>

<details id="dx-diffusion-reaction-equation">
<summary><strong>Diffusion-reaction 1D</strong></summary>

DeepXDE example: [`diffusion_reaction.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_reaction.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.reaction.html))

$$\Large \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} + e^{-t}\left(\tfrac{3}{2}\sin 2x + \tfrac{8}{3}\sin 3x + \tfrac{15}{4}\sin 4x + \tfrac{63}{8}\sin 8x\right), \qquad x \in [-\pi, \pi],\ t \in [0, 1]$$

- **Initial and boundary conditions:** imposed exactly by $\hat u = t(\pi^2 - x^2)N + \sum_k \frac{\sin(kx)}{k}$
- **Exact solution:** $u = e^{-t}\left(\sin x + \tfrac{\sin 2x}{2} + \tfrac{\sin 3x}{3} + \tfrac{\sin 4x}{4} + \tfrac{\sin 8x}{8}\right)$
- **Network:** FNN [2, 30, 30, 30, 30, 30, 30, 1], tanh, Glorot uniform
- **Training:** Adam 20,000; 320 domain points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Diffusion-reaction 1D**

DeepXDE's own script, run unchanged on the same GPU, ends with a similar loss (and the same Adam loss spikes near the end of training).

**Result (RTX 4090):** $L_2$ = 9.9 × 10⁻⁸ ($t = 0$), 1.4 × 10⁻² ($t = 0.25$), 3.4 × 10⁻² ($t = 0.5$), 6.7 × 10⁻² ($t = 0.75$), 1.1 × 10⁻¹ ($t = 1$)

*u — PINN solution over time*

![Diffusion-reaction 1D — PINN solution over time (u)](assets/Images/DeepXDE/Diffusion_reaction_equation/animation.gif)

*PINN vs reference, error analysis*

![Diffusion-reaction 1D — PINN vs reference, error analysis](assets/Images/DeepXDE/Diffusion_reaction_equation/surface_comparison.png)

</details>

<details id="dx-fractional-diffusion-equation">
<summary><strong>Fractional diffusion 1D</strong></summary>

DeepXDE example: [`fractional_diffusion_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_diffusion_1d.py)

$$\Large \frac{\partial u}{\partial t} = D_{0+}^{\alpha}u + D_{1-}^{\alpha}u - e^{-t}\left(x^3(1-x)^3 + f_\alpha(x)\right), \qquad x \in [0, 1],\ t \in [0, 1],\ \alpha = 1.8$$

- **Equation in the panel:** `du_t - frac(u) + exp(-t)*(x**3*(1 - x)**3 + ...)` (DeepXDE's TimeFPDE)
- **Fractional settings:** 52-point static mesh in x at each time
- **Boundary and initial conditions:** $u(0, t) = u(1, t) = 0$, $u(x, 0) = x^3(1-x)^3$, both exact through $\hat u = x(1-x)\ t\ N + x^3(1-x)^3$
- **Exact solution:** $u = e^{-t}x^3(1-x)^3$
- **Constants:** `alpha = 1.8`
- **Network:** FNN [2, 20, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 10,000; 400 domain, 2 boundary, 2 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Fractional diffusion 1D**

**Result (RTX 4090):** $L_2$ = 1.0 × 10⁻⁷ ($t = 0$), 1.4 × 10⁻³ ($t = 0.25$), 5.8 × 10⁻³ ($t = 0.5$), 2.4 × 10⁻³ ($t = 0.75$), 3.3 × 10⁻³ ($t = 1$)

*u — PINN solution over time*

![Fractional diffusion 1D — PINN solution over time (u)](assets/Images/DeepXDE/Fractional_diffusion_equation/animation.gif)

*PINN vs reference, error analysis*

![Fractional diffusion 1D — PINN vs reference, error analysis](assets/Images/DeepXDE/Fractional_diffusion_equation/surface_comparison.png)

</details>

<details id="dx-heat-equation">
<summary><strong>Heat 1D – Dirichlet</strong></summary>

DeepXDE example: [`heat.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/heat.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html))

$$\Large \frac{\partial u}{\partial t} = 0.4\frac{\partial^2 u}{\partial x^2}, \qquad x \in [0, 1],\ t \in [0, 1]$$

- **Initial condition:** $u(x, 0) = \sin(\pi x)$; **boundary conditions:** $u(0, t) = u(1, t) = 0$
- **Exact solution:** $u = e^{-0.4\pi^2 t}\sin(\pi x)$
- **Constants:** `a = 0.4`, `L = 1`, `n = 1`
- **Network:** FNN [2, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 20,000 → L-BFGS 15,000; 2,540 domain, 80 boundary, 160 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Heat 1D – Dirichlet**

The solution decays to about 2% of its initial size by $t = 1$, so the relative error grows over time even though the absolute error stays small.

**Result (RTX 4090):** $L_2$ = 4.2 × 10⁻⁴ ($t = 0$), 9.1 × 10⁻⁴ ($t = 0.25$), 2.2 × 10⁻³ ($t = 0.5$), 9.5 × 10⁻³ ($t = 0.75$), 5.0 × 10⁻² ($t = 1$)

*u — PINN solution over time*

![Heat 1D – Dirichlet — PINN solution over time (u)](assets/Images/DeepXDE/Heat_equation/animation.gif)

*PINN vs reference, error analysis*

![Heat 1D – Dirichlet — PINN vs reference, error analysis](assets/Images/DeepXDE/Heat_equation/surface_comparison.png)

*Inverse mode* — pick the example from the **Inverse** list: the diffusivity $a$ is unknown (initial guess 1, true value 0.4) and is learned together with $u$ from $u$ measured at $t = 1$ (the exact solution, [`reference_data/1D/heat/t_1.0.txt`](reference_data/1D/heat/t_1.0.txt)):

$$\Large \frac{\partial u}{\partial t} = a\ \frac{\partial^2 u}{\partial x^2}$$

**Inverse result (RTX 4090):** $a$ = 0.3984 (true 0.4); $L_2$ = 1.0 × 10⁻³ ($t = 0$), 3.2 × 10⁻³ ($t = 0.25$), 1.2 × 10⁻² ($t = 0.5$), 1.5 × 10⁻² ($t = 0.75$), 2.7 × 10⁻³ ($t = 1$)

*Inverse mode — inferred parameter during training*

![Heat 1D – Dirichlet inverse mode — inferred parameter during training](assets/Images/DeepXDE/Heat_equation/inverse_param_convergence_animation.gif)

</details>

<details id="dx-heat-equation-with-resampling">
<summary><strong>Heat 1D – resampling</strong></summary>

DeepXDE example: [`heat_resample.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/heat_resample.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.resample.html))

$$\Large \frac{\partial u}{\partial t} = 0.4\frac{\partial^2 u}{\partial x^2}, \qquad x \in [0, 1],\ t \in [0, 1]$$

- **Initial / boundary conditions:** $u(x, 0) = \sin(\pi x)$, $u(0, t) = u(1, t) = 0$
- **Point resampling:** the collocation points are redrawn every 10 iterations (DeepXDE's `PDEPointResampler`)
- **Exact solution:** $u = e^{-0.4\pi^2 t}\sin(\pi x)$
- **Constants:** `a = 0.4`, `L = 1`, `n = 1`
- **Network:** FNN [2, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 200,000 → L-BFGS 15,000; 2,540 domain, 80 boundary, 160 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Heat 1D – resampling**

**Result (RTX 4090):** $L_2$ = 1.6 × 10⁻⁴ ($t = 0$), 4.6 × 10⁻⁴ ($t = 0.25$), 5.8 × 10⁻⁴ ($t = 0.5$), 4.1 × 10⁻³ ($t = 0.75$), 1.4 × 10⁻² ($t = 1$)

*u — PINN solution over time*

![Heat 1D – resampling — PINN solution over time (u)](assets/Images/DeepXDE/Heat_equation_with_resampling/animation.gif)

*PINN vs reference, error analysis*

![Heat 1D – resampling — PINN vs reference, error analysis](assets/Images/DeepXDE/Heat_equation_with_resampling/surface_comparison.png)

</details>

<details id="dx-klein-gordon-equation">
<summary><strong>Klein-Gordon 1D</strong></summary>

DeepXDE example: [`Klein_Gordon.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Klein_Gordon.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/klein.gordon.html))

$$\Large \frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} + u^2 = -x\cos t + x^2\cos^2 t, \qquad x \in [-1, 1],\ t \in [0, 10]$$

- **Initial conditions:** $u(x, 0) = x$ and $\partial u/\partial t\ (x, 0) = 0$ (the second as an Operator BC applied at the initial time)
- **Boundary condition:** $u = x\cos t$
- **Training:** Adam with inverse-time decay ($10^{-3}$, steps 3000, rate 0.9), then L-BFGS
- **Exact solution:** $u = x\cos t$
- **Constants:** `alpha = -1`, `beta = 0`, `gamma = 1`, `k = 2`
- **Network:** FNN [2, 40, 40, 1], tanh, Glorot uniform
- **Training:** Adam 20,000 → L-BFGS 15,000; 30,000 domain, 1,500 boundary, 1,500 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Klein-Gordon 1D**

**Result (RTX 4090):** $L_2$ = 1.4 × 10⁻³ ($t = 0$), 1.6 × 10⁻² ($t = 2.5$), 8.5 × 10⁻² ($t = 5$), 6.5 × 10⁻² ($t = 7.5$), 8.1 × 10⁻² ($t = 10$)

*u — PINN solution over time*

![Klein-Gordon 1D — PINN solution over time (u)](assets/Images/DeepXDE/Klein_Gordon_equation/animation.gif)

*PINN vs reference, error analysis*

![Klein-Gordon 1D — PINN vs reference, error analysis](assets/Images/DeepXDE/Klein_Gordon_equation/surface_comparison.png)

</details>

<details id="dx-schr-dinger-equation">
<summary><strong>Schrödinger 1D</strong></summary>

DeepXDE example: [`Schrodinger.ipynb`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Schrodinger.ipynb)

$$\Large i\frac{\partial h}{\partial t} + \frac{1}{2}\frac{\partial^2 h}{\partial x^2} + |h|^2 h = 0, \qquad x \in [-5, 5],\ t \in [0, \pi/2]$$

$$\Large \frac{\partial u}{\partial t} + 0.5\frac{\partial^2 v}{\partial x^2} + (u^2 + v^2)\ v = 0$$

$$\Large \frac{\partial v}{\partial t} - 0.5\frac{\partial^2 u}{\partial x^2} - (u^2 + v^2)\ u = 0$$

- **Solved as:** two real outputs, $h = u + iv$: the second and third equations above are the real and imaginary parts, entered in PINNStudio as PDE 1 (`du_t + 0.5*dv_xx + (u**2 + v**2)*v`) and PDE 2 (`dv_t - 0.5*du_xx - (u**2 + v**2)*u`)
- **Initial condition:** $h(x, 0) = 2\ \mathrm{sech}(x)$; **boundary conditions:** periodic in $u$, $v$ and their $x$-derivatives
- **Compared:** $|h| = \sqrt{u^2 + v^2}$ against a Fourier pseudo-spectral (split-step) solution in [`reference_data/1D/schrodinger`](reference_data/1D/schrodinger), made by [`tools/make_exact_reference_data.py`](tools/make_exact_reference_data.py)
- **Network:** FNN [2, 100, 100, 100, 100, 2], tanh, Glorot normal
- **Training:** Adam 10,000 → L-BFGS 10,000; 10,000 domain, 20 boundary, 200 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Schrödinger 1D**

**Result (RTX 4090):** $L_2$ of $|h|$ = 1.5 × 10⁻³ ($t = 0$), 1.1 × 10⁻³ ($t = 0.25$), 1.1 × 10⁻³ ($t = 0.5$), 1.4 × 10⁻³ ($t = 0.75$), 1.4 × 10⁻³ ($t = 1$), 1.1 × 10⁻³ ($t = 1.25$), 1.7 × 10⁻³ ($t = 1.5$)

*|h| — PINN solution over time*

![Schrödinger 1D — PINN solution over time (|h|)](assets/Images/DeepXDE/Schr_dinger_equation/animation.gif)

*|h| — PINN vs reference, error analysis*

![Schrödinger 1D — PINN vs reference, error analysis (|h|)](assets/Images/DeepXDE/Schr_dinger_equation/surface_comparison.png)

*Inverse mode* — pick the example from the **Inverse** list: the coefficient $c$ of the second-derivative terms (0.5 above; initial guess 1) is unknown and is learned from $|h| = \sqrt{u^2 + v^2}$ measured at $t = 0.75$, with the same training as above:

$$\Large \frac{\partial u}{\partial t} + c\frac{\partial^2 v}{\partial x^2} + (u^2 + v^2) v = 0, \qquad \frac{\partial v}{\partial t} - c\frac{\partial^2 u}{\partial x^2} - (u^2 + v^2) u = 0$$

**Why $t = 0.75$.** Measured at $t = 1.5$ only, $|h|$ does not determine $c$: solving the equation with $c = 1.95$ gives an $|h|$ at $t = 1.5$ within 1% of the one for $c = 0.5$, and training from the guess $c = 1$ converges to 1.95, fitting the data at $t = 1.5$ while missing every other time by 27–60%. At $t = 0.75$ (or $0.5$) $|h|$ changes with $c$ in one direction only, and the true value is recovered.

**Inverse result (RTX 4090):** $c$ = 0.50017 (true 0.5); $L_2$ of $|h|$ = 1.3 × 10⁻³ ($t = 0$), 9.7 × 10⁻⁴ ($t = 0.25$), 1.4 × 10⁻³ ($t = 0.5$), 2.2 × 10⁻⁴ ($t = 0.75$), 8.4 × 10⁻⁴ ($t = 1$), 1.3 × 10⁻³ ($t = 1.25$), 1.5 × 10⁻³ ($t = 1.5$)

*Inverse mode — inferred parameter during training*

![Schrödinger 1D inverse mode — inferred parameter during training](assets/Images/DeepXDE/Schr_dinger_equation/inverse_param_convergence_animation.gif)

</details>

<details id="dx-wave-propagation-space-time-multi-scale-fourier-features">
<summary><strong>Wave 1D – space-time multi-scale Fourier features</strong></summary>

DeepXDE example: [`wave_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/wave_1d.py)

$$\Large \frac{\partial^2 u}{\partial t^2} = C^2\frac{\partial^2 u}{\partial x^2}, \qquad x \in [0, 1],\ t \in [0, 1],\ C = 10$$

- **Boundary conditions:** $u(0, t) = u(1, t) = 0$
- **Initial conditions:** $u(x, 0) = \sin(\pi x) + \sin(A\pi x)$ ($A = 2$); $\partial u/\partial t(x, 0) = 0$ as an Operator BC `du_t` applied at the initial time
- **Exact solution:** $u = \sin(\pi x)\cos(C\pi t) + \sin(A\pi x)\cos(AC\pi t)$ — ten and twenty oscillations in time
- **Network:** space-time multi-scale Fourier features (STMsFFN): space scale $\sigma_x = 1$, time scales $\sigma_t = 1, 10$; inputs rescaled to zero mean, unit variance by the input-transform formula `(x - 0.5)*2*sqrt(3); (t - 0.5)*2*sqrt(3)`
- **Loss weights:** set from the initial losses, $w_i = 5 / L_i(0)$ (*Scale by initial losses*); learning rate $10^{-3}/(1 + 0.9\ n/2000)$; collocation points redrawn every iteration
- **Constants:** `A = 2`, `C = 10`
- **Network:** STMsFFN [2, 100, 100, 100, 1], Fourier scales σ = 1, time scales 1, 10, tanh, Glorot uniform
- **Training:** Adam 10,000; 360 domain, 360 boundary, 360 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Wave 1D – space-time multi-scale Fourier features**

Collocation points are redrawn every iteration from a pseudo-random distribution. DeepXDE's script keeps the default Hammersley distribution, which is deterministic, so its resampler redraws the same 360 points: run that way, the network fits them (training loss 4 × 10⁻⁵) but not the wave in between (test loss near 1, $L_2$ ≈ 1 for t > 0). This is a hard problem in 10,000 iterations — the error grows with t; longer training improves it.

**Result (RTX 4090):** $L_2$ = 7.7 × 10⁻⁴ ($t = 0$), 4.5 × 10⁻² ($t = 0.25$), 9.8 × 10⁻² ($t = 0.5$), 1.0 × 10⁻¹ ($t = 0.75$), 1.4 × 10⁻¹ ($t = 1$)

*u — PINN solution over time*

![Wave 1D – space-time multi-scale Fourier features — PINN solution over time (u)](assets/Images/DeepXDE/Wave_propagation_space_time_multi_scale_Fourier_features/animation.gif)

*PINN vs reference, error analysis*

![Wave 1D – space-time multi-scale Fourier features — PINN vs reference, error analysis](assets/Images/DeepXDE/Wave_propagation_space_time_multi_scale_Fourier_features/surface_comparison.png)

</details>

<details id="dx-inverse-diffusion-equation">
<summary><strong>Diffusion 1D – unknown coefficient</strong></summary>

DeepXDE example: [`diffusion_1d_inverse.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/diffusion_1d_inverse.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/diffusion.1d.inverse.html))

$$\Large \frac{\partial u}{\partial t} = C\frac{\partial^2 u}{\partial x^2} - e^{-t}\left(\sin(\pi x) - \pi^2\sin(\pi x)\right), \qquad x \in [-1, 1],\ t \in [0, 1]$$

- **Unknown:** $C$ (true value 1, initial guess 2)
- **Data:** 10 measurements of $u$ at $t = 1$, plus the initial and boundary conditions
- **Exact solution:** $u = e^{-t}\sin(\pi x)$
- **Network:** FNN [2, 32, 32, 32, 1], tanh, Glorot uniform
- **Training:** Adam 50,000; 40 domain, 20 boundary, 10 initial points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Diffusion 1D – unknown coefficient**

**Result (RTX 4090):** $C$ = 0.9988 (true 1); $L_2$ = 1.2 × 10⁻³ ($t = 0$), 1.8 × 10⁻³ ($t = 0.25$), 1.9 × 10⁻³ ($t = 0.5$), 2.8 × 10⁻³ ($t = 0.75$), 3.1 × 10⁻³ ($t = 1$)

*u — PINN solution over time*

![Diffusion 1D – unknown coefficient — PINN solution over time (u)](assets/Images/DeepXDE/Inverse_diffusion_equation/animation.gif)

*PINN vs reference, error analysis*

![Diffusion 1D – unknown coefficient — PINN vs reference, error analysis](assets/Images/DeepXDE/Inverse_diffusion_equation/surface_comparison.png)

*Inferred parameters during training*

![Diffusion 1D – unknown coefficient — inferred parameters during training](assets/Images/DeepXDE/Inverse_diffusion_equation/param_plot.png)

</details>

<details id="dx-inverse-diffusion-reaction-system">
<summary><strong>Diffusion-reaction 1D – unknown k_f, D</strong></summary>

DeepXDE example: [`reaction_inverse.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/reaction_inverse.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/reaction.inverse.html))

$$\Large \frac{\partial C_A}{\partial t} = D\frac{\partial^2 C_A}{\partial x^2} - k_f C_A C_B^2, \qquad \frac{\partial C_B}{\partial t} = D\frac{\partial^2 C_B}{\partial x^2} - 2k_f C_A C_B^2$$

- **Domain:** $x \in [0, 1]$, $t \in [0, 10]$; $C_A = C_B = e^{-20x}$ at $t = 0$, $C_A = C_B = 1 - x$ on the boundary
- **Unknowns:** $k_f$ (true 0.1) and $D$ (true $2\times 10^{-3}$, written as $10^{-3}D$ with $D = 2$)
- **Data:** $C_A$ and $C_B$ on a space-time grid, computed by `tools/make_deepxde_reference_data.py`
- **Network:** FNN [2, 20, 20, 20, 2], tanh, Glorot uniform
- **Training:** Adam 80,000; 2,000 domain, 100 boundary, 100 initial points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Diffusion-reaction 1D – unknown k_f, D**

**Result (RTX 4090):** $k_f$ = 0.0992 (true 0.1), $D$ = 1.995 (true 2)

*ca — PINN solution over time*

![Diffusion-reaction 1D – unknown k_f, D — PINN solution over time (ca)](assets/Images/DeepXDE/Inverse_diffusion_reaction_system/animation.gif)

*ca — inferred parameters during training*

![Diffusion-reaction 1D – unknown k_f, D — inferred parameters during training (ca)](assets/Images/DeepXDE/Inverse_diffusion_reaction_system/param_plot.png)

</details>

### 2D examples

*Click an example to expand it.*

<details id="dx-fractional-poisson-equation-2d">
<summary><strong>Fractional Poisson 2D – disk</strong></summary>

DeepXDE example: [`fractional_Poisson_2d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_2d.py)

$$\Large -(-\Delta)^{\alpha/2}u = 2^{\alpha}\Gamma(2 + \tfrac{\alpha}{2})\Gamma(1 + \tfrac{\alpha}{2})\left(1 - (1 + \tfrac{\alpha}{2})|x|^2\right), \qquad |x| < 1,\ \alpha = 1.8$$

- **Equation in the panel:** `frac(u)*gamma((1 - alpha)/2)*gamma((2 + alpha)/2)/(2*pi**1.5) - ...` — `frac(u)` times $C(\alpha, 2)$ is $-(-\Delta)^{\alpha/2}u$
- **Fractional settings:** 8 directions, 100 Grünwald-Letnikov points per unit length, dynamic mesh
- **Geometry:** unit disk; $u = 0$ on the circle through $\hat u = (1 - x^2 - y^2)N$
- **Exact solution:** $u = (1 - |x|^2)^{1 + \alpha/2}$
- **Constants:** `alpha = 1.8`
- **Network:** FNN [2, 20, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 20,000; 100 domain points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Fractional Poisson 2D – disk**

**Result (RTX 4090):** $L_2$ = 3.4 × 10⁻³

*PINN vs reference, error analysis*

![Fractional Poisson 2D – disk — PINN vs reference, error analysis](assets/Images/DeepXDE/Fractional_Poisson_equation_2D/surface_comparison.png)

</details>

<details id="dx-helmholtz-2d-dirichlet">
<summary><strong>Helmholtz 2D – Dirichlet</strong></summary>

DeepXDE example: [`Helmholtz_Dirichlet_2d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.dirichlet.html))

$$\Large -\nabla^2 u - k_0^2 u = k_0^2\sin(k_0 x)\sin(k_0 y), \qquad k_0 = 4\pi,\ (x, y) \in [0, 1]^2$$

- **Boundary conditions:** $u = 0$ on the boundary, imposed exactly by $\hat u = x(1-x)y(1-y)N$
- **Exact solution:** $u = \sin(k_0 x)\sin(k_0 y)$
- **Constants:** `n = 2`, `k0 = 2*pi*n`
- **Network:** FNN [2, 150, 150, 150, 1], sin, Glorot uniform
- **Training:** Adam 5,000; 400 domain, 80 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Helmholtz 2D – Dirichlet**

**Result (RTX 4090):** $L_2$ = 1.9 × 10⁻³

*PINN vs reference, error analysis*

![Helmholtz 2D – Dirichlet — PINN vs reference, error analysis](assets/Images/DeepXDE/Helmholtz_2D_Dirichlet/surface_comparison.png)

</details>

<details id="dx-helmholtz-2d-hyperparameter-optimization">
<summary><strong>Helmholtz 2D – hyperparameter search</strong></summary>

DeepXDE example: [`Helmholtz_Dirichlet_2d_HPO.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d_HPO.py)

$$\Large -\nabla^2 u - k_0^2 u = k_0^2\sin(k_0 x)\sin(k_0 y), \qquad k_0 = 4\pi,\ (x, y) \in [0, 1]^2$$

- **Same problem** as Helmholtz 2D (Dirichlet), with $\hat u = x(1-x)y(1-y)N$
- **Parameter Sweep, mode Bayesian search:** learning rate $10^{-4}$–$5\times10^{-2}$ (log), hidden layers 1–10, neurons per layer 5–500, activation sin / sigmoid / tanh
- **Budget:** 50 runs of 10,000 Adam iterations each — the baseline (lr $10^{-3}$, 4 × 50, sin), 10 random runs, then 39 chosen by the Gaussian-process model, minimizing the final loss
- **Output:** one folder per run, `bayes_search.csv` and `bayes_convergence.png`; the best run is printed at the end of the Training Log
- **Constants:** `n = 2`, `k0 = 2*pi*n`
- **Network:** FNN [2, 50, 50, 50, 50, 1], sin, Glorot uniform
- **Training:** Adam 10,000; 400 domain, 80 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Helmholtz 2D – hyperparameter search**

**Result (RTX 4090):** the baseline (lr $10^{-3}$, 4 × 50, sin) ends at final loss 5.7 ($L_2$ = 3.3 × 10⁻¹); the best of the 50 runs — lr $1.5\times10^{-3}$, 3 hidden layers of 87, sin — reaches final loss 0.020 and **$L_2$ = 1.3 × 10⁻³** (about 40 minutes for all 50 runs)

*PINN vs reference, error analysis*

![Helmholtz 2D – hyperparameter search — PINN vs reference, error analysis](assets/Images/DeepXDE/Helmholtz_2D_hyperparameter_optimization/surface_comparison.png)

*Bayesian search: final loss of every run and the best so far*

![Helmholtz 2D – hyperparameter search — Bayesian search: final loss of every run and the best so far](assets/Images/DeepXDE/Helmholtz_2D_hyperparameter_optimization/bayes_convergence.png)

</details>

<details id="dx-helmholtz-sound-hard-scattering-absorbing-bc">
<summary><strong>Helmholtz 2D – sound-hard scattering</strong></summary>

DeepXDE example: [`Helmholtz_Sound_hard_ABC_2d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Sound_hard_ABC_2d.py)

$$\Large -\nabla^2 u - k_0^2 u = 0, \qquad u = u_{re} + i\ u_{im},\ k_0 = 2$$

- **Geometry:** Custom — the square $[-\pi, \pi]^2$ minus a disk of radius $R = \pi/4$
- **Scattered field** of the plane wave $e^{ik_0x}$ off a sound-hard disk: two outputs `u_re`, `u_im`, one equation each
- **On the disk** (Neumann): $\partial u/\partial n = -\partial e^{ik_0x}/\partial n$, i.e. `-k0*x*sin(k0*x)/R` and `k0*x*cos(k0*x)/R`
- **On the outer square** (absorbing, Robin): $\partial u/\partial n = ik_0u$, i.e. `-k0*u_im` on the `u_re` row and `k0*u_re` on the `u_im` row
- **Reference:** the Bessel/Hankel series solution of DeepXDE's script
- **Constants:** `k0 = 2`, `R = pi/4`
- **Network:** FNN [2, 350, 350, 350, 2], tanh, Glorot uniform
- **Training:** Adam 10,000; 1,600 domain, 320 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Helmholtz 2D – sound-hard scattering**

**Result (RTX 4090):** $L_2$ = 7.8 × 10⁻² ($u_{re}$), 6.7 × 10⁻² ($u_{im}$) — DeepXDE's own script, run unchanged on the same GPU, ends at 9.1 × 10⁻² (best 7.1 × 10⁻²)

*u_re — PINN vs reference, error analysis*

![Helmholtz 2D – sound-hard scattering — PINN vs reference, error analysis (u_re)](assets/Images/DeepXDE/Helmholtz_sound_hard_scattering_absorbing_BC/surface_comparison_u_re.png)

*u_im — PINN vs reference, error analysis*

![Helmholtz 2D – sound-hard scattering — PINN vs reference, error analysis (u_im)](assets/Images/DeepXDE/Helmholtz_sound_hard_scattering_absorbing_BC/surface_comparison_u_im.png)

</details>

<details id="dx-helmholtz-2d-with-a-hole">
<summary><strong>Helmholtz 2D – square with hole</strong></summary>

DeepXDE example: [`Helmholtz_Neumann_2d_hole.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Neumann_2d_hole.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.neumann.hole.html))

$$\Large -\nabla^2 u - k_0^2 u = k_0^2\sin(k_0 x)\sin(k_0 y), \qquad k_0 = 2\pi$$

- **Geometry:** Custom — the square $[-0.5, 0.5]^2$ minus a disk of radius 0.25
- **Boundary conditions:** Dirichlet (exact solution) on the outer square; Neumann (exact normal derivative) on the hole
- **Exact solution:** $u = \sin(k_0 x)\sin(k_0 y)$
- **Constants:** `n = 1`, `k0 = 2*pi*n`, `R = 1/4`
- **Network:** FNN [2, 350, 350, 350, 1], sin, Glorot uniform
- **Training:** Adam 5,000; 225 domain, 240 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Helmholtz 2D – square with hole**

**Result (RTX 4090):** $L_2$ = 4.6 × 10⁻²

*PINN vs reference, error analysis*

![Helmholtz 2D – square with hole — PINN vs reference, error analysis](assets/Images/DeepXDE/Helmholtz_2D_with_a_hole/surface_comparison.png)

</details>

<details id="dx-laplace-equation-on-a-disk">
<summary><strong>Laplace 2D – disk</strong></summary>

DeepXDE example: [`Laplace_disk.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Laplace_disk.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/laplace.disk.html))

$$\Large r\frac{\partial u}{\partial r} + r^2\frac{\partial^2 u}{\partial r^2} + \frac{\partial^2 u}{\partial \theta^2} = 0, \qquad r \in [0, 1],\ \theta \in [0, 2\pi]$$

- **Boundary condition:** $u(1, \theta) = \cos\theta$
- **Input features:** $(r\sin\theta,\ r\cos\theta)$ through an input-transform formula, which makes the solution periodic in $\theta$
- **Exact solution:** $u = r\cos\theta$
- **In PINNStudio:** $x$ plays $r$ and $y$ plays $\theta$, so the plots show the $(r, \theta)$ rectangle
- **Network:** FNN [2, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 15,000; 2,540 domain, 80 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Laplace 2D – disk**

**Result (RTX 4090):** $L_2$ = 1.1 × 10⁻³

*PINN vs reference, error analysis*

![Laplace 2D – disk — PINN vs reference, error analysis](assets/Images/DeepXDE/Laplace_equation_on_a_disk/surface_comparison.png)

</details>

<details id="dx-linear-elastostatic-plate">
<summary><strong>Linear elasticity 2D – plate</strong></summary>

DeepXDE example: [`elasticity_plate.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/elasticity_plate.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/elasticity.plate.html))

$$\Large \frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} = f_x, \qquad \frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} = f_y$$

$$\Large \sigma_{xx} = (\lambda + 2\mu)\varepsilon_{xx} + \lambda\varepsilon_{yy}, \quad \sigma_{yy} = (\lambda + 2\mu)\varepsilon_{yy} + \lambda\varepsilon_{xx}, \quad \sigma_{xy} = 2\mu\varepsilon_{xy}$$

- **Domain:** $[0, 1]^2$; $\lambda = 1$, $\mu = 0.5$, $Q = 4$ (constants `lmbd`, `mu`, `Q`); body forces $f_x$, $f_y$ as in DeepXDE's script
- **Outputs:** five, $u_x, u_y, \sigma_{xx}, \sigma_{yy}, \sigma_{xy}$, with five equations (two momentum, three constitutive)
- **Boundary conditions:** imposed exactly by an output transform, e.g. $\hat u_x = y(1-y)N_1$, $\hat\sigma_{yy} = (1-y)N_4 + (\lambda+2\mu)Q\sin(\pi x)$
- **Exact solution:** $u_x = \cos(2\pi x)\sin(\pi y)$, $u_y = \sin(\pi x)\ Qy^4/4$
- **Constants:** `lmbd = 1`, `mu = 0.5`, `Q = 4`
- **Network:** PFNN, a separate [40, 40, 40, 40] subnetwork for each of the 5 outputs, tanh, Glorot uniform
- **Training:** Adam 20,000 → L-BFGS 20,000; 500 domain, 500 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Linear elasticity 2D – plate**

**Note — more training than DeepXDE's script.** DeepXDE's script trains 5,000 Adam iterations, which are not enough for this problem: with exactly those settings the $L_2$ errors are 3.5 × 10⁻¹, 5.9 × 10⁻¹, 1.9 × 10⁻¹, 4.8 × 10⁻¹ and 1.6 for $u_x, u_y, \sigma_{xx}, \sigma_{yy}, \sigma_{xy}$ (DeepXDE's own script, run unchanged on the same GPU, also ends at 3.3 × 10⁻¹–3.6 × 10⁻¹). This example therefore trains **20,000 Adam + 20,000 L-BFGS iterations** (Training Phases 1 and 2; about 9 minutes on an RTX 4090), and the results below are for these settings.

**Result (RTX 4090):** $L_2$ = 4.9 × 10⁻² ($u_x$), 4.3 × 10⁻² ($u_y$), 7.2 × 10⁻³ ($\sigma_{xx}$), 2.0 × 10⁻² ($\sigma_{yy}$), 6.7 × 10⁻² ($\sigma_{xy}$) — 20,000 Adam + 20,000 L-BFGS (DeepXDE's 5,000 Adam alone: 3.5 × 10⁻¹, 5.9 × 10⁻¹, 1.9 × 10⁻¹, 4.8 × 10⁻¹, 1.6)

*ux — PINN vs reference, error analysis*

![Linear elasticity 2D – plate — PINN vs reference, error analysis (ux)](assets/Images/DeepXDE/Linear_elastostatic_plate/surface_comparison_ux.png)

*uy — PINN vs reference, error analysis*

![Linear elasticity 2D – plate — PINN vs reference, error analysis (uy)](assets/Images/DeepXDE/Linear_elastostatic_plate/surface_comparison_uy.png)

*sxx — PINN vs reference, error analysis*

![Linear elasticity 2D – plate — PINN vs reference, error analysis (sxx)](assets/Images/DeepXDE/Linear_elastostatic_plate/surface_comparison_sxx.png)

*syy — PINN vs reference, error analysis*

![Linear elasticity 2D – plate — PINN vs reference, error analysis (syy)](assets/Images/DeepXDE/Linear_elastostatic_plate/surface_comparison_syy.png)

*sxy — PINN vs reference, error analysis*

![Linear elasticity 2D – plate — PINN vs reference, error analysis (sxy)](assets/Images/DeepXDE/Linear_elastostatic_plate/surface_comparison_sxy.png)

</details>

<details id="dx-linear-elastostatic-plate-soft-bcs">
<summary><strong>Linear elasticity 2D – plate, soft BCs</strong></summary>

DeepXDE example: [`elasticity_plate.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/elasticity_plate.py)

$$\Large \frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} = f_x, \qquad \frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} = f_y$$

- **Same problem** as the plate above (constants `lmbd`, `mu`, `Q`, five outputs and equations)
- **Boundary conditions as rows** (DeepXDE's `BC_type = \"soft\"`): $u_x = 0$ at top and bottom; $u_y = 0$ at left, bottom and right; $\sigma_{xx} = 0$ at left and right; $\sigma_{yy} = (\lambda + 2\mu)Q\sin(\pi x)$ at the top
- **Constants:** `lmbd = 1`, `mu = 0.5`, `Q = 4`
- **Network:** PFNN, a separate [40, 40, 40, 40] subnetwork for each of the 5 outputs, tanh, Glorot uniform
- **Training:** Adam 20,000 → L-BFGS 20,000; 500 domain, 500 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Linear elasticity 2D – plate, soft BCs**

**Note — more training than DeepXDE's script.** As for the plate above, DeepXDE's 5,000 Adam iterations are not enough ($L_2$ errors 2 × 10⁻²–8 × 10⁻² with exactly those settings), so this example trains **20,000 Adam + 20,000 L-BFGS iterations** (about 7 minutes on an RTX 4090); the results below are for these settings. With soft boundary conditions the plate trains more easily than with the hard output transform.

**Result (RTX 4090):** $L_2$ = 4.9 × 10⁻⁴ ($u_x$), 1.1 × 10⁻³ ($u_y$), 6.4 × 10⁻⁴ ($\sigma_{xx}$), 9.6 × 10⁻⁴ ($\sigma_{yy}$), 2.7 × 10⁻³ ($\sigma_{xy}$) — 20,000 Adam + 20,000 L-BFGS (DeepXDE's 5,000 Adam alone: 2.4 × 10⁻² to 7.8 × 10⁻²)

*ux — PINN vs reference, error analysis*

![Linear elasticity 2D – plate, soft BCs — PINN vs reference, error analysis (ux)](assets/Images/DeepXDE/Linear_elastostatic_plate_soft_BCs/surface_comparison_ux.png)

*uy — PINN vs reference, error analysis*

![Linear elasticity 2D – plate, soft BCs — PINN vs reference, error analysis (uy)](assets/Images/DeepXDE/Linear_elastostatic_plate_soft_BCs/surface_comparison_uy.png)

*sxx — PINN vs reference, error analysis*

![Linear elasticity 2D – plate, soft BCs — PINN vs reference, error analysis (sxx)](assets/Images/DeepXDE/Linear_elastostatic_plate_soft_BCs/surface_comparison_sxx.png)

*syy — PINN vs reference, error analysis*

![Linear elasticity 2D – plate, soft BCs — PINN vs reference, error analysis (syy)](assets/Images/DeepXDE/Linear_elastostatic_plate_soft_BCs/surface_comparison_syy.png)

*sxy — PINN vs reference, error analysis*

![Linear elasticity 2D – plate, soft BCs — PINN vs reference, error analysis (sxy)](assets/Images/DeepXDE/Linear_elastostatic_plate_soft_BCs/surface_comparison_sxy.png)

</details>

<details id="dx-kovasznay-flow">
<summary><strong>Navier-Stokes 2D – Kovasznay flow</strong></summary>

DeepXDE example: [`Kovasznay_flow.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Kovasznay_flow.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html))

$$\Large (\mathbf u\cdot\nabla)\mathbf u + \nabla p = \frac{1}{Re}\nabla^2\mathbf u, \qquad \nabla\cdot\mathbf u = 0, \qquad Re = 20$$

- **Domain:** $[-0.5, 1] \times [-0.5, 1.5]$; outputs $u, v, p$
- **Boundary conditions:** Dirichlet for $u$ and $v$ on the whole boundary, and for $p$ on the outlet $x = 1$
- **Exact solution:** $u = 1 - e^{\lambda x}\cos(2\pi y)$, $v = \frac{\lambda}{2\pi}e^{\lambda x}\sin(2\pi y)$, $p = \frac{1}{2}(1 - e^{2\lambda x})$, $\lambda = \frac{Re}{2} - \sqrt{\frac{Re^2}{4} + 4\pi^2}$
- **Constants:** `Re = 20`, `nu = 1/Re`, `l = 1/(2*nu) - sqrt(1/(4*nu**2) + 4*pi**2)`
- **Network:** FNN [2, 50, 50, 50, 50, 3], tanh, Glorot normal
- **Training:** Adam 30,000 → L-BFGS 15,000; 2,601 domain, 400 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Navier-Stokes 2D – Kovasznay flow**

**Result (RTX 4090):** $L_2$ = 2.1 × 10⁻⁴ ($u$), 9.0 × 10⁻⁴ ($v$), 5.4 × 10⁻⁴ ($p$)

*u — PINN vs reference, error analysis*

![Navier-Stokes 2D – Kovasznay flow — PINN vs reference, error analysis (u)](assets/Images/DeepXDE/Kovasznay_flow/surface_comparison_u.png)

*p — PINN vs reference, error analysis*

![Navier-Stokes 2D – Kovasznay flow — PINN vs reference, error analysis (p)](assets/Images/DeepXDE/Kovasznay_flow/surface_comparison_p.png)

</details>

<details id="2d-poisson-disk">
<summary><strong>Poisson 2D – disk</strong></summary>

The Poisson equation on the unit disk, steady-state (no time axis).

$$\Large -\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} = 1, \qquad x^2 + y^2 \le 1$$

- **Boundary conditions:** Dirichlet, $u = 0$ on the boundary circle
- **Geometry:** Disk, center $(0,0)$, radius $1$
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Poisson 2D – disk**

A companion to the L-Shape template above, on a smooth (curved, non-singular) boundary instead.

**Result (RTX 4090):** $L_2$ = 2.8 × 10⁻⁴, against the reference data in [`reference_data/2D/poisson_disk`](reference_data/2D/poisson_disk). Inverse mode: source term = 1.00002 (true 1).

![2D Poisson (Disk) — PINN vs Ground Truth, Error Analysis surface comparison](assets/Images/2D-Poisson-Disk-Shaped/surface_comparison.png)

</details>

<details id="dx-poisson-over-an-l-shaped-domain">
<summary><strong>Poisson 2D – L-shape</strong></summary>

DeepXDE example: [`Poisson_Lshape.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Lshape.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.Lshape.html))

$$\Large -\nabla^2 u = 1, \qquad \Omega = [-1, 1]^2 \setminus [0, 1]\times[-1, 0]$$

- **Boundary conditions:** Dirichlet, $u = 0$ on the whole boundary
- **Geometry:** Polygon (L-shape)
- **Network:** FNN [2, 50, 50, 50, 50, 1], tanh, Glorot uniform
- **Training:** Adam 50,000 → L-BFGS 15,000; 1,200 domain, 120 boundary points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Poisson 2D – L-shape**

**Result (RTX 4090):** $L_2$ = 3.5 × 10⁻²

*PINN vs reference, error analysis*

![Poisson 2D – L-shape — PINN vs reference, error analysis](assets/Images/DeepXDE/Poisson_over_an_L_shaped_domain/surface_comparison.png)

</details>

<details id="2d-poisson-l-shape">
<summary><strong>Poisson 2D – L-shape, Lu et al.</strong></summary>

A classic re-entrant-corner benchmark — the Poisson equation on an L-shaped domain (a unit square with a quadrant notched out), steady-state (no time axis). This is the same benchmark used as a demonstration example in the original DeepXDE paper — see Lu, Meng, Mao & Karniadakis (2021) under [References](#references).

$$\Large -\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} = 1, \qquad (x, y) \in \Omega_{L}$$

- **Boundary conditions:** Dirichlet, $u = 0$ on the whole boundary
- **Geometry:** Polygon, vertices $(0,0), (1,0), (1,-1), (-1,-1), (-1,1), (0,1)$
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Poisson 2D – L-shape, Lu et al.**

The re-entrant corner at the origin produces a solution singularity that's a standard stress-test for numerical solvers, PINNs included.

**Result (RTX 4090):** $L_2$ = 3.5 × 10⁻², against the reference data in [`reference_data/2D/poisson_lshape`](reference_data/2D/poisson_lshape). Inverse mode: source term = 0.99987 (true 1).

![2D Poisson (L-Shape) — PINN vs Ground Truth, Error Analysis surface comparison](assets/Images/2D-Poisson-L-Shaped/surface_comparison.png)

</details>

<details id="dx-inverse-fractional-poisson-equation-2d">
<summary><strong>Fractional Poisson 2D – unknown order α</strong></summary>

DeepXDE example: [`fractional_Poisson_2d_inverse.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/fractional_Poisson_2d_inverse.py)

$$\Large -(-\Delta)^{\alpha/2}u = 2^{\alpha_0}\Gamma(2 + \tfrac{\alpha_0}{2})\Gamma(1 + \tfrac{\alpha_0}{2})\left(1 - (1 + \tfrac{\alpha_0}{2})|x|^2\right), \qquad |x| < 1$$

- **Unknown:** the order $\alpha$ (true 1.8, start 1.5); $C(\alpha, 2)$ written with `lgamma` so it stays differentiable in α
- **Data:** 30 random measurements of $u = (1 - |x|^2)^{1 + \alpha_0/2}$, loss weight 100
- **Fractional settings:** 8 directions, 100 points per unit length, dynamic mesh
- **Constants:** `a0 = 1.8`
- **Network:** FNN [2, 20, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 10,000; 64 domain points
- **In PINNStudio:** Quick Examples → Category **Inverse** → **Fractional Poisson 2D – unknown order α**

**Result (RTX 4090):** $\alpha$ = 1.805 (true 1.8); $L_2$ of $u$ = 2.2 × 10⁻³

*PINN vs reference, error analysis*

![Fractional Poisson 2D – unknown order α — PINN vs reference, error analysis](assets/Images/DeepXDE/Inverse_fractional_Poisson_equation_2D/surface_comparison.png)

*Inferred parameters during training*

![Fractional Poisson 2D – unknown order α — inferred parameters during training](assets/Images/DeepXDE/Inverse_fractional_Poisson_equation_2D/param_plot.png)

</details>

<details id="2d-allen-cahn-mattey--ghosh">
<summary><strong>Allen-Cahn 2D – Mattey & Ghosh</strong></summary>

Benchmark problem after Mattey & Ghosh (2022) — see [References](#references).

$$\Large \frac{\partial u}{\partial t} = c_1^2\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) - c_2(u^3 - u), \qquad (x, y) \in [0, 1]^2,\ t \in (0, 1]$$

where $c_1^2 = 0.0001$ and $c_2 = 1$.

- **Initial condition:** $u(x, y, 0) = \sin(4\pi x)\cos(4\pi y)$
- **Boundary conditions:** Periodic
- **Geometry:** Rectangle
- **Time-Adaptive default:** on — $t \in [0,1]$ split into 4 steps of 0.25, L-BFGS transfer learning between steps
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Allen-Cahn 2D – Mattey & Ghosh**

**Result (RTX 4090):** $L_2$ = 6.0 × 10⁻³ ($t = 0$), 5.6 × 10⁻³ ($t = 0.25$), 5.5 × 10⁻³ ($t = 0.5$), 5.6 × 10⁻³ ($t = 0.75$), 6.7 × 10⁻³ ($t = 1$), against the reference data in [`reference_data/2D/allen_cahn_mattey`](reference_data/2D/allen_cahn_mattey). Inverse mode: $c_2$ = 1.050 (true 1); $c_1^2$ = 1.66 × 10⁻⁴ (true 1 × 10⁻⁴) is not recovered — the small diffusion coefficient is only weakly determined by the data, as in the 1D Allen-Cahn template.

![2D Allen-Cahn (Mattey & Ghosh) — PINN vs Ground Truth, Error Analysis surface comparison](assets/Images/2D_Allen_Cahn_Mattey/surface_comparison.png)

*Restore & Visualize* — the trained model restored from its saved checkpoint and re-animated without retraining:

![2D Allen-Cahn (Mattey & Ghosh) — restored solution animation](assets/Images/2D_Allen_Cahn_Mattey/restored_animation.gif)

*Inverse mode* — same template, switched to **Inverse**, jointly recovering both $c_1^2$ (interfacial-thickness-squared) and $c_2$ (reaction) as unknown parameters; $c_2$'s initial guess starts at $0.1$ rather than $1.0$, since $1.0$ is already its true value. The live convergence animation below shows both estimates settling onto their true values during training:

![2D Allen-Cahn (Mattey & Ghosh) inverse parameter convergence animation](assets/Images/2D_Allen_Cahn_Mattey/param_convergence_animation.gif)

</details>

<details id="2d-allen-cahn-wight--zhao">
<summary><strong>Allen-Cahn 2D – Wight & Zhao</strong></summary>

Benchmark problem after Wight & Zhao (2021) — see [References](#references). This is the paper's own shrinking-circle-interface benchmark, posed there for a field it calls $\phi$ (PINNStudio's $u$) with rate constant $\lambda$ and interface-width parameter $\varepsilon$ (eq. 3.11):

$$\Large \frac{\partial u}{\partial t} = \lambda\left(\varepsilon^2\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) - u^3 + u\right), \qquad (x, y) \in [0, 1]^2,\ t \in [0, 10]$$

where $\lambda = 10$ and $\varepsilon = 0.025$, matching the paper's own values ($\lambda\varepsilon^2 = 0.00625$ is the effective diffusion coefficient).

- **Initial condition:** a smooth circular interface, $u(x, y, 0) = \tanh\left(\dfrac{0.35 - \sqrt{(x-0.5)^2 + (y-0.5)^2}}{2\varepsilon}\right)$ (paper's eq. 3.12)
- **Boundary conditions:** Periodic
- **Geometry:** Rectangle
- **Time-Adaptive default:** on — $t \in [0,10]$ is a wide window for a single pass, so this template splits it into 10 steps of 1 (`0→1, 1→2, ..., 9→10`), L-BFGS transfer learning between steps
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Allen-Cahn 2D – Wight & Zhao**

**Result (RTX 4090):** $L_2$ = 3.1 × 10⁻⁴ ($t = 0$), 9.3 × 10⁻³ ($t = 2.5$), 2.1 × 10⁻² ($t = 5$), 7.9 × 10⁻³ ($t = 10$), against the reference data in [`reference_data/2D/allen_cahn_wight`](reference_data/2D/allen_cahn_wight). Inverse mode: $\lambda$ = 9.68 (true 10), $\lambda\varepsilon^2$ = 6.07 × 10⁻³ (true 6.25 × 10⁻³).

![2D Allen-Cahn (Wight & Zhao) — PINN vs Ground Truth, Error Analysis surface comparison](assets/Images/2D_AllenCahn-Wight/surface_comparison.png)

*Restore & Visualize* — the trained model restored from its saved checkpoint and re-animated without retraining:

![2D Allen-Cahn (Wight & Zhao) — restored solution animation](assets/Images/2D_AllenCahn-Wight/restored_animation.gif)

*Inverse mode* — same template, switched to **Inverse**, jointly recovering the two coefficients exactly as they appear in the PDE box — the reaction-term coefficient $\lambda$ and the lumped diffusion-term coefficient $\lambda\varepsilon^2$ (not $\varepsilon$ on its own):

$$\Large \frac{\partial u}{\partial t} = (\lambda\varepsilon^2)\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) - \lambda(u^3 - u)$$

with true values $\lambda = 10$ and $\lambda\varepsilon^2 = 0.00625$. The live convergence animation below shows both estimates settling onto their true values during training:

![2D Allen-Cahn (Wight & Zhao) inverse parameter convergence animation](assets/Images/2D_AllenCahn-Wight/param_convergence_animation.gif)

</details>

<details id="2d-burgers-mathias">
<summary><strong>Burgers 2D – Mathias et al.</strong></summary>

Physics after Mathias, de Almeida, de Barros, Coelho, et al. (2022) — see [References](#references). A coupled, two-output system for the velocity components $U$, $V$, with kinematic viscosity $\nu = \dfrac{0.01}{\pi}$:

$$\Large \frac{\partial U}{\partial t} + U\frac{\partial U}{\partial x} + V\frac{\partial U}{\partial y} = \nu\left(\frac{\partial^2 U}{\partial x^2} + \frac{\partial^2 U}{\partial y^2}\right)$$

$$\Large \frac{\partial V}{\partial t} + U\frac{\partial V}{\partial x} + V\frac{\partial V}{\partial y} = \nu\left(\frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2}\right)$$

$(x, y) \in [0, 1]^2$, $t \in [0, 1]$

- **Initial condition:** $U(x,y,0) = \sin(2\pi x)\sin(2\pi y)$, $\ V(x,y,0) = \sin(\pi x)\sin(\pi y)$
- **Boundary conditions:** Dirichlet, $U = V = 0$ on all four edges
- **Geometry:** Rectangle
- **Network:** 4 hidden layers × 128 neurons — wider than PINNStudio's global default (3 × 64), which gave better results for this coupled two-output velocity field
- **Time-Adaptive default:** on — $t \in [0,1]$ split into 4 steps of 0.25, L-BFGS transfer learning between steps
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Burgers 2D – Mathias et al.**

This template reproduces the paper's own PDE, domain, and initial/boundary conditions with PINNStudio's standard soft-constrained loss and plain MLP network — not the paper's own sparse-data augmentation or hard-constrained output layer, which are outside this template's scope.

**Result (RTX 4090):** $L_2$ of $U$ = 6.0 × 10⁻⁴ ($t = 0$), 1.9 × 10⁻² ($t = 0.25$), 1.1 × 10⁻² ($t = 0.5$), 1.1 × 10⁻² ($t = 0.75$), 1.1 × 10⁻² ($t = 1$); of $V$ = 4.3 × 10⁻⁴ ($t = 0$), 9.0 × 10⁻³ ($t = 0.25$), 7.5 × 10⁻³ ($t = 0.5$), 5.1 × 10⁻³ ($t = 0.75$), 4.7 × 10⁻³ ($t = 1$), against the reference data in [`reference_data/2D/burgers_mathias`](reference_data/2D/burgers_mathias).

PINN solution animated over time, for each of the two coupled outputs:

<table>
<tr>
<td width="50%" align="center">
<img src="assets/Images/2D_Burgers/solution_plot_u.gif" alt="2D Burgers (Mathias) — U solution animated over time" width="100%">
<sub><strong>U</strong></sub>
</td>
<td width="50%" align="center">
<img src="assets/Images/2D_Burgers/solution_plot_v.gif" alt="2D Burgers (Mathias) — V solution animated over time" width="100%">
<sub><strong>V</strong></sub>
</td>
</tr>
</table>

Error Analysis — PINN vs Ground Truth, for each output:

<table>
<tr>
<td width="50%" align="center">
<img src="assets/Images/2D_Burgers/surface_comparison_u.png" alt="2D Burgers (Mathias) — U surface comparison" width="100%">
<sub><strong>U</strong></sub>
</td>
<td width="50%" align="center">
<img src="assets/Images/2D_Burgers/surface_comparison_v.png" alt="2D Burgers (Mathias) — V surface comparison" width="100%">
<sub><strong>V</strong></sub>
</td>
</tr>
</table>

*Inverse mode* — same template, switched to **Inverse**, recovering the shared viscosity $\nu$ (true value $0.01/\pi \approx 3.18 \times 10^{-3}$, initial guess 1) from $U$ and $V$ measured at $t = 1$, each against its own output. $\nu$ converges to 3.44 × 10⁻³, and the $L_2$ error at $t = 1$ is 3.7 × 10⁻³ for $U$ and 3.1 × 10⁻³ for $V$:

![2D Burgers (Mathias) inverse parameter convergence animation](assets/Images/2D_Burgers/param_convergence_animation.gif)

</details>

<details id="2d-heat">
<summary><strong>Heat 2D</strong></summary>

$$\Large \frac{\partial u}{\partial t} = 0.4\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right), \qquad (x, y) \in [0, 1]^2,\ t \in [0, 1]$$

- **Initial condition:** $u(x, y, 0) = 0$
- **Boundary conditions:** Dirichlet $u = 1$ on the $x = 1$ edge; Neumann (insulated) on the other three
- **Geometry:** Rectangle
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Heat 2D**

**Result (RTX 4090):** $L_2$ = 9.1 × 10⁻³ ($t = 0.25$), 1.0 × 10⁻² ($t = 0.5$), 7.9 × 10⁻³ ($t = 0.75$), 3.9 × 10⁻³ ($t = 1$), against the reference data in [`reference_data/2D/heat`](reference_data/2D/heat) ($t = 0$ is left out: the initial condition is $u = 0$, so a relative error there divides by almost nothing). Inverse mode: diffusivity = 0.4005 (true 0.4).

![2D Heat — PINN vs Ground Truth, Error Analysis surface comparison](assets/Images/2D-Heat/surface_comparison.png)

*Restore & Visualize* — the trained model restored from its saved checkpoint and re-animated without retraining:

![2D Heat — restored solution animation](assets/Images/2D-Heat/restored_animation_2D_Heat.gif)

*Inverse mode* — same template, switched to **Inverse**, recovering the diffusivity coefficient (fixed at $0.4$ in Forward mode above) as an unknown parameter; the live convergence animation below shows the estimate settling onto its true value during training:

![2D Heat inverse parameter convergence animation](assets/Images/2D-Heat/param_convergence_animation.gif)

</details>

### 3D examples

*Click an example to expand it.*

<details id="dx-fractional-poisson-equation-3d">
<summary><strong>Fractional Poisson 3D – sphere</strong></summary>

DeepXDE example: [`fractional_Poisson_3d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_3d.py)

$$\Large -(-\Delta)^{\alpha/2}u = \frac{2^{\alpha}\Gamma(2 + \frac{\alpha}{2})\Gamma(\frac{3 + \alpha}{2})}{\Gamma(\frac{3}{2})}\left(1 - (1 + \tfrac{\alpha}{3})|x|^2\right), \qquad |x| < 1,\ \alpha = 1.8$$

- **Fractional settings:** 8 × 8 directions, 100 points per unit length, dynamic mesh; constant $C(\alpha, 3)$ written with `gamma`
- **Geometry:** unit sphere; $u = 0$ on the sphere through $\hat u = (1 - |x|^2)N$
- **Exact solution:** $u = (1 - |x|^2)^{1 + \alpha/2}$
- **Constants:** `alpha = 1.8`
- **Network:** FNN [3, 20, 20, 20, 20, 1], tanh, Glorot normal
- **Training:** Adam 10,000; 256 domain points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Fractional Poisson 3D – sphere**

**Result (RTX 4090):** $L_2$ = 8.2 × 10⁻³

*PINN vs reference, error analysis*

![Fractional Poisson 3D – sphere — PINN vs reference, error analysis](assets/Images/DeepXDE/Fractional_Poisson_equation_3D/surface_comparison.png)

</details>

<details id="3d-poisson-sphere">
<summary><strong>Poisson 3D – sphere</strong></summary>

The Poisson equation on the unit ball, steady-state (no time axis) — the 3D companion to the 2D Poisson (Disk) template above.

$$\Large -\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} - \frac{\partial^2 u}{\partial z^2} = 1, \qquad x^2 + y^2 + z^2 \le 1$$

- **Boundary conditions:** Dirichlet, $u = 0$ on the boundary sphere
- **Geometry:** Sphere, center $(0,0,0)$, radius $1$
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Poisson 3D – sphere**

**Result (RTX 4090):** $L_2$ = 1.5 × 10⁻³, against the reference data in [`reference_data/3D/poisson_sphere`](reference_data/3D/poisson_sphere). Inverse mode: source term = 1.00005 (true 1).

![3D Poisson (Sphere) — PINN vs Ground Truth, Error Analysis surface comparison](assets/Images/3D-Poisson-Sphere/surface_comparison.png)

*Inverse mode* — same template, switched to **Inverse**, recovering the source term (fixed at $1$ in Forward mode above) as an unknown parameter; the live convergence plot below shows the estimate settling onto its true value during training:

![3D Poisson (Sphere) inverse source-term parameter convergence](assets/Images/3D-Poisson-Sphere/sourceterm_inverse_plot.png)

</details>

<details id="3d-heat">
<summary><strong>Heat 3D</strong></summary>

The 2D Heat problem extended with a $z$ axis — diffusion in a unit cube.

$$\Large \frac{\partial u}{\partial t} = 0.4\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\right), \qquad (x, y, z) \in [0, 1]^3,\ t \in [0, 1]$$

- **Initial condition:** $u(x, y, z, 0) = 0$
- **Boundary conditions:** Dirichlet $u = 1$ on the $x = 1$ face; Neumann (insulated) on the other five
- **Geometry:** Cuboid
- **In PINNStudio:** Quick Examples → Category **Forward** (or **Inverse**) → **Heat 3D**

Time-Adaptive Training is fully supported for this template if you turn it on — see the note on 3D grid resolution under [Time-Adaptive training](#training) in Features before picking a large "IC grid resolution" value.

**Result (RTX 4090):** $L_2$ = 9.0 × 10⁻³ ($t = 0.2$), 5.7 × 10⁻³ ($t = 0.4$), 3.7 × 10⁻³ ($t = 0.6$), 2.8 × 10⁻³ ($t = 0.8$), 2.6 × 10⁻³ ($t = 1$), against the reference data in [`reference_data/3D/heat`](reference_data/3D/heat) ($t = 0$ left out, as for Heat 2D). Inverse mode: diffusivity = 0.4001 (true 0.4).

![3D Heat solution surface](assets/Images/3D_Heat/3D_Heat_Solution_Surface.png)

*Restore & Visualize* — the trained model restored from its saved checkpoint and re-animated without retraining:

![3D Heat — restored solution animation](assets/Images/3D_Heat/restored_animation.gif)

*Inverse mode* — same template, switched to **Inverse**, recovering the diffusivity coefficient (fixed at $0.4$ in Forward mode above) as an unknown parameter; the live convergence animation below shows the estimate settling onto its true value during training:

![3D Heat inverse parameter convergence animation](assets/Images/3D_Heat/param_convergence_animation.gif)

</details>

<details id="dx-beltrami-flow">
<summary><strong>Navier-Stokes 3D – Beltrami flow</strong></summary>

DeepXDE example: [`Beltrami_flow.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Beltrami_flow.py)

$$\Large \frac{\partial \mathbf u}{\partial t} + (\mathbf u\cdot\nabla)\mathbf u + \nabla p = \nabla^2\mathbf u, \qquad \nabla\cdot\mathbf u = 0, \qquad (x, y, z) \in [-1, 1]^3,\ t \in [0, 1]$$

- **Outputs:** $u, v, w, p$; Dirichlet BCs and initial conditions for the velocity (the exact Beltrami solution)
- **Loss weights:** 1 for the four equations, 100 for the BCs and ICs
- **Exact solution:** e.g. $u = -\left(e^{x}\sin(y + z) + e^{z}\cos(x + y)\right)e^{-t}$
- **Network:** FNN [4, 50, 50, 50, 50, 4], tanh, Glorot normal
- **Training:** Adam 30,000 → L-BFGS 15,000; 50,000 domain, 5,000 boundary, 5,000 initial points
- **In PINNStudio:** Quick Examples → Category **Forward** → **Navier-Stokes 3D – Beltrami flow**

**Why the pressure error is large.** This example is DeepXDE's script without changes: like the script, it gives boundary and initial conditions for the velocity $u, v, w$ only, and none for $p$. The pressure enters the equations only through its gradient $\nabla p$, so $p$ and $p + C$ satisfy them equally well for any constant $C$ — the training has nothing that fixes $C$. The PINN's pressure has the right shape but is shifted by a constant (about 24.7 in this run). The exact pressure decays like $e^{-2t}$ while the shift stays the same, so the relative error of $p$ grows with time: 5.6 at $t = 0$, $5.6\ e^{1} \approx 15$ at $t = 0.5$ and $5.6\ e^{2} \approx 41$ at $t = 1$ — exactly the reported values. DeepXDE's script computes the pressure error too but does not print it; it reports only $u, v, w$. Fixing $p$ at a single point (a Point Set BC row on $p$) would remove the free constant.

**Result (RTX 4090):** $L_2$ at $t = 0, 0.5, 1$: $u$ 1.9 × 10⁻⁴, 3.5 × 10⁻⁴, 1.4 × 10⁻³; $v$ 1.8 × 10⁻⁴, 2.7 × 10⁻⁴, 7.3 × 10⁻⁴; $w$ 1.9 × 10⁻⁴, 3.6 × 10⁻⁴, 1.2 × 10⁻³; $p$ 5.6, 15, 41 (a constant offset of about 24.7, see above)

*u — PINN solution over time*

![Navier-Stokes 3D – Beltrami flow — PINN solution over time (u)](assets/Images/DeepXDE/Beltrami_flow/animation_u.gif)

*v — PINN solution over time*

![Navier-Stokes 3D – Beltrami flow — PINN solution over time (v)](assets/Images/DeepXDE/Beltrami_flow/animation_v.gif)

*w — PINN solution over time*

![Navier-Stokes 3D – Beltrami flow — PINN solution over time (w)](assets/Images/DeepXDE/Beltrami_flow/animation_w.gif)

*p — PINN solution over time*

![Navier-Stokes 3D – Beltrami flow — PINN solution over time (p)](assets/Images/DeepXDE/Beltrami_flow/animation_p.gif)

*u — PINN vs reference, error analysis*

![Navier-Stokes 3D – Beltrami flow — PINN vs reference, error analysis (u)](assets/Images/DeepXDE/Beltrami_flow/surface_comparison_u.png)

*v — PINN vs reference, error analysis*

![Navier-Stokes 3D – Beltrami flow — PINN vs reference, error analysis (v)](assets/Images/DeepXDE/Beltrami_flow/surface_comparison_v.png)

*w — PINN vs reference, error analysis*

![Navier-Stokes 3D – Beltrami flow — PINN vs reference, error analysis (w)](assets/Images/DeepXDE/Beltrami_flow/surface_comparison_w.png)

*p — PINN vs reference, error analysis*

![Navier-Stokes 3D – Beltrami flow — PINN vs reference, error analysis (p)](assets/Images/DeepXDE/Beltrami_flow/surface_comparison_p.png)

</details>

> **Tip:** Accuracy can generally be improved by refining the time discretization — use more, smaller **Time Adaptive** step groups (a finer time step per phase) rather than one large training pass, or increase collocation points for finer spatial/adaptive refinement of the residual. Several templates above already default to Time-Adaptive for this reason; add or adjust step groups for any template — in any dimension — from the *Adaptive Training* panel.

## Features

### Problem Setup

<details>
<summary><strong>ODE, 1D, 2D, and 3D problems</strong></summary>

Problems are defined over `t` (ODEs), `(x, t)`, `(x, y, t)` or `(x, y, z, t)`, or without the time axis for steady-state problems in `x`, `(x, y)` or `(x, y, z)`. Changing the dimension rebuilds the related controls (geometry, boundary-condition rows, plot options), so no setting from the previous dimension is carried over by mistake.

**ODEs** have time as the only variable: derivatives are written `du_t`, `du_tt` (one equation per output for a system, e.g. `dy1_t - y2|dy2_t + y1`), initial values are set in the Initial Conditions panel, and *Start*/*End* in a boundary-condition row gives a condition at either end of the time range. An ODE can also contain integrals of the solution, `integral(f)` from t₀ to t with f written in the integration variable `s`, `t` and the outputs at s — for integro-differential equations such as `du_t + u - integral(exp(s - t)*u)`.

**Fractional equations**: `frac(u)` is the fractional operator of order α applied to output u (DeepXDE's FPDE discretization — in 1D the sum of the left and right Riemann–Liouville derivatives, in 2D/3D the integral of the directional derivative over all directions). The fractional settings (α, resolution, mesh) appear under the equations as soon as `frac(...)` is used, and α can be an unknown to learn from data. See the ODE and fractional examples under [Examples](#examples).

</details>

<details>
<summary><strong>Forward and inverse problems</strong></summary>

**Forward**: every coefficient is known and PINNStudio solves for the field. **Inverse**: one or more coefficients — or a whole unknown field — are learned from measured data while the field is solved for at the same time. Any number of unknowns can be learned in one problem, each with its own initial guess and, when it is known, its true value for comparison. Measured data comes from one or more files, each with its own loss weight, and can be a single output or any expression of the outputs (e.g. $|h| = \sqrt{u^2 + v^2}$).

Every PINNStudio template, and several DeepXDE examples, have an Inverse mode that loads with its measured-data file already set — pick the example from the **Inverse** list of Quick Examples.

</details>

<details>
<summary><strong>Equation editor for single equations and coupled systems</strong></summary>

Equations are typed as residuals, not chosen from presets. Derivatives use a plain `d<output>_<variables>` naming — `du_x`, `du_xx`, `du_t`, `du_xy`, up to mixed fourth-order terms such as `du_xxyy` — so anything from a diffusion equation to a coupled higher-order system can be written directly. Coupled systems work the same way: a second output `v` brings `dv_x`, `dv_t` and so on (the *Burgers 2D – Mathias et al.* and *Schrödinger 1D* examples are two-output systems).

The coordinates `x`, `y`, `z` and `t` can appear in an equation along with standard functions (`sin`, `exp`, `cosh`, ...), so source terms such as `-du_xx - pi**2*sin(pi*x)` are written as they read. The number of **equations** is set separately from the number of **outputs**: for example, three residuals for one output (DeepXDE's American put example), or one equation for two outputs when the second output is an unknown field to be learned.

**Constants** give names to numbers used across the problem — e.g. `Re = 20`, `nu = 1/Re` — and can be used in the equations, boundary and initial conditions, transforms and the exact-solution reference.

</details>

<details>
<summary><strong>Boundary and initial conditions</strong></summary>

Boundary conditions are rows in a panel, each with its type, the output(s) it applies to, and **where** it applies:

- **Types:** Dirichlet, Neumann, Robin, Periodic, Point Set (values at given points), Point Set Operator (an expression of the outputs and their derivatives at given points), Operator (any expression, applied on the boundary or at the initial time), and Interface 2D (DeepXDE's `Interface2DBC`, pairing two matching boundary pieces).
- **Where:** *Whole boundary*, *Left*, *Right*, *Bottom*, *Top* (and *Front*/*Back* in 3D), or any `True`/`False` expression in `x`, `y`, `z`. The side names and the domain limits (`x_min`, `x_max`, ...) follow your domain, so `left or right` or `x <= x_min` stay correct if the domain changes; a curved boundary piece uses an expression such as `np.isclose(x**2 + y**2, 0.25)`.
- Neumann and Robin values are the derivative along the outward normal, $du/dn$ ($+du/dx$ on a right side, $-du/dx$ on a left side).
- Operator conditions use the same `u`, `du_x`, `du_xx` notation as the equations. Applied at the initial time, they give a second initial condition such as $\partial u/\partial t = 0$ at $t = 0$ for a wave equation.
- Point Set rows take their points from a file, a typed list (`1`, `0.2, 0.5`, `0,0; 1,0.5`) or N random points, with values from the file or a formula; they can target several outputs at once and can be minibatched.

Initial conditions are given per output as an expression or loaded from a data file. Boundary and initial conditions can also be imposed exactly through an output transform (see *Input and output transforms* below).

</details>

<details>
<summary><strong>Geometry and domains</strong></summary>

1D problems use an interval; 2D problems a **Rectangle, Disk, Ellipse, Triangle** or any **Polygon** (given by its vertices); 3D problems a **Cuboid** or **Sphere**. Domain bounds accept expressions such as `pi/2`.

**Custom** geometry (2D and 3D) combines any number of these shapes with **Union**, **Subtract** and **Intersect** — for example a plate with a hole, a `Rectangle` with a `Disk` subtracted:

<p align="center">
<img src="assets/Images/View_Domain_Pictures/Custom_Geometry_Rectangle_minus_Disk.png" alt="PINNStudio — Custom geometry domain preview, Rectangle with a Disk subtracted out" width="55%">
</p>

</details>

<details>
<summary><strong>Collocation points and domain preview</strong></summary>

The numbers of domain, boundary, initial and test points are set independently, along with the sampling distribution (Hammersley, uniform, Halton, LHS, Sobol or pseudo-random). A domain preview shows the actual training points — domain, boundary and initial points, color-coded — on the geometry, with a second panel showing their distribution in time, in 2D and in 3D:

<table>
<tr>
<td width="33%" align="center">
<img src="assets/Images/View_Domain_Pictures/2D_Rectangle_combined.png" alt="PINNStudio — 2D domain preview, Rectangle geometry" width="100%">
<sub>2D preview — Rectangle</sub>
</td>
<td width="33%" align="center">
<img src="assets/Images/View_Domain_Pictures/2D_Disk_combined.png" alt="PINNStudio — 2D domain preview, Disk geometry" width="100%">
<sub>2D preview — Disk</sub>
</td>
<td width="33%" align="center">
<img src="assets/Images/View_Domain_Pictures/3D_Cuboid_combined.png" alt="PINNStudio — 3D domain preview, Cuboid geometry" width="100%">
<sub>3D preview — Cuboid</sub>
</td>
</tr>
</table>

**Extra training points (anchors)** can be added on top of the sampled ones, from a file or at random inside a box — for example to concentrate points near a front or a boundary layer.

</details>

### Training

<details>
<summary><strong>Network architecture</strong></summary>

The number of hidden layers, neurons per layer, activation function (tanh, sin, Sigmoid, ReLU, SiLU/Swish, ELU, GELU, SELU) and weight initializer (Glorot or He, uniform or normal) are all set in the panel. Four network types are available:

- **FNN** — a fully connected network shared by all outputs (the default).
- **PFNN** — a separate sub-network for each output, useful when outputs have very different scales or behavior.
- **MsFFN** and **STMsFFN** — multi-scale Fourier feature networks, which pass the inputs through random Fourier features at several scales (e.g. `1, 10`) so the network can capture slow and fast variations together.

</details>

<details>
<summary><strong>Input and output transforms</strong></summary>

Inputs and outputs can be rescaled ($x \mapsto x \times \text{scale} + \text{shift}$), which helps when a coordinate or the solution is far from order one. Each transform can also be a **formula**: an input formula sets the network's input features (e.g. `x*sin(y); x*cos(y)` for polar coordinates), and an output formula rewrites each output — typically to satisfy boundary or initial conditions exactly, e.g. `x*(1 - x)*u` gives $u = 0$ at $x = 0$ and $x = 1$ without a boundary loss term.

</details>

<details>
<summary><strong>Training phases, learning-rate schedules and loss weights</strong></summary>

Training is a sequence of **phases**, each with its own optimizer (Adam, AdamW, L-BFGS or NNCG), number of iterations, learning rate and loss weights — for example the common Adam-then-L-BFGS recipe, or several Adam phases with decreasing learning rates. Each phase's learning rate can decay by a step, cosine, exponential or inverse-time schedule. L-BFGS uses DeepXDE's default options unless you set your own, and the run's precision can be set to float64 when L-BFGS needs the extra precision to converge (float32 is faster).

Loss weights can be set per term (each equation, boundary condition, initial condition and measured-data file) or automatically from the first evaluation (**Scale by initial losses**, $w_i = c / L_i(0)$), so that every term starts at the same size.

**IC-guided pre-training** first fits the network to the initial condition alone, before the equations are switched on, which helps problems where training from a random start struggles to find the right solution.

</details>

<details>
<summary><strong>Residual-based adaptive refinement (RAR)</strong></summary>

RAR trains in rounds. After each round it evaluates the equation residual at a large set of candidate points, adds the points where the residual is largest to the training set, and trains again — concentrating effort on the hardest parts of the domain, such as a shock or a thin interface. It works for steady and time-dependent problems in every dimension. For a system of equations, the ranking can use the combined residual or a single equation (*Points from*), and RAR can stop once the mean residual drops below a set value.

Every round saves a checkpoint and two figures to `solution_results/rar_rounds/round_NN/`: the solution at that round (compared with reference data when it is available) and the training points, showing the original points, those added in earlier rounds and those added in this round. Side by side they show the added points landing where the error is largest:

<table>
<tr>
<td width="50%" align="center">
<img src="assets/Images/1D-Burgers/rar_collocation_points.png" alt="PINNStudio — Burgers 1D RAR round 2, collocation points: the points added this round (red) concentrate where the model is least accurate" width="100%">
<sub>Burgers 1D, RAR round 2 — points added (red) vs. earlier and original points</sub>
</td>
<td width="50%" align="center">
<img src="assets/Images/1D-Burgers/rar_solution_plot.png" alt="PINNStudio — Burgers 1D RAR round 2, PINN vs Reference solution and error" width="100%">
<sub>Burgers 1D, RAR round 2 — solution and error: the added points line up with the largest error</sub>
</td>
</tr>
</table>

<table>
<tr>
<td width="50%" align="center">
<img src="assets/Images/2D_AllenCahn-Wight/rar_collocation_points.png" alt="PINNStudio — Allen-Cahn 2D (Wight & Zhao) RAR round 1, collocation points: the points added this round concentrate near the sharpest part of the interface" width="100%">
<sub>Allen-Cahn 2D (Wight & Zhao), RAR round 1 — points added vs. original points</sub>
</td>
<td width="50%" align="center">
<img src="assets/Images/2D_AllenCahn-Wight/rar_solution_plot.png" alt="PINNStudio — Allen-Cahn 2D (Wight & Zhao) RAR round 1, PINN vs Reference solution and error" width="100%">
<sub>Allen-Cahn 2D (Wight & Zhao), RAR round 1 — solution and error: the added points follow the largest error</sub>
</td>
</tr>
</table>

</details>

<details>
<summary><strong>Time-adaptive training</strong></summary>

Splits the time range into steps and trains through them in order, each step starting from the network of the previous one (transfer learning) and taking its initial condition from the previous step's solution. This suits problems over a long time window or with fast dynamics that a single network struggles to fit in one pass; the Allen-Cahn templates use it by default. It works in 1D, 2D and 3D, and each step is compared with any reference data that falls inside its time window.

The grid that passes the solution from one step to the next is set per axis (*IC grid resolution*), so its size grows with the dimension — PINNStudio picks a smaller default in 3D and warns when a chosen value would be very large.

</details>

<details>
<summary><strong>Training callbacks and monitors</strong></summary>

- **Point resampling** redraws the collocation points every N iterations.
- **Early stopping** ends training once the monitored loss stops improving.
- **Model checkpoints** save the model at regular intervals, or only when the loss improves.
- **Training timer** stops a run after a set wall-clock time.
- **Training monitors** track any expression of the outputs and their derivatives (`u`, `du_x`, a flux, ...) at chosen points during training, alongside the loss.

</details>

<details>
<summary><strong>Inverse parameters during training</strong></summary>

In an inverse problem, each unknown is saved at regular intervals throughout training, including the final value where training stops, and its convergence can be shown as a plot or an animation — with the true value drawn as a reference line when it is known, both after training and later in Restore & Visualize.

</details>

### Parameter Sweep

<details>
<summary><strong>Run many training configurations from one setup, varying any parameter</strong></summary>

Instead of changing a value and pressing Solve again for each case, **Enable Parameter Sweep** runs a whole set of configurations of the problem you have set up, one after another, each as a full training run. The Solve and Stop buttons drive the sweep, and the Training Log shows every run in turn.

![Parameter Sweep panel — Burgers 1D network-size sweep (hidden layers × neurons, All combinations) after it finished: each run's result in the Training Log, the last run's loss and solution on the right](assets/Images/Parameter_Sweep/sweep_panel.png)

**Four ways to combine parameters:**
- **One-at-a-time** — vary each parameter separately, the others held at their current values (3 values of one parameter → 3 runs).
- **All combinations** — every value of every parameter (2 parameters with 3 and 2 values → 6 runs).
- **Specified combinations** — the *i*-th values of all parameters run together (lists of equal length).
- **Bayesian search** — each run's settings are chosen by a Gaussian-process model of the results so far, to reach the lowest final loss or $L_2$ error in as few runs as possible. Each parameter's range (linear or log) or list of choices is its search space; **Runs** sets the budget and **Random first** the number of random runs before the model takes over (see the Helmholtz hyperparameter-search example under [Examples](#examples)).

Each parameter takes a list of values or a linear/log range; text settings (activation, point distribution, ...) take a list of choices. **Almost any setting can be swept** — **🔄 Refresh** lists everything available for the current problem:

- **Network** — hidden layers, neurons per layer, activation, initializer, weight decay.
- **Collocation points** — domain, boundary, initial and test points, sampling distribution.
- **Training phases** — each phase's iterations, learning rate and optimizer.
- **Loss weights** — every equation, boundary and initial-condition weight.
- **Inverse** — each unknown's initial guess, each measured-data file's weight.
- **RAR** and **time-adaptive training** — rounds, candidate and added points, iterations; step counts, IC grid resolution.
- **Input and output transforms** — each scale value.

*Also run the current settings (baseline)* adds one run with the settings unchanged, as a reference point; it is on by default for One-at-a-time with two or more parameters and for Bayesian search.

**Results.** The sweep saves to one folder (e.g. `Poisson_1D-Dirichlet_sweep__2026-10-13_11-39-17`) with a subfolder per run named after what it set (`run_01__Phase-1-Iterations_1000`, …), each with its model, plots and Error Analysis results. The folder also holds `sweep_summary.csv` (the swept values, final loss and error metrics of every run) and **`sweep_error_vs_parameter.png`**, a plot of the error ($L_2$, plus MSE and maximum error when selected; for a time-dependent problem, the mean over the reference times) and the final loss against the swept values.

**Example — network size for Burgers 1D.** Load *Burgers 1D – Raissi et al.*, choose *All combinations*, and sweep **Network: Hidden layers** over 2, 4, 6, 8 and **Network: Neurons per layer** over 10, 20, 40 — the same study as Table 1 of Raissi et al. (2019). The 12 runs (everything else as the template sets it: Adam 15,000 + L-BFGS 10,000 iterations, 8,000 collocation points) took about 17 minutes on an RTX 4090:

![Parameter sweep summary — Burgers 1D, relative L₂ error (mean over t = 0, 0.25, 0.5, 0.75, 1) and final loss vs hidden layers, one line per neurons-per-layer value](assets/Images/Parameter_Sweep/burgers_network_size_sweep.png)

Relative $L_2$ error (mean over the five reference times) from `sweep_summary.csv`:

| Hidden layers | 10 neurons | 20 neurons | 40 neurons |
|---|---|---|---|
| 2 | 2.7 × 10⁻² | 1.1 × 10⁻² | 4.4 × 10⁻³ |
| 4 | 1.3 × 10⁻³ | 7.0 × 10⁻⁴ | 3.0 × 10⁻⁴ |
| 6 | 5.2 × 10⁻⁴ | 4.0 × 10⁻⁴ | 2.3 × 10⁻⁴ |
| 8 | 7.2 × 10⁻⁴ | 5.9 × 10⁻⁴ | 1.5 × 10⁻⁴ |

Two hidden layers are too few for the shock at $x = 0$; at four layers the error is 15–21× lower, and 40 neurons per layer gives the lowest error at every depth, down to 1.5 × 10⁻⁴ at eight layers. With 10 or 20 neurons, eight layers is slightly worse than six. (Raissi et al. trained with L-BFGS only and 10,000 collocation points, so their numbers differ; the trend is the same.)

</details>

### Analysis & Output

<details>
<summary><strong>Live training log and loss curves</strong></summary>

Training runs in the background and its output streams into the Training Log as it happens; **Stop** ends the run immediately. The loss curve can show the total training and test losses, each loss term separately, or both, on a log or linear axis (*Loss Plot Settings*).

</details>

<details>
<summary><strong>Error analysis against reference data</strong></summary>

A run can be compared with reference solution files at one or more times, or — when the problem has an exact solution — with a formula typed in (one per output, in `x`, `y`, `z`, `t`). For predicted values $u_{\text{pred}}$ and reference values $u_{\text{true}}$ at $N$ points, PINNStudio reports

$$\Large L_2 \text{ relative error} = \frac{\lVert u_{\text{pred}} - u_{\text{true}} \rVert_2}{\lVert u_{\text{true}} \rVert_2}, \qquad \text{MSE} = \frac{1}{N}\sum_{i=1}^{N}\left(u_{\text{pred},i} - u_{\text{true},i}\right)^2$$

$$\Large \text{Max error} = \max_i \left| u_{\text{pred},i} - u_{\text{true},i} \right|, \qquad \text{Mean absolute error} = \frac{1}{N}\sum_{i=1}^{N}\left| u_{\text{pred},i} - u_{\text{true},i} \right|$$

with line and surface comparisons of the prediction, the reference and the error. The relative $L_2$ error is reported by default; MSE and maximum error can be added, and the mean absolute error is always included. A reference can also be compared against a derivative or any expression of the outputs (e.g. `du_x`), not only a raw output. All but two of the examples come with an exact solution or reference data, so the comparison works as soon as an example is loaded.

</details>

<details>
<summary><strong>Plots and animations</strong></summary>

Results can be shown as surface or line plots, or as animations of either over time, with the colormap, contour levels, resolution, snapshots, figure size, font sizes and axis orientation all adjustable. Besides the raw outputs, **Custom...** plots any expression of the outputs and their derivatives (`sqrt(u**2 + v**2)`, `du_x`, ...), during training, in Error Analysis and in Restore & Visualize. For problems with several outputs, each output is also saved as its own figure. Numbers in plots are written in scientific notation.

</details>

<details>
<summary><strong>Data export and standalone scripts</strong></summary>

The predicted solution can be saved as data on a grid of your choice, for your own analysis. Every problem can also be exported as a standalone DeepXDE/PyTorch script (*Problem → Export as DeepXDE Script...*) — the same code PINNStudio runs — to read, run on a cluster, share with a collaborator, or use as a starting point for your own code.

</details>

<details>
<summary><strong>Saving problems, reproducibility and hardware</strong></summary>

A whole problem setup can be saved and reopened later (*Problem → Save Problem As* / *Open Saved Problem*). Runs use a fixed random seed by default, so a run can be repeated exactly, and each run saves its settings next to its results. The GPU to train on, its memory share, the float precision, the random seed and the L-BFGS options are set under *Settings*.

</details>

### Restore & Visualize

<details>
<summary><strong>Reload a trained model without retraining</strong></summary>

Point Restore & Visualize at a saved model and its `model_config.json`, and PINNStudio rebuilds the trained network and produces any of the plots or animations above — including derivative expressions you did not plot during training, Error Analysis against the original reference data, and, for an inverse run, the convergence of the unknowns — without training again.

</details>

## Solving Your Own Problem

The examples above cover 60 specific problems, but PINNStudio isn't limited to them — every field in those examples is just a starting point you can overwrite. Here's a complete walkthrough for a PDE that **isn't** one of the examples, to show the general path from "I have an equation" to "I have a trained PINN."

We'll use the **Fisher-KPP equation**, a classic reaction-diffusion model of a population (or concentration front) that diffuses and grows logistically toward a carrying capacity of 1:

$$\Large \frac{\partial u}{\partial t} = D\frac{\partial^2 u}{\partial x^2} + r\ u(1-u), \qquad x \in [0, 1],\ t \in [0, 1]$$

with $D = 0.01$, $r = 1$, a localized initial bump, and no-flux (Neumann) boundaries — none of the examples above has a logistic nonlinearity like this one.

1. **Dimension & geometry.** Leave Dimension on **1D** (the default) — no geometry selector is needed outside 2D/3D.
2. **PDE residual.** In the free-form PDE editor, enter the residual (moving everything to one side):
   ```
   du_t - 0.01*du_xx - 1.0*u*(1 - u)
   ```
   This is exactly the `d<output>_<vars>` convention described under [Free-form, multi-output PDE editor](#problem-setup) — `du_t` is $\partial u/\partial t$, `du_xx` is $\partial^2 u/\partial x^2$, and the coefficients $D=0.01$ and $r=1$ are just written inline.
3. **Domain.** Set `x_min = 0`, `x_max = 1`, `t_min = 0`, `t_max = 1`. Leave Steady-state **off** — this is a time-dependent problem.
4. **Initial condition.** A localized bump that will spread and saturate toward 1:
   ```
   exp(-50*(x-0.5)**2)
   ```
5. **Boundary conditions.** Add one row in the Boundary Conditions panel: **Neumann**, value `0` (no-flux — the population can't leave through either edge), and in **Where** pick *Left and right* from the list (it writes `left or right`). Two rows, `left` and `right`, would do the same.
6. **Network & training.** The defaults (a handful of hidden layers, Adam then L-BFGS) are a reasonable starting point for a problem this size — adjust layer/neuron counts or add Time-Adaptive stepping later if convergence needs help.
7. **Solve.** Click **Solve** and watch the Training Log stream progress; the loss and solution plots populate once training finishes. Try the axis-swap option in Plot Settings on the resulting Surface plot — it's the same option described under [Configurable result plotting](#analysis--output).
8. **Error Analysis (optional).** Since this isn't a bundled template, there's no reference data pre-loaded — Error Analysis is entirely optional here, but if you have your own reference solution (from a separate FEM/FD solver, say), point the Error Analysis dialog at it the same way the built-in templates do automatically.

<table>
<tr>
<td width="33%" align="center">
<img src="assets/Images/1D-Fisher-KPP/Fisher-KPP_line_plot.png" alt="Fisher-KPP — PINN solution, line snapshots at several times" width="100%">
<sub>Line snapshots over time</sub>
</td>
<td width="33%" align="center">
<img src="assets/Images/1D-Fisher-KPP/Fisher-KPP_surface_plot.png" alt="Fisher-KPP — PINN solution surface" width="100%">
<sub>Surface</sub>
</td>
<td width="33%" align="center">
<img src="assets/Images/1D-Fisher-KPP/Fisher-KPP_animation.gif" alt="Fisher-KPP — PINN solution animated over time, population front spreading" width="100%">
<sub>Animation</sub>
</td>
</tr>
</table>

From here, the same eight steps apply to essentially any PDE expressible with the derivative syntax under [Free-form, multi-output PDE editor](#problem-setup) — swap in your own residual, domain, and conditions.

## How It Works

PINNStudio is built on [DeepXDE](https://github.com/lululxvi/deepxde) and PyTorch, and it doesn't wrap them at runtime — it **generates code**. Every setting you choose maps to a field on a `PINNConfig` dataclass ([`pinnstudio/core/config.py`](pinnstudio/core/config.py)); clicking **Solve** passes that config to [`codegen.py`](pinnstudio/core/codegen.py), which writes out a complete, standalone DeepXDE/PyTorch script, and [`runner.py`](pinnstudio/core/runner.py) executes it as a subprocess, streaming stdout back into the Training Log panel in real time.

Because the output of every run is an ordinary Python script, you can take it and run it outside PINNStudio, hand it to a cluster job, or use it as a starting point for a hand-written DeepXDE project.

## Repository Structure

```text
pinnstudio/
├── pinnstudio/
│   ├── main.py            # Entry point
│   ├── ui/
│   │   └── main_window.py # PyQt6 interface — every tab, dialog, and control
│   └── core/
│       ├── config.py          # PINNConfig — the full problem definition
│       ├── codegen.py         # PINNConfig -> standalone DeepXDE/PyTorch script
│       ├── deepxde_examples.py # DeepXDE's documented examples as Quick Examples
│       ├── runner.py          # Runs the generated script, streams output to the Training Log
│       ├── sweep_registry.py  # Parameter Sweep's list of sweepable parameters
│       └── sweep_runner.py    # Builds and runs each sweep configuration in turn
├── assets/
│   ├── screenshots/        # README screenshots
│   └── results/             # Example output (solution images, demo GIFs)
├── reference_data/          # Bundled ground truth for the built-in templates
│   ├── 1D/
│   ├── 2D/
│   ├── 3D/
│   └── deepxde/             # Reference/measurement data for the DeepXDE examples
├── tools/                   # Reference-data generator, DeepXDE example runner
├── requirements.txt
├── setup.py
├── install.sh              # One-command setup (macOS/Linux)
├── install.bat              # One-command setup (Windows)
└── README.md
```

## Citation

If PINNStudio is useful in your work, please cite it — see [`CITATION.cff`](CITATION.cff):

```bibtex
@software{khan2026pinnstudio,
  author  = {Khan, Asfandyar and Mamivand, Mahmood},
  title   = {PINNStudio: A No-Code Scientific Computing Environment for Forward and Inverse Physics-Informed Neural Networks},
  year    = {2026},
  url     = {https://github.com/AsfandyarKhan72/PINNStudio}
}
```

## References

- Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208–228. https://doi.org/10.1137/19M1274067
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Mattey, R., & Ghosh, S. (2022). A novel sequential method to train physics informed neural networks for Allen-Cahn and Cahn-Hilliard equations. *Computer Methods in Applied Mechanics and Engineering*, 390, 114474. https://doi.org/10.1016/j.cma.2021.114474
- Wight, C. L., & Zhao, J. (2021). Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive physics informed neural networks. *Communications in Computational Physics*, 29(3), 930–954. https://doi.org/10.4208/cicp.OA-2020-0086
- Mathias, D. L., de Almeida, T. B. F., de Barros, G. F., Coelho, L. et al. (2022). Augmenting a Physics-Informed Neural Network for the 2D Burgers Equation by Addition of Solution Data Points. *Brazilian Conference on Intelligent Systems (BRACIS 2022)*. https://arxiv.org/abs/2301.07824

## Acknowledgment

PINNStudio is built on [DeepXDE](https://github.com/lululxvi/deepxde) (Lu et al., 2021), [PyTorch](https://pytorch.org) and PyQt6. We are especially grateful to Lu Lu and the DeepXDE developers: DeepXDE is the physics-informed learning library that every script PINNStudio generates runs on, and its documented examples are included among PINNStudio's examples, each linked to its original script.

The examples that reproduce published benchmarks follow the problem setups of the original papers: *Burgers 1D – Raissi et al.* follows Raissi, Perdikaris & Karniadakis (2019); *Burgers 2D – Mathias et al.* follows Mathias et al. (2022); the Allen-Cahn examples follow Wight & Zhao (2021) and Mattey & Ghosh (2022); and *Poisson 2D – L-shape, Lu et al.* follows the demonstration example of the DeepXDE paper (Lu et al., 2021) — see [References](#references).

Developed under the supervision of Prof. Mahmood Mamivand, Computational Materials Design Lab, Boise State University.

The authors appreciate the support of the National Science Foundation grant DMR-2142935. We would like to acknowledge the high-performance computing support of the Borah compute cluster (DOI: 10.18122/oit/3/boisestate) provided by Boise State University's Research Computing Department.

## Contributing

Bug reports, feature requests, and pull requests are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Contact

Asfandyar Khan
PhD Candidate, Materials Science and Engineering
Boise State University
Email: [asfandyarkhan@u.boisestate.edu](mailto:asfandyarkhan@u.boisestate.edu)

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

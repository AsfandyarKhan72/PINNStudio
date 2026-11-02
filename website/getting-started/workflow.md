# The PINNStudio workflow

This page gives an overview of a complete PINNStudio session, from an empty window to a trained model that
you can plot again, export and compare with other runs. Each step names the pages of the model tree (the
left column of the window) where it happens and links to the pages that describe it in detail. The
[First run](first-run.md) page goes through the same steps once with a shipped example.

![The PINNStudio window as it opens: the model tree on the left with Problem, Equations, Geometry, Collocation points, Initial conditions, Boundary conditions, Neural network, Training (with Loss weights and Adaptive training), Parameter sweep and Results (with Restore & visualize); the Problem page in the middle; the Start tab on the right with the new-problem buttons and the example list; and the empty Training Log below](../assets/screenshots/start.png){ .screenshot loading=lazy }

The steps, in the order in which a problem is usually set up:

1. [Start from an example or a new problem](#1-start-from-an-example-or-a-new-problem)
2. [Define the problem](#2-define-the-problem)
3. [Choose the network and the collocation points](#3-choose-the-network-and-the-collocation-points)
4. [Set up the training](#4-set-up-the-training)
5. [Solve and follow the run](#5-solve-and-follow-the-run)
6. [Read the results](#6-read-the-results)
7. [Plot, export and save](#7-plot-export-and-save)
8. [Come back later: restore and compare](#8-come-back-later-restore-and-compare)

You do not have to go through the steps strictly in order: every page can be changed at any time before
**Solve**, and settings that would stop a run are marked on their tree entries as you work.

## 1. Start from an example or a new problem

The **Start** tab offers new 1D, 2D, 3D and ODE problems, **Open a saved problem...**, and the shipped
examples. The **Examples** lists in the toolbar load an example as a forward or inverse problem, with its
**Default** or **Tuned** settings. Starting from an example that is close to your problem is often quicker
than starting from an empty one.

See [First run](first-run.md) and [The interface](../guide/interface.md).

## 2. Define the problem

- **Problem**: the dimension (ODE, 1D, 2D, 3D), time-dependent or stationary, forward or inverse
  ([Problem](../guide/problem.md)).
- **Equations**: the outputs, the equations written as residuals equal to zero, and named constants
  ([Equations, outputs and constants](../guide/equations.md)).
- **Geometry**: the spatial domain and the time interval ([Geometry and domain](../guide/geometry.md)).
- **Initial conditions** and **Boundary conditions**: each output's initial value, and one row per boundary
  condition ([Initial conditions](../guide/initial-conditions.md),
  [Boundary conditions](../guide/boundary-conditions.md)).
- **Inverse problem** (shown only for an inverse problem): the unknown parameters and the measured data they
  are fitted to ([Inverse problems and measured data](../guide/inverse.md)).

Check the problem before training: the **Geometry** view shows the domain and which part of the boundary
each condition selects, and the **Script** tab below the graphics shows the script your settings produce.

## 3. Choose the network and the collocation points

- **Neural network**: the size, the activation, the network type (for example an FNN or a multi-scale
  Fourier-feature network) and optional input and output transforms ([Neural network](../guide/network.md)).
- **Collocation points**: how many training points are sampled in the domain, on the boundary and at the
  initial time, and how ([Collocation points](../guide/points.md)).

## 4. Set up the training

- **Training**: the **Training phases** (by default Adam followed by L-BFGS), the precision, optional
  initial-condition pre-training and the training callbacks ([Training phases](../guide/training.md)).
- **Loss weights**: one weight per loss term, and the automatic weighting methods, per-point weights and
  causal training ([Loss weights](../guide/loss-weights.md)).
- **Adaptive training**: time-adaptive training (time windows) or residual-based adaptive refinement (RAR)
  ([Adaptive training](../guide/adaptive.md)).

Which of these methods to try depends on the problem; a plain run is the usual starting point. See
[Choosing a training strategy](../guide/choosing-a-strategy.md).

## 5. Solve and follow the run

**Solve** checks the settings, writes the training script and runs it as a separate process; if a setting
cannot be used, nothing starts and the Training Log says why. While it trains, the **Live loss** tab draws
the loss, the status bar shows the phase, the iteration and the time, and the **Training Log** shows what the
script prints. **Stop** ends the run.

See [Solve and the training log](../guide/solve.md).

## 6. Read the results

When the run ends, the **Results** tab of the graphics area shows the loss and the solution in **Overview**,
the errors against reference data in **Error analysis**, and every saved run in **History**. The error is
only computed when Error Analysis is set up (with the **Error Analysis** button on the **Results** page, from
reference files or an exact formula); most examples set it up when they are loaded. A small loss alone does
not show that the solution is accurate, so compare with a reference whenever you can.

![The Results tab after a short run of Burgers 1D – Raissi et al.: the Overview tab with the training and test loss on a logarithmic axis and the PINN solution u(x, t) as a colour map; the Training Log below reports the saved error metrics, Training complete! and the run folder](../assets/screenshots/results_overview.png){ .screenshot loading=lazy }

See [Results and plots](../guide/results.md), [Error Analysis](../guide/error-analysis.md) and
[Scope and limitations](../theory/limitations.md).

## 7. Plot, export and save

On the **Results** page, **Outputs to plot:**, **Plot type:** and **Plot settings** decide which figures are
saved and how they look, and **Export Solution** writes the trained solution as text files after a forward
run. **Export script** in the toolbar writes a standalone DeepXDE script for the current settings, which runs
without PINNStudio. **Save** stores the problem itself as a `.pinn.json` file.

See [Results and plots](../guide/results.md), [Exporting a script](../guide/export.md) and
[Save, reopen and export](save-and-export.md).

## 8. Come back later: restore and compare

Each run is saved in its own folder inside the **Save to** folder. **Restore & visualize** (under **Results**
in the tree) loads a trained model from such a folder and plots it again, with another plot type or output,
without retraining. **Parameter sweep** runs the problem several times over a range of settings (or as a
Bayesian search) and saves every run, so that settings and training methods can be compared by their
error; **Compare** in the **History** tab plots several runs together.

See [Restore & visualize](../guide/restore.md) and [Parameter sweeps](../guide/sweeps.md).

# User guide

This guide walks through every part of the PINNStudio window, in the order in which a problem is
usually set up: what kind of problem it is, its equations, its domain, its conditions, the network,
the training, and what to do with the results. Each page explains how to do a task ("to add a
Neumann condition on the right side…"), how the settings on that page depend on the settings on other
pages, and ends with a table of every setting on the page: its label in the window, what it means,
its default, and the field of the saved problem (the `PINNConfig` field, see
[Configuration schema](../developer/config.md)) it is stored in.

The mathematics behind each feature is in the [Feature reference](../reference/index.md); worked
problems are in the [Tutorials](../tutorials/index.md).


!!! warning "Scope"
    PINNStudio helps you set up, train and evaluate physics-informed neural networks. It does not
    guarantee that training converges, or that a trained network is accurate or physically valid for an
    arbitrary problem. Check results against a reference whenever you can ([Error Analysis](error-analysis.md)).

## The pages of the model tree

The left column of the window (the **model tree**) has one entry per part of the problem. Selecting an
entry shows its settings in the middle column.

| Tree entry | Guide page | What you set there |
|---|---|---|
| **Problem** | [Problem](problem.md) | Dimension (ODE, 1D, 2D, 3D), time-dependent or stationary, forward or inverse |
| **Equations** | [Equations, outputs and constants](equations.md) | Number of outputs and equations, output names, the residuals, named constants |
| **Inverse problem** (inverse only) | [Inverse problems and measured data](inverse.md) | Unknown parameters and the measured data they are fitted to |
| **Geometry** | [Geometry and domain](geometry.md) | The spatial domain and the time interval |
| **Collocation points** | [Collocation points](points.md) | How many training points, and how they are sampled |
| **Initial conditions** | [Initial conditions](initial-conditions.md) | Each output's value at the initial time |
| **Boundary conditions** | [Boundary conditions](boundary-conditions.md) | One row per condition: type, output, where, value |
| **Neural network** | [Neural network](network.md) | Size, activation, network type, input and output transforms |
| **Training** | [Training phases](training.md) | Optimizer phases, precision, IC pre-training, callbacks |
| **Loss weights** | [Loss weights](loss-weights.md) | One weight per loss term; per-point weights, causal training |
| **Adaptive training** | [Adaptive training](adaptive.md) | Time-adaptive training (time windows) or RAR |
| **Parameter sweep** | [Parameter sweeps](sweeps.md) | Several runs over a range of settings, or a Bayesian search |
| **Results** | [Results and plots](results.md) | Plot type, Save to folder, Error Analysis, solution export |
| **Restore & visualize** | [Restore & visualize](restore.md) | Plot a trained model again without retraining |

The other pages of the guide cover the parts of the window that are not tree entries:
[The interface](interface.md), [Solve and the training log](solve.md),
[Error Analysis](error-analysis.md), [Exporting a script](export.md) and the
[Settings menu](settings.md).

For an overview of the whole workflow see [The PINNStudio workflow](../getting-started/workflow.md); for when to use which training method see [Choosing a training strategy](choosing-a-strategy.md).

## How the settings depend on each other

Many settings change what other pages show. The most important links are:

| When you change… | …this changes |
|---|---|
| **Problem › Problem Dimension** | The coordinates the expressions may use (`t`; `x, t`; `x, y, t`; `x, y, z, t`), the **Shape** choices on the Geometry page, the **Where** choices of boundary conditions, the number of network inputs, and the plot types. Changing it by hand **clears every boundary-condition row** and resets **Examples** to **None**. |
| **Problem › Stationary** | Removes the time interval, the Initial conditions page, IC pre-training and time-adaptive training; the network loses its `t` input. |
| **Problem › Inverse** | Shows the Inverse problem page (unknown parameters and measured data), adds one data loss term per measured-data file, adds the parameter-convergence plot types and **Save parameters**, and removes time-adaptive training. |
| **Equations › Outputs** | Rebuilds the output names, the equation boxes, one initial-condition block per output, one output-transform row per output, and the output lists on the Results, Restore and Inverse pages. |
| **Equations › Equations** | The number of residual boxes and of **PDE** loss weights; the **Points from** list of RAR. |
| **Boundary conditions** rows | One **BC** loss weight per row (per phase when phases have their own weights). |
| **Initial conditions › IC for …** and **Exact** | One **IC** loss weight per active, non-exact initial condition. |
| **Training phases** | The phases' optimizers and loss functions are what RAR rounds and time windows use; with **Use same weights for all phases** unticked, each phase gets its own set of loss weights. |
| **Adaptive training › Method** | RAR adds its own training rounds after the phases; time-adaptive training repeats the phases in every time window. |
| **Parameter sweep › Enable Parameter Sweep** | **Solve** becomes **Run Sweep**. |

Settings that would stop **Solve** are marked on their tree entries as you work, and a button
**N settings to fix** appears in the toolbar (see [The interface](interface.md#settings-that-would-stop-solve)).

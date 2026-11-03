# Examples

PINNStudio ships **64 examples**: 15 PINNStudio templates and 49 examples reproduced from DeepXDE's documentation. Each page below is generated from the example's own definition in the software (`pinnstudio/core/example_catalog.py`, `deepxde_examples.py`, `pinnstudio_examples.py` and the templates as the main window loads them), so the equations, conditions and settings shown are the ones PINNStudio loads.

Every example is loaded from the toolbar's **Examples:** list: choose **Forward** or **Inverse**, pick the example under its **ODE**, **1D**, **2D** or **3D** heading, and choose **Default** or **Tuned** settings. *Default* is DeepXDE's own settings for a DeepXDE example, the paper's method for a template taken from a paper, and the template's own settings otherwise; *Tuned* is PINNStudio's best setting from the tuning study, where one exists.

!!! warning "Scope"
    An example is a starting point. PINNStudio sets up, trains and evaluates the PINN; it does not guarantee convergence or accuracy, and results vary from run to run. How the examples' errors are measured and recorded is described in [Reproducibility](../reproducibility/benchmarks.md).

## Browse by group

<div class="grid cards" markdown>

-   **[ODEs](#odes)** (7)

    ---

    Ordinary differential equations in t only.

-   **[1D PDEs](#1d-pdes)** (33)

    ---

    One space dimension, steady or in time.

-   **[2D PDEs](#2d-pdes)** (18)

    ---

    Two space dimensions.

-   **[3D PDEs](#3d-pdes)** (6)

    ---

    Three space dimensions.

-   **[Coupled systems](#coupled-systems)** (17)

    ---

    More than one governing equation.

-   **[Forward problems](#forward)** (54)

    ---

    Listed under **Forward** in the Examples list.

-   **[Inverse problems](#inverse)** (36)

    ---

    Listed under **Inverse**: DeepXDE's inverse examples and every example with an Inverse mode.

-   **[Phase-field](#phase-field)** (8)

    ---

    Allen–Cahn and Cahn–Hilliard equations.

-   **[Fluid mechanics](#fluid-mechanics)** (6)

    ---

    Burgers, Navier–Stokes (Kovasznay, Beltrami) and Brinkman–Forchheimer.

-   **[Solid mechanics and elasticity](#solid-mechanics)** (3)

    ---

    Euler beam and linear elastostatic plate.

-   **[Advanced training strategies](#advanced-training)** (12)

    ---

    Default or Tuned settings that use time windows, RAR, causal training, RBA or SA-PINN point weights, or multi-scale Fourier features.

</div>

## ODEs { #odes }

Ordinary differential equations in t only. 7 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Integro-differential ODE](ode/integro-differential-ode.md) | ODE | Time-dependent | Forward | — |
| [Integro-differential ODE – Volterra kernel](ode/integro-differential-ode-volterra-kernel.md) | ODE | Time-dependent | Forward | — |
| [Linear ODE system](ode/linear-ode-system.md) | ODE | Time-dependent | Forward / Inverse | yes |
| [Lotka-Volterra ODE](ode/lotka-volterra-ode.md) | ODE | Time-dependent | Forward / Inverse | — |
| [Second-order ODE](ode/second-order-ode.md) | ODE | Time-dependent | Forward / Inverse | yes |
| [Lorenz ODE – unknown C₁, C₂, C₃](ode/lorenz-ode-unknown-c1-c2-c3.md) | ODE | Time-dependent | Inverse | — |
| [Lorenz ODE – unknown C₁, C₂, C₃, exogenous input](ode/lorenz-ode-unknown-c1-c2-c3-exogenous-input.md) | ODE | Time-dependent | Inverse | — |

## 1D PDEs { #1d-pdes }

One space dimension, steady or in time. 33 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Euler beam 1D](1d/euler-beam-1d.md) | 1D | Steady | Forward / Inverse | yes |
| [Fractional Poisson 1D](1d/fractional-poisson-1d.md) | 1D | Steady | Forward | yes |
| [Poisson 1D – Dirichlet](1d/poisson-1d-dirichlet.md) | 1D | Steady | Forward | — |
| [Poisson 1D – Dirichlet/Neumann](1d/poisson-1d-dirichlet-neumann.md) | 1D | Steady | Forward | — |
| [Poisson 1D – Dirichlet/Periodic](1d/poisson-1d-dirichlet-periodic.md) | 1D | Steady | Forward | yes |
| [Poisson 1D – Dirichlet/point-set operator](1d/poisson-1d-dirichlet-point-set-operator.md) | 1D | Steady | Forward | — |
| [Poisson 1D – Dirichlet/Robin](1d/poisson-1d-dirichlet-robin.md) | 1D | Steady | Forward | — |
| [Poisson 1D – hard BCs](1d/poisson-1d-hard-bcs.md) | 1D | Steady | Forward | — |
| [Poisson 1D – multi-scale Fourier features](1d/poisson-1d-multi-scale-fourier-features.md) | 1D | Steady | Forward | — |
| [Brinkman-Forchheimer 1D – unknown ν_e, K](1d/brinkman-forchheimer-1d-unknown-nu-e-k.md) | 1D | Steady | Inverse | yes |
| [Diffusion-reaction 1D – unknown rate field](1d/diffusion-reaction-1d-unknown-rate-field.md) | 1D | Steady | Inverse | yes |
| [Fractional Poisson 1D – unknown order α](1d/fractional-poisson-1d-unknown-order-alpha.md) | 1D | Steady | Inverse | — |
| [Poisson 1D – unknown forcing field](1d/poisson-1d-unknown-forcing-field.md) | 1D | Steady | Inverse | yes |
| [Poisson 1D – unknown forcing field (batches)](1d/poisson-1d-unknown-forcing-field-batches.md) | 1D | Steady | Inverse | yes |
| [Allen-Cahn 1D – Wight & Zhao](1d/allen-cahn-1d-wight-zhao.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 1D – Wight & Zhao](1d/cahn-hilliard-1d-wight-zhao.md) | 1D | Time-dependent | Forward | yes |
| [Cahn-Hilliard 1D – Mattey & Ghosh](1d/cahn-hilliard-1d-mattey-ghosh.md) | 1D | Time-dependent | Forward | yes |
| [Black-Scholes 1D – American put](1d/black-scholes-1d-american-put.md) | 1D | Time-dependent | Forward | — |
| [Black-Scholes 1D – European call](1d/black-scholes-1d-european-call.md) | 1D | Time-dependent | Forward | — |
| [Burgers 1D – Raissi et al.](1d/burgers-1d-raissi-et-al.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Burgers 1D – RAR](1d/burgers-1d-rar.md) | 1D | Time-dependent | Forward | — |
| [Diffusion 1D](1d/diffusion-1d.md) | 1D | Time-dependent | Forward | yes |
| [Diffusion 1D – hard IC and BCs](1d/diffusion-1d-hard-ic-and-bcs.md) | 1D | Time-dependent | Forward | — |
| [Diffusion 1D – resampling](1d/diffusion-1d-resampling.md) | 1D | Time-dependent | Forward | yes |
| [Diffusion-reaction 1D](1d/diffusion-reaction-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Fractional diffusion 1D](1d/fractional-diffusion-1d.md) | 1D | Time-dependent | Forward | — |
| [Heat 1D – Dirichlet](1d/heat-1d-dirichlet.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Heat 1D – resampling](1d/heat-1d-resampling.md) | 1D | Time-dependent | Forward | yes |
| [Klein-Gordon 1D](1d/klein-gordon-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Schrödinger 1D](1d/schrodinger-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Wave 1D – space-time multi-scale Fourier features](1d/wave-1d-space-time-multi-scale-fourier-features.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Diffusion 1D – unknown coefficient](1d/diffusion-1d-unknown-coefficient.md) | 1D | Time-dependent | Inverse | yes |
| [Diffusion-reaction 1D – unknown k_f, D](1d/diffusion-reaction-1d-unknown-k-f-d.md) | 1D | Time-dependent | Inverse | — |

## 2D PDEs { #2d-pdes }

Two space dimensions. 18 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Fractional Poisson 2D – disk](2d/fractional-poisson-2d-disk.md) | 2D | Steady | Forward | — |
| [Helmholtz 2D – Dirichlet](2d/helmholtz-2d-dirichlet.md) | 2D | Steady | Forward / Inverse | yes |
| [Helmholtz 2D – hyperparameter search](2d/helmholtz-2d-hyperparameter-search.md) | 2D | Steady | Forward | — |
| [Helmholtz 2D – sound-hard scattering](2d/helmholtz-2d-sound-hard-scattering.md) | 2D | Steady | Forward | — |
| [Helmholtz 2D – square with hole](2d/helmholtz-2d-square-with-hole.md) | 2D | Steady | Forward | yes |
| [Laplace 2D – disk](2d/laplace-2d-disk.md) | 2D | Steady | Forward | — |
| [Linear elasticity 2D – plate](2d/linear-elasticity-2d-plate.md) | 2D | Steady | Forward / Inverse | yes |
| [Linear elasticity 2D – plate, soft BCs](2d/linear-elasticity-2d-plate-soft-bcs.md) | 2D | Steady | Forward | — |
| [Navier-Stokes 2D – Kovasznay flow](2d/navier-stokes-2d-kovasznay-flow.md) | 2D | Steady | Forward / Inverse | yes |
| [Poisson 2D – disk](2d/poisson-2d-disk.md) | 2D | Steady | Forward / Inverse | — |
| [Poisson 2D – L-shape](2d/poisson-2d-l-shape.md) | 2D | Steady | Forward / Inverse | yes |
| [Fractional Poisson 2D – unknown order α](2d/fractional-poisson-2d-unknown-order-alpha.md) | 2D | Steady | Inverse | — |
| [Allen-Cahn 2D – Wight & Zhao](2d/allen-cahn-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Allen-Cahn 2D – Mattey & Ghosh](2d/allen-cahn-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Wight & Zhao](2d/cahn-hilliard-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Mattey & Ghosh](2d/cahn-hilliard-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Burgers 2D – Mathias et al.](2d/burgers-2d-mathias-et-al.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Heat 2D](2d/heat-2d.md) | 2D | Time-dependent | Forward / Inverse | yes |

## 3D PDEs { #3d-pdes }

Three space dimensions. 6 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Fractional Poisson 3D – sphere](3d/fractional-poisson-3d-sphere.md) | 3D | Steady | Forward | — |
| [Poisson 3D – cube with spherical cut-out](3d/poisson-3d-cube-with-spherical-cut-out.md) | 3D | Steady | Forward | yes |
| [Poisson 3D – sphere](3d/poisson-3d-sphere.md) | 3D | Steady | Forward / Inverse | — |
| [Allen-Cahn 3D – Wight & Zhao](3d/allen-cahn-3d-wight-zhao.md) | 3D | Time-dependent | Forward / Inverse | yes |
| [Heat 3D](3d/heat-3d.md) | 3D | Time-dependent | Forward / Inverse | yes |
| [Navier-Stokes 3D – Beltrami flow](3d/navier-stokes-3d-beltrami-flow.md) | 3D | Time-dependent | Forward / Inverse | yes |

## Coupled systems { #coupled-systems }

More than one governing equation. 17 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Linear ODE system](ode/linear-ode-system.md) | ODE | Time-dependent | Forward / Inverse | yes |
| [Lotka-Volterra ODE](ode/lotka-volterra-ode.md) | ODE | Time-dependent | Forward / Inverse | — |
| [Lorenz ODE – unknown C₁, C₂, C₃](ode/lorenz-ode-unknown-c1-c2-c3.md) | ODE | Time-dependent | Inverse | — |
| [Lorenz ODE – unknown C₁, C₂, C₃, exogenous input](ode/lorenz-ode-unknown-c1-c2-c3-exogenous-input.md) | ODE | Time-dependent | Inverse | — |
| [Cahn-Hilliard 1D – Wight & Zhao](1d/cahn-hilliard-1d-wight-zhao.md) | 1D | Time-dependent | Forward | yes |
| [Cahn-Hilliard 1D – Mattey & Ghosh](1d/cahn-hilliard-1d-mattey-ghosh.md) | 1D | Time-dependent | Forward | yes |
| [Black-Scholes 1D – American put](1d/black-scholes-1d-american-put.md) | 1D | Time-dependent | Forward | — |
| [Schrödinger 1D](1d/schrodinger-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Diffusion-reaction 1D – unknown k_f, D](1d/diffusion-reaction-1d-unknown-k-f-d.md) | 1D | Time-dependent | Inverse | — |
| [Helmholtz 2D – sound-hard scattering](2d/helmholtz-2d-sound-hard-scattering.md) | 2D | Steady | Forward | — |
| [Linear elasticity 2D – plate](2d/linear-elasticity-2d-plate.md) | 2D | Steady | Forward / Inverse | yes |
| [Linear elasticity 2D – plate, soft BCs](2d/linear-elasticity-2d-plate-soft-bcs.md) | 2D | Steady | Forward | — |
| [Navier-Stokes 2D – Kovasznay flow](2d/navier-stokes-2d-kovasznay-flow.md) | 2D | Steady | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Wight & Zhao](2d/cahn-hilliard-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Mattey & Ghosh](2d/cahn-hilliard-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Burgers 2D – Mathias et al.](2d/burgers-2d-mathias-et-al.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Navier-Stokes 3D – Beltrami flow](3d/navier-stokes-3d-beltrami-flow.md) | 3D | Time-dependent | Forward / Inverse | yes |

## Forward problems { #forward }

Listed under **Forward** in the Examples list. 54 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Integro-differential ODE](ode/integro-differential-ode.md) | ODE | Time-dependent | Forward | — |
| [Integro-differential ODE – Volterra kernel](ode/integro-differential-ode-volterra-kernel.md) | ODE | Time-dependent | Forward | — |
| [Linear ODE system](ode/linear-ode-system.md) | ODE | Time-dependent | Forward / Inverse | yes |
| [Lotka-Volterra ODE](ode/lotka-volterra-ode.md) | ODE | Time-dependent | Forward / Inverse | — |
| [Second-order ODE](ode/second-order-ode.md) | ODE | Time-dependent | Forward / Inverse | yes |
| [Euler beam 1D](1d/euler-beam-1d.md) | 1D | Steady | Forward / Inverse | yes |
| [Fractional Poisson 1D](1d/fractional-poisson-1d.md) | 1D | Steady | Forward | yes |
| [Poisson 1D – Dirichlet](1d/poisson-1d-dirichlet.md) | 1D | Steady | Forward | — |
| [Poisson 1D – Dirichlet/Neumann](1d/poisson-1d-dirichlet-neumann.md) | 1D | Steady | Forward | — |
| [Poisson 1D – Dirichlet/Periodic](1d/poisson-1d-dirichlet-periodic.md) | 1D | Steady | Forward | yes |
| [Poisson 1D – Dirichlet/point-set operator](1d/poisson-1d-dirichlet-point-set-operator.md) | 1D | Steady | Forward | — |
| [Poisson 1D – Dirichlet/Robin](1d/poisson-1d-dirichlet-robin.md) | 1D | Steady | Forward | — |
| [Poisson 1D – hard BCs](1d/poisson-1d-hard-bcs.md) | 1D | Steady | Forward | — |
| [Poisson 1D – multi-scale Fourier features](1d/poisson-1d-multi-scale-fourier-features.md) | 1D | Steady | Forward | — |
| [Allen-Cahn 1D – Wight & Zhao](1d/allen-cahn-1d-wight-zhao.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 1D – Wight & Zhao](1d/cahn-hilliard-1d-wight-zhao.md) | 1D | Time-dependent | Forward | yes |
| [Cahn-Hilliard 1D – Mattey & Ghosh](1d/cahn-hilliard-1d-mattey-ghosh.md) | 1D | Time-dependent | Forward | yes |
| [Black-Scholes 1D – American put](1d/black-scholes-1d-american-put.md) | 1D | Time-dependent | Forward | — |
| [Black-Scholes 1D – European call](1d/black-scholes-1d-european-call.md) | 1D | Time-dependent | Forward | — |
| [Burgers 1D – Raissi et al.](1d/burgers-1d-raissi-et-al.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Burgers 1D – RAR](1d/burgers-1d-rar.md) | 1D | Time-dependent | Forward | — |
| [Diffusion 1D](1d/diffusion-1d.md) | 1D | Time-dependent | Forward | yes |
| [Diffusion 1D – hard IC and BCs](1d/diffusion-1d-hard-ic-and-bcs.md) | 1D | Time-dependent | Forward | — |
| [Diffusion 1D – resampling](1d/diffusion-1d-resampling.md) | 1D | Time-dependent | Forward | yes |
| [Diffusion-reaction 1D](1d/diffusion-reaction-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Fractional diffusion 1D](1d/fractional-diffusion-1d.md) | 1D | Time-dependent | Forward | — |
| [Heat 1D – Dirichlet](1d/heat-1d-dirichlet.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Heat 1D – resampling](1d/heat-1d-resampling.md) | 1D | Time-dependent | Forward | yes |
| [Klein-Gordon 1D](1d/klein-gordon-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Schrödinger 1D](1d/schrodinger-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Wave 1D – space-time multi-scale Fourier features](1d/wave-1d-space-time-multi-scale-fourier-features.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Fractional Poisson 2D – disk](2d/fractional-poisson-2d-disk.md) | 2D | Steady | Forward | — |
| [Helmholtz 2D – Dirichlet](2d/helmholtz-2d-dirichlet.md) | 2D | Steady | Forward / Inverse | yes |
| [Helmholtz 2D – hyperparameter search](2d/helmholtz-2d-hyperparameter-search.md) | 2D | Steady | Forward | — |
| [Helmholtz 2D – sound-hard scattering](2d/helmholtz-2d-sound-hard-scattering.md) | 2D | Steady | Forward | — |
| [Helmholtz 2D – square with hole](2d/helmholtz-2d-square-with-hole.md) | 2D | Steady | Forward | yes |
| [Laplace 2D – disk](2d/laplace-2d-disk.md) | 2D | Steady | Forward | — |
| [Linear elasticity 2D – plate](2d/linear-elasticity-2d-plate.md) | 2D | Steady | Forward / Inverse | yes |
| [Linear elasticity 2D – plate, soft BCs](2d/linear-elasticity-2d-plate-soft-bcs.md) | 2D | Steady | Forward | — |
| [Navier-Stokes 2D – Kovasznay flow](2d/navier-stokes-2d-kovasznay-flow.md) | 2D | Steady | Forward / Inverse | yes |
| [Poisson 2D – disk](2d/poisson-2d-disk.md) | 2D | Steady | Forward / Inverse | — |
| [Poisson 2D – L-shape](2d/poisson-2d-l-shape.md) | 2D | Steady | Forward / Inverse | yes |
| [Allen-Cahn 2D – Wight & Zhao](2d/allen-cahn-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Allen-Cahn 2D – Mattey & Ghosh](2d/allen-cahn-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Wight & Zhao](2d/cahn-hilliard-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Mattey & Ghosh](2d/cahn-hilliard-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Burgers 2D – Mathias et al.](2d/burgers-2d-mathias-et-al.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Heat 2D](2d/heat-2d.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Fractional Poisson 3D – sphere](3d/fractional-poisson-3d-sphere.md) | 3D | Steady | Forward | — |
| [Poisson 3D – cube with spherical cut-out](3d/poisson-3d-cube-with-spherical-cut-out.md) | 3D | Steady | Forward | yes |
| [Poisson 3D – sphere](3d/poisson-3d-sphere.md) | 3D | Steady | Forward / Inverse | — |
| [Allen-Cahn 3D – Wight & Zhao](3d/allen-cahn-3d-wight-zhao.md) | 3D | Time-dependent | Forward / Inverse | yes |
| [Heat 3D](3d/heat-3d.md) | 3D | Time-dependent | Forward / Inverse | yes |
| [Navier-Stokes 3D – Beltrami flow](3d/navier-stokes-3d-beltrami-flow.md) | 3D | Time-dependent | Forward / Inverse | yes |

## Inverse problems { #inverse }

Listed under **Inverse**: DeepXDE's inverse examples and every example with an Inverse mode. 36 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Linear ODE system](ode/linear-ode-system.md) | ODE | Time-dependent | Forward / Inverse | yes |
| [Lotka-Volterra ODE](ode/lotka-volterra-ode.md) | ODE | Time-dependent | Forward / Inverse | — |
| [Second-order ODE](ode/second-order-ode.md) | ODE | Time-dependent | Forward / Inverse | yes |
| [Lorenz ODE – unknown C₁, C₂, C₃](ode/lorenz-ode-unknown-c1-c2-c3.md) | ODE | Time-dependent | Inverse | — |
| [Lorenz ODE – unknown C₁, C₂, C₃, exogenous input](ode/lorenz-ode-unknown-c1-c2-c3-exogenous-input.md) | ODE | Time-dependent | Inverse | — |
| [Euler beam 1D](1d/euler-beam-1d.md) | 1D | Steady | Forward / Inverse | yes |
| [Brinkman-Forchheimer 1D – unknown ν_e, K](1d/brinkman-forchheimer-1d-unknown-nu-e-k.md) | 1D | Steady | Inverse | yes |
| [Diffusion-reaction 1D – unknown rate field](1d/diffusion-reaction-1d-unknown-rate-field.md) | 1D | Steady | Inverse | yes |
| [Fractional Poisson 1D – unknown order α](1d/fractional-poisson-1d-unknown-order-alpha.md) | 1D | Steady | Inverse | — |
| [Poisson 1D – unknown forcing field](1d/poisson-1d-unknown-forcing-field.md) | 1D | Steady | Inverse | yes |
| [Poisson 1D – unknown forcing field (batches)](1d/poisson-1d-unknown-forcing-field-batches.md) | 1D | Steady | Inverse | yes |
| [Allen-Cahn 1D – Wight & Zhao](1d/allen-cahn-1d-wight-zhao.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Burgers 1D – Raissi et al.](1d/burgers-1d-raissi-et-al.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Diffusion-reaction 1D](1d/diffusion-reaction-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Heat 1D – Dirichlet](1d/heat-1d-dirichlet.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Klein-Gordon 1D](1d/klein-gordon-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Schrödinger 1D](1d/schrodinger-1d.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Wave 1D – space-time multi-scale Fourier features](1d/wave-1d-space-time-multi-scale-fourier-features.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Diffusion 1D – unknown coefficient](1d/diffusion-1d-unknown-coefficient.md) | 1D | Time-dependent | Inverse | yes |
| [Diffusion-reaction 1D – unknown k_f, D](1d/diffusion-reaction-1d-unknown-k-f-d.md) | 1D | Time-dependent | Inverse | — |
| [Helmholtz 2D – Dirichlet](2d/helmholtz-2d-dirichlet.md) | 2D | Steady | Forward / Inverse | yes |
| [Linear elasticity 2D – plate](2d/linear-elasticity-2d-plate.md) | 2D | Steady | Forward / Inverse | yes |
| [Navier-Stokes 2D – Kovasznay flow](2d/navier-stokes-2d-kovasznay-flow.md) | 2D | Steady | Forward / Inverse | yes |
| [Poisson 2D – disk](2d/poisson-2d-disk.md) | 2D | Steady | Forward / Inverse | — |
| [Poisson 2D – L-shape](2d/poisson-2d-l-shape.md) | 2D | Steady | Forward / Inverse | yes |
| [Fractional Poisson 2D – unknown order α](2d/fractional-poisson-2d-unknown-order-alpha.md) | 2D | Steady | Inverse | — |
| [Allen-Cahn 2D – Wight & Zhao](2d/allen-cahn-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Allen-Cahn 2D – Mattey & Ghosh](2d/allen-cahn-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Wight & Zhao](2d/cahn-hilliard-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Mattey & Ghosh](2d/cahn-hilliard-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Burgers 2D – Mathias et al.](2d/burgers-2d-mathias-et-al.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Heat 2D](2d/heat-2d.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Poisson 3D – sphere](3d/poisson-3d-sphere.md) | 3D | Steady | Forward / Inverse | — |
| [Allen-Cahn 3D – Wight & Zhao](3d/allen-cahn-3d-wight-zhao.md) | 3D | Time-dependent | Forward / Inverse | yes |
| [Heat 3D](3d/heat-3d.md) | 3D | Time-dependent | Forward / Inverse | yes |
| [Navier-Stokes 3D – Beltrami flow](3d/navier-stokes-3d-beltrami-flow.md) | 3D | Time-dependent | Forward / Inverse | yes |

## Phase-field { #phase-field }

Allen–Cahn and Cahn–Hilliard equations. 8 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Allen-Cahn 1D – Wight & Zhao](1d/allen-cahn-1d-wight-zhao.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 1D – Wight & Zhao](1d/cahn-hilliard-1d-wight-zhao.md) | 1D | Time-dependent | Forward | yes |
| [Cahn-Hilliard 1D – Mattey & Ghosh](1d/cahn-hilliard-1d-mattey-ghosh.md) | 1D | Time-dependent | Forward | yes |
| [Allen-Cahn 2D – Wight & Zhao](2d/allen-cahn-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Allen-Cahn 2D – Mattey & Ghosh](2d/allen-cahn-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Wight & Zhao](2d/cahn-hilliard-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Cahn-Hilliard 2D – Mattey & Ghosh](2d/cahn-hilliard-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Allen-Cahn 3D – Wight & Zhao](3d/allen-cahn-3d-wight-zhao.md) | 3D | Time-dependent | Forward / Inverse | yes |

## Fluid mechanics { #fluid-mechanics }

Burgers, Navier–Stokes (Kovasznay, Beltrami) and Brinkman–Forchheimer. 6 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Brinkman-Forchheimer 1D – unknown ν_e, K](1d/brinkman-forchheimer-1d-unknown-nu-e-k.md) | 1D | Steady | Inverse | yes |
| [Burgers 1D – Raissi et al.](1d/burgers-1d-raissi-et-al.md) | 1D | Time-dependent | Forward / Inverse | yes |
| [Burgers 1D – RAR](1d/burgers-1d-rar.md) | 1D | Time-dependent | Forward | — |
| [Navier-Stokes 2D – Kovasznay flow](2d/navier-stokes-2d-kovasznay-flow.md) | 2D | Steady | Forward / Inverse | yes |
| [Burgers 2D – Mathias et al.](2d/burgers-2d-mathias-et-al.md) | 2D | Time-dependent | Forward / Inverse | yes |
| [Navier-Stokes 3D – Beltrami flow](3d/navier-stokes-3d-beltrami-flow.md) | 3D | Time-dependent | Forward / Inverse | yes |

## Solid mechanics and elasticity { #solid-mechanics }

Euler beam and linear elastostatic plate. 3 examples.

| Example | Dimension | Regime | Modes | Tuned |
|---|---|---|---|---|
| [Euler beam 1D](1d/euler-beam-1d.md) | 1D | Steady | Forward / Inverse | yes |
| [Linear elasticity 2D – plate](2d/linear-elasticity-2d-plate.md) | 2D | Steady | Forward / Inverse | yes |
| [Linear elasticity 2D – plate, soft BCs](2d/linear-elasticity-2d-plate-soft-bcs.md) | 2D | Steady | Forward | — |

## Advanced training strategies { #advanced-training }

Default or Tuned settings that use time windows, RAR, causal training, RBA or SA-PINN point weights, or multi-scale Fourier features. 12 examples.

| Example | Dimension | Regime | Modes | Tuned | Strategies |
|---|---|---|---|---|---|
| [Poisson 1D – multi-scale Fourier features](1d/poisson-1d-multi-scale-fourier-features.md) | 1D | Steady | Forward | — | Fourier features |
| [Allen-Cahn 1D – Wight & Zhao](1d/allen-cahn-1d-wight-zhao.md) | 1D | Time-dependent | Forward / Inverse | yes | Fourier features, RAR, causal training |
| [Cahn-Hilliard 1D – Wight & Zhao](1d/cahn-hilliard-1d-wight-zhao.md) | 1D | Time-dependent | Forward | yes | SA-PINN, time windows |
| [Cahn-Hilliard 1D – Mattey & Ghosh](1d/cahn-hilliard-1d-mattey-ghosh.md) | 1D | Time-dependent | Forward | yes | time windows |
| [Burgers 1D – RAR](1d/burgers-1d-rar.md) | 1D | Time-dependent | Forward | — | RAR |
| [Wave 1D – space-time multi-scale Fourier features](1d/wave-1d-space-time-multi-scale-fourier-features.md) | 1D | Time-dependent | Forward / Inverse | yes | Fourier features |
| [Allen-Cahn 2D – Wight & Zhao](2d/allen-cahn-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes | RBA, time windows |
| [Allen-Cahn 2D – Mattey & Ghosh](2d/allen-cahn-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes | SA-PINN, time windows |
| [Cahn-Hilliard 2D – Wight & Zhao](2d/cahn-hilliard-2d-wight-zhao.md) | 2D | Time-dependent | Forward / Inverse | yes | time windows |
| [Cahn-Hilliard 2D – Mattey & Ghosh](2d/cahn-hilliard-2d-mattey-ghosh.md) | 2D | Time-dependent | Forward / Inverse | yes | SA-PINN, time windows |
| [Burgers 2D – Mathias et al.](2d/burgers-2d-mathias-et-al.md) | 2D | Time-dependent | Forward / Inverse | yes | time windows |
| [Allen-Cahn 3D – Wight & Zhao](3d/allen-cahn-3d-wight-zhao.md) | 3D | Time-dependent | Forward / Inverse | yes | time windows |

# 9. Exact boundary and initial constraints

Initial and boundary conditions are normally loss terms: the network is penalized when it violates them, and they
hold only approximately. They can instead be built into the network, so that they hold exactly whatever the network
learns ("hard constraints"; see [Exact constraints](../theory/exact-constraints.md)). The loss then contains only the
PDE residual (and any conditions left as loss terms). This tutorial shows the three ways PINNStudio offers.

**What you will learn**

- impose initial and boundary conditions with an **output transform** formula (shipped example);
- make an initial condition exact with **Exact (built into the network)**;
- where exact periodicity is set;
- how exact constraints change the loss.

## Prerequisites

- [Tutorial 2](02-burgers.md) and [Tutorial 4](04-coupled.md).

## Part A: an output transform that satisfies the IC and BCs

The example **Diffusion 1D – hard IC and BCs** (DeepXDE's `diffusion_1d_exactBC`) solves
\( u_t = u_{xx} - e^{-t}(\sin\pi x - \pi^2\sin\pi x) \) on \( x \in [-1, 1] \), \( t \in [0, 1] \), with
\( u(x, 0) = \sin\pi x \) and \( u(\pm 1, t) = 0 \). The exact solution is \( u = e^{-t}\sin\pi x \).

1. Load **Forward › Diffusion 1D – hard IC and BCs › Default**.
2. Open **Neural network**. **Enable output transform** is ticked, **Mode:** is **Formula**, and
   **Output 1 =** is `t*(1 - x**2)*u + sin(pi*x)`. Here `u` is the raw network output. At \( t = 0 \) the formula gives
   \( \sin\pi x \); at \( x = \pm 1 \) it gives \( \sin(\pm\pi) = 0 \): both conditions hold for any network.

    ![The Neural network page with Enable output transform ticked, Mode Formula, and Output 1 = t*(1 - x**2)*u + sin(pi*x)](../assets/screenshots/tut09_network.png){ .screenshot loading=lazy }

3. Open **Boundary conditions**: there are no rows, and the page says
   `No rows: the boundary conditions are imposed exactly by the output transform formula (Network panel → Output transform).`
   On **Initial conditions**, **IC for u** is unticked, so the IC is not a loss term either.

    ![The Boundary conditions page of the hard-constraint example, with no rows and a note that the conditions are imposed by the output transform](../assets/screenshots/tut09_bcs.png){ .screenshot loading=lazy }

4. The example trains with one Adam phase of 10000 iterations and compares with the exact solution (configured in
   **Error Analysis** as a formula). Click **Solve**.

Writing such a formula yourself: multiply the raw output by a function that vanishes where the condition must hold,
and add a function that has the required values there. For \( u = 0 \) at \( x = 0 \) and \( x = 1 \) the in-app hint gives
`x*(1-x)*u`.

## Part B: an exact initial condition from a checkbox

For the initial condition there is a shortcut that needs no formula.

5. Start a **1D problem** (Start tab). Keep **x:** and **t:** from 0 to 1, one output `u`, and type
   `du_t - 0.1*du_xx` as the equation and `sin(pi*x)` as the initial condition.
6. On **Initial conditions**, tick **Exact (built into the network)**. **IC for u** is unticked and greyed out: the
   network output becomes \( u = \mathrm{IC}(x) + \tau N \), with \( \tau = (t - t_{\min})/(t_{\max} - t_{\min}) \), which equals
   the initial condition at \( t_{\min} \) exactly.

    ![The Initial conditions page with Exact (built into the network) ticked and IC for u greyed out](../assets/screenshots/tut09_ic_exact.png){ .screenshot loading=lazy }

7. The boundary values stay ordinary loss terms here: on **Boundary conditions** click **Add Boundary Condition** and
   set **Dirichlet BC**, **Output:** 1, **Where:** **Whole boundary**, **Value:** `0`.
8. On **Results** › **Error Analysis**, choose **Exact solution formula**, enter `exp(-0.1*pi**2*t)*sin(pi*x)` and
   `0, 0.5, 1` in **Compare at times:**. Click **Run Error Analysis**, then **Solve**.

!!! warning "Exact IC and an output-transform formula do not combine"
    If you also tick **Enable output transform** with a **Formula**, PINNStudio refuses to start:
    `Exact initial condition can't be combined with your own output transform formula -- write the condition into the formula yourself (IC + t*u), or turn the formula off.`
    To make both the IC and the boundary values exact, untick **Exact (built into the network)** and write one formula
    that satisfies both, as in Part A (for this problem, `sin(pi*x) + t*x*(1 - x)*u`).

The exact IC needs the IC as a formula (not **Load IC from file**) and cannot be combined with time-adaptive
training.

## Part C: exact periodicity

For a solution periodic in space, tick **Exact periodicity (periodic inputs)** on the **Boundary conditions** page.
Every axis with a periodic condition is then fed to the network through cosine and sine of the coordinate, so the
network is periodic by construction. [Tutorial 5](05-phase-field.md) uses a periodic boundary condition on
Allen–Cahn.

## Expected results

With an exact initial condition, the error at \( t = 0 \) is at the level of floating-point round-off, whatever the
network learned. With exact boundary values, the boundary-condition columns disappear from the loss table.

Our test runs on a CPU (all verified against the exact solution with Error Analysis):

| Run | Iterations | Relative L2 at \( t = 0 \) | at \( t = 0.5 \) | at \( t = 1 \) |
|---|---|---|---|---|
| Part A, example (formula for IC and BCs) | 3000 Adam (20 s) | 9.2 × 10⁻⁸ | 1.8 × 10⁻³ | 3.7 × 10⁻³ |
| Part B, exact IC checkbox + Dirichlet row | 3000 Adam + 1000 L-BFGS (100 s) | 8.8 × 10⁻⁸ | 7.6 × 10⁻⁵ | 6.9 × 10⁻⁴ |
| Part B alternative, one formula `sin(pi*x) + t*x*(1 - x)*u` | 3000 Adam + 1000 L-BFGS (95 s) | 8.8 × 10⁻⁸ | 2.5 × 10⁻⁵ | 7.3 × 10⁻⁵ |

The example's own setting is 10000 Adam iterations; Part B uses the default phases (10000 + 10000). The two Part B runs
are not a benchmark of the two approaches (one seed, short training), but both show the exact initial condition.

## Troubleshooting

| Symptom | What to check |
|---|---|
| The condition is not satisfied exactly | Evaluate the formula on the boundary by hand: the multiplying factor must be zero there and the added term must equal the boundary value. |
| Training is slow or stalls | A multiplying factor that is very small inside the domain makes the raw output hard to train; scale it (e.g. `4*x*(1 - x)`). |
| **Exact (built into the network)** is not available | The IC comes from a file, or time-adaptive training is on. |
| A transform and a boundary row for the same condition | Remove the row: the condition is already exact. |

## Next steps

- [Exact constraints](../reference/exact-constraints.md) and [Networks and Fourier features](../reference/networks.md)
  in the reference.
- [Tutorial 10](10-rar.md): adaptive refinement.

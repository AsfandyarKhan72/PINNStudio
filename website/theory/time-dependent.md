# Time-dependent training

Training a PINN on a long time interval at once often fails for evolution equations with sharp or slowly
developing features (phase-field equations are a common example): the network fits a trivial or wrong solution
that still has a small residual. Splitting the interval into windows and training them in order, and keeping
the earlier part of the solution fixed while later parts are learnt, are standard remedies
([Wight & Zhao, 2021](../about/references.md#wight2021); [Mattey & Ghosh, 2022](../about/references.md#mattey2022)).
In PINNStudio these are set with **Method:** **Time Adaptive Training** on the **Adaptive training** page; see
[Time windows and bc-PINN](../reference/time-windows.md) and
[IC pre-training and guided learning](../reference/pretraining.md). [Causal training](loss-weighting.md#causal-training)
addresses the same difficulty by weighting instead.

## Time windows (time marching)

The **Time step groups:** rows each give a start, an end and a number of steps; every group is split into that
many equal windows, and the windows of all groups form the sequence
\([t_0, t_1], [t_1, t_2], \dots, [t_{K-1}, t_K]\). For window \(k\) the script builds a new DeepXDE problem on
\(\Omega \times [t_k, t_{k+1}]\), with the same equations, boundary conditions, number of points and Training
Phases as the whole problem, and trains a network \(u^{(k)}\).

**Initial condition of a window.** The first window uses the problem's initial conditions. Every later window
starts from the previous window's solution at its end time: on a grid \(\{\mathbf{x}_g\}\) with **IC grid
resolution:** points per axis (squared in 2D, cubed in 3D; for a non-box shape only the grid points in the
domain), each output is held to

\[
\mathcal{L}^{(k)}_{0,i} = \ell\Big(\big\{\, u^{(k)}_{\theta,i}(\mathbf{x}_g, t_k) - u^{(k-1)}_{i}(\mathbf{x}_g, t_k) \,\big\}_g\Big),
\]

one point-set term per output (also for an output that had no initial condition in the first window). Its
weight is the weight of that output's initial condition, or 1 for an output without one; the equation and
boundary weights are kept. In 2D and 3D these grid points are also added as residual points.

The solution over the whole interval is the piecewise function \(u(\mathbf{x}, t) = u^{(k)}(\mathbf{x}, t)\) for
\(t \in [t_k, t_{k+1}]\). Each window's models are saved in its own folder, and Error Analysis evaluates each
reference time with the network of its window.

!!! info "Weights of later windows: changed in v1.6.0 (in development)"
    In v1.5.0 every loss weight was replaced by 1 from the second window on; the development version maps the
    weights as described above.

### Transfer learning

Without transfer learning every window starts from a freshly initialised network. With
**Enable transfer learning (warm start from previous step)** each window starts from the weights of the
previous window's network; **Transfer optimizer:** chooses which state is passed on: **Adam** the network after
the previous window's last Adam phase, **L-BFGS** the network as the previous window ended.

## One network through all windows

!!! info "New in v1.6.0 (in development)"
    The growing time range, backward compatibility (bc-PINN), initial-condition guided learning and resampling
    in each window are new in the development version. With any of them on, **Export as DeepXDE Script...**
    writes the full script that Solve runs.

These options need transfer learning (the configuration is refused otherwise), so that one network is trained
through all windows.

### Growing time range

With **Growing time range: each window trains on [start, t] (one network)**, window \(k\) uses residual and
boundary points on \([t_0, t_{k+1}]\) and the true initial condition at \(t_0\), instead of the previous
window's solution at \(t_k\). This is "approach I" of [Wight & Zhao (2021)](../about/references.md#wight2021):
the training interval grows, and the network is always fitted to the whole range learnt so far.

### Backward compatibility (bc-PINN)

With **Backward compatibility (bc-PINN): keep the solution on earlier windows**
([Mattey & Ghosh, 2022](../about/references.md#mattey2022)), after window \(k\) the script stores the network's
prediction on the time range trained so far: at \(S\) spatial points (the window grid, or a random subset of
**Points:** of them, drawn with the run's seed) and at the times

\[
\mathcal{T}^{\text{bc}}_k = \bigcup_{l \le k} \Big\{ t_l + m\,\frac{t_{l+1} - t_l}{n_t} : m = 0, \dots, n_t - 1 \Big\} \cup \{ t_{k+1} \},
\qquad n_t = \textbf{Times/window:},
\]

all predicted by the newest network. Every later window adds one point-set term per output,

\[
\mathcal{L}^{(k+1)}_{\text{bc},i} = \ell\Big(\big\{\, u_{\theta,i}(\mathbf{x}_s, \tau) - \tilde u_i(\mathbf{x}_s, \tau) \,\big\}_{s,\ \tau \in \mathcal{T}^{\text{bc}}_k}\Big),
\]

with weight **Weight:** (default 100), so the network keeps the solution it has already learnt on earlier
windows while it learns the next one.

### Initial-condition guided learning

**IC-guided learning:** \(= f > 0\) follows the initial-condition guided learning of Mattey & Ghosh: before a
window's own training, the network is fitted for \(f\) times the iterations of the first Adam (or AdamW)
Training Phase, with Adam at that phase's learning rate, to the window's starting profile held constant over
the window,

\[
\min_\theta\ \sum_i \ell\Big(\big\{\, u_{\theta,i}(\mathbf{x}_g, \tau_m) - u^{\text{start}}_i(\mathbf{x}_g) \,\big\}_{g,m}\Big),
\qquad \tau_m \in \text{10 evenly spaced times in the window},
\]

with the equation residual weighted 0. The starting profile is the initial condition in the first window (and
in every window with a growing range), for the outputs that have one, and the previous window's solution at
\(t_k\) otherwise. The fit gives the window's training a starting point that already has the right initial
state.

## IC pre-training

**Enable IC-guided pre-training** (on the **Training** page) is a separate, simpler step for the whole run (or
the first window): before the main training the network is trained on the initial-condition terms alone (on
**Initial points:** initial points, the equation residual weighted 0), with its own **Optimizer:**,
**Iterations:**, **Learning rate:** and **Loss function:**, or restored from a checkpoint. It needs at least
one initial condition given as a loss term.

## Choosing between the options

There is no setting that works for every problem. Time windows trade one hard optimisation for several easier
ones, but errors made in one window are passed on to the next through the initial condition of the next
window and can accumulate; bc-PINN reduces the drift of earlier windows at the cost of more loss terms; the
growing range trains on ever more points. Compare with a plain run and with a reference solution
([Error measures](metrics.md)). Exact initial conditions and scaling by initial losses cannot be combined with
time windows.

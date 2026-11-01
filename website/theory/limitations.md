# Scope and limitations

PINNStudio helps set up, train and evaluate physics-informed neural networks: it turns a problem description
into a DeepXDE script, runs it, and compares the result with reference data. It does **not** guarantee that
training converges, that the trained network is physically valid, or that it is accurate for an arbitrary
problem. A finished run with a small loss is a candidate solution that has to be checked.

## Known failure modes of PINNs

These are properties of the method, not of PINNStudio; the literature cited on the other theory pages
discusses them in detail.

- **Optimisation failure.** The loss is non-convex. Training can stall in a poor local minimum or converge to a
  trivial solution (for example \(u \equiv 0\) or a constant) that satisfies the equation but not the
  conditions well. Long time intervals, stiff reaction terms and phase-field equations are typical cases
  ([Time-dependent training](time-dependent.md), [causal training](loss-weighting.md#causal-training)).
- **Imbalanced loss terms.** Terms of very different size, or with gradients of very different size, let one
  term dominate; the others are then fitted poorly ([Wang, Teng & Perdikaris, 2021](../about/references.md#wang2021gradient);
  [Adaptive loss weighting](loss-weighting.md)).
- **Spectral bias.** Fully connected networks learn smooth, low-frequency components first; high frequencies,
  multi-scale solutions, sharp fronts, boundary layers and shocks are learnt slowly or not at all
  (Fourier-feature networks, adaptive sampling and more points can help).
- **Residual is not error.** A small residual at the collocation points does not bound the error between
  them, and says nothing about a region without points. The network is not reliable outside the training domain
  (in space or time).
- **High-order derivatives.** Every derivative is a derivative of the network; fourth-order terms are expensive
  and amplify the network's own roughness.
- **Precision.** Single precision (float32) limits how far the loss can be reduced, and L-BFGS may stop
  early; float64 is slower but can go further.
- **Inverse problems.** Inferred coefficients can be non-identifiable (several values fit the data equally well),
  sensitive to noise in the data and to the initial guess, and biased when the equation is not the right
  model for the data ([Forward and inverse problems](forward-inverse.md#identifiability)).
- **Randomness.** Results depend on the random initialisation and on the sampled points. A single run does not
  show how much a result varies.

## What you should check

1. **The problem itself.** Check the equations, conditions and units on the **Geometry** view and in the
   generated script (**Script** tab or **Export as DeepXDE Script...**): that every boundary condition selects
   the intended part of the boundary, that the initial condition is right, and that each unknown appears where it
   belongs.
2. **Against a reference.** Compare with an analytical solution or an independent numerical solution of
   sufficient resolution ([Error measures](metrics.md)), at several times for a time-dependent problem.
3. **Sensitivity.** Repeat with more collocation points, a larger network and more iterations; a converged
   result should not change much. Repeat with other random seeds (**Settings › Random Seed...**, or a sweep over
   **Random seed**) and report the spread.
4. **Physical sense.** Check bounds, symmetries and conserved quantities the solution should have (for
   example a phase field between \(-1\) and \(1\), mass conservation for Cahn-Hilliard).
5. **The loss terms.** Plot the individual terms (**Settings › Plot Settings...**, **Loss** tab, **Show:**
   **Each loss term**): a run where one term stays large has not solved the problem even when the total is
   small.
6. **Inverse problems.** Start from several initial guesses, test the setup first on synthetic data generated
   with known coefficients, and check whether the inferred value changes when data are removed or noise is
   added.

## Limitations of the implementation

The following follow from how the current (development) code works and are stated on the theory pages where
they arise.

**Inverse problems**

- Unknown coefficients are unconstrained scalars (`dde.Variable`): no bounds, no positivity, no automatic
  re-parametrisation. An unknown function must be modelled as an extra network output.

**Losses and optimisation**

- Of the **Loss function:** choices, only `MSE` and `MAE` give a usable loss with the PyTorch backend
  ([Losses](losses.md#loss-functions)).
- Weight decay is ignored for PFNN, MsFFN and STMsFFN networks and cannot be used with L-BFGS or NNCG.
- **Scale by initial losses** cannot be combined with time windows.

**Adaptive methods**

- RAR rounds train with a constant learning rate (the phase's learning-rate decay is not applied), draw fresh
  pseudo-random candidates each round, rank a system of equations by the sum of absolute residuals, and
  implement only the greedy variant ([Adaptive sampling](adaptive-sampling.md)).
- Resampling inside time windows always draws 10,000 candidates (no control in the window) and ranks by all
  equations together.
- Per-point weights (RBA, SA-PINN) act only on the equation residuals, not on boundary, initial or data terms;
  they start again from 1 whenever the training points change, and cannot be combined with point resampling.
- Causal training bins random training points into time chunks (not a fixed temporal grid), includes the
  sampled boundary and initial points in the chunk losses, and with the MAE loss weights by \(\sqrt{w}\) instead
  of \(w\) ([Adaptive loss weighting](loss-weighting.md#differences-from-the-paper)). It is not available for
  steady problems or ODEs.

**Constraints and time windows**

- The exact initial condition uses the linear factor \(\tau = (t - t_{\min})/(t_{\max} - t_{\min})\) only, needs
  the condition as a formula, and cannot be combined with time windows. Exact periodicity uses one harmonic and
  makes every output periodic in that axis ([Exact constraints](exact-constraints.md)).
- Time windows pass the solution from one window to the next only at the grid points of **IC grid
  resolution:**; a coarse grid loses detail, and errors can accumulate from window to window.
- With a growing time range, backward compatibility, initial-condition guided learning or resampling in each
  window, the exported script is the full script Solve runs, not the short tutorial-style script.
- Problems with fractional operators cannot use RAR, time windows or IC pre-training.

**Search and evaluation**

- Bayesian search uses a simplified Gaussian-process model (one length scale from five values, fixed
  amplitude, random candidates) and places categorical choices on an ordered axis ([Bayesian optimization](bayesian.md)).
- Error measures are computed at the reference points only; a reference time without a matching window
  network is compared with zeros ([Error measures](metrics.md)).
- The random seed makes CPU runs repeatable; GPU runs are not made bit-for-bit deterministic.
- `PINNConfig.validate()` refuses many impossible combinations before a run, but it is not exhaustive. A
  combination it does not catch may fail inside the training script or train something other than intended.

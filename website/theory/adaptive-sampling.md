# Adaptive sampling

A fixed set of residual points can miss the regions where the solution changes quickly (fronts, boundary
layers, interfaces). Adaptive sampling adds points where the current network satisfies the equation worst.
PINNStudio implements two such schemes: residual-based adaptive refinement (RAR) for a whole run, and
resampling inside each time window. Both are on the **Adaptive training** page; see also
[RAR](../reference/rar.md) and [Time windows and bc-PINN](../reference/time-windows.md).

## Residual-based adaptive refinement (RAR)

RAR was introduced with DeepXDE ([Lu et al., 2021](../about/references.md#lu2021deepxde)): after training, the
residual is evaluated at many random points, the points with the largest residual are added to the training
set, and training continues. [Wu et al. (2023)](../about/references.md#wu2023rar) compare it with other sampling
strategies and call this greedy variant RAR-G.

### The procedure as implemented

With **Method:** set to **Residual-based Adaptive Refinement (RAR)**, the generated script first runs all
Training Phases and then repeats, for \(c = 1, \dots, R\) (\(R\) = **RAR training rounds:**):

1. **Candidates.** Draw \(N_c\) = **Residual sampling points:** random points
   \(\{\mathbf{z}_n\}\) in the space-time domain with the geometry's own `random_points` (uniform pseudo-random
   sampling; for a non-box shape, only points inside the shape).

2. **Score.** Evaluate the residuals of the current network at the candidates and score each point by

    \[
    s(\mathbf{z}_n) = \sum_{j=1}^{J} \lvert r_j(\mathbf{z}_n; \theta) \rvert ,
    \]

    the sum of the absolute residuals of all equations, or by \(\lvert r_{j^\ast}(\mathbf{z}_n)\rvert\) of a single
    equation \(j^\ast\) chosen under **Points from:** (default: **All outputs (combined)**).

3. **Stop test.** If **Stop below mean residual:** is a value \(\tau > 0\) and
   \(\frac{1}{N_c}\sum_n s(\mathbf{z}_n) < \tau\), refinement ends here (no points are added in this round).

4. **Add points.** The \(k\) = **Points added per round:** candidates with the largest score are added to the
   residual points (DeepXDE anchors). Points added in earlier rounds stay; boundary and initial points are not
   changed.

5. **Retrain.** Train for **Adam iterations:** with the optimizer, learning rate, loss function and loss weights
   of the last first-order Training Phase, then, if **L-BFGS iterations:** is above 0, with L-BFGS using the loss
   function and weights of the last L-BFGS phase.

After each round the Training Log prints the maximum and mean score of the candidates, and with a save folder
each round's model and a plot of the collocation points are saved.

### Differences from the original description

- Lu et al. loop until the mean residual falls below a tolerance. In PINNStudio the number of rounds \(R\) is
  always a cap; the tolerance test (off by default) can end the loop earlier.
- The candidate set is drawn anew in every round.
- The learning rate in a round is constant: it is the phase's initial learning rate, and the phase's
  learning-rate decay is not applied in RAR rounds.
- Only the greedy variant is implemented; the residual-distribution sampling (RAR-D) of Wu et al. is not.
- The score of a system of equations is the sum of absolute residuals, which mixes equations of different
  scales. Use **Points from:** to rank by one equation when the residuals differ by orders of magnitude.

With causal training or per-point weights switched on, the candidates are scored by the plain, unweighted
residual ([Adaptive loss weighting](loss-weighting.md)).

## Resampling in each time window

!!! info "New in v1.6.0 (in development)"
    Resampling rounds inside time windows are new in the development version.

With **Time Adaptive Training** ([Time-dependent training](time-dependent.md)), **Resampling rounds:**
\(> 0\) applies the same idea inside each window, after the window's own training, following the
"approach I with resampling" of [Wight & Zhao (2021)](../about/references.md#wight2021). In every round:

1. draw **Candidates per round:** random candidates (10,000 by default) in the window's space-time domain;
2. score them by \(s = \sum_j \lvert r_j \rvert\) (all equations; **Points from:** does not apply here);
3. add the \(k\) = **Points per round:** best-scoring candidates to the window's residual points;
4. train for **Adam iterations per round:** with the optimizer, learning rate and loss function of the last
   first-order Training Phase.

Resampling cannot be combined with per-point weights (RBA, SA-PINN): the weights belong to fixed points, and
the configuration is refused with an explanation.

## Non-adaptive resampling

The **Point Resampling** training callback (**Periodically resample collocation points**) is DeepXDE's
`PDEPointResampler`: every given number of steps it redraws the residual points (and, if ticked, the boundary
points) from the same distribution. It does not look at the residual; its purpose is to expose the network to
more points over training without enlarging the set used at each step.

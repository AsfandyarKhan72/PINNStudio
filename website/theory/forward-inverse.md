# Forward and inverse problems

## Forward problems

In a **forward** problem every coefficient of the equations, the domain and the conditions are known, and the
network is trained to approximate the solution \(u\). The loss is the one of
[Losses and automatic differentiation](losses.md): equation residuals plus boundary- and initial-condition
terms. See [Forward problems](../reference/forward.md).

## Inverse problems

In an **inverse** problem some coefficients \(\lambda = (\lambda_1, \dots, \lambda_Q)\) of the equations are
unknown, and measurements of the solution are available. PINNs treat the unknowns as additional trainable
parameters and fit them together with the network ([Raissi, Perdikaris & Karniadakis, 2019](../about/references.md#raissi2019);
[Lu et al., 2021](../about/references.md#lu2021deepxde)):

\[
(\theta^\star, \lambda^\star) = \arg\min_{\theta, \lambda}\ \mathcal{L}(\theta, \lambda),
\qquad
\mathcal{L}(\theta, \lambda) = \sum_j w_{r,j}\,\mathcal{L}_{r,j}(\theta, \lambda)
+ \sum_b w_b\,\mathcal{L}_b(\theta) + \sum_i w_{0,i}\,\mathcal{L}_{0,i}(\theta)
+ \sum_f w_{d,f}\,\mathcal{L}_{d,f}(\theta).
\]

The unknowns enter only the residual terms, through the equations in which their names appear.

### Unknown coefficients as trainable variables

Each row of **Unknown parameters** (on the **Inverse problem** page of the model tree) becomes, in the generated
script,

```python
D = dde.Variable(1.0)          # Name: D, Initial guess: 1.0
```

a scalar PyTorch tensor with gradient tracking. All unknowns are passed to every `model.compile(...)` call as
`external_trainable_variables`, so every Training Phase (and every RAR round) updates them with the same
optimizer and learning rate as the network weights. The name is written into an equation like any other
symbol, for example `du_t - D*du_xx`. The **True value** column is never used in training; it only draws the
dashed reference line in the parameter convergence plot and gives the relative parameter error of a
sweep ([Error measures](metrics.md)).

!!! note "Differences from some published setups"
    PINNStudio trains \(\lambda\) directly: there are no bounds, no positivity constraint and no
    re-parametrisation (such as \(\lambda = e^{s}\) or a scaled variable). The **Log scale** option only changes
    the axis of the convergence plot. Weight decay is not applied to the unknowns (DeepXDE excludes external
    trainable variables from it). If an unknown must stay positive or spans orders of magnitude, write the
    re-parametrisation into the equation yourself, for example a variable `s` used as `exp(s)*du_xx`.

The value of every unknown is recorded in the parameter history at the period chosen under
**Save parameters:** (`Every 100 iters`, `Every 1000 iters`; with `No`, 1/200 of the total number of
iterations of the Training Phases) and after every RAR round; the parameter convergence plot is drawn from it, and the last entry is reported as
`Final inferred D: ...` in the Training Log.

An unknown *function* rather than a constant (for example an unknown forcing field \(f(x)\)) is modelled as one
more network output that has no boundary or initial condition of its own and appears in the equation, as in
DeepXDE's inverse example with an unknown forcing field, which PINNStudio ships as an example.

### The observation loss

Each **Measured data** file holds points and measured values. It becomes one DeepXDE point-set condition and
one loss term, with its own **Loss weight:**:

\[
\mathcal{L}_{d,f}(\theta) = \ell\big(\{\, u_{\theta,i_f}(\mathbf{x}_n, t_n) - d_n \,\}_{n=1}^{N_f}\big)
\qquad \text{(}\texttt{PointSetBC}\text{, } \textbf{Measures:} \text{ an output } i_f\text{)},
\]

or, when **Measures:** is `Custom...`, an expression \(G\) of the outputs and their derivatives,

\[
\mathcal{L}_{d,f}(\theta) = \ell\big(\{\, G[u_\theta](\mathbf{x}_n, t_n) - d_n \,\}_{n=1}^{N_f}\big)
\qquad \text{(}\texttt{PointSetOperatorBC}\text{)},
\]

with \(\ell\) the loss function of the training phase (normally the mean squared error). With
**Check the equation at the measured points too (anchors)** ticked (the default) the measured points are also
added to the residual points. **Use:** can reduce the data to a grid of a given number of points per space
axis or to a random subset (with the run's seed); the files are not changed. See
[Observational data](../reference/observations.md) and [Inverse problems](../reference/inverse.md).

## Identifiability

Fitting the data well does not guarantee that the inferred coefficients are the true ones. If several
values of \(\lambda\) produce the same solution at the measured points (for example a wave speed that only
enters through a quantity the data do not resolve, or a coefficient multiplying a term that is zero for
the measured solution), the problem is not identifiable and the result depends on the initial guess. Check
the convergence plot, repeat with different initial guesses, and, where a true value is known, compare with
it. The [Scope and limitations](limitations.md) page lists further checks.

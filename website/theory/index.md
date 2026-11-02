# Theory

These pages explain the mathematics behind the settings PINNStudio offers, written to match what the
generated training scripts actually compute. Where PINNStudio's implementation differs from the method as
published, the difference is stated on the page.

PINNStudio writes and runs a [DeepXDE](https://deepxde.readthedocs.io/) script with the PyTorch backend
([Lu et al., 2021](../about/references.md#lu2021deepxde)). Most of the theory below is therefore the theory of
DeepXDE's physics-informed neural networks; the methods that PINNStudio adds to the generated code (causal
training, per-point weights, time windows with backward compatibility, exact constraints, Bayesian search)
are described in their own detail.

| Page | Topic | Related settings |
|---|---|---|
| [Physics-informed neural networks](pinns.md) | The network \(u_\theta\), collocation points, the training problem | **Neural network**, **Collocation points** |
| [Losses and automatic differentiation](losses.md) | Residuals, condition losses, loss functions, the weighted total loss | **Training**, **Loss weights** |
| [Forward and inverse problems](forward-inverse.md) | Unknown coefficients as trainable variables, the observation loss | **Problem**, inverse settings |
| [Error measures](metrics.md) | Relative \(L_2\) error, MSE, maximum error, parameter error | **Error Analysis** |
| [Adaptive sampling](adaptive-sampling.md) | RAR and resampling inside time windows | **Adaptive training** |
| [Adaptive loss weighting](loss-weighting.md) | Fixed weights, scaling by initial losses, RBA, SA-PINN, causal training | **Loss weights** |
| [Exact constraints](exact-constraints.md) | Exact initial conditions, exact periodicity | **Initial conditions**, **Boundary conditions** |
| [Time-dependent training](time-dependent.md) | Time windows, transfer learning, bc-PINN, guided learning | **Adaptive training** |
| [Bayesian optimization](bayesian.md) | Gaussian-process search over settings; parameter sweeps | **Parameter sweep** |
| [Scope and limitations](limitations.md) | What PINNStudio does not guarantee, known failure modes | all |

!!! note "Scope"
    PINNStudio helps set up, train and evaluate PINNs. It does not guarantee that training converges, that
    the trained network is physically valid, or that it is accurate for an arbitrary problem. See
    [Scope and limitations](limitations.md).

## Notation

| Symbol | Meaning |
|---|---|
| \(\mathbf{x} \in \Omega \subset \mathbb{R}^d\) | spatial coordinates (\(d = 1, 2, 3\)); the problem's **Geometry** |
| \(t \in [t_{\min}, t_{\max}]\) | time (absent for a steady problem) |
| \(u_\theta(\mathbf{x}, t)\) | the network's outputs, one per output name, with parameters \(\theta\) |
| \(\mathcal{N}[u](\mathbf{x}, t) = 0\) | a governing equation as typed on the **Equations** page, written as a residual |
| \(\mathcal{B}[u] = g\) on \(\partial\Omega\) | a boundary condition (one row of the **Boundary conditions** page) |
| \(u(\mathbf{x}, t_{\min}) = u_0(\mathbf{x})\) | an initial condition |
| \(\lambda\) | unknown physical coefficients of an inverse problem |
| \(r(\mathbf{x}, t)\) | the residual of an equation at a point |
| \(\mathcal{L}_k\), \(w_k\) | the \(k\)-th loss term and its weight |

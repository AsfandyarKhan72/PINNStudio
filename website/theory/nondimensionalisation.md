# Scaling and nondimensional form

PINNStudio has no units: every value is a plain number. How the problem is scaled is therefore up to you, and it
matters more for a PINN than for a mesh solver. This page says why, and how to write a problem in nondimensional
form before entering it.

## Why scale

A PINN's loss is a sum of terms: the equation residuals, the boundary and initial conditions, and any data. The
optimizer sees only their sum.

- **Terms of very different size.** In \(\rho c_p\, T_t = k\, T_{xx}\) with SI values (\(\rho c_p \approx 4\cdot
  10^6\), \(k \approx 0.6\)), the residual is millions of times larger than a boundary error of one kelvin. Training
  then fits the equation and ignores the conditions, or the reverse. [Loss weights](loss-weighting.md) can
  compensate, but only for a ratio that stays the same during training.
- **Inputs far from 1.** The network's first layer multiplies the coordinates by weights initialised for inputs of
  order 1. A domain \(x \in [0, 0.001]\) looks constant to it; \(t \in [0, 3600]\) saturates `tanh` at once.
- **Outputs far from 1.** The last layer starts with outputs of order 1. A field of size \(10^5\) needs weights
  that the initialization and the learning rate do not reach in a reasonable number of steps.
- **Derivatives.** Every derivative in the residual brings a factor of one over the length or time scale. Second
  derivatives on a domain of length \(10^{-3}\) are \(10^6\) times the first ones.

In nondimensional form all inputs, outputs and terms are of order 1, and the remaining parameters (Reynolds, Péclet,
Damköhler numbers ...) say directly which terms dominate.

## How to do it

1. **Choose reference scales.** A length \(L\) (the domain size, or a feature size), a time \(T\), and a scale
   \(U\) for each output (a boundary value, the largest expected value, or a difference such as
   \(T_{\text{hot}} - T_{\text{cold}}\)). Often a time scale follows from the equation, as below.
2. **Substitute.** Write \(x = L\,\hat x\), \(t = T\,\hat t\), \(u = U\,\hat u\) (with an offset when the values do
   not start at zero: \(u = u_0 + U\,\hat u\)). Each derivative scales accordingly:
   \(u_t = (U/T)\,\hat u_{\hat t}\), \(u_{xx} = (U/L^2)\,\hat u_{\hat x \hat x}\).
3. **Divide** the equation by the coefficient of one term, so that the others carry the dimensionless groups.
4. **Enter the scaled problem**: the nondimensional equation, the domain in \(\hat x, \hat t\), the scaled boundary
   and initial values, and reference or observation data divided by the same scales.
5. **Convert back** when reading the results: \(x = L\,\hat x\), \(u = u_0 + U\,\hat u\).

### Example: heat conduction

\(\rho c_p\, T_t = k\, T_{xx}\) on \(x \in [0, L]\), with \(T(0) = T_{\text{hot}}\), \(T(L) = T_{\text{cold}}\). With
\(\hat x = x/L\), \(\hat T = (T - T_{\text{cold}})/(T_{\text{hot}} - T_{\text{cold}})\) and the diffusion time
\(t_d = L^2 \rho c_p / k\) as time scale (\(\hat t = t/t_d\)):

\[
\hat T_{\hat t} = \hat T_{\hat x \hat x}, \qquad \hat x \in [0, 1], \qquad \hat T(0) = 1, \quad \hat T(1) = 0 .
\]

In PINNStudio: the equation `dT_t - dT_xx` (the output named `T`), the domain 0 to 1 in x, and the time range
\(t_{\text{end}}/t_d\). Every material constant has gone into the time scale.

### Inverse problems

An unknown coefficient is learnt as a plain number ([Inverse problems](../guide/inverse.md)) and starts near its
initial value. Scale it like the rest, so that the value sought is of order 1: estimate \(\hat k = k/k_0\) with a
\(k_0\) of the expected size, not \(k\) itself. A coefficient of \(10^{-5}\) learnt directly barely moves.

## What PINNStudio does for you

- **Input transform** (Neural network page, **Enable input transform**, mode **Scale/shift**): the network sees
  `x_raw × scale + shift`. With scale \(2/3600\) and shift \(-1\), a time range \([0, 3600]\) reaches the
  network as \([-1, 1]\) without rewriting the equation
  ([Neural network](../guide/network.md#rescale-or-transform-the-inputs)).
- **Output transform** (**Enable output transform**, **Scale/shift**): `y_raw × scale + shift`, so the network can
  produce values of order 1 for a field of order \(10^5\)
  ([Neural network](../guide/network.md#rescale-the-outputs-or-impose-conditions-exactly-with-an-output-formula)).
- **Loss weights** and the adaptive weighting methods balance the terms during training
  ([Adaptive loss weighting](loss-weighting.md)).

These help with the network's inputs and outputs. They do not change the equation: derivatives are still taken
with respect to the original coordinates, so residual terms of very different size remain. Writing the equation
in nondimensional form is the more reliable fix.

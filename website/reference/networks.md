---
search:
  boost: 2
---

# Networks and Fourier features

## What it does

The **Neural network** page sets the network \( N_\theta \) that represents the solution: its size, type
(FNN, PFNN, MsFFN, STMsFFN), activation and initializer, and optional input and output transforms.
Weight decay (L2 regularization) is set in **Settings › Optimizer Settings...**.

## When to use it

- **FNN** (default): one fully connected network for all outputs.
- **PFNN**: a separate sub-network per output, merged only at the output layer; can help when the
  outputs differ a lot in size or behaviour.
- **MsFFN / STMsFFN**: multi-scale random Fourier features of the inputs (Wang, Wang & Perdikaris,
  [2021](../about/references.md#wang2021fourier)), for solutions with both slow and fast variation;
  STMsFFN uses separate features in space and time.
- **Input/output transforms**: rescale inputs or outputs, feed hand-made features, or build hard
  constraints into the output.

## How to enable it

**Neural network** page. **Network type:** chooses the architecture; **Fourier scales (σ):** (and
**Time scales (σₜ):** for STMsFFN) appear for the Fourier networks. **Enable input transform** and
**Enable output transform** open their rows, with **Mode:** **Scale/shift** or **Formula**.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Hidden layers:** | Number of hidden layers | 3 | 1–50 | `layers` |
| **Neurons per layer:** | Width of every hidden layer | 64 | 1–4096 | `layers` |
| **Widths per layer:** | Widths of each hidden layer, e.g. `128, 64, 32`; overrides the two above | blank | positive integers | `layers` |
| **Activation:** | Hidden-layer activation | tanh | tanh, sin, Sigmoid, ReLU, SiLU, Swish, ELU, GELU, SELU | `activation` |
| **Kernel initializer:** | Weight initialization | Glorot uniform | Glorot uniform, Glorot normal, He uniform, He normal, zeros | `kernel_initializer` |
| **Network type:** | Architecture | FNN | FNN, PFNN, MsFFN, STMsFFN | `network_type` |
| **Fourier scales (σ):** | Scales of the Fourier features | `1, 10` | positive numbers, comma-separated | `msffn_sigmas` |
| **Time scales (σₜ):** | Scales of the time features (STMsFFN) | `1, 10` | positive numbers | `msffn_sigmas_t` |
| **Enable input transform** | Transform the inputs | off | — | `input_transform_enabled`, `input_transform_mode` |
| input **Scale/shift** rows | \( x \mapsto a x + b \) per input | ×1 +0 | −10⁶–10⁶ | `input_transform_scale`, `input_transform_shift` |
| input **Formula** | Network input features, `;`-separated, in `x, y, z, t` | blank | expressions | `input_feature_formula` |
| **Enable output transform** | Transform the outputs | off | — | `output_transform_enabled`, `output_transform_mode` |
| output **Scale/shift** rows | \( y \mapsto a y + b \) per output | ×1 +0 | −10⁶–10⁶ | `output_transform_scale`, `output_transform_shift` |
| output **Formula** rows | One formula per output in the coordinates and the raw outputs by name; blank = unchanged | blank | expressions | `output_transform_formula` (`|`-separated) |
| **Weight decay (L2):** (Settings › Optimizer Settings...) | L2 regularization of the network weights | 0 | ≥ 0 | `weight_decay` |

`layers` is stored as the full list `[inputs, hidden…, outputs]`, e.g. `[2, 64, 64, 64, 1]`.

## Formulation

**FNN** (`dde.nn.FNN`): \( N_\theta = W_L\,\sigma(\cdots\sigma(W_1 \mathbf{z} + b_1)\cdots) + b_L \) on the
(possibly transformed) inputs \( \mathbf{z} \).

**PFNN** (`dde.nn.PFNN`): each hidden layer of width \( W \) becomes one layer of width \( W \) per output,
so every output has its own sub-network.

**MsFFN.** With the first hidden layer of width \( 2m \) and scales \( \sigma_1, \dots, \sigma_K \),
fixed random matrices \( B_k \in \mathbb{R}^{d\times m} \), \( (B_k)_{ij} \sim \mathcal{N}(0, \sigma_k^2) \),
map the inputs to

\[
\gamma_k(\mathbf{z}) = \big[\cos(\mathbf{z} B_k),\ \sin(\mathbf{z} B_k)\big] \in \mathbb{R}^{2m}
\]

— written **without a factor \( 2\pi \)** inside the cosine and sine. Each \( \gamma_k \) passes through the
same remaining hidden layers (shared weights), the \( K \) results are concatenated and a linear layer gives
the outputs. **STMsFFN** builds space features from all inputs but the last and time features from the last
input (\( t \)) with the time scales, passes each through the shared hidden layers, multiplies every
(space, time) pair point-wise, and concatenates the products. The matrices \( B_k \) are stored with the
model, so a restored model has the same features. These networks are PINNStudio's own implementation in
the generated script, written to match DeepXDE's `MsFFN`/`STMsFFN`.

**Input transform.** Scale/shift: \( \tilde{\mathbf{x}} = \mathbf{a}\odot\mathbf{x} + \mathbf{b} \).
Formula: the network's input is the list of feature expressions (DeepXDE's `apply_feature_transform`);
the first layer is sized to the number of features. **Output transform.** Scale/shift:
\( u = a\,N_\theta + b \). Formula: \( u_i = F_i(\mathbf{x}, t, N_\theta) \) with the raw outputs by name
(DeepXDE's `apply_output_transform`), e.g. `x*(1-x)*u` for \( u = 0 \) at \( x = 0, 1 \).

**Weight decay** is given to DeepXDE as `regularization=("l2", λ)` on an FNN, which DeepXDE's PyTorch
backend applies as the optimizer's weight decay (L2 penalty for Adam, decoupled decay for AdamW).

## Defaults and limitations

- MsFFN and STMsFFN need an even first-hidden-layer width of at least 2 (half cosine, half sine features).
- Weight decay applies to Adam and AdamW phases only; it is passed to DeepXDE's FNN and is not available
  for PFNN, MsFFN or STMsFFN.
- ReLU has a zero second derivative and is a poor choice for equations with second derivatives (the page's
  help says so).

## Supported problem types

All. STMsFFN needs a time-dependent PDE (not an ODE, not stationary).

## Compatibility with other features

Refused by `validate()` or before Solve:

> STMsFFN separates space and time, so it needs a time-dependent problem; use MsFFN for a steady one.

> STMsFFN needs space and time; an ODE has only t -- use MsFFN.

> STMsFFN takes the network's last input as time: with an input-feature formula, end it with a feature of t alone (e.g.
> 'cos(x); sin(x); t') or use MsFFN.

> MsFFN: the first hidden layer needs an even number of neurons (half cosine, half sine Fourier features);
> got ….

> Weight decay can't be used with a PFNN network (DeepXDE's PFNN has no regularization): set it to 0, or
> use an FNN. (likewise MsFFN, STMsFFN)

> Weight decay is set > 0 (Settings > Optimizer Settings), but this run uses L-BFGS and/or NNCG somewhere
> (Training Phases, or RAR's optional L-BFGS polish) — both reject any nonzero weight decay.

> A Training Phase is set to use AdamW, which requires a nonzero weight decay.

Exact periodicity replaces the input transform, and an exact initial condition cannot be combined with an
output **Formula** (see [Exact constraints](exact-constraints.md)).

!!! info "New in v1.6.0 (in development)"
    The STMsFFN check of an input-feature formula, and a weight-decay check that looks only at the Training
    Phases (a hidden optimizer left over from older problems no longer blocks an AdamW-only run).

## Common mistakes and troubleshooting

- **Typed widths not accepted**: **Widths per layer:** turns red unless it is a list of positive whole
  numbers; while it holds widths, **Hidden layers:** and **Neurons per layer:** are disabled.
- **MsFFN does not help**: the scales set which frequencies the features represent; with input coordinates
  of order 1, \( \sigma \) of order of the solution's highest angular frequency is a starting point.

## Example

**Poisson 1D – multi-scale Fourier features** (MsFFN), **Wave 1D – space-time multi-scale Fourier
features** (STMsFFN, with an input-feature formula), **Linear elasticity 2D – plate** (PFNN),
**Laplace 2D – disk** (input-feature formula in polar coordinates) and **Poisson 1D – hard BCs** (output
formula) in **Examples**. See the [Neural network guide page](../guide/network.md).

## See also

- [Theory › PINNs](../theory/pinns.md)
- Examples that use it: [Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md), [Brinkman-Forchheimer 1D – unknown ν_e, K](../examples/1d/brinkman-forchheimer-1d-unknown-nu-e-k.md), [Cahn-Hilliard 1D – Wight & Zhao](../examples/1d/cahn-hilliard-1d-wight-zhao.md), [Diffusion 1D – hard IC and BCs](../examples/1d/diffusion-1d-hard-ic-and-bcs.md), [Diffusion-reaction 1D](../examples/1d/diffusion-reaction-1d.md), [Fractional diffusion 1D](../examples/1d/fractional-diffusion-1d.md) and 19 more in the [example gallery](../examples/index.md)

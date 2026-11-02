# Neural network

The **Neural network** page sets the network that approximates the solution: its size, its activation
function, its type, and optional transforms of its inputs and outputs.

![The Neural network page of the 1D Burgers example: Hidden layers 8, Neurons per layer 20, Widths per layer empty, Activation tanh, Kernel initializer Glorot uniform, Network type FNN, and the Enable input transform and Enable output transform check boxes](../assets/screenshots/burgers_network.png){ .screenshot loading=lazy }

Background: [Networks and Fourier features](../reference/networks.md),
[Exact constraints](../reference/exact-constraints.md).

## Inputs and outputs are set by the problem

You do not set the network's input and output sizes here:

- **inputs** are the coordinates: 1 (`t`) for an ODE; `x, t` in 1D, `x, y, t` in 2D, `x, y, z, t` in 3D
  for a time-dependent problem; one fewer (no `t`) for a stationary problem;
- **outputs** are the **Outputs** of the [Equations](equations.md) page.

The saved network is the list `layers = [inputs, hidden widths…, outputs]`, for example
`[2, 64, 64, 64, 1]` for the 1D default.

## Choose the size

**Hidden layers** and **Neurons per layer** give a network with that many layers of equal width. For
layers of different widths, type them in **Widths per layer**, for example `128, 64, 32`; this overrides
the two boxes above (they are greyed out, and **Hidden layers** shows the number of widths). A width
that is not a positive whole number turns the box red.

Most problems in the examples train with 3 to 6 layers of 32 to 100 neurons. A wider network can fit
sharper features but trains more slowly.

## Choose the activation and initializer

**Activation** offers tanh, sin, Sigmoid, ReLU, SiLU, Swish, ELU, GELU and SELU (DeepXDE's names).
tanh is the usual choice, since the residual needs smooth derivatives of the network; sin can suit
oscillating solutions; ReLU is a poor choice because its second derivative is zero.
**Kernel initializer** sets the starting weights: Glorot uniform (default), Glorot normal, He uniform,
He normal or zeros.

## Choose the network type

| **Network type** | What it is |
|---|---|
| **FNN** | One fully connected network for all outputs |
| **PFNN** | A separate sub-network per output, which can help when outputs differ a lot in size |
| **MsFFN** | Multi-scale Fourier feature network: random Fourier features of the inputs at the scales in **Fourier scales (σ)**, for solutions with both slow and fast variations |
| **STMsFFN** | The same with separate space and time features; **Time scales (σₜ)** sets the time scales. Needs a time-dependent problem with space (not an ODE) |

For MsFFN and STMsFFN the first hidden layer needs an even number of neurons (at least 2). Large σ
lets the network represent fast oscillations. Weight decay (**Settings › Optimizer Settings...**) can only
be used with an FNN: DeepXDE's PFNN, MsFFN and STMsFFN have no regularization, and **Solve** refuses the
combination.

!!! info "New in v1.6.0 (in development)"
    STMsFFN takes the network's last input as time. With an input-feature formula that does not end
    with `t`, **Solve** now refuses with an explanation.

## Rescale or transform the inputs

Tick **Enable input transform** and choose a **Mode:**

- **Scale/shift** — one row per input (`x:`, `y:`, …, `t:`): the network sees
  `x_raw × scale + shift`. Scale 1 and shift 0 change nothing.
- **Formula** — the network's input features, separated by `;`, written in `x, y, z, t`, for example
  `x*sin(y); x*cos(y)`. The first layer then takes one input per feature.

**Exact periodicity** on the Boundary conditions page builds its own periodic input features; the input
transform must then stay off (the page says so).

## Rescale the outputs, or impose conditions exactly with an output formula

Tick **Enable output transform** and choose a **Mode:**

- **Scale/shift** — one row per output (`Output 1:`, …): `y_raw × scale + shift`, useful when an output
  is much larger or smaller than 1.
- **Formula** — one formula per output (`Output 1 =` …) in the coordinates and the raw network output by
  name. Blank leaves that output unchanged.

### Impose conditions exactly with an output formula

A formula can build a condition into the network so that it holds exactly. For example
`x*(1-x)*u` makes the output zero at `x = 0` and `x = 1` whatever the network learns, so no boundary
rows are needed there. An exact initial condition (Initial conditions page) uses the output transform
too, as \( \mathrm{IC} + \tau N \) together with any scale/shift set here; it cannot be combined with
your own formula. The page shows a note while that is the case.

## Settings on this page

| GUI label | Meaning | Default (new problem) | Range | Saved as |
|---|---|---|---|---|
| **Hidden layers:** | Number of hidden layers | 3 | 1–50 | `layers` |
| **Neurons per layer:** | Width of each hidden layer | 64 | 1–4096 | `layers` |
| **Widths per layer:** | Comma-separated widths, overriding the two above | blank | | `layers` |
| **Activation:** | Activation function | tanh | | `activation` |
| **Kernel initializer:** | Initial weights | Glorot uniform | | `kernel_initializer` |
| **Network type:** | FNN, PFNN, MsFFN, STMsFFN | FNN | | `network_type` |
| **Fourier scales (σ):** | MsFFN/STMsFFN feature scales | `1, 10` | | `msffn_sigmas` |
| **Time scales (σₜ):** | STMsFFN time-feature scales | `1, 10` | | `msffn_sigmas_t` |
| **Enable input transform** | Transform the inputs | off | | `input_transform_enabled` |
| **Mode:** (input) | Scale/shift or Formula | Scale/shift | | `input_transform_mode` (`scale`, `formula`) |
| × … + … (per input) | Scale and shift | 1, 0 | | `input_transform_scale`, `input_transform_shift` |
| (input formula) | Input features, `;`-separated | blank | | `input_feature_formula` |
| **Enable output transform** | Transform the outputs | off | | `output_transform_enabled` |
| **Mode:** (output) | Scale/shift or Formula | Scale/shift | | `output_transform_mode` |
| × … + … (per output) | Scale and shift | 1, 0 | | `output_transform_scale`, `output_transform_shift` |
| **Output N =** | Output formula | blank (unchanged) | | `output_transform_formula` (separated by a vertical bar) |

# Physics-informed neural networks

A physics-informed neural network (PINN) represents the solution of a differential equation by a neural
network and trains the network so that it satisfies the equation, the boundary and initial conditions and,
for an inverse problem, measured data ([Raissi, Perdikaris & Karniadakis, 2019](../about/references.md#raissi2019);
[Lu et al., 2021](../about/references.md#lu2021deepxde)). No mesh is needed: the equation is enforced at a set of
points, and derivatives of the network are obtained exactly by automatic differentiation.

## The problem

PINNStudio solves problems of the form

\[
\begin{aligned}
\mathcal{N}_j[u](\mathbf{x}, t) &= 0, && \mathbf{x} \in \Omega,\ t \in (t_{\min}, t_{\max}],\quad j = 1, \dots, J,\\
\mathcal{B}_b[u](\mathbf{x}, t) &= g_b(\mathbf{x}, t), && \mathbf{x} \in \Gamma_b \subset \partial\Omega,\\
u_i(\mathbf{x}, t_{\min}) &= u_{0,i}(\mathbf{x}), && \mathbf{x} \in \Omega,
\end{aligned}
\]

where \(u = (u_1, \dots, u_m)\) are the outputs named on the **Equations** page, each \(\mathcal{N}_j\) is one
equation typed there (for example `du_t + u*du_x - 0.01/pi*du_xx`, which is
\( u_t + u u_x - \tfrac{0.01}{\pi} u_{xx} \)), each \(\mathcal{B}_b\) is one row of the **Boundary conditions**
page (Dirichlet, Neumann, Robin, periodic, operator, point-set or interface condition), and the initial
conditions come from the **Initial conditions** page. A steady problem has no \(t\) and no initial condition;
an ODE problem has \(t\) as its only variable. The number of equations \(J\) is normally the number of outputs
\(m\) (see [ODEs and PDEs](../reference/equations.md) and [Coupled systems](../reference/coupled.md)).

## The network

The solution is approximated by a network \(u_\theta : (\mathbf{x}, t) \mapsto \mathbb{R}^m\) with parameters
\(\theta\) (weights and biases). PINNStudio offers four architectures on the **Neural network** page
(**Network type:**); see [Networks and Fourier features](../reference/networks.md).

**FNN** (fully connected network, `dde.nn.FNN`). With input \(\mathbf{z}^{(0)} = (\mathbf{x}, t)\),

\[
\mathbf{z}^{(\ell)} = \sigma\!\left(W^{(\ell)} \mathbf{z}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right),\quad \ell = 1, \dots, L-1,
\qquad u_\theta = W^{(L)} \mathbf{z}^{(L-1)} + \mathbf{b}^{(L)},
\]

with the activation \(\sigma\) chosen under **Activation:** (`tanh`, `sin`, `Sigmoid`, `ReLU`, `SiLU`, `Swish`,
`ELU`, `GELU`, `SELU`) and the weights initialised as chosen under **Kernel initializer:** (Glorot or He,
uniform or normal, or zeros). **Hidden layers:** and **Neurons per layer:** set \(L-1\) and the widths. The
last layer is linear. All outputs share the hidden layers.

**PFNN** (`dde.nn.PFNN`) gives every output its own hidden layers of the chosen widths; the sub-networks are
independent.

**MsFFN** (multi-scale Fourier feature network; [Wang, Wang & Perdikaris, 2021](../about/references.md#wang2021fourier)).
For each scale \(\sigma_k\) listed under **Fourier scales (σ):** a fixed random matrix
\(B_k \in \mathbb{R}^{n_{\text{in}} \times F}\) with entries drawn from \(\mathcal{N}(0, \sigma_k^2)\) maps the input to
Fourier features

\[
\gamma_k(\mathbf{z}) = \left[\cos(\mathbf{z} B_k),\ \sin(\mathbf{z} B_k)\right] \in \mathbb{R}^{2F},
\]

each \(\gamma_k\) is passed through the same hidden layers \(H\) (shared weights), and the output is a linear
layer applied to the concatenation \([H(\gamma_1), \dots, H(\gamma_K)]\). In PINNStudio the first "hidden layer"
entry is the number of Fourier features \(2F\) (half cosine, half sine), so it must be even; the matrices
\(B_k\) are not trained and are saved with the network. **STMsFFN** (space-time version, for time-dependent
problems) builds separate features of \(\mathbf{x}\) (scales **Fourier scales (σ):**) and of \(t\) (scales
**Time scales (σₜ):**), multiplies every space branch with every time branch point-wise,
\(H(\gamma^{x}_k) \odot H(\gamma^{t}_l)\), and applies the linear output layer to the concatenation of all
products, as in the space-time network of the same paper.

**Input and output transforms.** The **Neural network** page can also replace the raw inputs by features
(for example `x*sin(y); x*cos(y)`) and apply a formula to the raw outputs. PINNStudio uses the same
mechanism for its [exact constraints](exact-constraints.md).

## Collocation points

The equation is enforced on finite point sets, the **Collocation points** page
([reference](../guide/points.md)):

| Set | Count | Used for |
|---|---|---|
| \(\mathcal{T}_r \subset \Omega \times (t_{\min}, t_{\max}]\) | **Domain points:** | the equation residuals |
| \(\mathcal{T}_b \subset \partial\Omega \times [t_{\min}, t_{\max}]\) | **Boundary points:** | the boundary conditions |
| \(\mathcal{T}_0 \subset \Omega \times \{t_{\min}\}\) | **Initial points:** | the initial conditions |
| \(\mathcal{T}_{\text{test}}\) | **Test points:** | the test loss reported in the Training Log (not trained on) |

DeepXDE draws these sets once before training with the method chosen under **Point distribution:**
(Hammersley, the default, and Halton, Sobol and LHS give evenly spread quasi-random points; `uniform` an
evenly spaced grid; `pseudo` independent random points). A boundary condition with a point list or a points
file, and measured data, add their own points. Extra points ("anchors") can be added to the residual set from a
file or a box, and RAR adds points during training ([Adaptive sampling](adaptive-sampling.md)). The
**Point Resampling** training callback redraws the residual (and optionally the boundary) points every given
number of steps.

## Training

Training minimises a weighted sum of mean losses over these point sets,

\[
\theta^\star = \arg\min_\theta \ \mathcal{L}(\theta), \qquad
\mathcal{L}(\theta) = \sum_{k} w_k \, \mathcal{L}_k(\theta),
\]

with gradient-based optimizers run in sequence (the **Training phases**: Adam, AdamW, L-BFGS, NNCG; see
[Optimizers and training phases](../reference/optimizers.md)). The loss terms and the derivatives they
need are described in [Losses and automatic differentiation](losses.md).

Two run settings affect what is computed:

- **Precision.** The script uses single precision (float32) unless **Settings › Float Precision...** selects
  float64 (for the whole run, or only for L-BFGS phases).
- **Random seed.** With **Settings › Random Seed...** on (the default, seed 2026) the script calls
  `dde.config.set_random_seed`, which seeds Python, NumPy and PyTorch, so point sampling and weight
  initialisation repeat between runs. GPU runs are not made bit-for-bit deterministic (cuDNN
  determinism is not switched on).

!!! note
    The training problem is non-convex. A small final loss does not by itself show that the network is
    close to the solution; compare with a reference solution ([Error measures](metrics.md)) and read
    [Scope and limitations](limitations.md).

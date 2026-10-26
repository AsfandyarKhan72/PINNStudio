DeepXDE example: [`Helmholtz_Dirichlet_2d_HPO.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d_HPO.py)

$$\Large -\nabla^2 u - k_0^2 u = k_0^2\sin(k_0 x)\sin(k_0 y), \qquad k_0 = 4\pi,\ (x, y) \in [0, 1]^2$$

- **Parameter Sweep, mode Bayesian search:** learning rate $10^{-4}$ to $5\times10^{-2}$ (log), hidden layers 1–10, neurons per layer 5–500, activation sin / sigmoid / tanh
- **Constants:** `n = 2`, `k0 = 2*pi*n`
- **In PINNStudio:** Examples → **Forward** → **Helmholtz 2D – hyperparameter search**

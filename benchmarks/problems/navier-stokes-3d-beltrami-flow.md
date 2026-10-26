DeepXDE example: [`Beltrami_flow.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Beltrami_flow.py)

$$\Large \frac{\partial \mathbf u}{\partial t} + (\mathbf u\cdot\nabla)\mathbf u + \nabla p = \nabla^2\mathbf u, \qquad \nabla\cdot\mathbf u = 0, \qquad (x, y, z) \in [-1, 1]^3,\ t \in [0, 1]$$

- **Exact solution:** e.g. $u = -\left(e^{x}\sin(y + z) + e^{z}\cos(x + y)\right)e^{-t}$
- **In PINNStudio:** Examples → **Forward** → **Navier-Stokes 3D – Beltrami flow**

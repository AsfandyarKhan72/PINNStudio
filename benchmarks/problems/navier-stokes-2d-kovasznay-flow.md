DeepXDE example: [`Kovasznay_flow.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Kovasznay_flow.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html))

$$\Large (\mathbf u\cdot\nabla)\mathbf u + \nabla p = \frac{1}{Re}\nabla^2\mathbf u, \qquad \nabla\cdot\mathbf u = 0, \qquad Re = 20$$

- **Domain:** $[-0.5, 1] \times [-0.5, 1.5]$; outputs $u, v, p$
- **Boundary conditions:** Dirichlet for $u$ and $v$ on the whole boundary, and for $p$ on the outlet $x = 1$
- **Exact solution:** $u = 1 - e^{\lambda x}\cos(2\pi y)$, $v = \frac{\lambda}{2\pi}e^{\lambda x}\sin(2\pi y)$, $p = \frac{1}{2}(1 - e^{2\lambda x})$, $\lambda = \frac{Re}{2} - \sqrt{\frac{Re^2}{4} + 4\pi^2}$
- **Constants:** `Re = 20`, `nu = 1/Re`, `l = 1/(2*nu) - sqrt(1/(4*nu**2) + 4*pi**2)`
- **In PINNStudio:** Examples → **Forward** → **Navier-Stokes 2D – Kovasznay flow**

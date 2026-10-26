The Poisson equation on the unit ball, steady-state (no time axis) — the 3D companion to the Poisson 2D – disk template.

$$\Large -\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} - \frac{\partial^2 u}{\partial z^2} = 1, \qquad x^2 + y^2 + z^2 \le 1$$

- **Boundary conditions:** Dirichlet, $u = 0$ on the boundary sphere
- **Geometry:** Sphere, center $(0,0,0)$, radius $1$
- **In PINNStudio:** Examples → **Forward** (or **Inverse**) → **Poisson 3D – sphere**

*Inverse mode* — same template, switched to **Inverse**, recovering the source term (fixed at $1$ in Forward mode) as an unknown parameter from 300 measured points

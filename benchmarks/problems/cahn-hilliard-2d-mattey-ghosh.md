Benchmark problem after [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474), section 6.2, IBVP-2: a pattern of stripes that sharpens into the two phases,

$$\Large \frac{\partial h}{\partial t'} = 0.005\left(\frac{\partial^2 \mu}{\partial x^2} + \frac{\partial^2 \mu}{\partial y^2}\right), \qquad \mu = h^3 - h - \alpha\left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right), \qquad (x, y) \in [0, 1]^2,\ t' \in [0, 1]$$

- **Initial condition:** $h(x, y, 0) = 0.4\cos(4\pi x)\cos(4\pi(x + y))$; $\mu$ has none
- **Boundary conditions:** Periodic, for $h$ and $\mu$
- **Geometry:** Rectangle
- **In PINNStudio:** Examples → **Forward** (or **Inverse**) → **Cahn-Hilliard 2D – Mattey & Ghosh**

*Inverse mode* — same template, switched to **Inverse**, recovering $\alpha$ and $\kappa$ (initial guesses 1 and 0.1, true values 0.0002 and 1) from $h$ measured at $t' = 1$.

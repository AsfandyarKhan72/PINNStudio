Benchmark problem after [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474), section 5.2 (eq. 25):

$$\Large \frac{\partial h}{\partial t} = \frac{\partial^2 \mu}{\partial x^2}, \qquad \mu = \kappa (h^3 - h) - \alpha\kappa \frac{\partial^2 h}{\partial x^2}, \qquad x \in [-1, 1],\ t \in [0, 1]$$

where $\alpha = 0.02$ and $\kappa = 1$. As in the paper (its phase-space form, eq. 19) and the Wight & Zhao example, $\mu$ is a second output, so no residual needs a fourth derivative. The same paper's 1D Allen-Cahn problem is the [Allen-Cahn 1D – Wight & Zhao](#1d-allen-cahn) example, and its 2D Allen-Cahn IBVP-1 is [Allen-Cahn 2D – Mattey & Ghosh](#2d-allen-cahn-mattey--ghosh).

- **Initial condition:** $h(x, 0) = \cos(\pi x) - \exp(-4(\pi x)^2)$; $\mu$ has none
- **Boundary conditions:** Periodic in value and slope, for $h$ and $\mu$
- **Geometry:** Interval
- **In PINNStudio:** Examples → **Forward** → **Cahn-Hilliard 1D – Mattey & Ghosh**

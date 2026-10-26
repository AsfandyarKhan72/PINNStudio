Benchmark problem after [Wight & Zhao (2021)](https://doi.org/10.4208/cicp.OA-2020-0086), section 3.2.

$$\Large \frac{\partial u}{\partial t} = \frac{\partial^2 \mu}{\partial x^2}, \qquad \mu = \gamma_2 (u^3 - u) - \gamma_1 \frac{\partial^2 u}{\partial x^2}, \qquad x \in [-1, 1],\ t \in [0, 1]$$

where $\gamma_1 = 10^{-6}$ and $\gamma_2 = 0.01$. The fourth-order equation is solved in mixed form: the chemical potential $\mu$ is a second output, so no residual needs a fourth derivative.

- **Initial condition:** $u(x, 0) = -\cos(2\pi x)$; $\mu$ has none, since its equation has no time derivative and determines $\mu$ at each time
- **Boundary conditions:** Periodic in value and slope, for $u$ and $\mu$
- **Geometry:** Interval
- **In PINNStudio:** Examples → **Forward** → **Cahn-Hilliard 1D – Wight & Zhao**

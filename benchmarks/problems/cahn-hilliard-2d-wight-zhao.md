Benchmark problem after [Wight & Zhao (2021)](https://doi.org/10.4208/cicp.OA-2020-0086), section 3.3 (eqs. 3.20–3.21): two bubbles that touch and merge into one, written for a field $\phi$,

$$\Large \frac{\partial \phi}{\partial t} = \lambda\left(\frac{\partial^2 \mu}{\partial x^2} + \frac{\partial^2 \mu}{\partial y^2}\right), \qquad \mu = \phi^3 - \phi - \varepsilon^2\left(\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2}\right), \qquad (x, y) \in [-1, 1]^2,\ t \in [0, 1]$$

where $\lambda = 1$ and $\varepsilon = 0.05$. As in the 1D example, the chemical potential $\mu$ is a second output (the paper's network is also $`(\phi, \mu)`$), so no residual needs a fourth derivative.

- **Initial condition:** the paper's two bubbles of radius $r = 0.4$ centred at $(\pm 0.28, 0)$, $\phi = \max\left(\tanh\frac{r - R_1}{2\varepsilon}, \tanh\frac{r - R_2}{2\varepsilon}\right) = \tanh\frac{r - \min(R_1, R_2)}{2\varepsilon}$, with the minimum written smoothly, $\min(R_1, R_2) \approx \tfrac12\left(R_1 + R_2 - \sqrt{(R_1 - R_2)^2 + 0.0004}\right)$. The plain minimum has a corner where the bubbles touch, and the curvature at that corner drives the merging. A network built on the exact initial condition cannot capture curvature at a corner, and with it every setup learned an almost frozen solution (16% / 30% error at $t = 0.25$ / 1, about what leaving $\phi$ unchanged gives). The smoothing changes $\phi$ by 0.5% (relative $L_2$), only along the line where the bubbles meet; the reference data uses the same formula. $\mu$ has no initial condition.
- **Boundary conditions:** Periodic, for $\phi$ and $\mu$
- **Geometry:** Rectangle
- **In PINNStudio:** Examples → **Forward** (or **Inverse**) → **Cahn-Hilliard 2D – Wight & Zhao**

*Inverse mode* — same template, switched to **Inverse**, recovering $\varepsilon^2$ (initial guess 1, true value 0.0025) from $\phi$ measured at $t = 1$.

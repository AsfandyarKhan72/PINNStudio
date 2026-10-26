Benchmark problem after [Wight & Zhao (2021)](https://doi.org/10.4208/cicp.OA-2020-0086), section 3.2 (eq. 3.13): a spherical drop inside the periodic cube $[0, 1]^3$ that shrinks under its own curvature,

$$\Large \frac{\partial \phi}{\partial t} = \lambda\left(\varepsilon^2\left(\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} + \frac{\partial^2 \phi}{\partial z^2}\right) - \phi^3 + \phi\right), \qquad (x, y, z) \in [0, 1]^3,\ t \in [0, 1]$$

where $\lambda = 10$ and $\varepsilon = 0.05$ ($\lambda\varepsilon^2 = 0.025$). Its radius goes from 0.35 to about 0.14 at $t = 1$.

- **Initial condition:** $\phi(x, y, z, 0) = \tanh\left(\dfrac{0.35 - \sqrt{(x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2}}{2\varepsilon}\right)$ (written with $+10^{-6}$ under the square root, which keeps its derivatives finite at the centre)
- **Boundary conditions:** Periodic in x, y and z
- **Geometry:** Cuboid
- **In PINNStudio:** Examples → **Forward** (or **Inverse**) → **Allen-Cahn 3D – Wight & Zhao**

*Inverse mode* — same template, switched to **Inverse**, recovering the two coefficients as they are written in the PDE box, the reaction coefficient $\lambda$ and the diffusion coefficient $\lambda\varepsilon^2$ (initial guesses 1, true values 10 and 0.025), from $\phi$ measured at $t = 1$ on every second point of the $51^3$ reference grid (*Measured data → Use: Grid, 26 points per axis*, 17,576 points).

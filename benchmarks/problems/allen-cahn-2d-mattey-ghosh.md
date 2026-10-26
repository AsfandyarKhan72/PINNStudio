Benchmark problem after [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474).

$$\Large \frac{\partial h}{\partial t} = c_1^2\left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right) - c_2(h^3 - h), \qquad (x, y) \in [0, 1]^2,\ t \in (0, 1]$$

where $c_1^2 = 0.0001$ and $c_2 = 1$.

- **Initial condition:** $h(x, y, 0) = \sin(4\pi x)\cos(4\pi y)$
- **Boundary conditions:** Periodic
- **Geometry:** Rectangle
- **In PINNStudio:** Examples → **Forward** (or **Inverse**) → **Allen-Cahn 2D – Mattey & Ghosh**

*Restore & visualize* — the trained model restored from its saved checkpoint and re-animated without retraining:

*Inverse mode* — same template, switched to **Inverse**, jointly recovering both $c_1^2$ (interfacial-thickness-squared) and $c_2$ (reaction) as unknown parameters; $c_2$'s initial guess starts at $0.1$ rather than $1.0$, since $1.0$ is already its true value. The template measures $h$ at every reference time after $t = 0$ ($t$ = 0.25, 0.5, 0.75 and 1, four data files, each with weight 100) and trains with Adam 20,000 → L-BFGS 20,000 (Forward keeps L-BFGS 10,000).

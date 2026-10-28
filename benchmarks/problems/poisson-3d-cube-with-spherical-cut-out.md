A Poisson problem on a domain built from two primitives: a cube with a sphere subtracted from its top. It shows a combined 3D geometry with a different kind of boundary condition on each part of its boundary, and is checked against the exact solution $u = e^{-(x^2 + y^2 + z^2)}$.

$$\Large -\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} - \frac{\partial^2 u}{\partial z^2} = \left(6 - 4(x^2 + y^2 + z^2)\right) e^{-(x^2 + y^2 + z^2)}$$

- **Geometry:** Custom: the cube $[-1, 1]^3$ minus the sphere of radius $R = 0.7$ centred at $(0, 0, 1)$ (*Cuboid − Sphere*)
- **Boundary conditions:** Dirichlet, $u = e^{-(x^2 + y^2 + z^2)}$ on the cube's faces (the named sides *left or right or bottom or top or front or back*); Neumann, $\partial u/\partial n = 2u\,(x^2 + y^2 + z(z - 1))/R$ on the spherical surface (the coordinate expression `np.isclose(x**2 + y**2 + (z - 1)**2, R**2)`; $n$ points out of the domain, into the sphere)
- **In PINNStudio:** Examples → **Forward** → **Poisson 3D – cube with spherical cut-out**

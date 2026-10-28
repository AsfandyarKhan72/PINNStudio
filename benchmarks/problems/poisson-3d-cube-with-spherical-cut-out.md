A Poisson problem on a domain built from two primitives: a cube with a sphere subtracted from its top. It shows a combined 3D geometry and boundary conditions on different parts of its boundary, and is checked against the exact solution $u = \sin(\pi x)\sin(\pi y)\sin(\pi z)$.

$$\Large -\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} - \frac{\partial^2 u}{\partial z^2} = 3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z)$$

- **Geometry:** Custom: the cube $[-1, 1]^3$ minus the sphere of radius $R = 0.7$ centred at $(0, 0, 1)$ (*Cuboid − Sphere*)
- **Boundary conditions:** Dirichlet, $u = 0$ on the cube's faces (the named sides *left or right or bottom or top or front or back*), and $u = \sin(\pi x)\sin(\pi y)\sin(\pi z)$ on the spherical surface (the coordinate expression `np.isclose(x**2 + y**2 + (z - 1)**2, R**2)`)
- **In PINNStudio:** Examples → **Forward** → **Poisson 3D – cube with spherical cut-out**

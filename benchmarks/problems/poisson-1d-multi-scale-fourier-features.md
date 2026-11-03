DeepXDE example: [`Poisson_multiscale_1d.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_multiscale_1d.py)

$$\Large \frac{d^2u}{dx^2} = -(\pi A)^2\sin(\pi A x) - 0.1(\pi B)^2\sin(\pi B x), \qquad x \in [0, 1],\ A = 2,\ B = 50$$

- **Boundary conditions:** $u(0) = u(1) = 0$
- **Exact solution:** $u = \sin(2\pi x) + 0.1\sin(50\pi x)$ — a slow wave plus a fast, small one
- **Constants:** `A = 2`, `B = 50`
- **In PINNStudio:** Examples → **Forward** → **Poisson 1D – multi-scale Fourier features**

A plain network of the same size learns the slow wave and largely misses the fast one; the Fourier features at scale 10 allow the PINN to represent the sin(50πx) component.

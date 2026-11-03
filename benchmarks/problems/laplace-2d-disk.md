DeepXDE example: [`Laplace_disk.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Laplace_disk.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/laplace.disk.html))

$$\Large r\frac{\partial u}{\partial r} + r^2\frac{\partial^2 u}{\partial r^2} + \frac{\partial^2 u}{\partial \theta^2} = 0, \qquad r \in [0, 1],\ \theta \in [0, 2\pi]$$

- **Boundary condition:** $u(1, \theta) = \cos\theta$
- **Exact solution:** $u = r\cos\theta$
- **In PINNStudio:** $x$ plays $r$ and $y$ plays $\theta$, so the plots show the $(r, \theta)$ rectangle
- **In PINNStudio:** Examples → **Forward** → **Laplace 2D – disk**

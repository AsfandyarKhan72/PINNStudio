DeepXDE example: [`heat.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/heat.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html))

$$\Large \frac{\partial u}{\partial t} = 0.4\frac{\partial^2 u}{\partial x^2}, \qquad x \in [0, 1],\ t \in [0, 1]$$

- **Initial condition:** $u(x, 0) = \sin(\pi x)$; **boundary conditions:** $u(0, t) = u(1, t) = 0$
- **Exact solution:** $u = e^{-0.4\pi^2 t}\sin(\pi x)$
- **Constants:** `a = 0.4`, `L = 1`, `n = 1`
- **In PINNStudio:** Examples → **Forward** (or **Inverse**) → **Heat 1D – Dirichlet**

The solution decays to about 2% of its initial size by $t = 1$, so the relative error grows over time even though the absolute error stays small.

*Inverse mode* — pick the example from the **Inverse** list: the diffusivity $a$ is unknown (initial guess 1, true value 0.4) and is learned together with $u$ from $u$ measured at $t = 1$ (the exact solution, [`reference_data/1D/heat/t_1.0.txt`](reference_data/1D/heat/t_1.0.txt)).

$$\Large \frac{\partial u}{\partial t} = a\ \frac{\partial^2 u}{\partial x^2}$$

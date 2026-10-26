DeepXDE example: [`Schrodinger.ipynb`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Schrodinger.ipynb)

$$\Large i\frac{\partial h}{\partial t} + \frac{1}{2}\frac{\partial^2 h}{\partial x^2} + |h|^2 h = 0, \qquad x \in [-5, 5],\ t \in [0, \pi/2]$$

$$\Large \frac{\partial u}{\partial t} + 0.5\frac{\partial^2 v}{\partial x^2} + (u^2 + v^2)\ v = 0$$

$$\Large \frac{\partial v}{\partial t} - 0.5\frac{\partial^2 u}{\partial x^2} - (u^2 + v^2)\ u = 0$$

- **Initial condition:** $h(x, 0) = 2\ \mathrm{sech}(x)$; **boundary conditions:** periodic in $u$, $v$ and their $x$-derivatives
- **In PINNStudio:** Examples → **Forward** (or **Inverse**) → **Schrödinger 1D**

*Inverse mode* — pick the example from the **Inverse** list: the coefficient $c$ of the second-derivative terms (0.5 in Forward mode; initial guess 1) is unknown and is learned from $|h| = \sqrt{u^2 + v^2}$ measured at $t = 0.75$, with the same training as the forward problem.

$$\Large \frac{\partial u}{\partial t} + c\frac{\partial^2 v}{\partial x^2} + (u^2 + v^2) v = 0, \qquad \frac{\partial v}{\partial t} - c\frac{\partial^2 u}{\partial x^2} - (u^2 + v^2) u = 0$$

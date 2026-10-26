DeepXDE example: [`Volterra_IDE.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py)

$$\Large \frac{du}{dt} + u(t) = \int_0^t e^{s - t}\ u(s)\ ds, \qquad t \in [0, 5]$$

- **Problem dimension:** ODE (t) — `du_t + u - integral(exp(s - t)*u)`, the kernel and $u$ written in the integration variable `s`
- **Initial condition:** $u(0) = 1$
- **Integral:** 20-point Gauss-Legendre quadrature; 10 uniformly spaced collocation points; L-BFGS only, as in DeepXDE's script
- **Exact solution:** $u = e^{-t}\cosh t$
- **In PINNStudio:** Examples → **Forward** → **Integro-differential ODE – Volterra kernel**

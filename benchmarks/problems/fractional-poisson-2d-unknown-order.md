DeepXDE example: [`fractional_Poisson_2d_inverse.py`](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/fractional_Poisson_2d_inverse.py)

$$\Large -(-\Delta)^{\alpha/2}u = 2^{\alpha_0}\Gamma(2 + \tfrac{\alpha_0}{2})\Gamma(1 + \tfrac{\alpha_0}{2})\left(1 - (1 + \tfrac{\alpha_0}{2})|x|^2\right), \qquad |x| < 1$$

- **Unknown:** the order $\alpha$ (true 1.8, start 1.5); $C(\alpha, 2)$ written with `lgamma` so it stays differentiable in α
- **Data:** 30 random measurements of $u = (1 - |x|^2)^{1 + \alpha_0/2}$, loss weight 100
- **Constants:** `a0 = 1.8`
- **In PINNStudio:** Examples → **Inverse** → **Fractional Poisson 2D – unknown order α**

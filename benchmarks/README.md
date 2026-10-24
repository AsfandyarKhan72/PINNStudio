# PINNStudio benchmarks

*Every example of PINNStudio, forward and inverse, with the result of its reference settings and PINNStudio's best. Updated as the runs finish.*

For a **DeepXDE example**, the reference is DeepXDE's own settings run in PINNStudio, and the best is the tuned run of the [tuning study](tuning_study.md). For a **template from a paper**, the table lists a plain PINN (soft initial and boundary conditions, the whole time interval, equal loss weights), the error the paper reports, the paper's method run in PINNStudio, and PINNStudio's best ([how the papers' methods are set up](paper_comparison.md)). Where a DeepXDE example has no inverse problem of its own, its inverse row is PINNStudio's inverse mode of it.

The error is the relative $L_2$ error of the main output (u, φ, h, …) at the last reference time, against the example's reference data; papers report one error over all times, so their rows also give that. So far 37 of the 65 examples have been run in the tuning study; the others show the README's result until they are.

![Every example: the reference settings and PINNStudio's best, forward and inverse, and the error of each learned parameter](figures/benchmarks_summary.png)

*Open grey markers: DeepXDE's settings, or the paper's method run in PINNStudio; filled: PINNStudio's best (one filled marker where the two are the same or the problem is not tuned yet). Right: the error of each learned parameter, labelled with its symbol.*

## ODEs

<details id="bench-integro-differential-ode">
<summary><strong>Integro-differential ODE</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-integro-differential-equation)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.5 × 10<sup>-4</sup> | 1.5 × 10<sup>-4</sup> (not tuned yet) |

*Forward, the README's result:*

![Integro-differential ODE: Forward, the README's result](../assets/Images/DeepXDE/Integro_differential_equation/line_comparison.png)

</details>

<details id="bench-integro-differential-ode-volterra-kernel">
<summary><strong>Integro-differential ODE – Volterra kernel</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-volterra-integro-differential-equation)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.5 × 10<sup>-3</sup> | 2.5 × 10<sup>-3</sup> (not tuned yet) |

*Forward, the README's result:*

![Integro-differential ODE – Volterra kernel: Forward, the README's result](../assets/Images/DeepXDE/Volterra_integro_differential_equation/line_comparison.png)

</details>

<details id="bench-linear-ode-system">
<summary><strong>Linear ODE system</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-ode-system) · [inverse runs in the tuning study](tuning_study.md#tune-linear-ode-system-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.6 × 10<sup>-3</sup> | 3.6 × 10<sup>-3</sup> (not tuned yet) |
| Inverse $L_2$ | 3.3 × 10<sup>-4</sup> | 2.0 × 10<sup>-4</sup> |
| Parameters found (true) | ω = 0.999776 (1) | ω = 0.999912 (1) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, the README's result:*

![Linear ODE system: Forward, the README's result](../assets/Images/DeepXDE/ODE_system/line_comparison_y1.png)

*Inverse, DeepXDE's settings:*

![Linear ODE system: Inverse, DeepXDE's settings](figures/tuning/ODE_system_inv/own_line_comparison_y1.png)

![Linear ODE system: Inverse, DeepXDE's settings](figures/tuning/ODE_system_inv/own_line_comparison_y2.png)

![Linear ODE system: Inverse, DeepXDE's settings](figures/tuning/ODE_system_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Linear ODE system: Inverse, PINNStudio best](figures/tuning/ODE_system_inv/tuned_line_comparison_y1.png)

![Linear ODE system: Inverse, PINNStudio best](figures/tuning/ODE_system_inv/tuned_line_comparison_y2.png)

![Linear ODE system: Inverse, PINNStudio best](figures/tuning/ODE_system_inv/tuned_param_plot.png)

</details>

<details id="bench-lotka-volterra-ode">
<summary><strong>Lotka-Volterra ODE</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-lotka-volterra-equation) · [inverse runs in the tuning study](tuning_study.md#tune-lotka-volterra-ode-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.1 × 10<sup>-3</sup> | 2.1 × 10<sup>-3</sup> (not tuned yet) |
| Inverse $L_2$ | 1.5 × 10<sup>-5</sup> | 1.5 × 10<sup>-5</sup> |
| Parameters found (true) | a = 2.00001 (2); d = 1.06 (1.06) | a = 2.00001 (2); d = 1.06 (1.06) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, the README's result:*

![Lotka-Volterra ODE: Forward, the README's result](../assets/Images/DeepXDE/Lotka_Volterra_equation/line_comparison_r.png)

*Inverse, DeepXDE's settings:*

![Lotka-Volterra ODE: Inverse, DeepXDE's settings](figures/tuning/Lotka_Volterra_equation_inv/own_line_comparison_p.png)

![Lotka-Volterra ODE: Inverse, DeepXDE's settings](figures/tuning/Lotka_Volterra_equation_inv/own_line_comparison_r.png)

![Lotka-Volterra ODE: Inverse, DeepXDE's settings](figures/tuning/Lotka_Volterra_equation_inv/own_param_plot.png)

</details>

<details id="bench-second-order-ode">
<summary><strong>Second-order ODE</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-second-order-ode) · [inverse runs in the tuning study](tuning_study.md#tune-second-order-ode-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 6.9 × 10<sup>-4</sup> | 6.9 × 10<sup>-4</sup> (not tuned yet) |
| Inverse $L_2$ | 1.6 × 10<sup>-3</sup> | 8.9 × 10<sup>-6</sup> |
| Parameters found (true) | a = 9.93327 (10); b = 9.6526 (9) | a = 10.0003 (10); b = 8.99262 (9) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, the README's result:*

![Second-order ODE: Forward, the README's result](../assets/Images/DeepXDE/Second_order_ODE/line_comparison.png)

*Inverse, DeepXDE's settings:*

![Second-order ODE: Inverse, DeepXDE's settings](figures/tuning/Second_order_ODE_inv/own_line_comparison.png)

![Second-order ODE: Inverse, DeepXDE's settings](figures/tuning/Second_order_ODE_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Second-order ODE: Inverse, PINNStudio best](figures/tuning/Second_order_ODE_inv/tuned_line_comparison.png)

![Second-order ODE: Inverse, PINNStudio best](figures/tuning/Second_order_ODE_inv/tuned_param_plot.png)

</details>

<details id="bench-lorenz-ode-unknown-c-c-c">
<summary><strong>Lorenz ODE – unknown C₁, C₂, C₃</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-lorenz-system)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 1.0 × 10<sup>-4</sup> | 1.0 × 10<sup>-4</sup> (not tuned yet) |
| Parameters found (true) | C₁ = 10 (10); C₂ = 14.9998 (15); C₃ = 2.66664 (2.66667) | C₁ = 10 (10); C₂ = 14.9998 (15); C₃ = 2.66664 (2.66667) |

*Inverse, the README's result:*

![Lorenz ODE – unknown C₁, C₂, C₃: Inverse, the README's result](../assets/Images/DeepXDE/Inverse_Lorenz_system/line_comparison_y1.png)

</details>

<details id="bench-lorenz-ode-unknown-c-c-c-exogenous-input">
<summary><strong>Lorenz ODE – unknown C₁, C₂, C₃, exogenous input</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-lorenz-system-with-exogenous-input)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 1.6 × 10<sup>-3</sup> | 1.6 × 10<sup>-3</sup> (not tuned yet) |
| Parameters found (true) | C₁ = 9.974 (10); C₂ = 14.996 (15); C₃ = 2.6669 (2.66667) | C₁ = 9.974 (10); C₂ = 14.996 (15); C₃ = 2.6669 (2.66667) |

*Inverse, the README's result:*

![Lorenz ODE – unknown C₁, C₂, C₃, exogenous input: Inverse, the README's result](../assets/Images/DeepXDE/Inverse_Lorenz_system_with_exogenous_input/line_comparison_y1.png)

</details>

## 1D, steady

<details id="bench-euler-beam-1d">
<summary><strong>Euler beam 1D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-euler-beam) · [forward runs in the tuning study](tuning_study.md#tune-euler-beam-1d) · [inverse runs in the tuning study](tuning_study.md#tune-euler-beam-1d-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.5 × 10<sup>-3</sup> | 2.8 × 10<sup>-4</sup> |
| Inverse $L_2$ | 1.6 × 10<sup>-3</sup> | 3.8 × 10<sup>-5</sup> |
| Parameters found (true) | EI = 1.02349 (1) | EI = 1.00075 (1) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, DeepXDE's settings:*

![Euler beam 1D: Forward, DeepXDE's settings](figures/tuning/Euler_beam/own_line_comparison.png)

*Forward, PINNStudio best:*

![Euler beam 1D: Forward, PINNStudio best](figures/tuning/Euler_beam/tuned_line_comparison.png)

*Inverse, DeepXDE's settings:*

![Euler beam 1D: Inverse, DeepXDE's settings](figures/tuning/Euler_beam_inv/own_line_comparison.png)

![Euler beam 1D: Inverse, DeepXDE's settings](figures/tuning/Euler_beam_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Euler beam 1D: Inverse, PINNStudio best](figures/tuning/Euler_beam_inv/tuned_line_comparison.png)

![Euler beam 1D: Inverse, PINNStudio best](figures/tuning/Euler_beam_inv/tuned_param_plot.png)

</details>

<details id="bench-fractional-poisson-1d">
<summary><strong>Fractional Poisson 1D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-fractional-poisson-equation-1d) · [forward runs in the tuning study](tuning_study.md#tune-fractional-poisson-1d)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.3 × 10<sup>-2</sup> | 1.2 × 10<sup>-2</sup> |

*Forward, DeepXDE's settings:*

![Fractional Poisson 1D: Forward, DeepXDE's settings](figures/tuning/Fractional_Poisson_equation_1D/own_line_comparison.png)

*Forward, PINNStudio best:*

![Fractional Poisson 1D: Forward, PINNStudio best](figures/tuning/Fractional_Poisson_equation_1D/tuned_line_comparison.png)

</details>

<details id="bench-poisson-1d-dirichlet">
<summary><strong>Poisson 1D – Dirichlet</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-poisson-1d-dirichlet)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.7 × 10<sup>-4</sup> | 3.7 × 10<sup>-4</sup> (not tuned yet) |

*Forward, the README's result:*

![Poisson 1D – Dirichlet: Forward, the README's result](../assets/Images/DeepXDE/Poisson_1D_Dirichlet/line_comparison.png)

</details>

<details id="bench-poisson-1d-dirichlet-neumann">
<summary><strong>Poisson 1D – Dirichlet/Neumann</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-poisson-1d-dirichlet-neumann)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.3 × 10<sup>-4</sup> | 2.3 × 10<sup>-4</sup> (not tuned yet) |

*Forward, the README's result:*

![Poisson 1D – Dirichlet/Neumann: Forward, the README's result](../assets/Images/DeepXDE/Poisson_1D_Dirichlet_Neumann/line_comparison.png)

</details>

<details id="bench-poisson-1d-dirichlet-periodic">
<summary><strong>Poisson 1D – Dirichlet/Periodic</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-poisson-1d-dirichlet-periodic) · [forward runs in the tuning study](tuning_study.md#tune-poisson-1d-dirichlet-periodic)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 9.6 × 10<sup>-3</sup> | 6.4 × 10<sup>-6</sup> |

*Forward, DeepXDE's settings:*

![Poisson 1D – Dirichlet/Periodic: Forward, DeepXDE's settings](figures/tuning/Poisson_1D_Dirichlet_Periodic/own_line_comparison.png)

*Forward, PINNStudio best:*

![Poisson 1D – Dirichlet/Periodic: Forward, PINNStudio best](figures/tuning/Poisson_1D_Dirichlet_Periodic/tuned_line_comparison.png)

</details>

<details id="bench-poisson-1d-dirichlet-point-set-operator">
<summary><strong>Poisson 1D – Dirichlet/point-set operator</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-poisson-1d-dirichlet-pointsetoperator)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.6 × 10<sup>-5</sup> | 2.6 × 10<sup>-5</sup> (not tuned yet) |

*Forward, the README's result:*

![Poisson 1D – Dirichlet/point-set operator: Forward, the README's result](../assets/Images/DeepXDE/Poisson_1D_Dirichlet_PointSetOperator/line_comparison.png)

</details>

<details id="bench-poisson-1d-dirichlet-robin">
<summary><strong>Poisson 1D – Dirichlet/Robin</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-poisson-1d-dirichlet-robin)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.4 × 10<sup>-4</sup> | 2.4 × 10<sup>-4</sup> (not tuned yet) |

*Forward, the README's result:*

![Poisson 1D – Dirichlet/Robin: Forward, the README's result](../assets/Images/DeepXDE/Poisson_1D_Dirichlet_Robin/line_comparison.png)

</details>

<details id="bench-poisson-1d-hard-bcs">
<summary><strong>Poisson 1D – hard BCs</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-poisson-1d-hard-boundary-conditions)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.2 × 10<sup>-6</sup> | 2.2 × 10<sup>-6</sup> (not tuned yet) |

*Forward, the README's result:*

![Poisson 1D – hard BCs: Forward, the README's result](../assets/Images/DeepXDE/Poisson_1D_hard_boundary_conditions/line_comparison.png)

</details>

<details id="bench-poisson-1d-multi-scale-fourier-features">
<summary><strong>Poisson 1D – multi-scale Fourier features</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-poisson-1d-multi-scale-fourier-features)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.9 × 10<sup>-3</sup> | 3.9 × 10<sup>-3</sup> (not tuned yet) |

*Forward, the README's result:*

![Poisson 1D – multi-scale Fourier features: Forward, the README's result](../assets/Images/DeepXDE/Poisson_1D_multi_scale_Fourier_features/line_comparison.png)

</details>

<details id="bench-brinkman-forchheimer-1d-unknown-e-k">
<summary><strong>Brinkman-Forchheimer 1D – unknown ν_e, K</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-brinkman-forchheimer-model) · [forward runs in the tuning study](tuning_study.md#tune-brinkman-forchheimer-1d-unknown-e-k)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 2.0 × 10<sup>-2</sup> | 1.2 × 10<sup>-2</sup> |
| Parameters found (true) | νₑ = 0.00127637 (0.001); K = 0.00103624 (0.001) | νₑ = 0.00117644 (0.001); K = 0.00100487 (0.001) |

*Inverse, DeepXDE's settings:*

![Brinkman-Forchheimer 1D – unknown ν_e, K: Inverse, DeepXDE's settings](figures/tuning/Inverse_Brinkman_Forchheimer_model/own_line_comparison.png)

![Brinkman-Forchheimer 1D – unknown ν_e, K: Inverse, DeepXDE's settings](figures/tuning/Inverse_Brinkman_Forchheimer_model/own_param_plot.png)

*Inverse, PINNStudio best:*

![Brinkman-Forchheimer 1D – unknown ν_e, K: Inverse, PINNStudio best](figures/tuning/Inverse_Brinkman_Forchheimer_model/tuned_line_comparison.png)

![Brinkman-Forchheimer 1D – unknown ν_e, K: Inverse, PINNStudio best](figures/tuning/Inverse_Brinkman_Forchheimer_model/tuned_param_plot.png)

</details>

<details id="bench-diffusion-reaction-1d-unknown-rate-field">
<summary><strong>Diffusion-reaction 1D – unknown rate field</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-diffusion-reaction-with-space-dependent-rate) · [forward runs in the tuning study](tuning_study.md#tune-diffusion-reaction-1d-unknown-rate-field)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 4.0 × 10<sup>-2</sup> | 2.8 × 10<sup>-2</sup> |
| Parameters found (true) | – | – |

*Inverse, DeepXDE's settings:*

![Diffusion-reaction 1D – unknown rate field: Inverse, DeepXDE's settings](figures/tuning/Inverse_diffusion_reaction_with_space_dependent_rate/own_line_comparison.png)

*Inverse, PINNStudio best:*

![Diffusion-reaction 1D – unknown rate field: Inverse, PINNStudio best](figures/tuning/Inverse_diffusion_reaction_with_space_dependent_rate/tuned_line_comparison.png)

</details>

<details id="bench-fractional-poisson-1d-unknown-order">
<summary><strong>Fractional Poisson 1D – unknown order α</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-fractional-poisson-equation-1d)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 4.9 × 10<sup>-3</sup> | 4.9 × 10<sup>-3</sup> (not tuned yet) |
| Parameters found (true) | α = 1.79183 (1.8) | α = 1.79183 (1.8) |

*Inverse, the README's result:*

![Fractional Poisson 1D – unknown order α: Inverse, the README's result](../assets/Images/DeepXDE/Inverse_fractional_Poisson_equation_1D/line_comparison.png)

</details>

<details id="bench-poisson-1d-unknown-forcing-field">
<summary><strong>Poisson 1D – unknown forcing field</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-poisson-with-unknown-forcing-field) · [forward runs in the tuning study](tuning_study.md#tune-poisson-1d-unknown-forcing-field)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 5.4 × 10<sup>-4</sup> | 2.1 × 10<sup>-4</sup> |
| Parameters found (true) | – | – |

*Inverse, DeepXDE's settings:*

![Poisson 1D – unknown forcing field: Inverse, DeepXDE's settings](figures/tuning/Inverse_Poisson_with_unknown_forcing_field/own_line_comparison_q.png)

![Poisson 1D – unknown forcing field: Inverse, DeepXDE's settings](figures/tuning/Inverse_Poisson_with_unknown_forcing_field/own_line_comparison_u.png)

*Inverse, PINNStudio best:*

![Poisson 1D – unknown forcing field: Inverse, PINNStudio best](figures/tuning/Inverse_Poisson_with_unknown_forcing_field/tuned_line_comparison_q.png)

![Poisson 1D – unknown forcing field: Inverse, PINNStudio best](figures/tuning/Inverse_Poisson_with_unknown_forcing_field/tuned_line_comparison_u.png)

</details>

<details id="bench-poisson-1d-unknown-forcing-field-batches">
<summary><strong>Poisson 1D – unknown forcing field (batches)</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-poisson-with-unknown-forcing-field-batches) · [forward runs in the tuning study](tuning_study.md#tune-poisson-1d-unknown-forcing-field-batches)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 4.5 × 10<sup>-3</sup> | 7.8 × 10<sup>-4</sup> |
| Parameters found (true) | – | – |

*Inverse, DeepXDE's settings:*

![Poisson 1D – unknown forcing field (batches): Inverse, DeepXDE's settings](figures/tuning/Inverse_Poisson_with_unknown_forcing_field_batches/own_line_comparison_q.png)

![Poisson 1D – unknown forcing field (batches): Inverse, DeepXDE's settings](figures/tuning/Inverse_Poisson_with_unknown_forcing_field_batches/own_line_comparison_u.png)

*Inverse, PINNStudio best:*

![Poisson 1D – unknown forcing field (batches): Inverse, PINNStudio best](figures/tuning/Inverse_Poisson_with_unknown_forcing_field_batches/tuned_line_comparison_q.png)

![Poisson 1D – unknown forcing field (batches): Inverse, PINNStudio best](figures/tuning/Inverse_Poisson_with_unknown_forcing_field_batches/tuned_line_comparison_u.png)

</details>

## 1D, time-dependent

<details id="bench-allen-cahn-1d">
<summary><strong>Allen-Cahn 1D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-allen-cahn-equation) · [forward runs in the tuning study](tuning_study.md#tune-allen-cahn-1d) · [inverse runs in the tuning study](tuning_study.md#tune-allen-cahn-1d-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.4 × 10<sup>-2</sup> | 1.2 × 10<sup>-2</sup> |
| Inverse $L_2$ | 5.4 × 10<sup>-4</sup> | 1.1 × 10<sup>-4</sup> |
| Parameters found (true) | d = 0.00100154 (0.001); c = 4.9979 (5) | d = 0.00100055 (0.001); c = 5.00002 (5) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, DeepXDE's settings:*

![Allen-Cahn 1D: Forward, DeepXDE's settings](figures/tuning/Allen_Cahn_equation/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Allen-Cahn 1D: Forward, PINNStudio best](figures/tuning/Allen_Cahn_equation/tuned_surface_comparison.png)

*Inverse, DeepXDE's settings:*

![Allen-Cahn 1D: Inverse, DeepXDE's settings](figures/tuning/Allen_Cahn_equation_inv/own_surface_comparison.png)

![Allen-Cahn 1D: Inverse, DeepXDE's settings](figures/tuning/Allen_Cahn_equation_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Allen-Cahn 1D: Inverse, PINNStudio best](figures/tuning/Allen_Cahn_equation_inv/tuned_surface_comparison.png)

![Allen-Cahn 1D: Inverse, PINNStudio best](figures/tuning/Allen_Cahn_equation_inv/tuned_param_plot.png)

</details>

<details id="bench-allen-cahn-1d-wight-zhao-time-adaptive">
<summary><strong>Allen-Cahn 1D – Wight & Zhao (time-adaptive)</strong> · PINNStudio template, after Wight & Zhao (2021)</summary>

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Wight & Zhao's method: reported in Wight & Zhao (2021), Table 3.1 | – | 2.3 × 10<sup>-2</sup> | |
| Wight & Zhao's method, run in PINNStudio | – | – | |
| Mattey & Ghosh's bc-PINN: reported in Mattey & Ghosh (2022), Table 2 | – | 1.7 × 10<sup>-2</sup> | |
| Mattey & Ghosh's bc-PINN, run in PINNStudio | – | – | |
| The template as it loads | 4.8 × 10<sup>-3</sup> | – | |
| **PINNStudio best** (forward) | 4.8 × 10<sup>-3</sup> | – | |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

</details>

<details id="bench-allen-cahn-1d-wight-zhao-whole-interval">
<summary><strong>Allen-Cahn 1D – Wight & Zhao (whole interval)</strong> · PINNStudio template, after Wight & Zhao (2021)</summary>

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Wight & Zhao's method: reported in Wight & Zhao (2021), Table 3.1 | – | 2.3 × 10<sup>-2</sup> | |
| Wight & Zhao's method, run in PINNStudio | – | – | |
| Mattey & Ghosh's bc-PINN: reported in Mattey & Ghosh (2022), Table 2 | – | 1.7 × 10<sup>-2</sup> | |
| Mattey & Ghosh's bc-PINN, run in PINNStudio | – | – | |
| The template as it loads | 1.4 × 10<sup>-3</sup> | – | |
| **PINNStudio best** (forward) | 1.4 × 10<sup>-3</sup> | – | |
| Inverse, the template's own settings | 2.1 × 10<sup>-4</sup> | – | γ₁ = 9.81621e-05 (0.0001); γ₂ = 4.99953 (5) |
| **PINNStudio best** (inverse) | 2.1 × 10<sup>-4</sup> | – | γ₁ = 9.81621e-05 (0.0001); γ₂ = 4.99953 (5) |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

</details>

<details id="bench-cahn-hilliard-1d-wight-zhao">
<summary><strong>Cahn-Hilliard 1D – Wight & Zhao</strong> · PINNStudio template, after Wight & Zhao (2021)</summary>

[problem and settings in the README](../README.md#1d-cahn-hilliard)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Wight & Zhao's method: reported in Wight & Zhao (2021), Sec. 3.2 | – | 9.5 × 10<sup>-3</sup> | |
| Wight & Zhao's method, run in PINNStudio | – | – | |
| The template as it loads | 6.0 × 10<sup>-4</sup> | – | |
| **PINNStudio best** (forward) | 6.0 × 10<sup>-4</sup> | – | |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the README's result:*

![Cahn-Hilliard 1D – Wight & Zhao: Forward, the README's result](../assets/Images/1D-CahnHilliard/surface_comparison_u.png)

</details>

<details id="bench-cahn-hilliard-1d-mattey-ghosh">
<summary><strong>Cahn-Hilliard 1D – Mattey & Ghosh</strong> · PINNStudio template, after Mattey & Ghosh (2022)</summary>

[problem and settings in the README](../README.md#1d-cahn-hilliard-mattey--ghosh) · [forward runs in the tuning study](tuning_study.md#tune-cahn-hilliard-1d-mattey-ghosh-pinnstudio-template-best-setup)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Mattey & Ghosh's bc-PINN: reported in Mattey & Ghosh (2022), Sec. 5.2 | – | 1.9 × 10<sup>-2</sup> | |
| Mattey & Ghosh's bc-PINN, run in PINNStudio | – | – | |
| The template as it loads | 4.8 × 10<sup>-3</sup> | – | |
| **PINNStudio best** (forward) | 3.9 × 10<sup>-3</sup> | – | |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the README's result:*

![Cahn-Hilliard 1D – Mattey & Ghosh: Forward, the README's result](../assets/Images/1D-CahnHilliard-Mattey/surface_comparison_u.png)

*Forward, PINNStudio best:*

![Cahn-Hilliard 1D – Mattey & Ghosh: Forward, PINNStudio best](figures/tuning/Template_1D_Cahn_Hilliard_Mattey_Ghosh_best_setup/own_surface_comparison_mu.png)

![Cahn-Hilliard 1D – Mattey & Ghosh: Forward, PINNStudio best](figures/tuning/Template_1D_Cahn_Hilliard_Mattey_Ghosh_best_setup/own_surface_comparison_u.png)

</details>

<details id="bench-black-scholes-1d-american-put">
<summary><strong>Black-Scholes 1D – American put</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-american-put-option)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|

</details>

<details id="bench-black-scholes-1d-european-call">
<summary><strong>Black-Scholes 1D – European call</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-black-scholes-equation-call-option)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 4.3 × 10<sup>-4</sup> | 4.3 × 10<sup>-4</sup> (not tuned yet) |

*Forward, the README's result:*

![Black-Scholes 1D – European call: Forward, the README's result](../assets/Images/DeepXDE/Black_Scholes_equation_call_option/animation.gif)

</details>

<details id="bench-burgers-1d-raissi-et-al">
<summary><strong>Burgers 1D – Raissi et al.</strong> · PINNStudio template, after Raissi et al. (2019)</summary>

[problem and settings in the README](../README.md#1d-burgers) · [forward runs in the tuning study](tuning_study.md#tune-burgers-1d-raissi-et-al-pinnstudio-template-best-setup)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Raissi et al.'s settings: reported in Raissi et al. (2019), Sec. 2.1 | – | 6.7 × 10<sup>-4</sup> | |
| Raissi et al.'s settings, run in PINNStudio | – | – | |
| The template as it loads | 1.3 × 10<sup>-3</sup> | – | |
| **PINNStudio best** (forward) | 5.2 × 10<sup>-4</sup> | – | |
| Inverse, the template's own settings | 7.8 × 10<sup>-4</sup> | – | λ₁ = 0.996107 (1); λ₂ = 0.0100386 (0.01) |
| **PINNStudio best** (inverse) | 7.8 × 10<sup>-4</sup> | – | λ₁ = 0.996107 (1); λ₂ = 0.0100386 (0.01) |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the README's result:*

![Burgers 1D – Raissi et al.: Forward, the README's result](../assets/Images/1D-Burgers/surface_comparison.png)

*Forward, PINNStudio best:*

![Burgers 1D – Raissi et al.: Forward, PINNStudio best](figures/tuning/Template_1D_Burgers_best_setup/tuned_surface_comparison.png)

*Inverse, the README's result:*

![Burgers 1D – Raissi et al.: Inverse, the README's result](../assets/Images/1D-Burgers/param_convergence_animation.gif)

</details>

<details id="bench-burgers-1d-rar">
<summary><strong>Burgers 1D – RAR</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-burgers-equation-with-rar)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 8.4 × 10<sup>-4</sup> | 8.4 × 10<sup>-4</sup> (not tuned yet) |

*Forward, the README's result:*

![Burgers 1D – RAR: Forward, the README's result](../assets/Images/DeepXDE/Burgers_equation_with_RAR/animation.gif)

</details>

<details id="bench-diffusion-1d">
<summary><strong>Diffusion 1D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-diffusion-equation) · [forward runs in the tuning study](tuning_study.md#tune-diffusion-1d)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.3 × 10<sup>-2</sup> | 1.5 × 10<sup>-3</sup> |

*Forward, DeepXDE's settings:*

![Diffusion 1D: Forward, DeepXDE's settings](figures/tuning/Diffusion_equation/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Diffusion 1D: Forward, PINNStudio best](figures/tuning/Diffusion_equation/tuned_surface_comparison.png)

</details>

<details id="bench-diffusion-1d-hard-ic-and-bcs">
<summary><strong>Diffusion 1D – hard IC and BCs</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-diffusion-equation-with-hard-ic-and-bc)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.8 × 10<sup>-3</sup> | 2.8 × 10<sup>-3</sup> (not tuned yet) |

*Forward, the README's result:*

![Diffusion 1D – hard IC and BCs: Forward, the README's result](../assets/Images/DeepXDE/Diffusion_equation_with_hard_IC_and_BC/animation.gif)

</details>

<details id="bench-diffusion-1d-resampling">
<summary><strong>Diffusion 1D – resampling</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-diffusion-equation-with-resampling) · [forward runs in the tuning study](tuning_study.md#tune-diffusion-1d-resampling)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 5.4 × 10<sup>-2</sup> | 5.8 × 10<sup>-3</sup> |

*Forward, DeepXDE's settings:*

![Diffusion 1D – resampling: Forward, DeepXDE's settings](figures/tuning/Diffusion_equation_with_resampling/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Diffusion 1D – resampling: Forward, PINNStudio best](figures/tuning/Diffusion_equation_with_resampling/tuned_surface_comparison.png)

</details>

<details id="bench-diffusion-reaction-1d">
<summary><strong>Diffusion-reaction 1D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-diffusion-reaction-equation) · [forward runs in the tuning study](tuning_study.md#tune-diffusion-reaction-1d) · [inverse runs in the tuning study](tuning_study.md#tune-diffusion-reaction-1d-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 8.8 × 10<sup>-2</sup> | 8.5 × 10<sup>-5</sup> |
| Inverse $L_2$ | 6.2 × 10<sup>-4</sup> | 1.4 × 10<sup>-4</sup> |
| Parameters found (true) | D = 0.998802 (1) | D = 0.999773 (1) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, DeepXDE's settings:*

![Diffusion-reaction 1D: Forward, DeepXDE's settings](figures/tuning/Diffusion_reaction_equation/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Diffusion-reaction 1D: Forward, PINNStudio best](figures/tuning/Diffusion_reaction_equation/tuned_surface_comparison.png)

*Inverse, DeepXDE's settings:*

![Diffusion-reaction 1D: Inverse, DeepXDE's settings](figures/tuning/Diffusion_reaction_equation_inv/own_surface_comparison.png)

![Diffusion-reaction 1D: Inverse, DeepXDE's settings](figures/tuning/Diffusion_reaction_equation_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Diffusion-reaction 1D: Inverse, PINNStudio best](figures/tuning/Diffusion_reaction_equation_inv/tuned_surface_comparison.png)

![Diffusion-reaction 1D: Inverse, PINNStudio best](figures/tuning/Diffusion_reaction_equation_inv/tuned_param_plot.png)

</details>

<details id="bench-fractional-diffusion-1d">
<summary><strong>Fractional diffusion 1D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-fractional-diffusion-equation)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.3 × 10<sup>-3</sup> | 3.3 × 10<sup>-3</sup> (not tuned yet) |

*Forward, the README's result:*

![Fractional diffusion 1D: Forward, the README's result](../assets/Images/DeepXDE/Fractional_diffusion_equation/animation.gif)

</details>

<details id="bench-heat-1d-dirichlet">
<summary><strong>Heat 1D – Dirichlet</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-heat-equation) · [forward runs in the tuning study](tuning_study.md#tune-heat-1d-dirichlet) · [inverse runs in the tuning study](tuning_study.md#tune-heat-1d-dirichlet-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 5.0 × 10<sup>-2</sup> | 1.3 × 10<sup>-2</sup> |
| Inverse $L_2$ | 5.1 × 10<sup>-3</sup> | 1.1 × 10<sup>-3</sup> |
| Parameters found (true) | a = 0.397247 (0.4) | a = 0.400406 (0.4) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, DeepXDE's settings:*

![Heat 1D – Dirichlet: Forward, DeepXDE's settings](figures/tuning/Heat_equation/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Heat 1D – Dirichlet: Forward, PINNStudio best](figures/tuning/Heat_equation/tuned_surface_comparison.png)

*Inverse, DeepXDE's settings:*

![Heat 1D – Dirichlet: Inverse, DeepXDE's settings](figures/tuning/Heat_equation_inv/own_surface_comparison.png)

![Heat 1D – Dirichlet: Inverse, DeepXDE's settings](figures/tuning/Heat_equation_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Heat 1D – Dirichlet: Inverse, PINNStudio best](figures/tuning/Heat_equation_inv/tuned_surface_comparison.png)

![Heat 1D – Dirichlet: Inverse, PINNStudio best](figures/tuning/Heat_equation_inv/tuned_param_plot.png)

</details>

<details id="bench-heat-1d-resampling">
<summary><strong>Heat 1D – resampling</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-heat-equation-with-resampling) · [forward runs in the tuning study](tuning_study.md#tune-heat-1d-resampling)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.4 × 10<sup>-2</sup> | 3.3 × 10<sup>-3</sup> |

*Forward, DeepXDE's settings:*

![Heat 1D – resampling: Forward, DeepXDE's settings](figures/tuning/Heat_equation_with_resampling/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Heat 1D – resampling: Forward, PINNStudio best](figures/tuning/Heat_equation_with_resampling/tuned_surface_comparison.png)

</details>

<details id="bench-klein-gordon-1d">
<summary><strong>Klein-Gordon 1D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-klein-gordon-equation) · [forward runs in the tuning study](tuning_study.md#tune-klein-gordon-1d)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 6.9 × 10<sup>-2</sup> | 2.3 × 10<sup>-3</sup> |

*Forward, DeepXDE's settings:*

![Klein-Gordon 1D: Forward, DeepXDE's settings](figures/tuning/Klein_Gordon_equation/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Klein-Gordon 1D: Forward, PINNStudio best](figures/tuning/Klein_Gordon_equation/tuned_surface_comparison.png)

</details>

<details id="bench-schr-dinger-1d">
<summary><strong>Schrödinger 1D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-schr-dinger-equation) · [inverse runs in the tuning study](tuning_study.md#tune-schr-dinger-1d-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.8 × 10<sup>-3</sup> | 1.8 × 10<sup>-3</sup> (not tuned yet) |
| Inverse $L_2$ | 1.8 × 10<sup>-1</sup> | 9.9 × 10<sup>-4</sup> |
| Parameters found (true) | c = 0.514806 (0.5) | c = 0.500036 (0.5) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, the README's result:*

![Schrödinger 1D: Forward, the README's result](../assets/Images/DeepXDE/Schr_dinger_equation/animation.gif)

*Inverse, DeepXDE's settings:*

![Schrödinger 1D: Inverse, DeepXDE's settings](figures/tuning/Schr_dinger_equation_inv/own_surface_comparison.png)

![Schrödinger 1D: Inverse, DeepXDE's settings](figures/tuning/Schr_dinger_equation_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Schrödinger 1D: Inverse, PINNStudio best](figures/tuning/Schr_dinger_equation_inv/tuned_surface_comparison.png)

![Schrödinger 1D: Inverse, PINNStudio best](figures/tuning/Schr_dinger_equation_inv/tuned_param_plot.png)

</details>

<details id="bench-wave-1d-space-time-multi-scale-fourier-features">
<summary><strong>Wave 1D – space-time multi-scale Fourier features</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-wave-propagation-space-time-multi-scale-fourier-features) · [forward runs in the tuning study](tuning_study.md#tune-wave-1d-space-time-multi-scale-fourier-features)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.0 × 10<sup>-1</sup> | 1.5 × 10<sup>-3</sup> |

*Forward, DeepXDE's settings:*

![Wave 1D – space-time multi-scale Fourier features: Forward, DeepXDE's settings](figures/tuning/Wave_propagation_space_time_multi_scale_Fourier_features/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Wave 1D – space-time multi-scale Fourier features: Forward, PINNStudio best](figures/tuning/Wave_propagation_space_time_multi_scale_Fourier_features/tuned_surface_comparison.png)

</details>

<details id="bench-diffusion-1d-unknown-coefficient">
<summary><strong>Diffusion 1D – unknown coefficient</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-diffusion-equation) · [forward runs in the tuning study](tuning_study.md#tune-diffusion-1d-unknown-coefficient)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 6.2 × 10<sup>-3</sup> | 2.0 × 10<sup>-4</sup> |
| Parameters found (true) | C = 0.999553 (1) | C = 1.00017 (1) |

*Inverse, DeepXDE's settings:*

![Diffusion 1D – unknown coefficient: Inverse, DeepXDE's settings](figures/tuning/Inverse_diffusion_equation/own_surface_comparison.png)

![Diffusion 1D – unknown coefficient: Inverse, DeepXDE's settings](figures/tuning/Inverse_diffusion_equation/own_param_plot.png)

*Inverse, PINNStudio best:*

![Diffusion 1D – unknown coefficient: Inverse, PINNStudio best](figures/tuning/Inverse_diffusion_equation/tuned_surface_comparison.png)

![Diffusion 1D – unknown coefficient: Inverse, PINNStudio best](figures/tuning/Inverse_diffusion_equation/tuned_param_plot.png)

</details>

<details id="bench-diffusion-reaction-1d-unknown-k-f-d">
<summary><strong>Diffusion-reaction 1D – unknown k_f, D</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-diffusion-reaction-system)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | – | – (not tuned yet) |
| Parameters found (true) | k_f = 0.0992 (0.1); D = 1.993 (2) | k_f = 0.0992 (0.1); D = 1.993 (2) |

*Inverse, the README's result:*

![Diffusion-reaction 1D – unknown k_f, D: Inverse, the README's result](../assets/Images/DeepXDE/Inverse_diffusion_reaction_system/animation.gif)

</details>

## 2D, steady

<details id="bench-fractional-poisson-2d-disk">
<summary><strong>Fractional Poisson 2D – disk</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-fractional-poisson-equation-2d)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.2 × 10<sup>-3</sup> | 1.2 × 10<sup>-3</sup> (not tuned yet) |

*Forward, the README's result:*

![Fractional Poisson 2D – disk: Forward, the README's result](../assets/Images/DeepXDE/Fractional_Poisson_equation_2D/surface_comparison.png)

</details>

<details id="bench-helmholtz-2d-dirichlet">
<summary><strong>Helmholtz 2D – Dirichlet</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-helmholtz-2d-dirichlet) · [inverse runs in the tuning study](tuning_study.md#tune-helmholtz-2d-dirichlet-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.9 × 10<sup>-3</sup> | 1.9 × 10<sup>-3</sup> (not tuned yet) |
| Inverse $L_2$ | 3.5 × 10<sup>-4</sup> | 8.7 × 10<sup>-5</sup> |
| Parameters found (true) | k₀ = 12.565 (12.5664) | k₀ = 12.5669 (12.5664) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, the README's result:*

![Helmholtz 2D – Dirichlet: Forward, the README's result](../assets/Images/DeepXDE/Helmholtz_2D_Dirichlet/surface_comparison.png)

</details>

<details id="bench-helmholtz-2d-hyperparameter-search">
<summary><strong>Helmholtz 2D – hyperparameter search</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-helmholtz-2d-hyperparameter-optimization)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.3 × 10<sup>-3</sup> | 1.3 × 10<sup>-3</sup> (not tuned yet) |

</details>

<details id="bench-helmholtz-2d-sound-hard-scattering">
<summary><strong>Helmholtz 2D – sound-hard scattering</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-helmholtz-sound-hard-scattering-absorbing-bc) · [forward runs in the tuning study](tuning_study.md#tune-helmholtz-2d-sound-hard-scattering)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 8.0 × 10<sup>-2</sup> | 7.7 × 10<sup>-2</sup> |

*Forward, DeepXDE's settings:*

![Helmholtz 2D – sound-hard scattering: Forward, DeepXDE's settings](figures/tuning/Helmholtz_sound_hard_scattering_absorbing_BC/own_surface_comparison_u_im.png)

![Helmholtz 2D – sound-hard scattering: Forward, DeepXDE's settings](figures/tuning/Helmholtz_sound_hard_scattering_absorbing_BC/own_surface_comparison_u_re.png)

*Forward, PINNStudio best:*

![Helmholtz 2D – sound-hard scattering: Forward, PINNStudio best](figures/tuning/Helmholtz_sound_hard_scattering_absorbing_BC/tuned_surface_comparison_u_im.png)

![Helmholtz 2D – sound-hard scattering: Forward, PINNStudio best](figures/tuning/Helmholtz_sound_hard_scattering_absorbing_BC/tuned_surface_comparison_u_re.png)

</details>

<details id="bench-helmholtz-2d-square-with-hole">
<summary><strong>Helmholtz 2D – square with hole</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-helmholtz-2d-with-a-hole) · [forward runs in the tuning study](tuning_study.md#tune-helmholtz-2d-square-with-hole)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.1 × 10<sup>-2</sup> | 7.0 × 10<sup>-4</sup> |

*Forward, DeepXDE's settings:*

![Helmholtz 2D – square with hole: Forward, DeepXDE's settings](figures/tuning/Helmholtz_2D_with_a_hole/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Helmholtz 2D – square with hole: Forward, PINNStudio best](figures/tuning/Helmholtz_2D_with_a_hole/tuned_surface_comparison.png)

</details>

<details id="bench-laplace-2d-disk">
<summary><strong>Laplace 2D – disk</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-laplace-equation-on-a-disk)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.1 × 10<sup>-3</sup> | 1.1 × 10<sup>-3</sup> (not tuned yet) |

*Forward, the README's result:*

![Laplace 2D – disk: Forward, the README's result](../assets/Images/DeepXDE/Laplace_equation_on_a_disk/surface_comparison.png)

</details>

<details id="bench-linear-elasticity-2d-plate">
<summary><strong>Linear elasticity 2D – plate</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-linear-elastostatic-plate) · [forward runs in the tuning study](tuning_study.md#tune-linear-elasticity-2d-plate) · [inverse runs in the tuning study](tuning_study.md#tune-linear-elasticity-2d-plate-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 6.7 × 10<sup>-2</sup> | 1.3 × 10<sup>-4</sup> |
| Inverse $L_2$ | 1.9 × 10<sup>-3</sup> | 1.9 × 10<sup>-3</sup> |
| Parameters found (true) | λ = 0.999744 (1); μ = 0.500088 (0.5) | λ = 0.999744 (1); μ = 0.500088 (0.5) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, DeepXDE's settings:*

![Linear elasticity 2D – plate: Forward, DeepXDE's settings](figures/tuning/Linear_elastostatic_plate/own_surface_comparison_sxx.png)

![Linear elasticity 2D – plate: Forward, DeepXDE's settings](figures/tuning/Linear_elastostatic_plate/own_surface_comparison_sxy.png)

![Linear elasticity 2D – plate: Forward, DeepXDE's settings](figures/tuning/Linear_elastostatic_plate/own_surface_comparison_syy.png)

![Linear elasticity 2D – plate: Forward, DeepXDE's settings](figures/tuning/Linear_elastostatic_plate/own_surface_comparison_ux.png)

![Linear elasticity 2D – plate: Forward, DeepXDE's settings](figures/tuning/Linear_elastostatic_plate/own_surface_comparison_uy.png)

*Forward, PINNStudio best:*

![Linear elasticity 2D – plate: Forward, PINNStudio best](figures/tuning/Linear_elastostatic_plate/tuned_surface_comparison_sxx.png)

![Linear elasticity 2D – plate: Forward, PINNStudio best](figures/tuning/Linear_elastostatic_plate/tuned_surface_comparison_sxy.png)

![Linear elasticity 2D – plate: Forward, PINNStudio best](figures/tuning/Linear_elastostatic_plate/tuned_surface_comparison_syy.png)

![Linear elasticity 2D – plate: Forward, PINNStudio best](figures/tuning/Linear_elastostatic_plate/tuned_surface_comparison_ux.png)

![Linear elasticity 2D – plate: Forward, PINNStudio best](figures/tuning/Linear_elastostatic_plate/tuned_surface_comparison_uy.png)

</details>

<details id="bench-linear-elasticity-2d-plate-soft-bcs">
<summary><strong>Linear elasticity 2D – plate, soft BCs</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-linear-elastostatic-plate-soft-bcs)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 4.9 × 10<sup>-4</sup> | 4.9 × 10<sup>-4</sup> (not tuned yet) |

*Forward, the README's result:*

![Linear elasticity 2D – plate, soft BCs: Forward, the README's result](../assets/Images/DeepXDE/Linear_elastostatic_plate_soft_BCs/surface_comparison_ux.png)

</details>

<details id="bench-navier-stokes-2d-kovasznay-flow">
<summary><strong>Navier-Stokes 2D – Kovasznay flow</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-kovasznay-flow) · [inverse runs in the tuning study](tuning_study.md#tune-navier-stokes-2d-kovasznay-flow-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.1 × 10<sup>-4</sup> | 2.1 × 10<sup>-4</sup> (not tuned yet) |
| Inverse $L_2$ | 9.1 × 10<sup>-5</sup> | 3.7 × 10<sup>-5</sup> |
| Parameters found (true) | Re = 20.0026 (20) | Re = 19.9982 (20) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, the README's result:*

![Navier-Stokes 2D – Kovasznay flow: Forward, the README's result](../assets/Images/DeepXDE/Kovasznay_flow/surface_comparison_u.png)

*Inverse, DeepXDE's settings:*

![Navier-Stokes 2D – Kovasznay flow: Inverse, DeepXDE's settings](figures/tuning/Kovasznay_flow_inv/own_surface_comparison_u.png)

![Navier-Stokes 2D – Kovasznay flow: Inverse, DeepXDE's settings](figures/tuning/Kovasznay_flow_inv/own_surface_comparison_v.png)

![Navier-Stokes 2D – Kovasznay flow: Inverse, DeepXDE's settings](figures/tuning/Kovasznay_flow_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Navier-Stokes 2D – Kovasznay flow: Inverse, PINNStudio best](figures/tuning/Kovasznay_flow_inv/tuned_surface_comparison_u.png)

![Navier-Stokes 2D – Kovasznay flow: Inverse, PINNStudio best](figures/tuning/Kovasznay_flow_inv/tuned_surface_comparison_v.png)

![Navier-Stokes 2D – Kovasznay flow: Inverse, PINNStudio best](figures/tuning/Kovasznay_flow_inv/tuned_param_plot.png)

</details>

<details id="bench-poisson-2d-disk">
<summary><strong>Poisson 2D – disk</strong> · PINNStudio template</summary>

[problem and settings in the README](../README.md#2d-poisson-disk) · [inverse runs in the tuning study](tuning_study.md#tune-poisson-2d-disk-pinnstudio-template-inverse-mode)

| | The template's own settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.8 × 10<sup>-4</sup> | 2.8 × 10<sup>-4</sup> (not tuned yet) |
| Inverse $L_2$ | 5.4 × 10<sup>-4</sup> | 5.4 × 10<sup>-4</sup> |
| Parameters found (true) | f = 1.00002 (1) | f = 1.00002 (1) |

*Forward, the README's result:*

![Poisson 2D – disk: Forward, the README's result](../assets/Images/2D-Poisson-Disk-Shaped/surface_comparison.png)

*Inverse, the template's own settings:*

![Poisson 2D – disk: Inverse, the template's own settings](figures/tuning/Template_2D_Poisson_Disk_inv/own_surface_comparison.png)

![Poisson 2D – disk: Inverse, the template's own settings](figures/tuning/Template_2D_Poisson_Disk_inv/own_param_plot.png)

</details>

<details id="bench-poisson-2d-l-shape">
<summary><strong>Poisson 2D – L-shape</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-poisson-over-an-l-shaped-domain) · [forward runs in the tuning study](tuning_study.md#tune-poisson-2d-l-shape)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.5 × 10<sup>-2</sup> | 2.3 × 10<sup>-2</sup> |

*Forward, DeepXDE's settings:*

![Poisson 2D – L-shape: Forward, DeepXDE's settings](figures/tuning/Poisson_over_an_L_shaped_domain/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Poisson 2D – L-shape: Forward, PINNStudio best](figures/tuning/Poisson_over_an_L_shaped_domain/tuned_surface_comparison.png)

</details>

<details id="bench-fractional-poisson-2d-unknown-order">
<summary><strong>Fractional Poisson 2D – unknown order α</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-inverse-fractional-poisson-equation-2d)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Inverse $L_2$ | 2.3 × 10<sup>-3</sup> | 2.3 × 10<sup>-3</sup> (not tuned yet) |
| Parameters found (true) | α = 1.80445 (1.8) | α = 1.80445 (1.8) |

*Inverse, the README's result:*

![Fractional Poisson 2D – unknown order α: Inverse, the README's result](../assets/Images/DeepXDE/Inverse_fractional_Poisson_equation_2D/surface_comparison.png)

</details>

## 2D, time-dependent

<details id="bench-allen-cahn-2d-wight-zhao">
<summary><strong>Allen-Cahn 2D – Wight & Zhao</strong> · PINNStudio template, after Wight & Zhao (2021)</summary>

[problem and settings in the README](../README.md#2d-allen-cahn-wight--zhao) · [forward runs in the tuning study](tuning_study.md#tune-allen-cahn-2d-wight-zhao-pinnstudio-template) · [forward runs in the tuning study](tuning_study.md#tune-allen-cahn-2d-wight-zhao-pinnstudio-template-best-setup) · [inverse runs in the tuning study](tuning_study.md#tune-allen-cahn-2d-wight-zhao-pinnstudio-template-inverse-mode)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Wight & Zhao's method: reported in Wight & Zhao (2021), Fig. 3.8 | – | figures only | |
| Wight & Zhao's method, run in PINNStudio | – | – | |
| The template as it loads | 7.9 × 10<sup>-3</sup> | – | |
| **PINNStudio best** (forward) | 1.6 × 10<sup>-3</sup> | – | |
| Inverse, the template's own settings | 3.1 × 10<sup>-4</sup> | – | λ = 9.54428 (10); λε² = 0.00598505 (0.00625) |
| **PINNStudio best** (inverse) | 3.1 × 10<sup>-4</sup> | – | λ = 9.54428 (10); λε² = 0.00598505 (0.00625) |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the template's own settings:*

![Allen-Cahn 2D – Wight & Zhao: Forward, the template's own settings](figures/tuning/Template_2D_Allen_Cahn_Wight_Zhao/own_surface_comparison.png)

*Forward, PINNStudio best:*

![Allen-Cahn 2D – Wight & Zhao: Forward, PINNStudio best](figures/tuning/Template_2D_Allen_Cahn_Wight_Zhao_best_setup/own_surface_comparison.png)

*Inverse, the template's own settings:*

![Allen-Cahn 2D – Wight & Zhao: Inverse, the template's own settings](figures/tuning/Template_2D_Allen_Cahn_Wight_Zhao_inv/own_surface_comparison.png)

![Allen-Cahn 2D – Wight & Zhao: Inverse, the template's own settings](figures/tuning/Template_2D_Allen_Cahn_Wight_Zhao_inv/own_param_plot.png)

</details>

<details id="bench-allen-cahn-2d-mattey-ghosh">
<summary><strong>Allen-Cahn 2D – Mattey & Ghosh</strong> · PINNStudio template, after Mattey & Ghosh (2022)</summary>

[problem and settings in the README](../README.md#2d-allen-cahn-mattey--ghosh) · [forward runs in the tuning study](tuning_study.md#tune-allen-cahn-2d-mattey-ghosh-pinnstudio-template-best-setup) · [inverse runs in the tuning study](tuning_study.md#tune-allen-cahn-2d-mattey-ghosh-pinnstudio-template-inverse-mode)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Mattey & Ghosh's bc-PINN: reported in Mattey & Ghosh (2022), Sec. 6.1.1 | – | 2.5 × 10<sup>-2</sup> | |
| Mattey & Ghosh's bc-PINN, run in PINNStudio | – | – | |
| The template as it loads | 6.6 × 10<sup>-3</sup> | – | |
| **PINNStudio best** (forward) | 3.6 × 10<sup>-4</sup> | – | |
| Inverse, the template's own settings | 8.9 × 10<sup>-4</sup> | – | c₁² = 0.000105791 (0.0001); c₂ = 1.00367 (1) |
| **PINNStudio best** (inverse) | 8.9 × 10<sup>-4</sup> | – | c₁² = 0.000105791 (0.0001); c₂ = 1.00367 (1) |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the README's result:*

![Allen-Cahn 2D – Mattey & Ghosh: Forward, the README's result](../assets/Images/2D_Allen_Cahn_Mattey/surface_comparison.png)

*Forward, PINNStudio best:*

![Allen-Cahn 2D – Mattey & Ghosh: Forward, PINNStudio best](figures/tuning/Template_2D_Allen_Cahn_Mattey_Ghosh_best_setup/own_surface_comparison.png)

*Inverse, the template's own settings:*

![Allen-Cahn 2D – Mattey & Ghosh: Inverse, the template's own settings](figures/tuning/Template_2D_Allen_Cahn_Mattey_Ghosh_inv/own_surface_comparison.png)

![Allen-Cahn 2D – Mattey & Ghosh: Inverse, the template's own settings](figures/tuning/Template_2D_Allen_Cahn_Mattey_Ghosh_inv/own_param_plot.png)

</details>

<details id="bench-cahn-hilliard-2d-wight-zhao">
<summary><strong>Cahn-Hilliard 2D – Wight & Zhao</strong> · PINNStudio template, after Wight & Zhao (2021)</summary>

[problem and settings in the README](../README.md#2d-cahn-hilliard-wight--zhao) · [forward runs in the tuning study](tuning_study.md#tune-cahn-hilliard-2d-wight-zhao-pinnstudio-template) · [inverse runs in the tuning study](tuning_study.md#tune-cahn-hilliard-2d-wight-zhao-pinnstudio-template-inverse-mode)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Wight & Zhao's method: reported in Wight & Zhao (2021), Fig. 3.13 | – | figures only | |
| Wight & Zhao's method, run in PINNStudio | – | – | |
| The template as it loads | 1.1 × 10<sup>-2</sup> | – | |
| **PINNStudio best** (forward) | 8.0 × 10<sup>-3</sup> | – | |
| Inverse, the template's own settings | 6.9 × 10<sup>-5</sup> | – | ε² = 0.00251042 (0.0025) |
| **PINNStudio best** (inverse) | 6.9 × 10<sup>-5</sup> | – | ε² = 0.00251042 (0.0025) |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the template's own settings:*

![Cahn-Hilliard 2D – Wight & Zhao: Forward, the template's own settings](figures/tuning/Template_2D_Cahn_Hilliard_Wight_Zhao/own_surface_comparison_mu.png)

![Cahn-Hilliard 2D – Wight & Zhao: Forward, the template's own settings](figures/tuning/Template_2D_Cahn_Hilliard_Wight_Zhao/own_surface_comparison_u.png)

*Forward, PINNStudio best:*

![Cahn-Hilliard 2D – Wight & Zhao: Forward, PINNStudio best](figures/tuning/Template_2D_Cahn_Hilliard_Wight_Zhao/tuned_surface_comparison_mu.png)

![Cahn-Hilliard 2D – Wight & Zhao: Forward, PINNStudio best](figures/tuning/Template_2D_Cahn_Hilliard_Wight_Zhao/tuned_surface_comparison_u.png)

*Inverse, the template's own settings:*

![Cahn-Hilliard 2D – Wight & Zhao: Inverse, the template's own settings](figures/tuning/Template_2D_Cahn_Hilliard_Wight_Zhao_inv/own_surface_comparison_mu.png)

![Cahn-Hilliard 2D – Wight & Zhao: Inverse, the template's own settings](figures/tuning/Template_2D_Cahn_Hilliard_Wight_Zhao_inv/own_surface_comparison_u.png)

![Cahn-Hilliard 2D – Wight & Zhao: Inverse, the template's own settings](figures/tuning/Template_2D_Cahn_Hilliard_Wight_Zhao_inv/own_param_plot.png)

</details>

<details id="bench-cahn-hilliard-2d-mattey-ghosh">
<summary><strong>Cahn-Hilliard 2D – Mattey & Ghosh</strong> · PINNStudio template, after Mattey & Ghosh (2022)</summary>

[problem and settings in the README](../README.md#2d-cahn-hilliard-mattey--ghosh)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Mattey & Ghosh's bc-PINN: reported in Mattey & Ghosh (2022), Sec. 6.2.1 | – | figures only | |
| Mattey & Ghosh's bc-PINN, run in PINNStudio | – | – | |
| The template as it loads | 2.0 × 10<sup>-3</sup> | – | |
| **PINNStudio best** (forward) | 2.0 × 10<sup>-3</sup> | – | |
| Inverse, the template's own settings | – | – | α = 0.000188 (0.0002); κ = 1.012 (1) |
| **PINNStudio best** (inverse) | – | – | α = 0.000188 (0.0002); κ = 1.012 (1) |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the README's result:*

![Cahn-Hilliard 2D – Mattey & Ghosh: Forward, the README's result](../assets/Images/2D_CahnHilliard-Mattey/surface_comparison_u.png)

*Inverse, the README's result:*

![Cahn-Hilliard 2D – Mattey & Ghosh: Inverse, the README's result](../assets/Images/2D_CahnHilliard-Mattey/param_convergence_animation.gif)

</details>

<details id="bench-burgers-2d-mathias-et-al">
<summary><strong>Burgers 2D – Mathias et al.</strong> · PINNStudio template, after Mathias et al. (2022)</summary>

[problem and settings in the README](../README.md#2d-burgers-mathias) · [forward runs in the tuning study](tuning_study.md#tune-burgers-2d-mathias-et-al-pinnstudio-template) · [inverse runs in the tuning study](tuning_study.md#tune-burgers-2d-mathias-et-al-pinnstudio-template-inverse-mode)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Mathias et al.'s settings: reported in Mathias et al. (2022) | – | figures only | |
| Mathias et al.'s settings, run in PINNStudio | – | – | |
| The template as it loads | 1.1 × 10<sup>-2</sup> | – | |
| **PINNStudio best** (forward) | 8.4 × 10<sup>-3</sup> | – | |
| Inverse, the template's own settings | 3.5 × 10<sup>-3</sup> | – | ν = 0.00341319 (0.0031831) |
| **PINNStudio best** (inverse) | 3.5 × 10<sup>-3</sup> | – | ν = 0.00341319 (0.0031831) |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the template's own settings:*

![Burgers 2D – Mathias et al.: Forward, the template's own settings](figures/tuning/Template_2D_Burgers_Mathias/own_surface_comparison_u.png)

![Burgers 2D – Mathias et al.: Forward, the template's own settings](figures/tuning/Template_2D_Burgers_Mathias/own_surface_comparison_v.png)

*Forward, PINNStudio best:*

![Burgers 2D – Mathias et al.: Forward, PINNStudio best](figures/tuning/Template_2D_Burgers_Mathias/tuned_surface_comparison_u.png)

![Burgers 2D – Mathias et al.: Forward, PINNStudio best](figures/tuning/Template_2D_Burgers_Mathias/tuned_surface_comparison_v.png)

*Inverse, the template's own settings:*

![Burgers 2D – Mathias et al.: Inverse, the template's own settings](figures/tuning/Template_2D_Burgers_Mathias_inv/own_surface_comparison_u.png)

![Burgers 2D – Mathias et al.: Inverse, the template's own settings](figures/tuning/Template_2D_Burgers_Mathias_inv/own_surface_comparison_v.png)

![Burgers 2D – Mathias et al.: Inverse, the template's own settings](figures/tuning/Template_2D_Burgers_Mathias_inv/own_param_plot.png)

</details>

<details id="bench-heat-2d">
<summary><strong>Heat 2D</strong> · PINNStudio template</summary>

[problem and settings in the README](../README.md#2d-heat) · [inverse runs in the tuning study](tuning_study.md#tune-heat-2d-pinnstudio-template-inverse-mode)

| | The template's own settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 3.9 × 10<sup>-3</sup> | 3.9 × 10<sup>-3</sup> (not tuned yet) |
| Inverse $L_2$ | 5.8 × 10<sup>-4</sup> | 5.8 × 10<sup>-4</sup> |
| Parameters found (true) | α = 0.399893 (0.4) | α = 0.399893 (0.4) |

*Forward, the README's result:*

![Heat 2D: Forward, the README's result](../assets/Images/2D-Heat/surface_comparison.png)

*Inverse, the template's own settings:*

![Heat 2D: Inverse, the template's own settings](figures/tuning/Template_2D_Heat_inv/own_surface_comparison.png)

![Heat 2D: Inverse, the template's own settings](figures/tuning/Template_2D_Heat_inv/own_param_plot.png)

</details>

## 3D

<details id="bench-fractional-poisson-3d-sphere">
<summary><strong>Fractional Poisson 3D – sphere</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-fractional-poisson-equation-3d)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 6.7 × 10<sup>-3</sup> | 6.7 × 10<sup>-3</sup> (not tuned yet) |

*Forward, the README's result:*

![Fractional Poisson 3D – sphere: Forward, the README's result](../assets/Images/DeepXDE/Fractional_Poisson_equation_3D/surface_comparison.png)

</details>

<details id="bench-poisson-3d-sphere">
<summary><strong>Poisson 3D – sphere</strong> · PINNStudio template</summary>

[problem and settings in the README](../README.md#3d-poisson-sphere)

| | The template's own settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.5 × 10<sup>-3</sup> | 1.5 × 10<sup>-3</sup> (not tuned yet) |
| Inverse $L_2$ | 1.4 × 10<sup>-3</sup> | 1.4 × 10<sup>-3</sup> (not tuned yet) |
| Parameters found (true) | f = 1.00005 (1) | f = 1.00005 (1) |

*Forward, the README's result:*

![Poisson 3D – sphere: Forward, the README's result](../assets/Images/3D-Poisson-Sphere/surface_comparison.png)

*Inverse, the README's result:*

![Poisson 3D – sphere: Inverse, the README's result](../assets/Images/3D-Poisson-Sphere/param_convergence_animation.gif)

</details>

<details id="bench-allen-cahn-3d-wight-zhao">
<summary><strong>Allen-Cahn 3D – Wight & Zhao</strong> · PINNStudio template, after Wight & Zhao (2021)</summary>

[problem and settings in the README](../README.md#3d-allen-cahn-wight--zhao) · [inverse runs in the tuning study](tuning_study.md#tune-allen-cahn-3d-wight-zhao-pinnstudio-template-inverse-mode)

| | $L_2$, last time | $L_2$, all times | Parameters found (true) |
|---|---|---|---|
| Plain PINN (in PINNStudio) | – | – | |
| Wight & Zhao's method: reported in Wight & Zhao (2021), Fig. 3.9 | – | figures only | |
| Wight & Zhao's method, run in PINNStudio | – | – | |
| The template as it loads | 2.7 × 10<sup>-4</sup> | – | |
| **PINNStudio best** (forward) | 2.7 × 10<sup>-4</sup> | – | |
| Inverse, the template's own settings | 9.7 × 10<sup>-5</sup> | – | λ = 9.96899 (10); λε² = 0.0249817 (0.025) |
| **PINNStudio best** (inverse) | 9.7 × 10<sup>-5</sup> | – | λ = 9.96899 (10); λε² = 0.0249817 (0.025) |

*The plain PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*

*Forward, the README's result:*

![Allen-Cahn 3D – Wight & Zhao: Forward, the README's result](../assets/Images/3D_AllenCahn-Wight/slice_comparison.png)

*Inverse, the template's own settings:*

![Allen-Cahn 3D – Wight & Zhao: Inverse, the template's own settings](figures/tuning/Template_3D_Allen_Cahn_Wight_Zhao_inv/own_surface_comparison.png)

![Allen-Cahn 3D – Wight & Zhao: Inverse, the template's own settings](figures/tuning/Template_3D_Allen_Cahn_Wight_Zhao_inv/own_param_plot.png)

</details>

<details id="bench-heat-3d">
<summary><strong>Heat 3D</strong> · PINNStudio template</summary>

[problem and settings in the README](../README.md#3d-heat) · [inverse runs in the tuning study](tuning_study.md#tune-heat-3d-pinnstudio-template-inverse-mode)

| | The template's own settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 2.6 × 10<sup>-3</sup> | 2.6 × 10<sup>-3</sup> (not tuned yet) |
| Inverse $L_2$ | 1.3 × 10<sup>-3</sup> | 1.3 × 10<sup>-3</sup> |
| Parameters found (true) | α = 0.400303 (0.4) | α = 0.400303 (0.4) |

*Forward, the README's result:*

![Heat 3D: Forward, the README's result](../assets/Images/3D_Heat/3D_Heat_Solution_Surface.png)

*Inverse, the template's own settings:*

![Heat 3D: Inverse, the template's own settings](figures/tuning/Template_3D_Heat_inv/own_surface_comparison.png)

![Heat 3D: Inverse, the template's own settings](figures/tuning/Template_3D_Heat_inv/own_param_plot.png)

</details>

<details id="bench-navier-stokes-3d-beltrami-flow">
<summary><strong>Navier-Stokes 3D – Beltrami flow</strong> · DeepXDE example</summary>

[problem and settings in the README](../README.md#dx-beltrami-flow) · [inverse runs in the tuning study](tuning_study.md#tune-navier-stokes-3d-beltrami-flow-inverse-mode)

| | DeepXDE's settings | PINNStudio best |
|---|---|---|
| Forward $L_2$ | 1.4 × 10<sup>-3</sup> | 1.4 × 10<sup>-3</sup> (not tuned yet) |
| Inverse $L_2$ | 3.5 × 10<sup>-4</sup> | 2.2 × 10<sup>-4</sup> |
| Parameters found (true) | ν = 0.997797 (1) | ν = 0.998003 (1) |

DeepXDE's example is forward only; the inverse problem is PINNStudio's inverse mode of it, run with the example's settings and tuned.

*Forward, the README's result:*

![Navier-Stokes 3D – Beltrami flow: Forward, the README's result](../assets/Images/DeepXDE/Beltrami_flow/animation_u.gif)

*Inverse, DeepXDE's settings:*

![Navier-Stokes 3D – Beltrami flow: Inverse, DeepXDE's settings](figures/tuning/Beltrami_flow_inv/own_surface_comparison_u.png)

![Navier-Stokes 3D – Beltrami flow: Inverse, DeepXDE's settings](figures/tuning/Beltrami_flow_inv/own_surface_comparison_v.png)

![Navier-Stokes 3D – Beltrami flow: Inverse, DeepXDE's settings](figures/tuning/Beltrami_flow_inv/own_surface_comparison_w.png)

![Navier-Stokes 3D – Beltrami flow: Inverse, DeepXDE's settings](figures/tuning/Beltrami_flow_inv/own_param_plot.png)

*Inverse, PINNStudio best:*

![Navier-Stokes 3D – Beltrami flow: Inverse, PINNStudio best](figures/tuning/Beltrami_flow_inv/tuned_surface_comparison_u.png)

![Navier-Stokes 3D – Beltrami flow: Inverse, PINNStudio best](figures/tuning/Beltrami_flow_inv/tuned_surface_comparison_v.png)

![Navier-Stokes 3D – Beltrami flow: Inverse, PINNStudio best](figures/tuning/Beltrami_flow_inv/tuned_surface_comparison_w.png)

![Navier-Stokes 3D – Beltrami flow: Inverse, PINNStudio best](figures/tuning/Beltrami_flow_inv/tuned_param_plot.png)

</details>

---

Written by [`tools/make_benchmarks_readme.py`](../tools/make_benchmarks_readme.py) from the runs of [`tools/tuning_study.py`](../tools/tuning_study.py).

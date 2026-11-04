**How this result was obtained.** Unlike the other examples, this result is not one configuration selected against
the reference solution. The example runs PINNStudio's Bayesian Parameter Sweep: 50 training configurations (the
baseline, learning rate 10⁻³ with 4 × 50 sin, then 9 random and 40 Gaussian-process choices), and the configuration
with the **lowest final training loss** is selected (run 49: learning rate 1.5 × 10⁻³, 3 × 87, sin). Only then is
its relative L₂ error measured against the exact solution: 1.28 × 10⁻³. The baseline run alone reaches 3.3 × 10⁻¹.
Training seed 2026, search seed 0. The 50 runs, their losses and errors, and how to repeat the search are in
[the search record](benchmarks/records/helmholtz_hpo_2026-10-25/README.md).

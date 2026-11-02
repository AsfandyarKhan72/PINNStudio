# Reference data

The reference solutions in [`reference_data/`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data) were computed for PINNStudio by its author. None
were copied from DeepXDE or from published papers. Each file holds the coordinates (and time) and the reference value
at one time (`t_<time>.txt`, or `<output>_t_<time>.txt` for problems with several outputs), the layout
[Error Analysis](../guide/error-analysis.md) reads.

| Source | How it was produced | Generator |
|---|---|---|
| 1D templates (Burgers, heat, Schrödinger, Allen–Cahn) | Exact or high-accuracy solutions (for Burgers, the Cole–Hopf solution evaluated by quadrature) on the template's grid and times | `tools/make_exact_reference_data.py` |
| DeepXDE examples | DeepXDE's reference formula or data where the example has one; otherwise solved numerically (for example with SciPy's `solve_ivp` for ODE systems) | `tools/make_deepxde_reference_data.py` |
| Phase-field templates (3D Allen–Cahn, Cahn–Hilliard) | Fourier spectral discretization in space on periodic domains; in time, exponential time differencing for 3D Allen–Cahn (ETDRK4; Cox & Matthews 2002, Kassam & Trefethen 2005) and a stabilized semi-implicit step for Cahn–Hilliard | `tools/make_phase_field_reference_data.py` |
| Poisson templates | Disk and ball from their exact solutions; the L-shaped domain (no closed form) by the five-point finite-difference scheme on refined grids | `tools/make_poisson_reference_data.py` |
| 2D Burgers (Mathias et al.) | Chebyshev collocation in space | `tools/make_burgers2d_reference_data.py` |
| 2D and 3D heat, 2D Allen–Cahn (Mattey & Ghosh, Wight & Zhao) | No generator script in `tools/`; checked against independent solutions (a closed-form series for heat, Fourier spectral with ETDRK4 for Allen–Cahn) | `benchmarks/reference_regeneration/check_other_references.py` (checks only, writes nothing) |

## Validation

Numerical references are checked for convergence by refining the grid and the time step, and selected references are
checked against independent calculations. The regeneration records, with these checks, are kept in
[`benchmarks/reference_regeneration/`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/reference_regeneration).

!!! info "v1.6.0 (in development)"
    In the development version the reference data of 2D Burgers (Mathias et al.) and of the three Poisson templates
    (disk, L-shape, ball) are produced by the scripts above; earlier releases used solutions computed with COMSOL
    Multiphysics for these four templates. The comparison of the two is in
    [`benchmarks/reference_regeneration/`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/reference_regeneration).
    The 2D and 3D heat and 2D Allen–Cahn files have been checked against independent solutions. The final files and
    their provenance notes will be part of v1.6.0; results computed against earlier files are superseded.

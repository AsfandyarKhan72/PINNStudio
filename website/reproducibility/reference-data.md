# Reference data

The reference solutions in [`reference_data/`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data) were computed for PINNStudio by its author, with the
methods and scripts listed below. Each file holds the coordinates (and time) and the reference value
at one time (`t_<time>.txt`, or `<output>_t_<time>.txt` for problems with several outputs), the layout
[Error Analysis](../guide/error-analysis.md) reads.

| Source | How it was produced | Generator |
|---|---|---|
| 1D templates (Burgers, heat, Schrödinger, Allen–Cahn) | Exact or high-accuracy solutions (for Burgers, the Cole–Hopf solution evaluated by quadrature) on the template's grid and times | `tools/make_exact_reference_data.py` |
| DeepXDE examples | DeepXDE's reference formula or data where the example has one; otherwise solved numerically (for example with SciPy's `solve_ivp` for ODE systems) | `tools/make_deepxde_reference_data.py` |
| Phase-field templates (3D Allen–Cahn, Cahn–Hilliard) | Fourier spectral discretization in space on periodic domains; in time, exponential time differencing for 3D Allen–Cahn (ETDRK4; Cox & Matthews 2002, Kassam & Trefethen 2005) and a stabilized semi-implicit step for Cahn–Hilliard | `tools/make_phase_field_reference_data.py` |
| Poisson templates | Disk and ball from their exact solutions; the L-shaped domain (no closed form) by the five-point finite-difference scheme on refined grids | `tools/make_poisson_reference_data.py` |
| 2D Burgers (Mathias et al.) | Chebyshev collocation in space | `tools/make_burgers2d_reference_data.py` |
| 2D and 3D heat | The exact series solution (summed to 2000 terms; the initial condition itself at \(t = 0\)), at the points of the earlier files | `tools/make_heat_reference_data.py` |
| 2D Allen–Cahn (Mattey & Ghosh, Wight & Zhao) | Fourier pseudo-spectral discretization in space on the periodic square and ETDRK4 in time (Kassam & Trefethen 2005), in float64, evaluated as a Fourier series at the points of the earlier files; checked by refining the grid and the time step | `tools/make_allen_cahn_mattey_reference_data.py`, `tools/make_allen_cahn_wight_reference_data.py` |

The 2D and 3D heat and 2D Allen–Cahn files were also checked against independent calculations by
`benchmarks/reference_regeneration/check_other_references.py`, which writes nothing.

## Validation

Numerical references are checked for convergence by refining the grid and the time step, and selected references are
checked against independent calculations. The regeneration records, with these checks, are kept in
[`benchmarks/reference_regeneration/`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/reference_regeneration).

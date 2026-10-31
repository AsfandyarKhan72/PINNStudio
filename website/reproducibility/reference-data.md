# Reference data

The reference solutions in [`reference_data/`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data) were computed for PINNStudio by its author. None
were copied from DeepXDE or from published papers. Each file holds the coordinates (and time) and the reference value
at one time (`t_<time>.txt`, or `<output>_t_<time>.txt` for problems with several outputs), the layout
[Error Analysis](../guide/error-analysis.md) reads.

| Source | How it was produced | Generator |
|---|---|---|
| 1D templates (Burgers, heat, Schrödinger, Allen–Cahn) | Exact or high-accuracy solutions (for Burgers, the Cole–Hopf solution evaluated by quadrature) on the template's grid and times | `tools/make_exact_reference_data.py` |
| DeepXDE examples | DeepXDE's reference formula or data where the example has one; otherwise solved numerically (for example with SciPy's `solve_ivp` for ODE systems) | `tools/make_deepxde_reference_data.py` |
| Phase-field templates (Allen–Cahn, Cahn–Hilliard) | Fourier spectral discretization in space with exponential time differencing (ETDRK4; Cox & Matthews 2002, Kassam & Trefethen 2005) on periodic domains | `tools/make_phase_field_reference_data.py` |
| Poisson templates | Disk and ball from their exact solutions; the L-shaped domain (no closed form) by the five-point finite-difference scheme on refined grids | `tools/make_poisson_reference_data.py` |
| 2D Burgers (Mathias et al.) | Chebyshev collocation in space | `tools/make_burgers2d_reference_data.py` |
| Four templates | COMSOL Multiphysics | listed in [`THIRD_PARTY_LICENSES.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/THIRD_PARTY_LICENSES.md) |

## Validation

Numerical references are checked for convergence by refining the grid and the time step, and selected references are
checked against independent calculations. The regeneration records, with these checks, are kept in
[`benchmarks/reference_regeneration/`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/reference_regeneration).

!!! info "v1.6.0 (in development)"
    During the v1.6.0 audit the reference data of some templates (2D Allen–Cahn of Mattey & Ghosh and of Wight & Zhao,
    and the 2D and 3D heat equations) were regenerated and validated independently. The corrected files and their
    provenance notes will be part of v1.6.0; results computed against the earlier files are superseded.

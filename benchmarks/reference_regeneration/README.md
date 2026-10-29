# Reference data without COMSOL: validation report

Four reference datasets in `reference_data/` were computed with COMSOL Multiphysics: 2D Burgers (Mathias et al.),
Poisson on the unit disk, Poisson on the L-shaped domain and Poisson on the unit ball. This folder holds
regenerated versions made with Python scripts in `tools/` (no COMSOL), the evidence that they are correct, and
what replacing the COMSOL files would change. **Nothing in `reference_data/`, in the benchmark results or in the
figures has been changed.** The regenerated files are in `candidates/`, in the same layout and on the same points
as the files they would replace.

| Dataset | Script | Method | Regenerated vs COMSOL (relative L2) | Recommendation |
|---|---|---|---|---|
| Poisson, disk | `tools/make_poisson_reference_data.py` | exact solution | 2.6 × 10⁻⁸ | replace |
| Poisson, ball | `tools/make_poisson_reference_data.py` | exact solution | 1.9 × 10⁻⁶ | replace |
| Poisson, L-shape | `tools/make_poisson_reference_data.py` | finite differences, h = 1/1600, multigrid | 2.2 × 10⁻⁴ | replace |
| Burgers 2D (Mathias) | `tools/make_burgers2d_reference_data.py` | Chebyshev collocation, N = 512 | 1.0 × 10⁻² (1.8 × 10⁻² for u at t = 0.25) | replace, then rerun the Burgers 2D inverse runs |

The tables below come from `python benchmarks/reference_regeneration/validate.py`. The solutions on every grid
are in `convergence/`. `burgers_comparison.png` shows the Burgers fields side by side.

## Poisson on the disk and the ball

**Problem.** −Δu = 1 inside the domain and u = 0 on its boundary (the templates *Poisson 2D – disk* and
*Poisson 3D – sphere*).

**Method.** Both have exact solutions: u = (1 − r²)/4 on the unit disk and u = (1 − r²)/6 on the unit ball. The
script writes these on the points of the existing files: a 201 × 201 grid on [−1, 1]² for the disk and a 41³
grid for the ball, with NaN outside the domain, plus the 300 measurement points of `u_obs.txt`. The script
checks that the existing NaN points are exactly the points with r > 1.

**Comparison with COMSOL.** The disk agrees with COMSOL to 4.7 × 10⁻⁸ at worst, and the ball to 3.3 × 10⁻⁷. The
measurement files agree to 4.5 × 10⁻⁸ (disk), and exactly for the ball.

**Effect on reported values.** The errors reported for these templates are above 10⁻⁴, so a reference change
below 10⁻⁶ can't change any reported digit. That includes the inverse runs that learn from `u_obs.txt`.

## Poisson on the L-shaped domain

**Problem.** −Δu = 1 on [−1, 1]² without the quadrant (0, 1] × (0, 1], with u = 0 on the boundary. There is no
closed form: near the re-entrant corner the solution behaves like r^(2/3) sin(2θ/3), so its gradient is singular
there.

**Method.** The five-point finite-difference Laplacian is used on uniform grids of spacing h = 1/N, with N a
multiple of 100. These grids align with the corner and with every point of the 0.01 output grid, so no
interpolation is needed. The linear systems are solved with algebraic multigrid (pyamg) to a relative residual
below 10⁻¹². The script solves N = 100, 200, 400, 800 and 1600 (7.7 million unknowns) and writes the finest.

**Convergence.**

| N | max change from N/2 | max diff to N = 1600 | relative L2 vs COMSOL | max diff to COMSOL, r < 0.05 | max diff to COMSOL, r ≥ 0.2 |
|---|---|---|---|---|---|
| 100 | – | 1.5 × 10⁻³ | 4.8 × 10⁻⁴ | 1.2 × 10⁻³ | 1.2 × 10⁻⁴ |
| 200 | 9.5 × 10⁻⁴ | 5.9 × 10⁻⁴ | 5.9 × 10⁻⁵ | 4.1 × 10⁻⁴ | 1.1 × 10⁻⁵ |
| 400 | 3.9 × 10⁻⁴ | 2.0 × 10⁻⁴ | 1.2 × 10⁻⁴ | 2.4 × 10⁻⁴ | 3.5 × 10⁻⁵ |
| 800 | 1.4 × 10⁻⁴ | 5.4 × 10⁻⁵ | 1.9 × 10⁻⁴ | 3.3 × 10⁻⁴ | 5.3 × 10⁻⁵ |
| 1600 | 5.4 × 10⁻⁵ | – | 2.2 × 10⁻⁴ | 3.8 × 10⁻⁴ | 6.0 × 10⁻⁵ |

The change between successive grids falls by a factor of 2.4–2.7 per halving of h. That is the order 4/3 or a
little better that the corner singularity allows. Extrapolating that rate, the N = 1600 solution is within about
3 × 10⁻⁵ of the exact one everywhere, and much closer away from the corner. The largest value converges as
0.149329, 0.149379, 0.149399, 0.149407, 0.149410 (limit about 0.149412). COMSOL's largest value is 0.149384.

**Comparison with COMSOL.** COMSOL is closest to the finite-difference solution with N = 200, and moves away from
it as the grid is refined. Its error is therefore larger than the finite-difference error at N ≥ 400. The
largest difference (3.8 × 10⁻⁴, about 0.25% of max u) is at the points next to the corner, and away from the
corner the difference is below 6 × 10⁻⁵. This is consistent with a COMSOL mesh that wasn't refined at the corner.

**Effect on reported values.** The L-shape errors reported for PINNs are about 3 × 10⁻² (relative L2). A reference
change of 2.2 × 10⁻⁴ moves them by at most that amount, which is below the rounding of the reported values. The
DeepXDE L-shape inverse example reads `u_obs.txt`; its values change by 1 × 10⁻⁴ at most, far below the effect
of the training itself.

## Burgers 2D (Mathias et al.)

**Problem** (as the template defines it):

u_t + u u_x + v u_y = ν (u_xx + u_yy), v_t + u v_x + v v_y = ν (v_xx + v_yy), ν = 0.01/π, on [0, 1]² for
t ∈ [0, 1], with u = v = 0 on the four sides, u(x, y, 0) = sin(2πx) sin(2πy) and v(x, y, 0) = sin(πx) sin(πy).

The viscosity is small, so steep fronts form: a near-shock in u along x = 0.5 and a thin boundary layer in v
below y = 1.

**Method.** The solver uses Chebyshev collocation in x and y with N + 1 Gauss–Lobatto points per axis. The
unknowns are the interior values; the boundary values are zero. Time stepping is the integrating-factor
fourth-order Runge–Kutta method. The diffusion term is integrated exactly, with the matrix exponential of the
one-dimensional second-derivative matrix applied along each axis, and convection is treated with RK4. The output
is evaluated on the 101 × 101 grid of the existing files by barycentric interpolation of the Chebyshev
polynomial, which keeps spectral accuracy. NumPy is used on the CPU; with `--device cuda` the same steps run in
float64 with PyTorch on a GPU.

A first version used a sine series (odd extension across the walls). That is wrong for this system: the cross
terms v u_y and u v_x are even, not odd, across the walls, so the extended problem differs from the real one.
It gave errors of order 1 and has been replaced.

**Checks of the solver.**

1. The derivative and interpolation matrices reproduce sin(3x) and its derivative to 10⁻¹⁴.
2. An independent second-order finite-difference solver (h = 1/400, RK4; `convergence/burgers_fd400_t0.25.npy`)
   agrees with the N = 512 solution at t = 0.25 to 2.1 × 10⁻³ (u) and 6.0 × 10⁻⁴ (v). That is the size expected
   from its second-order error: with h = 1/200 the differences are 8.9 × 10⁻³ and 2.5 × 10⁻³, four times larger.
3. Halving the time step changes nothing above 10⁻⁸ (N = 128 and N = 384).
4. The spatial resolution converges spectrally:

Relative L2 difference to the N = 512 solution, u / v:

| N, dt | t = 0.25 | t = 0.5 | t = 0.75 | t = 1 |
|---|---|---|---|---|
| 128, 5 × 10⁻⁵ | 5.1 × 10⁻³ / 7.0 × 10⁻⁴ | 5.2 × 10⁻³ / 1.8 × 10⁻⁴ | 2.3 × 10⁻³ / 5.8 × 10⁻⁵ | 8.5 × 10⁻⁴ / 2.0 × 10⁻⁵ |
| 128, 2.5 × 10⁻⁵ | same to 3 digits | | | |
| 192, 2.5 × 10⁻⁵ | 2.2 × 10⁻³ / 1.2 × 10⁻⁴ | 1.2 × 10⁻³ / 2.1 × 10⁻⁵ | 2.4 × 10⁻⁴ / 3.9 × 10⁻⁶ | 4.2 × 10⁻⁵ / 7.2 × 10⁻⁷ |
| 256, 1.25 × 10⁻⁵ | 5.3 × 10⁻⁴ / 2.0 × 10⁻⁵ | 2.0 × 10⁻⁴ / 2.6 × 10⁻⁶ | 2.2 × 10⁻⁵ / 2.8 × 10⁻⁷ | 2.0 × 10⁻⁶ / 2.9 × 10⁻⁸ |
| 384, 5 × 10⁻⁶ | 3.5 × 10⁻⁵ / 9.7 × 10⁻⁷ | 5.8 × 10⁻⁶ / 5.8 × 10⁻⁸ | 1.8 × 10⁻⁷ / 2.5 × 10⁻⁹ | 1.3 × 10⁻⁸ / 2.0 × 10⁻⁹ |
| 384, 2.5 × 10⁻⁶ | same to 3 digits | | | |
| **COMSOL** | **1.8 × 10⁻² / 9.0 × 10⁻³** | **7.7 × 10⁻³ / 5.5 × 10⁻³** | **6.5 × 10⁻³ / 3.4 × 10⁻³** | **5.4 × 10⁻³ / 2.8 × 10⁻³** |

Each step in N reduces the difference by a larger factor than the one before (at t = 0.25 for u: 2.3, 4.2, then
15), as spectral convergence does. The step from 384 to 512 is smaller than 3.5 × 10⁻⁵ at every time. The hardest time is t = 0.25, when the
front is steepest. The N = 512 solution is therefore accurate to well below 10⁻⁵ at all output times. The N = 128,
192 and 256 runs were made on the CPU with NumPy, and N = 384 and 512 on an RTX 4090 with PyTorch; the two sets
continue the same convergence sequence.

**Comparison with COMSOL.** COMSOL differs from the converged solution by 1.0 × 10⁻² over all times and both
outputs. The difference is largest at t = 0.25: 1.8 × 10⁻² relative for u, and 0.075 at worst, where |u| ≤ 0.94.
COMSOL's max |u| there is 0.971 against 0.943, an overshoot at the front. The finite-difference check sides with
the Chebyshev solution (2 × 10⁻³ from it, 1.8 × 10⁻² from COMSOL). In `burgers_comparison.png` the COMSOL error
sits on the front at x = 0.5 and in the boundary layer under y = 1, which looks like a mesh too coarse to resolve
the fronts. At t = 1 the difference is smooth and small (max 3.9 × 10⁻³ in u).

**Effect on reported values.** `reeval_burgers2d.py` rebuilds every saved Burgers 2D PINN from its checkpoints
and computes its relative L2 error against both references. Against COMSOL it reproduces the error PINNStudio
logged for each run to all printed digits. One exception: the run of Mathias et al.'s own method, whose network
it doesn't rebuild correctly, so that row is left out. The full results are in
`convergence/burgers_pinn_reeval.json`.

| Run | u at t = 0.25 | u at t = 0.5 | u at t = 1 (the reported value) |
|---|---|---|---|
| forward, *Tuned* (more iterations), seed 2026 | 1.8 × 10⁻² → 3.2 × 10⁻³ | 8.5 × 10⁻³ → 4.4 × 10⁻³ | 8.4 × 10⁻³ → 6.2 × 10⁻³ |
| forward, *Tuned*, seed 1 | 1.8 × 10⁻² → 2.9 × 10⁻³ | 8.4 × 10⁻³ → 4.0 × 10⁻³ | 1.25 × 10⁻² → 1.03 × 10⁻² |
| forward, *Tuned*, seed 2 | 1.8 × 10⁻² → 3.3 × 10⁻³ | 9.1 × 10⁻³ → 5.0 × 10⁻³ | 9.7 × 10⁻³ → 8.3 × 10⁻³ |
| forward, base PINN of the template | 1.85 × 10⁻² → 5.8 × 10⁻³ | 1.06 × 10⁻² → 7.2 × 10⁻³ | 1.14 × 10⁻² → 9.9 × 10⁻³ |
| forward, plain PINN (no time windows) | 1.66 × 10⁻¹ → 1.63 × 10⁻¹ | 1.35 × 10⁻¹ → 1.34 × 10⁻¹ | 1.64 × 10⁻¹ → 1.64 × 10⁻¹ |
| inverse, *Tuned* (all changes), seed 2026 | 1.8 × 10⁻² → 6.3 × 10⁻³ | 1.0 × 10⁻² → 7.4 × 10⁻³ | 3.7 × 10⁻⁴ → 5.4 × 10⁻³ |
| inverse, measurements at all times | 4.1 × 10⁻³ → 1.8 × 10⁻² | 4.2 × 10⁻³ → 6.9 × 10⁻³ | 3.2 × 10⁻³ → 6.2 × 10⁻³ |

What this means for the paper and the benchmarks pages:

- **Burgers 2D forward, *Tuned*.** The value in Figure 4 and the tables (u at t = 1, median of three seeds) would
  go from 9.7 × 10⁻³ to 8.3 × 10⁻³. At earlier times the drop is larger, from 1.8 × 10⁻² to about 3 × 10⁻³ at
  t = 0.25. Part of the error reported until now was the reference's error, not the network's.
- **Burgers 2D, the paper's method and the plain PINN.** Their errors are about 0.17, so the change is in the
  third digit at most. For the paper's method the reported 1.7 × 10⁻¹ (against COMSOL) can shift by at most the
  reference difference at t = 1 (5 × 10⁻³ relative), so it stays 1.7 × 10⁻¹ or 1.6 × 10⁻¹.
- **Burgers 2D inverse.** These runs learn ν from measurements taken from the COMSOL files (`t_1.0_u.txt` and
  `t_1.0_v.txt`; the "times" variant uses all reference times). The networks fit COMSOL's data closely: 3.7 × 10⁻⁴
  at t = 1 against COMSOL, 5.4 × 10⁻³ against the true solution. So they have learned COMSOL's error along with
  the data, and the learned ν (0.003207 against the true 0.0031831, 0.77% off) may be partly COMSOL's error.
  These runs can only be judged again by training them on the regenerated data. That needs new GPU runs (the
  *Tuned* inverse with seeds 2026, 1 and 2, and the variants it is compared with, roughly an hour each).

## How to reproduce

    python tools/make_poisson_reference_data.py --out benchmarks/reference_regeneration/candidates
    python tools/make_burgers2d_reference_data.py --n 512 --dt 2.5e-6 --device cuda --out DIR   # 1.4 h on an RTX 4090
    python tools/make_burgers2d_reference_data.py --n 256 --dt 1.25e-5 --out DIR                 # 1.6 h on a CPU core
    python benchmarks/reference_regeneration/validate.py
    python benchmarks/reference_regeneration/make_figure.py

The Burgers convergence data in `convergence/burgers_N*.npz` are the `grid.npz` files these runs write. The
candidate Burgers files were written from `burgers_N512_dt2.5e-06.npz`. The finite-difference check is
`convergence/burgers_fd400_t0.25.npy`.

## If the candidates are accepted

1. Copy `candidates/2D/*` and `candidates/3D/*` over the matching folders of `reference_data/`.
2. Rerun the Burgers 2D inverse runs on the regenerated data, then regenerate the benchmarks pages and Figure 4.
   The forward Burgers 2D errors can be recomputed from the saved models without retraining
   (`reeval_burgers2d.py`). The Poisson errors don't change at the printed precision.
3. Update the provenance notes (README and THIRD_PARTY_LICENSES.md on the licensing branch), which describe these
   files as computed with COMSOL.

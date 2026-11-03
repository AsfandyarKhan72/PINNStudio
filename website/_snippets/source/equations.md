<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_pde_inputs` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Equations page: one expression per equation, outputs, constants
- `_simplify_pde_expr` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): typed expression to the residual's Python code

**Helpers**

- `_simplify_expr` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): math names and powers to NumPy/PyTorch syntax
- `_fold_numeric_calls` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): numeric calls such as sqrt(3) folded to values
- `_pde_coord_names` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): which coordinates (x, y, z, t) the equations use
- `_clean_needed_derivs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): derivatives (du_x, du_xx, ...) the exported residual computes
- `_n_equations` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): number of residuals, independent of the number of outputs
- `_constant_values` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): named constants
- `_integral_text` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): integral(...) terms
- `_frac_data_text` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): frac(...) terms (DeepXDE's fractional PDE data)
- [`pinnstudio/ui/equation.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/equation.py): live preview of the typed equations
- [`pinnstudio/core/constants.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/constants.py): constants table

**Tests**

- [`tests/test_round61_pde_coordinates.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round61_pde_coordinates.py): coordinates and math functions in equations
- [`tests/test_round63_equation_count.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round63_equation_count.py): number of equations and outputs
- [`tests/test_round70_constants.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round70_constants.py): named constants

Files on the `main` branch of the repository; each link names the function or class to look for in the file.

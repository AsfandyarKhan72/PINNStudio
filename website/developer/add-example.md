# Adding an example

The **Quick Examples** list (and the README's examples table) come from one catalogue,
`pinnstudio/core/example_catalog.py`, which collects three kinds of entries:

| Kind | Defined in | Loaded by |
|---|---|---|
| DeepXDE examples | `EXAMPLES` in `pinnstudio/core/deepxde_examples.py` | `deepxde_examples.build_config()` then `MainWindow._apply_config()` |
| PINNStudio templates defined by settings | `EXAMPLES` and `CATALOG` in `pinnstudio/core/pinnstudio_examples.py` | the same |
| Older PINNStudio templates | `_TEMPLATES` in `example_catalog.py`, set up widget by widget in `MainWindow._on_template_selected()` | `main_window.py` |

A new example should be one of the first two kinds: a plain dictionary of `PINNConfig` fields, so that it loads
exactly like a saved problem and needs no window code. The third kind exists for the templates written before
settings-based examples were possible.

## An example is a set of configuration fields

An `Example` (`deepxde_examples.py`) has a `name` (its internal key, kept stable because saved problems and
tests use it), a `category` (`FORWARD` or `INVERSE`), a `source` (the DeepXDE documentation page or script,
or `""`), the `settings` dictionary, a `note` stating any deviation from the original, and optionally an
`inverse` dictionary applied on top of `settings` when the example is chosen from the Inverse list.

The helper `_ex(name, category, source, *parts, note=..., inverse=..., inverse_note=...)` merges the parts, and
small helpers build the common pieces:

| Helper | Gives |
|---|---|
| `_steady_1d(a, b)`, `_time_1d(a, b, t0, t1)`, `_ode(t0, t1)` | dimension, time dependence, interval |
| `_net(layers, init, activation, network_type)` | the network |
| `_adam(iterations, lr)`, `_LBFGS`, `_NO_LBFGS`, `_scheduler(*_phase(...))` | the training |
| `_consts(("name", "value"), ...)` | named constants, as the original script defines them |
| `_bc(type, location, value, component, ...)`, `_bcs(*rows)` | boundary-condition rows (`custom_bc_json`) |
| `_formula_ref([...], times)` | Error Analysis against an exact solution |
| `_vars(...)`, `_obs(...)`, `_inv(...)` | an Inverse mode: unknowns, measured-data files, settings |

The 3D Poisson example in `pinnstudio_examples.py` is a complete, short model of a settings-based template:

```python
EXAMPLES.append(_dx._ex(
    "3D Poisson (Cube with spherical cut-out)", _dx.FORWARD, "",
    dict(problem_dim="3D", steady_state=True, geometry_type="Custom", ...,
         pde_expressions="-du_xx - du_yy - du_zz - (6 - 4*(x**2 + y**2 + z**2))*exp(-(x**2 + y**2 + z**2))",
         custom_bc_json=_dx._bcs(_dx._bc("dirichlet", "left or right or bottom or top or front or back", "..."),
                                 _dx._bc("neumann", "np.isclose(x**2 + y**2 + (z - 1)**2, R**2)", "...")),
         num_domain=8000, num_boundary=2000, num_test=5000),
    _dx._consts(("R", "0.7")),
    _dx._net([3, 64, 64, 64, 64, 1]), _dx._adam(15000), dict(optimizer2="lbfgs", iterations2=15000),
    _dx._formula_ref(["exp(-(x**2 + y**2 + z**2))"]),
    note="..."))
CATALOG["3D Poisson (Cube with spherical cut-out)"] = ("Poisson 3D – cube with spherical cut-out", "")
```

## Steps

1. **Write the settings.** Add the entry to `deepxde_examples.EXAMPLES` (a DeepXDE example: reproduce the
   script's equation, geometry, conditions, network, optimizers, iterations and loss weights, and state every
   deviation in `note`) or to `pinnstudio_examples.EXAMPLES` (a PINNStudio template). Only fields that differ
   from the `PINNConfig` defaults need to be given.
2. **Name it in the catalogue.** For a DeepXDE example add the shown name to `DEEPXDE_NAMES` in
   `example_catalog.py` (style "Equation Dimension – detail") and, for a documentation-page source, the
   script file to `_DOC_TO_SCRIPT`. For a template add `CATALOG[key] = (shown name, paper)` and the paper's
   link to `PAPERS`. The catalogue orders entries by dimension, regime and equation family by itself.
3. **Add an Inverse mode** (optional): for a DeepXDE forward example, an entry in `ADDED_INVERSE` in
   `deepxde_examples.py`; for a template, `inverse=` in `_ex`. Give each unknown an initial guess away from
   its true value (a test checks this) and say in the note which data the unknowns are learnt from.
4. **Reference data.** Prefer an exact solution (`_formula_ref`). Otherwise put the files under
   `reference_data/<1D|2D|3D|deepxde>/<problem>/` and refer to them with `@REF/...` paths, which
   `build_config()` resolves against the reference-data folder. The data must be reproducible: write or extend
   a generator script in `tools/` (`make_exact_reference_data.py`, `make_deepxde_reference_data.py`,
   `make_poisson_reference_data.py`, ...) and record the method, resolution and parameters there; nothing is
   copied from DeepXDE's repository. File formats are described in
   [Initial conditions and imported data](../reference/initial-conditions.md) and
   [Reference data](../reproducibility/reference-data.md).
5. **Settings: Default / Tuned** (optional): a template taken from a paper can get the paper's own settings
   in `paper_settings.py`; a tuned setup is stored in `pinnstudio/resources/tuned_settings.json` (produced
   by the tuning study, see `example_settings.py`). An example without either loads as it is.
6. **Test.** Run

    ```bash
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_deepxde_examples.py tests/test_templates.py -q
    ```

    `test_deepxde_examples.py` checks for every DeepXDE example, forward and inverse, that the configuration
    validates, that every data file exists, that both generators give valid Python, and that every unknown
    starts away from its true value. `test_templates.py` does the same for every template through the
    window. Then load the example in the window and press **Solve** once.

7. **Document.** Add a CHANGELOG entry. The README's examples section and the benchmark pages are generated
   (`tools/make_benchmarks_readme.py`); do not edit the generated text by hand.

## Local templates

Templates you keep for yourself go in `pinnstudio/core/local_examples.py`, with their data under
`reference_data/local/`. The file defines `EXAMPLES`, `CATALOG` and `PAPERS` in the same form as
`pinnstudio_examples.py` and is added to the lists when present. Both paths are ignored by git.

# Adding a training method

A training method is a setting that changes what the generated script does during training. Adding one
touches the configuration, the window, both code generators, validation, the sweep registry and the tests.
This page follows an existing method, residual-based attention (RBA, one of the **Per-point weights:** choices on
the **Loss weights** page; theory in [Adaptive loss weighting](../theory/loss-weighting.md#residual-based-attention-rba)),
through every step. Line positions change; search for the names given.

## 1. Configuration fields

Add the fields to `PINNConfig` in `pinnstudio/core/config.py`, with defaults that leave existing problems
unchanged and a comment above them (the [configuration table](config.md#field-reference) is generated from it):

```python
point_weights: str = "None"        # "None", "RBA" or "SA-PINN"
rba_gamma: float = 0.999
rba_eta: float = 0.01
point_weights_lbfgs: str = "plain"
```

A problem saved before the fields existed has no such keys and loads with these defaults, so it trains as before.
Prefer one field per user-visible choice over packing several into a string; use a JSON-encoded string only for
variable-length lists.

## 2. Validation

Add the checks to `PINNConfig.validate()`, as messages a user can act on. For per-point weights these are an
unknown method name, \(0 < \gamma < 1\), \(\eta > 0\), and combinations that cannot work: a point resampler
(the weights belong to fixed points), or resampling rounds inside time windows. Validation runs before every
Solve and before every run of a sweep, so a refused combination never reaches the generated script.

## 3. The widgets

The widgets are built in `MainWindow._build_adaptive_and_sweep_panels()` (`pinnstudio/ui/main_window.py`), under
"Automatic weighting" in the **Loss Weights** group:

- `self.point_weights_combo` with items whose *data* is the configuration value (`addItem("Residual-based attention (RBA)", "RBA")`);
- `self.rba_gamma`, `self.rba_eta` (`SciSpinBox`) in `self.rba_widget`, shown only when RBA is selected (`_pw_changed`);
- a tooltip that names the paper, says what the method does in one or two sentences, and when it helps or hurts.

Follow the existing patterns (row helpers, visibility handlers, `_set_combo_data`) rather than introducing new
ones; WRITING.md and the user guide quote the label texts, so choose them carefully.

## 4. Widgets ↔ configuration

- `MainWindow._build_config()`: `point_weights=self.point_weights_combo.currentData() or "None"`,
  `rba_gamma=self.rba_gamma.value()`, ...
- `MainWindow._apply_config()`: `self._set_combo_data(self.point_weights_combo, getattr(config, 'point_weights', 'None') or 'None')`,
  `self.rba_gamma.setValue(...)`, ...
- If a template can switch the method on, reset it when another template is loaded (`_tpl_set_extras`), so a
  setting never carries over from one example to the next.

## 5. Code generation, in both generators

The generated code for RBA is produced by `_training_methods_defs(config)` in `pinnstudio/core/codegen.py`. For
per-point weights it emits

- a state dictionary `_PW` with the method and its parameters;
- `_pw_wrap(pde)`, which wraps the residual function so that the residuals at the training points are multiplied
  by the weights (and the weights updated, for RBA);
- a DeepXDE callback class `_PointWeights` that marks the training points, triggers the update every step,
  and holds the weights during L-BFGS.

`_training_methods_cbs()` appends the callback to the callbacks list that `_build_train_cbs_code()` builds for
every `model.train(...)` call of the Training Phases. `_training_methods_defs` is inserted in **both**
`generate_script()` (Solve) and `generate_clean_script()` (Export), so the two scripts train the same model.

When adding a method:

- write the code once, in a helper used by both generators; if the short export script cannot support the method
  yet, make it fall back explicitly (as `_ta_clean_unsupported` does) rather than silently leaving it out;
- generate nothing when the method is off (`return ""`), so that existing scripts stay byte-for-byte the same;
- if the method can be expressed through an existing feature (as exact constraints are written as transform
  formulas), do it in `_effective_config()` instead;
- print one line to the Training Log when the method is active, with its parameters.

## 6. Sweep registry

To make the method sweepable, add a `SweepParam` in `pinnstudio/core/sweep_registry.py`:

```python
SweepParam(
    id="point_weights", label="Per-point weights", category="Training Methods",
    value_type="categorical", choices=["None", "RBA", "SA-PINN"],
    get_value=lambda c: str(getattr(c, "point_weights", "None") or "None"),
    set_value=_pw_set, is_available=lambda c: True,
),
```

`set_value` must leave a valid configuration (here `_pw_set` also sets `point_weights_lbfgs`), and
`is_available` must return False where the window would not offer the setting.

## 7. Tests

- A test of the whole path, like `tests/test_training_methods.py`: set the widgets, build the configuration,
  check the values, save and re-apply it as a `.pinn.json` round trip, check validation, generate both scripts,
  and run the generated method code on a tiny DeepXDE model. A test that trains belongs in `SLOW_FILES`
  ([Tests](tests.md)).
- Add the method to the export-parity checks (`tests/test_export_parity.py`) if it changes the generated code.
- A bug fix gets a regression test that fails before the fix, in the style of `tests/test_audit_v160.py`
  (code generation and validation) and `tests/test_audit_v160_runs.py` (short training runs).

## 8. Documentation

- a CHANGELOG entry under *Unreleased*;
- its entry under [Where each feature is in the code](structure.md#where-each-feature-is-in-the-code), with a
  `<!-- source: ... -->` comment, and its source links (`python tools/update_source_links.py`);
- this site: the [Feature reference](../reference/index.md) page, the [Theory](../theory/index.md) page with the
  formulas as implemented and any difference from the paper, the reference in
  [Scientific references](../about/references.md), and the configuration table
  (`python tools/docs/make_config_reference.py`).

#!/usr/bin/env python3
"""
DeepXDE Examples -- Quick Examples has a Category selector (PINNStudio /
DeepXDE Forward / DeepXDE Inverse). The DeepXDE categories list the
examples from DeepXDE's documentation, defined in
pinnstudio/core/deepxde_examples.py with DeepXDE's own settings.

Checks:
 - every example builds a PINNConfig that validates, whose reference /
   measurement files exist, and from which both script generators produce
   valid Python;
 - DeepXDE's default loss weights (all 1) are used unless the example sets
   its own;
 - the Category combo refills "Load example:" per category;
 - picking each example in the GUI fills every widget so that
   _build_config() gives back exactly the example's settings (nothing
   clamped or rounded by a widget, e.g. pi in a domain bound), the combo
   keeps showing the example, and the user's save folder is kept;
 - switching back to the PINNStudio category still loads its templates.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_deepxde_examples.py
"""
import ast
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core.codegen import _clean_loss_weights, generate_clean_script, generate_script
from pinnstudio.ui.main_window import REFERENCE_DATA_DIR, MainWindow


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def same(a, b):
    """Equal as settings: numbers to 1e-9, JSON compared as data (only the
    keys the example gives -- the GUI adds its defaults for the rest),
    comma-separated numbers as numbers, text ignoring spaces."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in b and same(v, b[k]) for k, v in a.items())
    if isinstance(a, str) and isinstance(b, str):
        sa, sb = a.strip(), b.strip()
        if sa.replace(" ", "") == sb.replace(" ", ""):
            return True
        try:
            return same(json.loads(sa), json.loads(sb))
        except ValueError:
            pass
        pa, pb = sa.split(","), sb.split(",")
        if len(pa) == len(pb) and all(_num(x) is not None for x in pa + pb):
            return all(same(_num(x), _num(y)) for x, y in zip(pa, pb))
        return False
    return a == b


def _paths(cfg):
    out = [p for (_t, p, _i) in ast.literal_eval(cfg.ea_files or "[]")]
    for row in json.loads(cfg.custom_bc_json or "[]"):
        if row.get("points_file"):
            out.append(row["points_file"])
    for row in json.loads(cfg.inverse_obs_files_json or "[]"):
        out.append(row["path"])
    if cfg.anchors_mode == "file":
        out.append(cfg.anchors_file)
    return out


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    check(len(dx.EXAMPLES) == len(dx.BY_NAME), "example names are unique")
    check(dx.names(dx.FORWARD) and dx.names(dx.INVERSE), "both categories have examples")

    # 1) Every example: valid config, files present, both scripts generate.
    for ex in dx.EXAMPLES:
        cfg = dx.build_config(ex, REFERENCE_DATA_DIR)
        errs = cfg.validate()
        check(not errs, f"{ex.name}: config validates {errs}")
        missing = [p for p in _paths(cfg) if not os.path.isfile(p)]
        check(not missing, f"{ex.name}: data files exist {missing}")
        for gen in (generate_script, generate_clean_script):
            try:
                ast.parse(gen(cfg))
                ok = True
            except Exception as e:  # noqa: BLE001
                ok = False
                print("   ", type(e).__name__, e)
            check(ok, f"{ex.name}: {gen.__name__} gives valid Python")

    # 2) DeepXDE's default loss weights.
    heat = dx.build_config(dx.BY_NAME["Heat equation"], REFERENCE_DATA_DIR)
    check(heat.loss_weights_multi == "1,1,1", f"Heat: PDE, BC, IC all weighted 1, got {heat.loss_weights_multi}")
    beam = dx.build_config(dx.BY_NAME["Euler beam"], REFERENCE_DATA_DIR)
    check(beam.loss_weights_multi == "1,1,1,1,1", f"Euler beam: 1 PDE + 4 BC weights, got {beam.loss_weights_multi}")

    # Equations != outputs: a phase's own weights are used when they have
    # one entry per equation (+ BCs + ICs), not per output.
    put = dx.build_config(dx.BY_NAME["American put option"], REFERENCE_DATA_DIR)
    bcs = json.loads(put.custom_bc_json)
    wts, n_exp = _clean_loss_weights(put, 1, bcs, put.ic_active.split(","), [])
    check(len(wts) == n_exp == 6, f"American put: 3 equations + 2 BCs + 1 IC weights, got {wts} / {n_exp}")
    live = generate_script(dx.build_config(dx.BY_NAME["Poisson 1D – hard boundary conditions"], REFERENCE_DATA_DIR))
    check("_sp.get('weights')" in live and "_sp['weights']" not in live,
          "a scheduler phase without its own weights uses the Loss Weights panel's")

    # 3) GUI: category combo and loading every example.
    w = MainWindow()
    w.save_dir_input.setText("/my/runs")
    cats = [w.quick_examples_category_combo.itemText(i) for i in range(w.quick_examples_category_combo.count())]
    check(cats == ["Forward", "Inverse"], f"categories {cats}")
    from pinnstudio.core import example_catalog as cat_mod

    def listed():
        return w.quick_examples_combo.keys()

    fwd_items = listed()
    shown_templates = set(cat_mod.TEMPLATE_KEYS) - set(cat_mod.HIDDEN_TEMPLATES)
    check(shown_templates <= set(fwd_items) and fwd_items[0] == "None", "Forward lists the PINNStudio templates")
    check(not set(cat_mod.HIDDEN_TEMPLATES) & set(fwd_items), "templates a DeepXDE example replaces are not listed")
    for cat, dx_cat in (("Forward", dx.FORWARD), ("Inverse", dx.INVERSE)):
        w.quick_examples_category_combo.setCurrentText(cat)
        check(listed() == ["None", *[e.key for e in cat_mod.for_category(cat)]], f"{cat}: lists its examples in catalogue order")
        want_templates = shown_templates - (set(cat_mod.FORWARD_ONLY_TEMPLATES) if cat == "Inverse" else set())
        check(set(dx.names(dx_cat)) <= set(listed()) and want_templates <= set(listed()),
              f"{cat}: every DeepXDE {cat.lower()} example and every template")
        if cat == "Inverse":
            check(not set(cat_mod.FORWARD_ONLY_TEMPLATES) & set(listed()), "Forward-only templates are not in Inverse")
        for name in dx.names(dx_cat):
            ex = dx.BY_NAME[name]
            want = dx.build_config(ex, REFERENCE_DATA_DIR)
            w.quick_examples_combo.setCurrentText(name)
            got = w._build_config()
            keys = set(ex.settings) | {"layers", "pde_expression", "ic_active", "output_names", "loss_weights_multi"}
            if not want.steady_state:
                keys.add("ic_expression")
            bad = []
            for k in sorted(keys):
                a, b = getattr(want, k), getattr(got, k)
                if k == "num_equations" and b == 0 and a == want.num_outputs:
                    continue  # 0 = "one per output"
                if not same(a, b):
                    bad.append(f"{k}: want {str(a)[:120]!r} got {str(b)[:120]!r}")
            check(not bad, f"{name}: GUI gives back the example's settings" + ("".join("\n      " + x for x in bad)))
            check(w.quick_examples_combo.currentText() == name, f"{name}: stays selected")
            check(w.save_dir_input.text() == "/my/runs", f"{name}: user's save folder kept")

    # 4) Back to the Forward list and a PINNStudio template.
    w.quick_examples_category_combo.setCurrentText("Forward")
    check(listed() == fwd_items, "Forward list restored")
    w.quick_examples_combo.setCurrentText("1D Heat")
    cfg = w._build_config()
    check(cfg.problem_dim == "1D" and "du_t" in cfg.pde_expressions and w.quick_examples_combo.currentText() == "1D Heat",
          "a PINNStudio template (here a hidden one, loaded by name) still loads after a DeepXDE example")

    # 5) Forward DeepXDE examples with an Inverse mode load it from the Inverse list.
    for name in [e.key for e in cat_mod.ENTRIES if e.source == cat_mod.DEEPXDE and e.problem == cat_mod.BOTH]:
        w.quick_examples_category_combo.setCurrentText("Inverse")
        check(name in listed(), f"{name}: listed under Inverse")
        w.quick_examples_combo.setCurrentText(name)
        got, want = w._build_config(), dx.build_config(dx.BY_NAME[name], REFERENCE_DATA_DIR, inverse=True)
        check(got.problem_type == "Inverse" and w.radio_inverse.isChecked()
              and same(got.inverse_variables_json, want.inverse_variables_json)
              and same(got.pde_expressions, want.pde_expressions), f"{name}: Inverse list loads its Inverse mode")
        w.quick_examples_category_combo.setCurrentText("Forward")
        w.quick_examples_combo.setCurrentText(name)
        check(w._build_config().problem_type == "Forward", f"{name}: Forward list loads the forward example")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL DEEPXDE EXAMPLES TESTS PASSED")
    return 0


def test_deepxde_examples():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())

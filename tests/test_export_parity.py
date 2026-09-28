#!/usr/bin/env python3
"""
Checks for two export/restore parity fixes:

1. generate_clean_script() ("Export as DeepXDE Script") previously always
   plotted and error-analyzed a raw output column, even for a template
   with a custom/derived plot field configured (e.g. 1D Schrodinger's
   |h| = sqrt(u**2+v**2)) -- silently different from what the Results
   panel/main Solve path actually show. It now defines the same kind of
   _extract_plot_field helper generate_script() already has, and an
   Error Analysis reference file tagged for that custom field is kept
   (one tagged for a different field is still correctly dropped, not
   silently compared against the wrong prediction).

2. The Restore panel's own Error Analysis script builder
   (_build_restore_ea_script) previously always compared against output
   column 0 regardless of which output was selected in the Restore
   panel's own "output:" combo, and had no 3D branch at all (a 3D
   restored model's reference data would be parsed with the wrong
   column layout). It now uses the selected output_idx (or a custom
   expression, if ever wired up from the UI) and has a real 3D data-
   loading/prediction branch; 3D surface-comparison plotting is left as
   an explicit, logged "not available yet" skip rather than silently
   producing a wrong or crashing plot.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_export_parity.py
"""
import os
import sys
import dataclasses
import ast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.codegen import generate_clean_script


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()

    # ---- generate_clean_script(): custom-field solution plot ----
    win.radio_1d.setChecked(True)
    win.quick_examples_combo.setCurrentText("1D Schrödinger")
    config = win._build_config()
    check(config.plot_custom_expr == "sqrt(u**2+v**2)",
          "1D Schrodinger should auto-configure its |h| custom plot field")

    script = generate_clean_script(config)
    ast.parse(script)
    check("_extract_plot_field" in script,
          "generate_clean_script() should define the custom-field extraction helper")
    check("sqrt(u**2+v**2)" in script,
          "generate_clean_script() should bake in the actual custom expression")

    # ---- generate_clean_script(): EA file selector matching ----
    matching = repr([(0.5, "/tmp/fake_h_0.5.txt", ["sqrt(u**2+v**2)", "|h|"])])
    c_match = dataclasses.replace(config, ea_files=matching, ea_do_line=True,
                                   ea_do_surface=False, t_min=0.0, t_max=1.0)
    s_match = generate_clean_script(c_match)
    ast.parse(s_match)
    check("fake_h_0.5.txt" in s_match,
          "an EA file tagged with the SAME custom expression should be kept")

    mismatching = repr([(0.5, "/tmp/fake_other.txt", ["u", "raw u"])])
    c_mismatch = dataclasses.replace(config, ea_files=mismatching, ea_do_line=True,
                                      ea_do_surface=False, t_min=0.0, t_max=1.0)
    s_mismatch = generate_clean_script(c_mismatch)
    ast.parse(s_mismatch)
    check("fake_other.txt" not in s_mismatch,
          "an EA file tagged for a DIFFERENT field should still be dropped, not silently compared")

    # ---- Non-custom template: old int-selector behavior preserved ----
    win2 = MainWindow()
    win2.radio_1d.setChecked(True)
    win2.quick_examples_combo.setCurrentText("1D Heat")
    heat_config = win2._build_config()
    heat_ea = repr([(0.5, "/tmp/fake_heat.txt", 0)])
    c_heat = dataclasses.replace(heat_config, ea_files=heat_ea, ea_do_line=True,
                                  ea_do_surface=False,
                                  t_min=heat_config.t_min, t_max=heat_config.t_max)
    s_heat = generate_clean_script(c_heat)
    ast.parse(s_heat)
    check("fake_heat.txt" in s_heat,
          "a plain int-selector EA file for a non-custom template should still be kept (unchanged behavior)")

    # ---- Restore panel's own EA script builder ----
    files = [(0.5, "/tmp/fake_1d.txt")]

    s_1d = win._build_restore_ea_script(
        files, "/tmp/save", is_2d=False, do_line=True, do_surface=True,
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, out_name="v",
        is_3d=False, output_idx=1, output_names="u,v",
    )
    ast.parse(s_1d)
    check("model.predict(_xt)[:, 0]" not in s_1d and "model.predict(_xt_c)[:, 0]" not in s_1d,
          "Restore EA script should no longer hardcode output column 0")
    check("pred[:, 1]" in s_1d,
          "Restore EA script should use the actually-selected output_idx")

    s_3d = win._build_restore_ea_script(
        files, "/tmp/save", is_2d=False, do_line=True, do_surface=True,
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, out_name="u",
        is_3d=True, output_idx=0, output_names="u",
    )
    ast.parse(s_3d)
    check("_ea_z_refs" in s_3d,
          "Restore EA script should have a real 3D reference-data branch now")
    check("skipping" in s_3d,
          "Restore EA script should explicitly skip (not silently mis-plot) 3D surface comparison")

    s_custom = win._build_restore_ea_script(
        files, "/tmp/save", is_2d=False, do_line=True, do_surface=False,
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, out_name="|h|",
        is_3d=False, output_idx=0, custom_expr="sqrt(u**2+v**2)", output_names="u,v",
    )
    ast.parse(s_custom)
    check("sqrt(u**2+v**2)" in s_custom,
          "Restore EA script's custom_expr parameter should be usable once wired up from the UI")

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL EXPORT-PARITY TESTS PASSED")
    return 0


def test_export_parity():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())

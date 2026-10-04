#!/usr/bin/env python3
"""
Custom Geometry (Phase 1): config validation, the GUI shape-builder panel
(add/remove/reorder shapes, JSON round-trip), the live-preview script
builder, and both codegen paths (generate_script's _build_geom() and
generate_clean_script's _clean_geom_line()) -- covering the CSG chain
(CSGUnion/CSGDifference/CSGIntersection) and the _DTypeSafeGeom
wrap_needed rule (wrapped only for a single buggy-type shape with no CSG
op; never wrapped for 2+ shapes, verified empirically against the
installed DeepXDE version -- see codegen.py's _build_custom_geom_code()
docstring).

This file only checks that the generated scripts are syntactically valid
Python with the right structure (ast.parse), the same way every other
test in this suite does -- it doesn't execute them, since CI runs
headless without torch/deepxde installed. An actual end-to-end DeepXDE
training run on a composite geometry (Triangle minus Disk, matching the
feature's own reference image) was verified manually in an environment
with deepxde installed and is not re-run here; see the Custom Geometry
delivery notes.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_custom_geometry.py
"""
import os
import sys
import ast
import json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.config import PINNConfig
from pinnstudio.core import codegen
from pinnstudio.core.codegen import generate_script, generate_clean_script, _build_custom_geom_code


TRIANGLE_MINUS_DISK = [
    {"type": "Triangle", "params": {"vertices_text": "0,0;2,0;1,2"}},
    {"type": "Disk", "params": {"cx": 1.0, "cy": 0.7, "r": 0.3}, "op": "subtract"},
]


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── Config-level validation ──────────────────────────────────
    c = PINNConfig()
    check(c.geom_custom_shapes_json == "[]", "geom_custom_shapes_json should default to '[]'")
    check(c.geometry_type != "Custom", "geometry_type should not default to Custom")
    check(c.validate() == [], "a default (non-Custom) config should validate cleanly")

    c_empty = PINNConfig()
    c_empty.geometry_type = "Custom"
    c_empty.geom_custom_shapes_json = "[]"
    check(any("no shapes" in e for e in c_empty.validate()),
          "Custom geometry with an empty shape list should fail validate()")

    c_badjson = PINNConfig()
    c_badjson.geometry_type = "Custom"
    c_badjson.geom_custom_shapes_json = "not json"
    check(any("not valid JSON" in e for e in c_badjson.validate()),
          "malformed geom_custom_shapes_json should fail validate()")

    c_badtype = PINNConfig()
    c_badtype.geometry_type = "Custom"
    c_badtype.geom_custom_shapes_json = json.dumps([{"type": "Blob", "params": {}}])
    check(any("unrecognized" in e for e in c_badtype.validate()),
          "an unrecognized shape type should fail validate()")

    c_ok = PINNConfig()
    c_ok.geometry_type = "Custom"
    c_ok.geom_custom_shapes_json = json.dumps(TRIANGLE_MINUS_DISK)
    check(c_ok.validate() == [], f"a valid Custom shape list should validate cleanly, got {c_ok.validate()}")

    # geom_custom_shapes_json is still checked for valid JSON regardless of
    # geometry_type (same as every other JSON-encoded field) -- only the
    # "has recognized shapes" check is gated on geometry_type == "Custom".
    c_irrelevant = PINNConfig()
    c_irrelevant.geometry_type = "Rectangle"
    c_irrelevant.geom_custom_shapes_json = "not json at all"
    check(any("not valid JSON" in e for e in c_irrelevant.validate()),
          "geom_custom_shapes_json should still be checked for valid JSON even when geometry_type != Custom")

    # ── GUI: shape-builder panel ─────────────────────────────────
    win = MainWindow()
    win.radio_2d.setChecked(True)
    win._on_dim_changed()

    check("Custom" in win.GEOM_TYPES_2D, "Custom should be offered in 2D")
    check("Custom" not in win.GEOM_TYPES_3D, "Custom is 2D-only in Phase 1 -- not offered in 3D")

    win.geometry_type_combo.setCurrentText("Custom")
    check(len(win.custom_geom_shape_rows) == 1,
          "selecting Custom for the first time should auto-seed one shape row")
    check(win.custom_geom_shape_rows[0]['op_row_widget'].isHidden(),
          "the first (only) shape's combine-op row should be hidden")

    r0 = win.custom_geom_shape_rows[0]
    r0['type_combo'].setCurrentText("Triangle")
    r0['tri_verts'].setText("0,0;2,0;1,2")
    win._add_custom_geom_shape()
    r1 = win.custom_geom_shape_rows[1]
    r1['type_combo'].setCurrentText("Disk")
    r1['disk_cx'].setValue(1.0); r1['disk_cy'].setValue(0.7); r1['disk_r'].setValue(0.3)
    r1['op_combo'].setCurrentIndex(r1['op_combo'].findData("subtract"))
    check(not r1['op_row_widget'].isHidden(), "the second shape's combine-op row should be visible")

    built_json = win._build_custom_geom_shapes_json()
    parsed = json.loads(built_json)
    check(parsed == TRIANGLE_MINUS_DISK,
          f"_build_custom_geom_shapes_json() mismatch: {parsed} != {TRIANGLE_MINUS_DISK}")

    # Reorder: move Disk (index 1) up -- now first, so its op row hides
    win._add_custom_geom_shape(shape_type="Rectangle", op="union")
    r2 = win.custom_geom_shape_rows[2]
    win._move_custom_geom_shape(r1, -1)
    check(win.custom_geom_shape_rows == [r1, r0, r2], "reorder (move up) should swap list order")
    check(r1['op_row_widget'].isHidden(), "the shape that's now first should hide its combine-op row")
    check(not r0['op_row_widget'].isHidden(), "the shape that's no longer first should show its combine-op row")
    out_of_range = list(win.custom_geom_shape_rows)
    win._move_custom_geom_shape(r1, -1)  # r1 is already first -- no-op
    check(win.custom_geom_shape_rows == out_of_range, "moving the first row further up should be a no-op")

    # Remove: delete the (now-first) row and confirm the new first row's
    # combine-op row gets hidden.
    r1['widget'].deleteLater()
    win.custom_geom_shape_rows.remove(r1)
    if win.custom_geom_shape_rows:
        win.custom_geom_shape_rows[0]['op_row_widget'].setVisible(False)
    check(win.custom_geom_shape_rows[0]['op_row_widget'].isHidden(),
          "after removing the first shape, the new first shape should hide its combine-op row")

    # Apply/round-trip: rebuild from a fresh JSON string
    win._apply_custom_geom_shapes_json(json.dumps(TRIANGLE_MINUS_DISK))
    check(len(win.custom_geom_shape_rows) == 2, "_apply_custom_geom_shapes_json should rebuild the row list")
    check(json.loads(win._build_custom_geom_shapes_json()) == TRIANGLE_MINUS_DISK,
          "round-trip through _apply/_build_custom_geom_shapes_json should be lossless")

    # Full _build_config / _apply_config round trip
    full_config = win._build_config()
    check(full_config.geometry_type == "Custom", "_build_config should record geometry_type = Custom")
    check(json.loads(full_config.geom_custom_shapes_json) == TRIANGLE_MINUS_DISK,
          "_build_config should serialize the shape list")
    check(full_config.validate() == [], f"the built config should validate cleanly, got {full_config.validate()}")

    win2 = MainWindow()
    win2._apply_config(full_config)
    check(win2.geometry_type_combo.currentText() == "Custom", "_apply_config should restore geometry_type")
    check(json.loads(win2._build_custom_geom_shapes_json()) == TRIANGLE_MINUS_DISK,
          "_apply_config should restore the same shape list on a fresh MainWindow")

    # Empty-list guard surfaces through the GUI too
    win3 = MainWindow()
    win3.radio_2d.setChecked(True)
    win3._on_dim_changed()
    win3.geometry_type_combo.setCurrentText("Custom")
    win3.custom_geom_shape_rows[0]['widget'].deleteLater()
    win3.custom_geom_shape_rows.clear()
    check(any("no shapes" in e for e in win3._build_config().validate()),
          "an emptied Custom shape list should surface a validate() error through the GUI")

    # ── Domain preview ────────────────────────────────────────────
    win4 = MainWindow()
    win4.radio_2d.setChecked(True)
    win4._on_dim_changed()
    win4.geometry_type_combo.setCurrentText("Custom")
    pr0 = win4.custom_geom_shape_rows[0]
    pr0['type_combo'].setCurrentText("Triangle")
    pr0['tri_verts'].setText("0,0;2,0;1,2")
    win4._add_custom_geom_shape()
    pr1 = win4.custom_geom_shape_rows[1]
    pr1['type_combo'].setCurrentText("Disk")
    pr1['disk_cx'].setValue(1.0); pr1['disk_cy'].setValue(0.7); pr1['disk_r'].setValue(0.3)
    pr1['op_combo'].setCurrentIndex(pr1['op_combo'].findData("subtract"))
    captured = {}
    win4._launch_preview_thread = lambda tmp: captured.__setitem__('tmp', tmp)
    win4._preview_domain()
    check('tmp' in captured, "_preview_domain() should reach script generation for a non-empty Custom geometry")
    if 'tmp' in captured:
        preview_script = open(captured['tmp']).read()
        os.unlink(captured['tmp'])
        try:
            ast.parse(preview_script)
        except SyntaxError as e:
            check(False, f"generated preview script is not valid Python: {e}")
        check("dde.geometry.CSGDifference(dde.geometry.Triangle" in preview_script,
              "preview script should build the CSG-difference chain")
        check("shape_patches = [" in preview_script and "for _sp in shape_patches:" in preview_script,
              "preview script should draw one outline patch per leaf shape")

    # Non-Custom preview paths still work after the patch-loop refactor
    for gtype in ("Rectangle", "Disk", "Ellipse", "Triangle"):
        winN = MainWindow()
        winN.radio_2d.setChecked(True)
        winN._on_dim_changed()
        winN.geometry_type_combo.setCurrentText(gtype)
        capturedN = {}
        winN._launch_preview_thread = lambda tmp, c=capturedN: c.__setitem__('tmp', tmp)
        winN._preview_domain()
        check('tmp' in capturedN, f"{gtype} preview should still reach script generation")
        if 'tmp' in capturedN:
            scriptN = open(capturedN['tmp']).read()
            os.unlink(capturedN['tmp'])
            try:
                ast.parse(scriptN)
            except SyntaxError as e:
                check(False, f"{gtype} preview script is not valid Python: {e}")

    # Empty shape list aborts cleanly (no script generated)
    win5 = MainWindow()
    win5.radio_2d.setChecked(True)
    win5._on_dim_changed()
    win5.geometry_type_combo.setCurrentText("Custom")
    win5.custom_geom_shape_rows[0]['widget'].deleteLater()
    win5.custom_geom_shape_rows.clear()
    captured5 = {}
    win5._launch_preview_thread = lambda tmp: captured5.__setitem__('tmp', tmp)
    win5._preview_domain()
    check('tmp' not in captured5, "previewing an empty Custom shape list should not launch a script")

    # ── Codegen: _build_custom_geom_code (shared helper) ──────────
    rect_only = PINNConfig()
    rect_only.geom_custom_shapes_json = json.dumps([
        {"type": "Rectangle", "params": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}}
    ])
    code, wrap = _build_custom_geom_code(rect_only)
    check(code == "dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])", f"unexpected Rectangle leaf code: {code}")
    check(wrap is False, "a single Rectangle should never need _DTypeSafeGeom wrapping")

    disk_only = PINNConfig()
    disk_only.geom_custom_shapes_json = json.dumps([{"type": "Disk", "params": {"cx": 0.5, "cy": 0.5, "r": 0.3}}])
    code, wrap = _build_custom_geom_code(disk_only)
    check(code == "dde.geometry.Disk([0.5, 0.5], 0.3)", f"unexpected Disk leaf code: {code}")
    check(wrap is True, "a single Disk (a known-buggy-dtype type) should need _DTypeSafeGeom wrapping")

    tri_minus_disk = PINNConfig()
    tri_minus_disk.geom_custom_shapes_json = json.dumps(TRIANGLE_MINUS_DISK)
    code, wrap = _build_custom_geom_code(tri_minus_disk)
    check(code == ("dde.geometry.CSGDifference("
                    "dde.geometry.Triangle([0.0, 0.0], [2.0, 0.0], [1.0, 2.0]), "
                    "dde.geometry.Disk([1.0, 0.7], 0.3))"), f"unexpected CSG chain: {code}")
    check(wrap is False, "a 2-shape CSG combination should never need _DTypeSafeGeom wrapping, "
                          "even though both leaves are individually buggy types")

    three_shape = PINNConfig()
    three_shape.geom_custom_shapes_json = json.dumps([
        {"type": "Rectangle", "params": {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0}},
        {"type": "Disk", "params": {"cx": 0.5, "cy": 0.5, "r": 0.3}, "op": "union"},
        {"type": "Ellipse", "params": {"cx": 1.5, "cy": 1.5, "a": 0.4, "b": 0.2, "angle": 0.0}, "op": "intersect"},
    ])
    code, wrap = _build_custom_geom_code(three_shape)
    check(code.startswith("dde.geometry.CSGIntersection(dde.geometry.CSGUnion("),
          f"3-shape chain should nest left-to-right in build order: {code}")
    check(wrap is False, "a 3-shape CSG combination should never need _DTypeSafeGeom wrapping")

    empty_shapes = PINNConfig()
    empty_shapes.geom_custom_shapes_json = "[]"
    code, wrap = _build_custom_geom_code(empty_shapes)
    check("dde.geometry.Rectangle" in code, "an empty shape list should fall back to a sane default Rectangle, not crash")

    malformed = PINNConfig()
    malformed.geom_custom_shapes_json = "not json"
    code, wrap = _build_custom_geom_code(malformed)
    check("dde.geometry.Rectangle" in code, "malformed JSON should fall back to a sane default, not crash codegen")

    # ── Codegen: generate_script() (live Solve/Sweep path) ────────
    live_config = PINNConfig()
    live_config.problem_dim = "2D"
    live_config.geometry_type = "Custom"
    live_config.geom_custom_shapes_json = json.dumps(TRIANGLE_MINUS_DISK)
    live_config.x_min, live_config.x_max = 0.0, 2.0
    live_config.y_min, live_config.y_max = 0.0, 2.0
    script = generate_script(live_config)
    try:
        ast.parse(script)
    except SyntaxError as e:
        check(False, f"generate_script() output is not valid Python for Custom geometry: {e}")
    check('elif _geom_type == "Custom":' in script, "generate_script() should add a Custom branch to _build_geom()")
    check("dde.geometry.CSGDifference(dde.geometry.Triangle" in script,
          "generate_script() should embed the CSG chain literally")
    check("_DTypeSafeGeom(dde.geometry.CSGDifference" not in script,
          "the 2-shape CSG case should not be wrapped in generate_script() either")

    live_single = PINNConfig()
    live_single.problem_dim = "2D"
    live_single.geometry_type = "Custom"
    live_single.geom_custom_shapes_json = json.dumps([{"type": "Disk", "params": {"cx": 0.5, "cy": 0.5, "r": 0.3}}])
    script_single = generate_script(live_single)
    ast.parse(script_single)
    check("_DTypeSafeGeom(dde.geometry.Disk" in script_single,
          "a single-Disk Custom geometry should be wrapped in generate_script() too")

    # ── Codegen: generate_clean_script() (Export Script path) ─────
    clean_config = PINNConfig()
    clean_config.problem_dim = "2D"
    clean_config.geometry_type = "Custom"
    clean_config.geom_custom_shapes_json = json.dumps(TRIANGLE_MINUS_DISK)
    clean_config.x_min, clean_config.x_max = 0.0, 2.0
    clean_config.y_min, clean_config.y_max = 0.0, 2.0
    clean_script = generate_clean_script(clean_config)
    try:
        ast.parse(clean_script)
    except SyntaxError as e:
        check(False, f"generate_clean_script() output is not valid Python for Custom geometry: {e}")
    check("geom = dde.geometry.CSGDifference(dde.geometry.Triangle" in clean_script,
          "generate_clean_script() should emit the single literal CSG-chain geom line")
    check("class _DTypeSafeGeom" not in clean_script,
          "the export script shouldn't define _DTypeSafeGeom when the 2-shape CSG case doesn't need it")

    clean_single = PINNConfig()
    clean_single.problem_dim = "2D"
    clean_single.geometry_type = "Custom"
    clean_single.geom_custom_shapes_json = json.dumps([{"type": "Disk", "params": {"cx": 0.5, "cy": 0.5, "r": 0.3}}])
    clean_script_single = generate_clean_script(clean_single)
    ast.parse(clean_script_single)
    check("class _DTypeSafeGeom" in clean_script_single,
          "a single-Disk Custom geometry's export script should define _DTypeSafeGeom")
    check("geom = _DTypeSafeGeom(dde.geometry.Disk" in clean_script_single,
          "a single-Disk Custom geometry's export script should wrap the geom line")

    # Parity: both codegen paths should agree on the CSG structure itself
    # (ignoring the live path's extra _geom_type runtime-dispatch scaffolding,
    # which the export path doesn't have -- see codegen.py's own comments).
    check("CSGDifference(dde.geometry.Triangle([0.0, 0.0], [2.0, 0.0], [1.0, 2.0]), dde.geometry.Disk([1.0, 0.7], 0.3))"
          in script and
          "CSGDifference(dde.geometry.Triangle([0.0, 0.0], [2.0, 0.0], [1.0, 2.0]), dde.geometry.Disk([1.0, 0.7], 0.3))"
          in clean_script,
          "generate_script() and generate_clean_script() should build the identical CSG chain")

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL CUSTOM GEOMETRY TESTS PASSED")
    return 0


def test_custom_geometry():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())

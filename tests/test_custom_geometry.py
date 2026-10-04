#!/usr/bin/env python3
"""
Custom Geometry: config validation, the GUI shape-builder panel
(add/remove/reorder shapes, JSON round-trip), the live-preview script
builder, and both codegen paths (generate_script's _build_geom() and
generate_clean_script's _clean_geom_line()) -- covering the CSG chain
(CSGUnion/CSGDifference/CSGIntersection) and the _DTypeSafeGeom
wrap_needed rule (wrapped only for a single buggy-type shape with no CSG
op; never wrapped for 2+ shapes, verified empirically against the
installed DeepXDE version -- see codegen.py's _build_custom_geom_code()
docstring).

Phase 1 covers 2D only (Rectangle/Disk/Ellipse/Triangle/Polygon). The
second section below (3D Custom Geometry) extends the same coverage to
Cuboid/Sphere in 3D: the dimension-aware shape-type lists, the
dimension-switch clearing of custom_geom_shape_rows, the Cuboid/Sphere
per-row widgets, the refactored 3D preview script builder
(_build_3d_preview_script now delegating to the shared
_render_3d_preview_script, with the new _custom_geom_3d_leaf_code
feeding the Custom-3D branch of _preview_domain), and both codegen
paths' CSG-chain + dtype-wrap logic for Cuboid/Sphere (Cuboid is never
dtype-buggy like Rectangle; Sphere alone is buggy like Disk/Ellipse/
Triangle/Polygon; any 2+-shape CSG combination of the two never needs
wrapping, verified empirically the same way the 2D leaf types were).

This file only checks that the generated scripts are syntactically valid
Python with the right structure (ast.parse), the same way every other
test in this suite does -- it doesn't execute them, since CI runs
headless without torch/deepxde installed. Actual end-to-end DeepXDE
training runs on composite geometries (2D: Triangle minus Disk, matching
the feature's own reference image; 3D: Cuboid minus Sphere) were
verified manually in an environment with deepxde installed and are not
re-run here; see the Custom Geometry delivery notes.

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

CUBOID_MINUS_SPHERE = [
    {"type": "Cuboid", "params": {
        "x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0, "z_min": 0.0, "z_max": 2.0}},
    {"type": "Sphere", "params": {"cx": 1.0, "cy": 1.0, "cz": 1.0, "r": 0.5}, "op": "subtract"},
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
    check("Custom" in win.GEOM_TYPES_3D, "Custom should also be offered in 3D")

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

    # ════════════════════════════════════════════════════════════
    # 3D Custom Geometry (Cuboid/Sphere + CSG)
    # ════════════════════════════════════════════════════════════

    # ── Config-level validation (3D shape types recognized) ──────
    c3_ok = PINNConfig()
    c3_ok.problem_dim = "3D"
    c3_ok.geometry_type = "Custom"
    c3_ok.geom_custom_shapes_json = json.dumps(CUBOID_MINUS_SPHERE)
    check(c3_ok.validate() == [], f"a valid 3D Custom shape list should validate cleanly, got {c3_ok.validate()}")

    c3_badtype = PINNConfig()
    c3_badtype.geometry_type = "Custom"
    c3_badtype.geom_custom_shapes_json = json.dumps([{"type": "Tetrahedron", "params": {}}])
    check(any("unrecognized" in e for e in c3_badtype.validate()),
          "an unrecognized 3D shape type should still fail validate()")

    # ── GUI: dimension-aware shape-type lists & row widgets ───────
    win6 = MainWindow()
    win6.radio_3d.setChecked(True)
    win6._on_dim_changed()
    win6.geometry_type_combo.setCurrentText("Custom")
    check(len(win6.custom_geom_shape_rows) == 1,
          "selecting Custom for the first time in 3D should auto-seed one shape row")
    r3_0 = win6.custom_geom_shape_rows[0]
    check(set(r3_0['type_combo'].itemText(i) for i in range(r3_0['type_combo'].count())) == {"Cuboid", "Sphere"},
          "a 3D shape row's type combo should only offer Cuboid/Sphere")
    check(r3_0['type_combo'].currentText() == "Cuboid",
          "the auto-seeded 3D row should default to Cuboid (the default shape_type isn't a 3D "
          "primitive, so the combo falls back to its first item)")

    r3_0['type_combo'].setCurrentText("Cuboid")
    r3_0['cuboid_xmin'].setValue(0.0); r3_0['cuboid_xmax'].setValue(2.0)
    r3_0['cuboid_ymin'].setValue(0.0); r3_0['cuboid_ymax'].setValue(2.0)
    r3_0['cuboid_zmin'].setValue(0.0); r3_0['cuboid_zmax'].setValue(2.0)
    win6._add_custom_geom_shape()
    r3_1 = win6.custom_geom_shape_rows[1]
    r3_1['type_combo'].setCurrentText("Sphere")
    r3_1['sphere_cx'].setValue(1.0); r3_1['sphere_cy'].setValue(1.0); r3_1['sphere_cz'].setValue(1.0)
    r3_1['sphere_r'].setValue(0.5)
    r3_1['op_combo'].setCurrentIndex(r3_1['op_combo'].findData("subtract"))

    built3_json = win6._build_custom_geom_shapes_json()
    parsed3 = json.loads(built3_json)
    check(parsed3 == CUBOID_MINUS_SPHERE,
          f"_build_custom_geom_shapes_json() mismatch for 3D: {parsed3} != {CUBOID_MINUS_SPHERE}")

    # Round trip through _apply_custom_geom_shapes_json
    win6._apply_custom_geom_shapes_json(json.dumps(CUBOID_MINUS_SPHERE))
    check(len(win6.custom_geom_shape_rows) == 2, "_apply_custom_geom_shapes_json should rebuild the 3D row list")
    check(json.loads(win6._build_custom_geom_shapes_json()) == CUBOID_MINUS_SPHERE,
          "round-trip through _apply/_build_custom_geom_shapes_json should be lossless for 3D shapes too")

    # Full _build_config / _apply_config round trip in 3D
    full3_config = win6._build_config()
    check(full3_config.geometry_type == "Custom", "_build_config should record geometry_type = Custom in 3D")
    check(json.loads(full3_config.geom_custom_shapes_json) == CUBOID_MINUS_SPHERE,
          "_build_config should serialize the 3D shape list")
    check(full3_config.validate() == [], f"the built 3D config should validate cleanly, got {full3_config.validate()}")

    win7 = MainWindow()
    win7.radio_3d.setChecked(True)
    win7._apply_config(full3_config)
    check(win7.geometry_type_combo.currentText() == "Custom", "_apply_config should restore geometry_type in 3D")
    check(json.loads(win7._build_custom_geom_shapes_json()) == CUBOID_MINUS_SPHERE,
          "_apply_config should restore the same 3D shape list on a fresh MainWindow")

    # ── Dimension-switch clearing ─────────────────────────────────
    win8 = MainWindow()
    win8.radio_2d.setChecked(True)
    win8._on_dim_changed()
    win8.geometry_type_combo.setCurrentText("Custom")
    check(len(win8.custom_geom_shape_rows) == 1, "2D Custom should auto-seed one (2D) shape row")
    win8.radio_3d.setChecked(True)
    win8._on_dim_changed()
    check(win8.custom_geom_shape_rows == [] or
          (len(win8.custom_geom_shape_rows) == 1 and
           win8.custom_geom_shape_rows[0]['type_combo'].currentText() in ("Cuboid", "Sphere")),
          "switching 2D -> 3D should clear stale 2D shape rows (then _on_geometry_type_changed "
          "re-seeds one 3D-appropriate row since Custom is still selected)")
    # And the re-seeded row (if any) only offers 3D primitives -- no leftover
    # Rectangle/Disk/etc. row could have survived the clear.
    for row in win8.custom_geom_shape_rows:
        check(set(row['type_combo'].itemText(i) for i in range(row['type_combo'].count())) == {"Cuboid", "Sphere"},
              "any row surviving a 2D->3D dimension switch must be a 3D-only row")

    # ── Domain preview: Custom-3D path (refactored builder) ───────
    win9 = MainWindow()
    win9.radio_3d.setChecked(True)
    win9._on_dim_changed()
    win9.geometry_type_combo.setCurrentText("Custom")
    win9._apply_custom_geom_shapes_json(json.dumps(CUBOID_MINUS_SPHERE))
    captured9 = {}
    win9._launch_preview_thread = lambda tmp: captured9.__setitem__('tmp', tmp)
    win9._preview_domain()
    check('tmp' in captured9, "_preview_domain() should reach script generation for a non-empty 3D Custom geometry")
    if 'tmp' in captured9:
        preview3_script = open(captured9['tmp']).read()
        os.unlink(captured9['tmp'])
        try:
            ast.parse(preview3_script)
        except SyntaxError as e:
            check(False, f"generated 3D Custom preview script is not valid Python: {e}")
        check("dde.geometry.CSGDifference(dde.geometry.Cuboid" in preview3_script,
              "3D Custom preview script should build the CSG-difference chain")
        check("'Custom  |" in preview3_script,
              "3D Custom preview script's title should use the geom_type_label parameter")
        check("_edges_1" in preview3_script, "3D Custom preview should draw the Cuboid leaf's box-edge outline")
        check("plot_wireframe" in preview3_script, "3D Custom preview should draw the Sphere leaf's wireframe outline")

    # Non-Custom 3D preview paths (plain Cuboid/Sphere) still work after the
    # _build_3d_preview_script / _render_3d_preview_script split.
    for gtype in ("Cuboid", "Sphere"):
        winN3 = MainWindow()
        winN3.radio_3d.setChecked(True)
        winN3._on_dim_changed()
        winN3.geometry_type_combo.setCurrentText(gtype)
        capturedN3 = {}
        winN3._launch_preview_thread = lambda tmp, c=capturedN3: c.__setitem__('tmp', tmp)
        winN3._preview_domain()
        check('tmp' in capturedN3, f"{gtype} (non-Custom) 3D preview should still reach script generation")
        if 'tmp' in capturedN3:
            scriptN3 = open(capturedN3['tmp']).read()
            os.unlink(capturedN3['tmp'])
            try:
                ast.parse(scriptN3)
            except SyntaxError as e:
                check(False, f"{gtype} 3D preview script is not valid Python: {e}")
            check(f"'{gtype}  |" in scriptN3, f"{gtype} 3D preview title should still use its own geom_type label")

    # Empty 3D Custom shape list aborts cleanly (no script generated)
    win10 = MainWindow()
    win10.radio_3d.setChecked(True)
    win10._on_dim_changed()
    win10.geometry_type_combo.setCurrentText("Custom")
    win10.custom_geom_shape_rows[0]['widget'].deleteLater()
    win10.custom_geom_shape_rows.clear()
    captured10 = {}
    win10._launch_preview_thread = lambda tmp: captured10.__setitem__('tmp', tmp)
    win10._preview_domain()
    check('tmp' not in captured10, "previewing an empty 3D Custom shape list should not launch a script")

    # ── Codegen: _build_custom_geom_code (3D leaf types) ──────────
    cuboid_only = PINNConfig()
    cuboid_only.geom_custom_shapes_json = json.dumps([{
        "type": "Cuboid",
        "params": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
    }])
    code, wrap = _build_custom_geom_code(cuboid_only)
    check(code == "dde.geometry.Cuboid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])", f"unexpected Cuboid leaf code: {code}")
    check(wrap is False, "a single Cuboid should never need _DTypeSafeGeom wrapping (matches Rectangle)")

    sphere_only = PINNConfig()
    sphere_only.geom_custom_shapes_json = json.dumps([
        {"type": "Sphere", "params": {"cx": 0.5, "cy": 0.5, "cz": 0.5, "r": 0.3}}
    ])
    code, wrap = _build_custom_geom_code(sphere_only)
    check(code == "dde.geometry.Sphere([0.5, 0.5, 0.5], 0.3)", f"unexpected Sphere leaf code: {code}")
    check(wrap is True, "a single Sphere (a known-buggy-dtype type) should need _DTypeSafeGeom wrapping")

    cuboid_minus_sphere_cfg = PINNConfig()
    cuboid_minus_sphere_cfg.geom_custom_shapes_json = json.dumps(CUBOID_MINUS_SPHERE)
    code, wrap = _build_custom_geom_code(cuboid_minus_sphere_cfg)
    check(code == ("dde.geometry.CSGDifference("
                    "dde.geometry.Cuboid([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), "
                    "dde.geometry.Sphere([1.0, 1.0, 1.0], 0.5))"), f"unexpected 3D CSG chain: {code}")
    check(wrap is False, "a 2-shape 3D CSG combination should never need _DTypeSafeGeom wrapping, "
                          "even though Sphere alone is a buggy type")

    empty_3d = PINNConfig()
    empty_3d.problem_dim = "3D"
    empty_3d.geom_custom_shapes_json = "[]"
    code, wrap = _build_custom_geom_code(empty_3d)
    check("dde.geometry.Cuboid" in code,
          "an empty shape list under problem_dim='3D' should fall back to a sane default Cuboid, not a 2D Rectangle")

    malformed_3d = PINNConfig()
    malformed_3d.problem_dim = "3D"
    malformed_3d.geom_custom_shapes_json = "not json"
    code, wrap = _build_custom_geom_code(malformed_3d)
    check("dde.geometry.Cuboid" in code, "malformed JSON under problem_dim='3D' should fall back to Cuboid, not crash")

    # ── Codegen: generate_script() (live Solve/Sweep path), 3D ────
    live3_config = PINNConfig()
    live3_config.problem_dim = "3D"
    live3_config.geometry_type = "Custom"
    live3_config.geom_custom_shapes_json = json.dumps(CUBOID_MINUS_SPHERE)
    script3 = generate_script(live3_config)
    try:
        ast.parse(script3)
    except SyntaxError as e:
        check(False, f"generate_script() output is not valid Python for 3D Custom geometry: {e}")
    check('elif _geom_type == "Custom":' in script3, "generate_script() should add a Custom branch in 3D too")
    check("dde.geometry.CSGDifference(dde.geometry.Cuboid" in script3,
          "generate_script() should embed the 3D CSG chain literally")
    check("_DTypeSafeGeom(dde.geometry.CSGDifference" not in script3,
          "the 2-shape 3D CSG case should not be wrapped in generate_script() either")

    live3_single = PINNConfig()
    live3_single.problem_dim = "3D"
    live3_single.geometry_type = "Custom"
    live3_single.geom_custom_shapes_json = json.dumps([
        {"type": "Sphere", "params": {"cx": 0.5, "cy": 0.5, "cz": 0.5, "r": 0.3}}
    ])
    script3_single = generate_script(live3_single)
    ast.parse(script3_single)
    check("_DTypeSafeGeom(dde.geometry.Sphere" in script3_single,
          "a single-Sphere 3D Custom geometry should be wrapped in generate_script() too")

    # ── Codegen: generate_clean_script() (Export Script path), 3D ─
    clean3_config = PINNConfig()
    clean3_config.problem_dim = "3D"
    clean3_config.geometry_type = "Custom"
    clean3_config.geom_custom_shapes_json = json.dumps(CUBOID_MINUS_SPHERE)
    clean_script3 = generate_clean_script(clean3_config)
    try:
        ast.parse(clean_script3)
    except SyntaxError as e:
        check(False, f"generate_clean_script() output is not valid Python for 3D Custom geometry: {e}")
    check("geom = dde.geometry.CSGDifference(dde.geometry.Cuboid" in clean_script3,
          "generate_clean_script() should emit the single literal 3D CSG-chain geom line")
    check("class _DTypeSafeGeom" not in clean_script3,
          "the export script shouldn't define _DTypeSafeGeom when the 2-shape 3D CSG case doesn't need it")

    clean3_single = PINNConfig()
    clean3_single.problem_dim = "3D"
    clean3_single.geometry_type = "Custom"
    clean3_single.geom_custom_shapes_json = json.dumps([
        {"type": "Sphere", "params": {"cx": 0.5, "cy": 0.5, "cz": 0.5, "r": 0.3}}
    ])
    clean_script3_single = generate_clean_script(clean3_single)
    ast.parse(clean_script3_single)
    check("class _DTypeSafeGeom" in clean_script3_single,
          "a single-Sphere 3D Custom geometry's export script should define _DTypeSafeGeom")
    check("geom = _DTypeSafeGeom(dde.geometry.Sphere" in clean_script3_single,
          "a single-Sphere 3D Custom geometry's export script should wrap the geom line")

    # Parity: both codegen paths should agree on the 3D CSG structure itself
    check("CSGDifference(dde.geometry.Cuboid([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), dde.geometry.Sphere([1.0, 1.0, 1.0], 0.5))"
          in script3 and
          "CSGDifference(dde.geometry.Cuboid([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), dde.geometry.Sphere([1.0, 1.0, 1.0], 0.5))"
          in clean_script3,
          "generate_script() and generate_clean_script() should build the identical 3D CSG chain")

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

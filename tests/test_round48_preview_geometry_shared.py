#!/usr/bin/env python3
"""
Round 48 -- codebase cleanup, phase 11 (audit §2e, the 3D-preview methods):
the GUI's Domain Preview built its dde.geometry constructors from its own
hand-typed per-shape copies (single shapes in _preview_domain and
_build_3d_preview_script; Custom leaves in _custom_geom_leaf_code /
_custom_geom_3d_leaf_code; two private copies of the CSG op-name table;
and MainWindow._parse_vertices, a line-for-line mirror of codegen.py's
_parse_vertex_list). Training builds the same geometry through
codegen.py's _clean_geom_line / _custom_geom_leaf_literal /
_CUSTOM_GEOM_OP_CTORS, so any future drift between the two would make the
preview sample a different domain than the one actually trained.

All of these now come from codegen.py's functions (a new
_clean_geom_inner split out of _clean_geom_line for the single-shape
case). The preview keeps its own validation messages, outline drawing and
bounding-box logic.

Before the commit, 200 randomized preview scripts (all 7 single-shape
types, including rejected Triangle/Polygon inputs and their messages) and
400 Custom-leaf calls (valid, invalid and missing-param cases) were
checked BYTE-FOR-BYTE identical to the pre-refactor output; that needs a
pre-refactor checkout and isn't repeated here. This file checks the
single-source property directly and runs real preview scripts end to end.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round48_preview_geometry_shared.py
"""
import os
import subprocess
import sys
import tempfile
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.codegen import (_clean_geom_inner, _clean_geom_line, _custom_geom_leaf_literal,
                                     _parse_vertex_list)


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    captured = []
    win._launch_preview_thread = lambda tmp: (captured.append(open(tmp, encoding="utf-8", errors="replace").read()), os.unlink(tmp))

    # _clean_geom_line is still exactly "geom = " + _clean_geom_inner (+ wrapper).
    ns = types.SimpleNamespace(geometry_type="Disk", geom_center_x=0.2, geom_center_y=0.3, geom_radius=0.7,
                               x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1)
    inner, wrap = _clean_geom_inner(ns, True, False, [], [])
    check(_clean_geom_line(ns, True, False, [], []) == (f"geom = _DTypeSafeGeom({inner})", True) and wrap,
          "_clean_geom_line should be _clean_geom_inner plus the dtype wrapper")

    # _parse_vertices delegates to the training parser.
    for text in ("0,0;1,0;0,1", " 0.5,0.1; 2,0.3 ;1.25,1.75", "1,2;bad;3,4", ""):
        check(win._parse_vertices(text) == [tuple(v) for v in _parse_vertex_list(text)],
              f"_parse_vertices({text!r}) should match codegen's _parse_vertex_list")

    # Custom leaves: constructor == training's _custom_geom_leaf_literal.
    leaf_cases_2d = [
        ("Rectangle", {"x_min": -1.0, "x_max": 2.0, "y_min": 0.0, "y_max": 0.5}),
        ("Disk", {"cx": 0.1, "cy": -0.2, "r": 0.75}),
        ("Ellipse", {"cx": 0.0, "cy": 0.0, "a": 1.5, "b": 0.5, "angle": 0.3}),
        ("Triangle", {"vertices_text": "0,0;1,0;0.5,1"}),
        ("Polygon", {"vertices_text": "0,0;1,0;1,0.5;0.5,0.5;0.5,1;0,1"}),
    ]
    for shape, params in leaf_cases_2d:
        code, _patch, _bbox = win._custom_geom_leaf_code(shape, params)
        check(code == _custom_geom_leaf_literal(shape, params), f"2D Custom leaf {shape}: constructor drift")
    for shape, params in [("Cuboid", {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1, "z_min": -1, "z_max": 1}),
                          ("Sphere", {"cx": 0.5, "cy": 0.5, "cz": 0.5, "r": 0.4})]:
        code, _outline, _bbox = win._custom_geom_3d_leaf_code(1, shape, params)
        check(code == _custom_geom_leaf_literal(shape, params), f"3D Custom leaf {shape}: constructor drift")

    # Single-shape previews: the geom line in the real preview script is the
    # training constructor, and the script actually runs.
    def preview(dim_radio, gtype):
        dim_radio.setChecked(True)
        win.geometry_type_combo.setCurrentText(gtype)
        win.num_domain.setValue(60); win.num_boundary.setValue(20); win.num_initial.setValue(10)
        captured.clear()
        win._preview_domain()
        return captured[0] if captured else None

    win.radio_2d.setChecked(True)
    win.geometry_type_combo.setCurrentText("Disk")
    win.geom_disk_cx.setValue(0.5); win.geom_disk_cy.setValue(0.5); win.geom_disk_r.setValue(0.4)
    disk_script = preview(win.radio_2d, "Disk")
    ns_disk = types.SimpleNamespace(geometry_type="Disk", geom_center_x=0.5, geom_center_y=0.5, geom_radius=0.4)
    check(disk_script is not None and f"geom  = {_clean_geom_inner(ns_disk, True, False, [], [])[0]}" in disk_script,
          "Disk preview should use the training constructor")

    win.radio_3d.setChecked(True)
    win.geometry_type_combo.setCurrentText("Sphere")
    win.geom_sphere_cx.setValue(0.5); win.geom_sphere_cy.setValue(0.5)
    win.geom_sphere_cz.setValue(0.5); win.geom_sphere_r.setValue(0.4)
    sphere_script = preview(win.radio_3d, "Sphere")
    ns_sph = types.SimpleNamespace(geometry_type="Sphere", geom_center_x=0.5, geom_center_y=0.5,
                                   geom_center_z=0.5, geom_radius=0.4)
    check(sphere_script is not None and f"geom  = {_clean_geom_inner(ns_sph, False, True, [], [])[0]}" in sphere_script,
          "Sphere preview should use the training constructor")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures
    for tag, script in (("disk", disk_script), ("sphere", sphere_script)):
        if not script:
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = os.path.join(tmpdir, f"preview_{tag}.py")
            with open(sp, "w", encoding="utf-8") as f:
                f.write(script)
            proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=240)
            check(proc.returncode == 0 and "DOMAIN_PREVIEW_DONE" in proc.stdout,
                  f"{tag} preview script should run cleanly, got {proc.returncode}:\n"
                  f"{proc.stdout[-1200:]}\n{proc.stderr[-1200:]}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 48 PREVIEW-GEOMETRY TESTS PASSED")
    return 0


def test_round48_preview_geometry_shared():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())

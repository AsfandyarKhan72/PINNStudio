<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_domain_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Geometry, Collocation points, Initial and Boundary conditions pages
- `MainWindow._on_geometry_type_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the shape and its parameters
- `MainWindow._build_custom_geom_shapes_json` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Custom geometry rows (combined shapes)

**Helpers**

- `_build_custom_geom_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): combined shapes to DeepXDE geometry code
- `_clean_geom_inner` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the geometry constructor in the exported script
- `_shape_bbox` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): bounding box of a non-box shape
- [`pinnstudio/ui/sketch_canvas.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_canvas.py): 2D Sketch
- [`pinnstudio/ui/sketch_edges.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_edges.py): edges of the sketch as boundary parts
- [`pinnstudio/ui/sketch_import.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_import.py): DXF/SVG import
- [`pinnstudio/ui/sketch3d.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch3d.py): 3D Sketch

**Tests**

- [`tests/test_custom_geometry.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_custom_geometry.py): Custom geometry: validation, panel, generated code
- [`tests/test_geometry_box.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_geometry_box.py): bounding box of non-box shapes
- [`tests/test_sketch_geometry.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_sketch_geometry.py): sketched shapes become Custom-geometry rows

Files on the `main` branch of the repository; each link names the function or class to look for in the file.

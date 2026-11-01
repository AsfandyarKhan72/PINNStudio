# Geometry and boundary conditions

This page shows where geometry types and boundary-condition rows are implemented, for anyone adding a shape,
a boundary-condition type or a way to select part of the boundary. The user-facing description is in
[Geometry](../reference/geometry.md) and [Boundary conditions](../reference/boundary-conditions.md).

## Geometry types

| Dimension | `geometry_type` | DeepXDE class |
|---|---|---|
| 1D, ODE | `Interval` | `dde.geometry.Interval` |
| 2D | `Rectangle`, `Disk`, `Ellipse`, `Triangle`, `Polygon` | `Rectangle`, `Disk`, `Ellipse`, `Triangle`, `Polygon` |
| 3D | `Cuboid`, `Sphere` | `Cuboid`, `Sphere` |
| 2D, 3D | `Custom` (also what the Sketch produces) | `CSGUnion` / `CSGDifference` / `CSGIntersection` of the shapes above |

A time-dependent problem wraps the geometry in `dde.geometry.GeometryXTime(geom, TimeDomain(t_min, t_max))`; a
steady problem uses the geometry itself.

### Where each part lives

- **Configuration** (`pinnstudio/core/config.py`): `geometry_type`; the box limits `x_min` ... `z_max`; the shape
  parameters `geom_center_x/y/z`, `geom_radius`, `geom_semi_major`, `geom_semi_minor`, `geom_angle`,
  `geom_triangle_vertices`, `geom_polygon_vertices`; and for `Custom` the ordered shape list
  `geom_custom_shapes_json` (`[{"type": ..., "op": "union"|"subtract"|"intersect", "params": {...}}, ...]`, the
  first entry being the starting region) with `geom_sketch` recording whether the list was drawn on the Sketch.
- **Window** (`pinnstudio/ui/main_window.py`): `GEOM_TYPES_2D`, `GEOM_TYPES_3D`, `CUSTOM_GEOM_SHAPE_TYPES_*`,
  `CUSTOM_GEOM_OPS`; the shape rows of the Custom builder; `_geometry_supported_for_training()`, the gate that
  refuses a type without code-generation support. The Sketch (`ui/sketch_canvas.py`, `ui/sketch3d.py`,
  `ui/sketch_import.py`) edits the same shape list.
- **Solve's script** (`generate_script`): the generated `_build_geom()` function constructs every type; a
  `Custom` list is turned into one nested constructor expression by `_build_custom_geom_code()` with
  `_custom_geom_leaf_literal()` for each primitive.
- **Exported script** (`generate_clean_script`): `_clean_geom_line()` / `_clean_geom_inner()` write the same
  constructor.
- **Shared fixes** in both scripts: `_DTypeSafeGeom` casts the points of geometries that return float64 to the
  configured precision; `_ellipse_box_fix()` corrects the ellipse's bounding box for a rotated ellipse.
- **Bounding box**: `_shape_bbox()` computes a non-box shape's bounding box; `_effective_config()` writes it
  into the domain fields, which the named sides, plot areas and Error Analysis grids use.

A new primitive needs a constructor in `_custom_geom_leaf_literal`, `_build_geom` and `_clean_geom_inner`, a
bounding box in `_shape_bbox`, an entry in the window's type lists and shape rows (and, if it can be drawn, in the
Sketch and in `ui/sketch_edges.py`, which finds the boundary pieces), and tests (see below). Prefer DeepXDE's own
geometry classes: PINNStudio does no geometry computation of its own for training, apart from the boundary
edge expressions of the Sketch.

## Boundary-condition rows

The **Boundary conditions** page is a list of rows, stored as `custom_bc_json`. Each row is a dictionary built by
`MainWindow._build_custom_bc_json()`:

| Key | Meaning |
|---|---|
| `type` | `dirichlet`, `neumann`, `robin`, `periodic`, `pointset`, `pointset_operator`, `operator`, `interface2d` |
| `component` / `components` | the output (or outputs) the condition applies to |
| `location` | the **Where:** expression selecting boundary points (named sides `left`, `right`, `bottom`, `top`, `front`, `back`, or an expression in `x`, `y`, `z`, `t`) |
| `value` | the condition's value or operator expression |
| `axis`, `deriv_order` | periodic conditions |
| `apply_at` | `boundary` or `initial` (an operator condition at the initial time) |
| `points_file`, `points_mode`, `points_list`, `points_count`, `points_where`, `values_mode`, `values_formula` | point-set conditions: where the points come from and their values |
| `batch_size`, `shuffle` | point-set batching (DeepXDE resamples the batch every 100 steps; a point resampler is added automatically) |
| `location2`, `direction` | the second edge and direction of an interface condition |

The window's labels for these types are in `CUSTOM_BC_TYPES`; which fields a type shows is decided by the
`_BC_NEEDS_*` sets.

In the generated scripts:

- **Solve's script**: one loop over the rows dispatches on `type` to `dde.icbc.DirichletBC`, `NeumannBC`,
  `RobinBC`, `PeriodicBC`, `PointSetBC`, `PointSetOperatorBC`, `OperatorBC` or `Interface2DBC`, and raises a
  `ValueError` for an unknown type. The same dispatch is repeated for every time window inside the time-window
  loop (`_gs_time_adaptive_loop_code`). Locations and values become functions through `_bc_loc_fn`,
  `_bc_init_loc_fn`, `_bc_val_fn`, `_bc_robin_fn` and `_bc_operator_fn`; the named sides come from
  `BC_SIDE_NAMES` and the domain limits.
- **Exported script**: `_clean_bc_row_code()` writes one plain DeepXDE call per row.
- **Preconditions** are checked at generation time (`_bc_check_periodic_preconditions`,
  `_bc_check_interface2d_preconditions`, `_bc_check_initial_operator_preconditions`) and in
  `PINNConfig.validate()` (a row whose **Where:** selects nothing).
- **Sketch edges**: `ui/sketch_edges.py` (2D) and `ui/sketch3d.py` (3D faces) number the pieces of the boundary
  and write for each one a **Where:** expression in `x`, `y` (`z`) and `np` that is true on that piece only, so the
  same expression works in both scripts.

The older per-side fields (`bc_left_types`, `bc_left_values`, ...) and the per-shape groups (`bc_boundary_json`,
`bc_edge_json`) are read only for problems saved before the row list existed.

A new boundary-condition type needs: an entry in `CUSTOM_BC_TYPES` and the `_BC_NEEDS_*` sets, its fields in
`_build_custom_bc_json` / `_apply_custom_bc_json`, a branch in both dispatch loops of Solve's script (the whole
interval and the time windows) and in `_clean_bc_row_code`, its loss-weight label (`_bc_weight_desc`), and tests.

## Tests

`tests/test_custom_geometry.py`, `test_geometry_box.py`, `test_poisson_3d_cutout.py`, `test_sketch_geometry.py`,
`test_sketch_edges.py`, `test_sketch_3d.py`, `test_sketch_import.py`, `test_line_slice_and_sketch_bcs.py`,
`test_round68_bc_where_names.py` and `test_restore_geometry_masking.py` cover geometry construction, boundary
selection and plotting on non-box domains. A new shape or condition should be added to them, and checked in
the window's **Geometry** view.

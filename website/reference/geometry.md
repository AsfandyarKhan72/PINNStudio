---
search:
  boost: 2
---

# Geometry

## What it does

The **Geometry** page sets the spatial domain and, for a time-dependent problem, the time interval. The
generated script builds the matching DeepXDE geometry (`dde.geometry.Interval`, `Rectangle`, `Cuboid`,
`Disk`, `Ellipse`, `Sphere`, `Triangle`, `Polygon`) and, for combined shapes, DeepXDE's constructive
solid geometry (`CSGUnion`, `CSGDifference`, `CSGIntersection`). A time-dependent problem uses
`dde.geometry.GeometryXTime(geom, TimeDomain(t_min, t_max))`.

## When to use it

For every problem. Pick the simplest shape that is the domain; build anything else from combinations or
draw it on the **Sketch** tab.

## How to enable it

**Geometry** page → **Shape** box → **Shape:**. The choices depend on the dimension set on the
**Problem** page:

| Dimension | **Shape:** choices |
|---|---|
| ODE | none (the variable is \( t \), set in the **Time interval** box) |
| 1D | none: the domain is the interval **x:** … **to** … |
| 2D | Rectangle, Disk, Ellipse, Triangle, Polygon, Custom, Sketch |
| 3D | Cuboid, Sphere, Custom, Sketch |

The middle box is named after the shape (**Rectangle dimensions**, **Shapes**, **Interval**, …); the
**Time interval** box holds **t:** … **to** ….

## Settings

| Shape | Labels | Default | Config fields |
|---|---|---|---|
| Interval, Rectangle, Cuboid | **x:**, **y:**, **z:** … **to** … | 0 to 1 on every axis | `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max` |
| Disk | **Center:** (x, y), **Radius:** | (0.5, 0.5), 0.5 | `geom_center_x`, `geom_center_y`, `geom_radius` |
| Ellipse | **Center:**, **Semi-axes:** (semi-major, semi-minor), **Angle (rad):** | (0.5, 0.5), 0.5 and 0.3, 0 | `geom_center_x`, `geom_center_y`, `geom_semi_major`, `geom_semi_minor`, `geom_angle` |
| Triangle | **Vertices:** three `x,y` pairs separated by `;` | `0,0;1,0;0,1` | `geom_triangle_vertices` |
| Polygon | **Vertices:** three or more `x,y` pairs in order around the boundary | `0,0;1,0;1,0.6;0.5,1;0,0.6` | `geom_polygon_vertices` |
| Sphere | **Center:** (x, y, z), **Radius:** | (0.5, 0.5, 0.5), 0.5 | `geom_center_x`, `geom_center_y`, `geom_center_z`, `geom_radius` |
| Custom | a list of shapes (**+ Add Shape**), each with **Combine:** and **Shape:** and that shape's own numbers | one Rectangle | `geom_custom_shapes_json`, `geometry_type = "Custom"` |
| Sketch | drawn on the **Sketch** tab | — | `geom_custom_shapes_json`, `geom_sketch = true` |
| (time) | **t:** … **to** … | 0 to 1 | `t_min`, `t_max` |

Domain limits accept any value from −10⁶ to 10⁶ and keep 10 decimals (so \( [0, \pi/2] \) is stored to
full precision). Radii and semi-axes must be positive. The ellipse angle is the rotation of the major
axis from the x axis in radians; a positive angle turns it clockwise (DeepXDE's convention).

### Custom: combined shapes

Shapes are combined top to bottom: the first shape is the starting region, and each later one is combined
with the result so far by its **Combine:** choice, **Union (add)**, **Subtract (cut hole)** or
**Intersect** (DeepXDE's `CSGUnion`, `CSGDifference`, `CSGIntersection`). Order matters: a triangle with a
circular hole is *Triangle*, then *Disk* with **Subtract (cut hole)**. In 2D the shapes are Rectangle,
Disk, Ellipse, Triangle and Polygon; in 3D Cuboid and Sphere. The ▲ ▼ buttons reorder shapes.

### Sketch

**Shape:** → **Sketch** opens the **Sketch** tab, which draws the same shape list as Custom (it trains
with the same code). The tools are **Select**, **Rectangle**, **Circle**, **Ellipse**, **Polygon** and
**Edges → BCs** (3D: **Box**, **Sphere**, **Faces → BCs**). **New shape:** chooses how the next shape is
combined (**Union (add)**, **Subtract (cut hole)**, **Intersect (keep overlap)**); **Grid:** (default
0.1) with **Snap to grid** sets the drawing grid; **Type:** takes exact numbers; **Import DXF/SVG…** reads
the closed outlines of a 2D drawing (outlines inside others become holes). In 3D, **View:** switches
between flat slices (**Slice at z =** …) and a rotatable 3D view, and **New box: z from … to …** sets a
new box's depth. With **Edges → BCs**, **＋ New BC on selected** adds a Boundary Conditions row whose
**Where:** picks the selected edges; **Add to**, **Remove from BC** and **Select edges without a BC**
edit those rows.

## Formulation

The geometry provides the sampling (`random_points`, `random_boundary_points`, or the chosen
**Point distribution:**), the boundary test `on_boundary`, and the outward normal used by Neumann and
Robin conditions. For a shape that is not a box (Disk, Ellipse, Triangle, Polygon, Sphere, Custom), the
scripts use the shape's own bounding box for the side names (`left`, `right`, …), the plot area and the
Error Analysis grid.

## Defaults and limitations

- A Polygon whose four vertices form an axis-aligned rectangle is rejected by DeepXDE ("The polygon is a
  rectangle. Use Rectangle instead."); use **Rectangle**.
- For time windows on a non-box shape, only the grid points inside the domain carry the solution from one
  window to the next (see [Time windows](time-windows.md)).
- Sampling on some DeepXDE shapes returns float64 arrays in a float32 run; the scripts wrap these shapes
  so every sampled array has the run's precision.

!!! info "New in v1.6.0 (in development)"
    Time windows on a disk, polygon or custom shape use only the grid points inside the domain (earlier
    the whole bounding box was used, so points outside the domain were trained on).

## Supported problem types

Every shape for forward and inverse, stationary and time-dependent problems of its dimension.

## Compatibility with other features

- **Periodic BC** needs an Interval, Rectangle or Cuboid:
  "Periodic BC requires an Interval (1D), Rectangle (2D), or Cuboid (3D) geometry (current geometry: …);
  DeepXDE doesn't implement periodic boundary matching for other geometry shapes." This includes
  [exact periodicity](exact-constraints.md).
- **Interface 2D BC** needs a 2D Rectangle or Polygon and two outputs:
  "Interface 2D BC requires a Rectangle or Polygon geometry (current geometry: …)."
- An empty Custom list is refused: "Geometry Type is 'Custom' but no shapes have been added; add at least
  one shape in the Custom geometry builder."
- A boundary-condition row whose last edge was removed in the sketch is refused: "BC *n* picks no part of
  the boundary (its last edge was taken out): pick its edges in the Geometry view, or delete the BC."

## Common mistakes and troubleshooting

- **A side name on a curved boundary**: `left` on a disk means \( x = \) the bounding box's \( x_{\min} \),
  which only one point of the circle touches; the condition then selects no training point and its loss is
  NaN (reported before training). Use **Whole boundary** or a custom expression such as
  `np.isclose(x**2 + y**2, 0.25)`.
- **Points outside the domain in a plot**: plots of non-box shapes draw only the inside of the domain; the
  rest of the bounding box is left blank.

## Example

**Poisson 2D – disk** (Disk), **Poisson 2D – L-shape** (Polygon), **Helmholtz 2D – square with hole**
(Custom: Rectangle − Disk), **Poisson 3D – sphere** (Sphere) and
**Poisson 3D – cube with spherical cut-out** (Custom: Cuboid − Sphere) in **Examples**. See
[Tutorial 8: Custom geometry](../tutorials/08-geometry.md) and the
[Geometry guide page](../guide/geometry.md).

## Related

- **Tutorial:** [8. Custom geometry](../tutorials/08-geometry.md)
- **Examples:** [Poisson 2D – disk](../examples/2d/poisson-2d-disk.md), [Poisson 2D – L-shape](../examples/2d/poisson-2d-l-shape.md), [Helmholtz 2D – square with hole](../examples/2d/helmholtz-2d-square-with-hole.md), [Poisson 3D – sphere](../examples/3d/poisson-3d-sphere.md), [Poisson 3D – cube with spherical cut-out](../examples/3d/poisson-3d-cube-with-spherical-cut-out.md)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/geometry.md"

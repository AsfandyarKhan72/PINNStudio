# Collocation points

A PINN is trained at a finite set of points. The **Collocation points** page sets how many points are
sampled and how they are spread:

- **domain points** inside the domain (and over the time interval), where the equations are checked;
- **boundary points** on the boundary, where the boundary conditions are checked;
- **initial points** at the initial time, where the initial conditions are checked;
- **test points**, which are not trained on: they give the *test loss*, a check of how well the network
  fits between the training points.

![The Collocation points page of the 1D Burgers example: Domain points 10000, Boundary points 50, Initial points 50, Test points 1000, Point distribution Hammersley, Extra training points None](../assets/screenshots/burgers_points.png){ .screenshot loading=lazy }

Background: [Geometry](../reference/geometry.md), [RAR](../reference/rar.md).

## Choose the numbers

More points resolve sharp features better but make every training step slower. The examples' numbers
are a good starting point for similar problems. A few points to keep in mind:

- **Initial points** only matter for a time-dependent problem with initial conditions as loss terms; a
  stationary problem has none.
- **Boundary points** are shared by all boundary-condition rows; each row uses the boundary points its
  **Where** picks. A row that picks no point at all is reported before training (see
  [Boundary conditions](boundary-conditions.md#check-that-each-row-picks-points)).
- Points of a **Point Set** boundary condition, the measured points of an inverse problem and the
  **Extra training points** below come on top of these numbers.

## Choose how the points are spread

**Point distribution** chooses DeepXDE's sampler: **Hammersley** (the default), **Sobol**, **Halton**
and **LHS** spread points evenly without a grid; **uniform** is a regular grid; **pseudo** is plain
random sampling. An **Interface 2D BC** row needs **uniform** and switches to it by itself.

## Add points where the solution is hard to fit

**Extra training points** adds points that are always part of training, on top of the sampled ones:

- **From a file** — a plain numeric text file with one column per input (`x[, y[, z]][, t]`),
  separated by commas, semicolons or spaces (header, comment, missing-value and wrong-width rows are skipped
  with a note in the log), for example points clustered near a front;
- **Random in a box** — **Count:** random points within **Ranges:**, given as `lo,hi` per input
  separated by `;` (blank means the whole domain), for example `0.4,0.6; 0,1` for a strip around
  `x = 0.5` in a 1D time-dependent problem.

To move points to where the residual is large automatically, use **RAR** on the
[Adaptive training](adaptive.md) page; to draw new random points during training, use
**Point Resampling** under Callbacks on the [Training](training.md#callbacks) page.

## Look at the points (2D and 3D)

In a 2D or 3D problem, tick **View domain & point distribution** to draw a sample of the points with the
current numbers and distribution in the two Overview panels: the domain in space, and the points over
time. The picture is redrawn when you change the numbers or the distribution.

## Settings on this page

| GUI label | Meaning | Default (new problem) | Range | Saved as |
|---|---|---|---|---|
| **Domain points:** | Points inside the domain | 2000 | 1 – 10,000,000 | `num_domain` |
| **Boundary points:** | Points on the boundary | 200 | 0 – 2,000,000 | `num_boundary` |
| **Initial points:** | Points at the initial time | 200 | 0 – 2,000,000 | `num_initial` |
| **Test points:** | Points for the test loss | 1000 | 10 – 10,000,000 | `num_test` |
| **Point distribution:** | Hammersley, uniform, Halton, LHS, Sobol or pseudo | Hammersley | | `point_distribution` |
| **Extra training points:** | None, From a file, Random in a box | None | | `anchors_mode` (`none`, `file`, `box`) |
| (file box) | Points file | empty | | `anchors_file` |
| **Count:** | Number of random extra points | 1000 | 1 – 1,000,000 | `anchors_count` |
| **Ranges:** | `lo,hi` per input, `;`-separated | blank (whole domain) | | `anchors_box` |
| **View domain & point distribution** | Draw the sampled points (2D/3D) | off | | not saved |

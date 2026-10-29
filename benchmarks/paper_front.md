## Section 3.3 — geometry and boundary conditions (Figure 3)

| | |
|---|---|
| **What it shows** | how a domain is built from primitives and how boundary conditions are assigned to parts of its boundary, on two of the configured examples: a 2D domain with a hole and a 3D domain with a spherical cut-out |
| **How it was made** | each example loaded from the Examples list as it ships, one boundary condition selected in the model tree (its edges or faces are highlighted and its settings shown), the PINNStudio window drawn at twice the screen resolution. To keep the text readable at page width, the window has PINNStudio's smallest size (1100 × 750 screen pixels), the model tree and the empty training log are collapsed with their splitters, as a user can, and the menu bar, the general help line under the graphics and the status bar are cropped off; nothing in the window is changed or drawn over. At 6.5 in wide the settings text prints at about 6 pt. (a) above (b) |
| **For users** | [README → Geometry](../README.md#problem-setup), the examples [Helmholtz 2D – sound-hard scattering](../README.md#dx-helmholtz-sound-hard-scattering-absorbing-bc) and [Poisson 3D – cube with spherical cut-out](README.md#bench-poisson-3d-cube-with-spherical-cut-out) |
| **Data and scripts** | `tools/make_geometry_figure.py` (writes `figures/paper_geometry.png` and `.pdf`); the examples are defined in `pinnstudio/core/deepxde_examples.py` and `pinnstudio/core/pinnstudio_examples.py` |
| **Status** | done; the 3D example trained on the GPU machine: relative L₂ error 3.7 × 10⁻⁴ with its own settings and 1.9 × 10⁻⁴ with the Tuned ones ([benchmark](README.md#bench-poisson-3d-cube-with-spherical-cut-out)) |

<details>
<summary>📄 <em>Figure 3 for the paper — geometry and boundary selection</em></summary>

![Figure 3: geometry and boundary selection in PINNStudio](figures/paper_geometry.png)

> **Figure 3.** Geometry definition and boundary selection in PINNStudio, shown for two of the configured examples. (a) Helmholtz 2D – sound-hard scattering: a square with a circular hole, built by subtracting a disk from a rectangle. The selected boundary condition (BC 3, a Robin condition on $u_{re}$) is assigned to the four outer edges E1–E4 (highlighted) through the named sides *left*, *right*, *bottom* and *top*; the Neumann conditions on the hole (E5) use a coordinate expression. (b) Poisson 3D – cube with spherical cut-out: a cube with a sphere subtracted, shown in the interactive 3D view. The selected boundary condition (BC 2, a Neumann condition) is assigned to the spherical surface (face F7, orange) through a coordinate expression, and BC 1 (Dirichlet) to the faces of the cube (blue). In both panels the settings of the selected condition (its type, the boundary selection under *Where* and its value) are shown on the left, and every edge or face takes the colour of its condition.

**The two examples.**

- *(a) Helmholtz 2D – sound-hard scattering* (DeepXDE's example `Helmholtz_Sound_hard_ABC_2d.py`): the scattered field of a plane wave around a sound-hard disk, with two outputs, the real and imaginary parts $u_{re}$ and $u_{im}$. Domain $[-\pi, \pi]^2$ minus the disk of radius $\pi/4$. Four boundary conditions, one per field and boundary part: Neumann on the hole for each part (BC 1, 2) and the first-order absorbing (Robin) condition on the outer square (BC 3, 4). It shows conditions set per solution field and both ways of choosing a boundary part, a named selection and a coordinate expression.
- *(b) Poisson 3D – cube with spherical cut-out* (a PINNStudio template): $-\nabla^2 u = (6 - 4r^2)\,e^{-r^2}$, $r^2 = x^2 + y^2 + z^2$, in the cube $[-1, 1]^3$ minus the sphere of radius 0.7 centred at $(0, 0, 1)$, with the exact solution $u = e^{-r^2}$. BC 1 sets $u$ on the cube's faces (Dirichlet, named sides) and BC 2 its normal derivative on the spherical surface (Neumann, a coordinate expression); Error Analysis compares against the exact solution.

**What to write in the paper (Section 3.3)** — a suggested change to the last sentence of the first paragraph and to the caption, so both say that the panels are configured examples:

> Figure 3 illustrates these tools for two of the configured examples: a two-dimensional domain with a circular hole and a three-dimensional domain with a spherical cut-out, each built from geometric primitives and shown with one of its boundary conditions selected.

</details>

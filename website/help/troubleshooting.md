# Troubleshooting

This page lists problems users run into, with what PINNStudio reports and what to do about it. The messages are quoted
as PINNStudio prints them. If your problem is not here, or a remedy does not help, please
[report it](../getting-started/reporting-bugs.md).

For installation problems (commands not found, Python missing, the installer cannot use the GPU), see also
[Common installation problems](../getting-started/installation.md#common-installation-problems).

## PINNStudio does not start

Start PINNStudio from a terminal (`pinnstudio`, or `./venv/bin/pinnstudio` / `.\venv\Scripts\pinnstudio.exe` for a
source installation) to see the error. When it is started from its icon on Windows there is no console, and the
output goes to `~/.pinnstudio/pinnstudio.log` in your home folder.

**Linux: "Could not load the Qt platform plugin "xcb"".** Qt 6 needs a few system libraries for the X11 platform
plugin that minimal installations may not have. PINNStudio's own Linux tests install these packages first (Ubuntu
names):

```bash
sudo apt-get install libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 \
  libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
  libxcb-xfixes0 libfontconfig1 libdbus-1-3
```

`libxcb-cursor0` is the one most often missing. On a computer without a display (a server or a cluster node) the
window cannot open; train there with an [exported script](../getting-started/save-and-export.md#exporting-the-script)
instead.

**Windows: `ImportError: DLL load failed` for PyQt6.** Newer PyQt6/Qt builds fail to load on some Windows versions.
PINNStudio therefore requires `PyQt6==6.6.1` and `PyQt6-Qt6==6.6.1` on Windows. If another package or a manual
upgrade replaced them, put the pinned versions back in the Python PINNStudio runs in:

```bash
pip install "PyQt6==6.6.1" "PyQt6-Qt6==6.6.1"
```

**Windows: `[WinError 1114] A dynamic link library (DLL) initialization routine failed` (`c10.dll`).** PyTorch fails
to load on Windows if PyQt6 is loaded first, because both bring a Microsoft C++ runtime. PINNStudio loads PyTorch
first when it starts (`pinnstudio` command, `python -m pinnstudio`, the icon). The error can still appear if you
import PINNStudio's window from your own script after PyQt6; import `torch` before PyQt6 there.

**The window opened, but disappeared or shows a message box "PINNStudio error".** An unexpected error is shown in a
message box and written to `~/.pinnstudio/pinnstudio.log`; PINNStudio keeps running. Please report it with the text
of the message ([Reporting a problem](../getting-started/reporting-bugs.md)).

**The layout is unusable** (panels collapsed or off-screen after a change of screen). **View › Reset Layout**
restores the default layout. **Settings › Display Settings...** has the theme and the UI scale.

**Typing `pinnstudio` starts an old version.** Another installation comes first on the PATH; see
[Common installation problems](../getting-started/installation.md#common-installation-problems).

## Missing or incompatible packages

PINNStudio needs DeepXDE (1.10.0 or newer), PyTorch (2.0 or newer), PyQt6, NumPy below 2.5, SciPy, Matplotlib and
pandas. The installers and `pip install pinnstudio` install them. Problems appear when packages are installed or
upgraded by hand afterwards:

- **`ModuleNotFoundError: No module named ...`** in the Training Log or the terminal: the package is missing from the
  Python PINNStudio runs in. The generated scripts run with the same Python as the window, so install the package
  there (`python -m pip install <package>` with that Python), or reinstall PINNStudio.
- **NumPy 2.5 or newer**: DeepXDE 1.15's 2D `Polygon`/`Triangle` geometry calls `np.cross` on two-component
  vectors, which NumPy 2.5 no longer allows (it raises `ValueError`). PINNStudio requires `numpy<2.5` for this
  reason; if a newer NumPy got installed, install an older one: `pip install "numpy<2.5"`.
- **Intel Mac**: PyTorch 2.2 (the last for Intel Macs) cannot use NumPy 2 ("Failed to initialize NumPy"). PINNStudio
  requires `numpy<2` there.
- **The NNCG optimizer** in a training phase needs DeepXDE 1.13.0 or newer. With an older DeepXDE, **Solve** stops
  with `⚠️ A Training Phase is set to use NNCG, which requires deepxde>=1.13.0 (you have ... installed).`; upgrade
  DeepXDE or choose Adam or L-BFGS for that phase.

## PyTorch does not use the GPU

Check first what PyTorch finds: **Settings › GPU / Hardware...** says either `Detected 1 GPU(s) -- 0: <name>` or
`No CUDA GPU detected -- training will run on CPU regardless of these settings.` The first lines of a run's Training
Log say the same (`✅ GPU: <name>`, or `⚠️ No GPU found — running on CPU`).

If you have an NVIDIA GPU but PyTorch does not see it:

1. Check that **Train on** in **Settings › GPU / Hardware...** is **GPU when there is one**, not **CPU**.
2. Check that the NVIDIA driver works: `nvidia-smi` must print the GPU and a *CUDA Version*.
3. Look at the terminal PINNStudio was started from. When `nvidia-smi` works but PyTorch cannot use the GPU, PINNStudio
   prints `An NVIDIA GPU was detected, but PyTorch cannot use it (PINNStudio runs on the CPU for now).`, followed by
   the commands that fix it for your installation: run the one-command installer again (installer copies), or
   install the PyTorch build for your driver with `pip` (see
   [CPU-only or NVIDIA GPU PyTorch](../getting-started/installation.md#cpu-only-or-nvidia-gpu-pytorch)). If it says
   `The NVIDIA driver is too old for the PyTorch builds available now.`, update the driver first.
4. With several GPUs, **GPU device index** in the same dialog chooses which one is used.

On a Mac, PINNStudio trains on the CPU on purpose (`[Info] Training on the CPU (Apple GPU / MPS not used)`).
PINNStudio's installers and GPU checks cover NVIDIA GPUs (CUDA) only.

## The GPU runs out of memory

A run that fails with `CUDA out of memory` (`OutOfMemoryError`) ends with the tip
`💡 Tip: the GPU ran out of memory. Fewer training points, a smaller network, a smaller Error Analysis/plot grid, or
Train on: CPU (slower, but uses the computer's memory) can help.`

What uses GPU memory, roughly in order:

- **Collocation points**: **Domain points**, **Boundary points**, **Initial points** and **Test points** on the
  **Collocation points** page, plus points added during training (RAR, resampling, extra training points).
  Second and higher derivatives in the equations multiply the memory per point.
- **The network**: hidden layers and neurons per layer (**Neural network**).
- **Evaluation after training**: the grid of the plots (**Settings › Plot Settings...**) and of the error analysis.
  A run that trains but fails while plotting needs a coarser grid, not fewer training points.

**Settings › GPU / Hardware...** › **Max GPU memory to reserve (%)** (95 % by default) caps the share of the GPU's
memory PyTorch may use. Lower it to leave memory for other programs on the same GPU; it does not make a run need less
memory. If nothing helps, **Train on: CPU** uses the computer's main memory instead (slower).

!!! info "New in v1.6.0 (in development)"
    In v1.5.0 a run that ran out of GPU memory ended with the tip to check the PDE syntax. v1.6.0 recognizes the
    out-of-memory error and prints the tip above.

## Solve does not start: settings to fix

Before every run, PINNStudio checks the settings. While you edit, the model tree marks the pages with a setting that
would stop **Solve**, and the toolbar shows a button such as **2 settings to fix**; clicking it goes to the first one.
When you press **Solve** anyway, the Training Log lists every problem after
`❌ Cannot start -- this configuration has invalid settings:`.

### Equations, output names and constants

On the **Equations** page, each equation is typeset as you type it. A name PINNStudio does not recognize is shown as
`Unknown name: k. Add it as a constant, or check the spelling.`, and an expression that cannot be read yet as
`This isn't a complete expression yet.` Common causes:

- a missing `*`: write `5*u`, not `5u`, and `u*du_x`, not `u du_x`;
- a coefficient that is neither a number nor a named constant (add it under the constants, or in an inverse problem
  as an unknown parameter);
- a derivative not written in the form `d<output>_<variables>` (for example `du_xx` or `dv_t`), or one that
  PINNStudio does not build, such as `du_xxy`. The second is refused with
  `Equations: du_xxy is not available (see Derivative syntax on the Equations page) -- write the equation with the
  derivatives PINNStudio provides, e.g. a third derivative as a new output.` (new in v1.6.0; see
  [Equations, outputs and constants](../guide/equations.md#write-the-equations)).

Output names must be usable in the equations. These messages come from the check:

- `Output name 'my u' can't be used in the equations: use letters, digits and _ (not starting with a digit), and not
  a Python word such as lambda.`
- `Output name 'y' is already used for something else (a coordinate, the array y the outputs are read from, a module,
  a math function or a side name) -- pick another name (e.g. u1, u2).`
- `Output name 'u' is used for two outputs -- give each output its own name.`
- For an ODE: `ODE problems use t as the variable (and x internally), so an output can't be named 'x' -- rename it
  (e.g. y1, y2).`
- For inverse problems: `Trainable variable "my var" can't be used as a name in the equations: ...`

!!! info "New in v1.6.0 (in development)"
    The output-name check is new. In v1.5.0 an output called `y` (natural in an ODE) silently trained a different
    problem in **Solve** and crashed the exported script; names such as `np`, `sin`, `lambda` or a name with a
    space crashed the scripts.

If a run fails with a Python error from the equations (`SyntaxError`, `NameError`), the run ends with
`❌ Error during training. Check log above.` and the tip `💡 Tip: Check PDE/IC syntax — use * for multiplication
(e.g. 5*u not 5u)`. The lines above it in the log name the expression.

### Incompatible training-method combinations

Some methods cannot be used together. The check refuses them with an explanation, for example:

| Combination | Message (shortened) |
|---|---|
| Causal training on a steady problem | `Causal training needs a time-dependent problem (it weights the residual by time).` |
| Causal training on an ODE | `Causal training isn't available for an ODE ...; turn it off.` |
| Per-point weights (RBA, SA-PINN) with the Point Resampler callback | `Per-point weights (RBA) belong to fixed training points, but the Point Resampler callback replaces the points every ... steps and the weights would start over each time -- turn the Point Resampler off, or set Per-point weights to None.` |
| Per-point weights with resampling in time windows | `Time-adaptive training: resampling adds collocation points, which per-point weights (...) can't follow -- ...` |
| Exact initial condition with time-adaptive training | `Exact initial condition can't be combined with time-adaptive training yet ... -- untick 'Exact' on the Initial Conditions page, or switch time-adaptive training off.` |
| Exact initial condition on a steady problem or an ODE | `Exact initial condition: needs a time-dependent PDE problem.` |
| Exact initial condition with your own output-transform formula | `Exact initial condition can't be combined with your own output transform formula -- ...` |
| Exact periodicity without a periodic boundary condition | `Exact periodicity: no boundary condition is periodic -- add a periodic BC row (or untick 'Exact periodicity' on the Boundary Conditions page).` |
| Exact periodicity with the network's input transform | `Exact periodicity can't be combined with the Network page's input transform -- ...` |
| IC pre-training when every initial condition is exact | `IC pre-training fits the network to the initial conditions given as loss terms, but no output has one ... -- untick IC pre-training.` |
| IC pre-training on an ODE or a steady problem | `IC pre-training needs a time-dependent PDE: ... -- untick IC pre-training.` |
| A boundary condition for an output the problem does not have | `BC 2 applies to output 3, but the problem has 2 output(s): pick one of them in the BC's Output box.` |
| A loss function other than MSE or MAE (from a problem saved by an older version) | `Loss function ...: not usable for a PINN (the equation residuals have target 0, or the loss does not exist with PyTorch) -- choose MSE or MAE in the Training phases.` |
| 'Scale by initial losses' with time-adaptive training | `'Scale by initial losses' isn't available with Time-Adaptive training -- turn one of them off.` |
| Growing time range or backward compatibility without transfer learning | `Time-adaptive training: Growing time range ... train one network through all the time windows, so they need 'transfer learning' ... -- tick it, or untick ...` |
| STMsFFN network on an ODE or a steady problem | `STMsFFN needs space and time; an ODE has only t -- use MsFFN.` / `STMsFFN separates space and time, so it needs a time-dependent problem; use MsFFN for a steady one.` |
| Weight decay with a PFNN, MsFFN or STMsFFN network | `Weight decay can't be used with a ... network (DeepXDE's ... has no regularization): set it to 0, or use an FNN.` |
| `frac(...)` with RAR, time-adaptive training or IC pre-training | `frac(...) problems can't use RAR, Time-Adaptive training or IC pre-training ...` |

Optimizer settings are checked when you press **Solve**: a nonzero weight decay
(**Settings › Optimizer Settings...**) cannot be used with an L-BFGS or NNCG phase, and an AdamW phase needs a
nonzero weight decay. Several of these checks are new in v1.6.0; the [Release history](../about/changelog.md) lists
them.

### Other messages

- `Inverse mode needs at least one observed-data file to fit against, but no observation row has a file selected.`:
  choose a measured-data file on the **Inverse problem** page.
- `x_min (...) must be less than x_max (...).` and the same for `y`, `z` and `t`: the domain or time interval is
  inverted.
- `Optimizer Scheduler is enabled but has no phases configured; ...` or `Every configured Optimizer Scheduler phase
  has 0 iterations; ...`: add a phase on the **Training** page (**Add Phase**) or give one a positive number of
  iterations.
- `❌ Cannot start -- the results folder's path is too long for Windows (...)`: choose a shorter **Save to** folder on
  the **Results** page, or turn on Windows' long paths (`LongPathsEnabled`).

## A boundary condition matches no points

A boundary condition whose **Where** selects no training point has no loss to compute: its loss term is NaN at any
weight. Training then goes on without it, and an L-BFGS phase stops inside PyTorch's line search with an
`IndexError`. Typical causes are a side name such as `left` on a disk or another curved shape (which has no flat
sides), for example after changing an example's domain, or a condition on an edge that no longer exists.

!!! info "New in v1.6.0 (in development)"
    Before training, both scripts now print which conditions are empty:

    ```text
    ⚠️ Condition(s) [2] (in loss order: Boundary Conditions rows, then initial conditions, then measured data)
    select no training point, so their loss is NaN -- check their 'Where' (e.g. a side such as 'left' on a disk or
    other curved shape). Change or remove them.
    ```

    The numbers count the rows of the **Boundary conditions** page first, then the initial conditions, then the
    measured data. In v1.5.0 there is no such message; look for a loss column that is `nan` in the loss table of the
    Training Log.

For a sketched or combined domain, a condition whose last edge was removed is refused before training:
`BC 2 picks no part of the boundary (its last edge was taken out): pick its edges in the Geometry view, or delete the
BC.`

## The loss becomes NaN or Inf

When the loss table in the Training Log shows `nan` or `inf`, find out where it starts:

- **NaN in one column from the first step**: that loss term cannot be computed. Check the condition belonging to the
  column (see the previous section for empty conditions) and its expression.
- **NaN everywhere from the first step**: an expression is undefined somewhere in the domain, for example a division
  by zero, `log` or `sqrt` of a value that can be zero or negative, or a function whose derivative is infinite at a
  point (the square root of a distance, at the centre). The equations, initial conditions, boundary values and
  output transforms all count. Shift or smooth the expression (for example add a small constant under a square
  root).
- **NaN after some training**: the training diverged. Lower the learning rate of the Adam phase (**Training** page),
  lower very large loss weights (**Loss weights**), or use float64 (**Settings › Float Precision...**, or
  **Precision** on the **Training** page), which also helps L-BFGS. Very large or very small coefficients in the
  equations are a frequent cause; scaling the equations so their residuals are of order 1 helps.

A run with a NaN loss still produces plots, but they are meaningless.

## Training does not converge, or converges to a wrong solution

PINNs do not always converge, and a small loss does not guarantee a correct solution. Problems with sharp fronts,
interfaces (phase fields), multiple scales, long time ranges or strong nonlinearity are hard for every PINN method;
some may not be solvable to useful accuracy with any setting PINNStudio offers. What helps, in the order the project's
[training methods study](../reproducibility/benchmarks.md) found most useful:

1. **Check the problem first.** Compare the typeset equations with the intended ones, check the units and the sign
   convention (PINNStudio's equations are residuals: everything on one side, `= 0`), and make sure the initial and
   boundary conditions are consistent with each other.
2. **Scale the equations** so that every residual is of order 1, and scale the outputs accordingly (**Neural
   network** › output transform).
3. **Train longer, or with more points.** Add iterations to the phases or an L-BFGS phase after Adam, and increase
   **Domain points**. Watch the test loss: if it rises while the training loss falls, more points are needed.
4. **Build the conditions into the network.** An exact initial condition and exact periodicity remove those loss
   terms ([Exact constraints](../reference/exact-constraints.md)).
5. **Use one automatic weighting method** (RBA, SA-PINN or causal training, on the **Loss weights** page; see
   [RBA and SA-PINN](../reference/point-weights.md) and [Causal training](../reference/causal.md)), or add points
   where the residual is large with [RAR](../reference/rar.md).
6. **Split long time ranges** into time windows ([Time windows and bc-PINN](../reference/time-windows.md)).
7. **Try another network**: more or wider layers, another activation, or Fourier-feature networks (MsFFN, STMsFFN)
   for multi-scale and oscillating solutions ([Networks and Fourier features](../reference/networks.md)).

These methods help on hard problems but can make results worse on problems a plain PINN already solves well. Compare
with a plain run, use a [parameter sweep](../guide/sweeps.md) to compare settings systematically, and judge the result
by its error against reference data, not by the loss. The examples' **Tuned** settings show what worked for each
example. See also [Scope and limitations](../theory/limitations.md).

## Error Analysis cannot find the reference data

**After loading an example.** Examples with reference data configure Error Analysis themselves
(`✅ Error analysis auto-configured — 5 ground truth files from template`). If this line is missing, the example's
files were not found in the installation's `reference_data` folder: reinstall PINNStudio. (Releases before 1.2.0
shipped without `reference_data`.)

**A problem saved on another computer.** A saved problem stores data files as paths. In v1.6.0, files inside
PINNStudio's `reference_data` folder are saved relative to it and are found on any installation; your own files must
be at the same path, or chosen again in the **Error Analysis** dialog (**Results** page). In v1.5.0, open the dialog
and choose the files again.

**Your own reference files.** In the **Error Analysis** dialog, **Reference files** expects one plain-text file per
time, with one row per point and the columns of the problem: the coordinates (x, then y and z), the time (not for a
steady problem), then the value, for example `x, t, u` for a 1D time-dependent problem. The time is read from the
file's first row and shown next to the file (`✅ t = 0.25 (… pts)`). A file that cannot be read shows `❌` and the
reason. If no row has a readable file, the log says `❌ Error Analysis: no valid reference files loaded.`

**No reference data at all.** Choose **Exact solution formula** in the dialog if the exact solution is known, or
compute reference files with another solver.

**After training.** **Run Error Analysis** sets up the error analysis for the next run
(`✅ Error Analysis configured — ... reference files, will run after training.`). To evaluate a model that has already
been trained, set up the error analysis and then load the model under **Restore & visualize**, which runs it for the
reference times inside the model's time range.

## An exported script fails outside PINNStudio

- **`ModuleNotFoundError`**: the Python you run it with lacks DeepXDE, PyTorch, NumPy, SciPy or Matplotlib. Run it
  with the Python PINNStudio runs in, or install PINNStudio (or these packages) in your environment. For a copy
  installed with the one-command installer, that Python is in the folder `uv tool dir` prints
  (`pinnstudio/bin/python`, or `pinnstudio\Scripts\python.exe` on Windows).
- **`FileNotFoundError` or an empty results folder**: the script contains the full paths of the results folder and
  of every data file as they were on the computer that exported it. Edit them near the top of the script (`save_dir`)
  and where the data files are read, or export the script again on the computer that runs it.
- **The script runs on a different device than expected**: the GPU settings of **Settings › GPU / Hardware...** are
  written into the script. To force the CPU, set `CUDA_VISIBLE_DEVICES=-1` in the environment, or export the script
  with **Train on: CPU**.
- **Different results from Solve**: on the same device and with the same seed, the export trains the same model as
  **Solve** for the built-in examples ([Export equivalence](../reproducibility/export-equivalence.md)). Small
  differences between runs on a GPU, or between CPU and GPU, are normal floating-point nondeterminism. A large
  difference is a bug: please [report it](../getting-started/reporting-bugs.md) with the `.pinn.json` file and the
  script.

!!! info "New in v1.6.0 (in development)"
    Several export problems of v1.5.0 are fixed in the development version, among them `NameError: _sol_dir` when
    Model Checkpoint was on, and the exported script ignoring the GPU index and memory fraction. See the
    [Release history](../about/changelog.md).

## A checkpoint cannot be restored

**Restore & visualize** needs a model file (`.pt`) and the run's configuration (**Config**). When you choose the
model file, PINNStudio looks for the configuration next to it: a `.json` file of the same name, then
`model_config.json` in the same folder (`solution_results` of the run), or for a time window its
`step_config.json`. If none is found, the log says `⚠️ No config found — please browse manually.`; choose the run's
`model_config.json` yourself.

- `❌ Please select a model file.` / `❌ Please select a model_config.json file.` / `❌ Please select a save
  directory.`: one of the three fields is empty.
- `❌ Could not read config: ...`: the file is not a run configuration (a `.pinn.json` problem file is not one).
- **PyTorch reports missing or unexpected keys, or sizes that do not match**: the model file and the configuration
  come from different runs (for example another network size or number of outputs). Use the two files from the same
  run folder.
- **The optimizer** is not chosen by hand: PINNStudio takes it from the model file's name (`model_adam...` or
  `model_lbfgs...`), otherwise from the last training phase in the configuration, and the log says
  `✅ Auto-detected optimizer from model_config.json: ...`. Keep the file names the run gave the checkpoints.

See [Restore & visualize](../guide/restore.md) and [Checkpoints and restoring](../reference/checkpoints.md).

## Custom geometry problems

- `Geometry Type is 'Custom' but no shapes have been added; add at least one shape in the Custom geometry builder.`:
  add a shape on the **Geometry** page.
- **The domain or the conditions are not what you intended**: look at the domain view, which appears when you select
  **Geometry**, **Collocation points** or a boundary condition in the model tree; selecting a condition shows the
  edges (faces in 3D) it covers.
- **A side name on a curved or combined shape**: side names such as `left` only exist on boxes; for other shapes pick
  the edges in the Sketch or write a **Where** expression. See
  [A boundary condition matches no points](#a-boundary-condition-matches-no-points).
- **Polygon and triangle geometries fail with `ValueError` from `np.cross`**: NumPy 2.5 or newer is installed; see
  [Missing or incompatible packages](#missing-or-incompatible-packages).
- **DXF or SVG import**: only closed outlines of plain geometry are read (lines, arcs, circles, ellipses, polylines
  and paths, with arcs and curves approximated by short straight pieces). DXF blocks, splines, text and dimensions,
  and SVG clipping, masks and strokes are ignored. Check the result in the Sketch before training. See
  [Geometry and domain](../guide/geometry.md).

## Training is slow

- **CPU threads.** On the CPU, PINNStudio's scripts use about one thread per performance core, at most 8
  (**Settings › GPU / Hardware...** › **CPU threads: Automatic**). More threads are usually slower on processors with
  efficiency cores. You can set a number of your own there; the environment variable `OMP_NUM_THREADS` overrides
  *Automatic*.
- **CPU or GPU.** A GPU pays off for larger problems (a few thousand points and more). A small problem, such as a 1D
  problem or an ODE with a few hundred points, is often faster on the CPU: **Train on: CPU**. PINNStudio prints a
  tip in the Training Log when a small 1D problem trains on the GPU (new in v1.6.0).
- **Precision.** float32 (the default) is faster than float64.
- **Problem size.** Training time grows with the number of points, the size of the network, the order of the
  derivatives and the number of outputs and equations. Start with fewer points and iterations, and increase them once
  the set-up works.
- **Callbacks.** **Early Stopping** ends a run once the loss stops improving; **Training Timer** stops it after a
  given time.
- **Windows: starting a run.** Starting a run takes longer than on Linux, most of it importing PyTorch and
  DeepXDE. Windows Defender scans every file Python loads, which added 1.5 to 2 s per run on a test laptop. If you
  are comfortable with it, an exclusion for the folder of an installer copy removes that (PowerShell as
  administrator: `Add-MpPreference -ExclusionPath "$env:APPDATA\uv\tools\pinnstudio"`; `uv tool dir` shows the
  folder if yours is elsewhere). It is optional; PINNStudio works the same without it.
- **Windows laptops: power mode.** Plug in and choose the *Best performance* power mode (Windows **Settings ›
  System › Power**). A small network on the GPU is limited by how fast the CPU hands it work, so a CPU that slows
  down on battery or when hot slows the GPU too; on the test laptop the 1D examples trained 1.5 to 3 times faster
  with **Train on:** **CPU**.

[Training speed](../reproducibility/environment.md#training-speed) summarizes the speed measurements, and
[Seeds, precision and hardware](../reproducibility/environment.md) has more on hardware and precision.

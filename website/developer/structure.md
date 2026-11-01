# Repository structure

```text
PINNStudio/
├── pinnstudio/              the application (the Python package)
│   ├── main.py              entry point: QApplication, MainWindow (console script `pinnstudio`)
│   ├── __main__.py          `python -m pinnstudio`
│   ├── shortcut.py          desktop shortcuts (--create-shortcut / --remove-shortcut)
│   ├── errors.py            error reporting for the window
│   ├── core/                configuration, code generation, running, examples (no Qt)
│   ├── ui/                  the PyQt6 interface
│   └── resources/           icons and tuned_settings.json (packaged)
├── reference_data/          reference solutions and measured data used by the examples (packaged)
├── tests/                   the pytest suite
├── tools/                   developer scripts: reference-data generators, checks, figures, docs tools
├── benchmarks/              recorded benchmark runs, studies and the pages generated from them
├── website/                 this documentation site (MkDocs + Material); mkdocs.yml at the root
├── docs/                    the single-page project website (index.html and its images)
├── assets/                  images used by the README and the benchmark pages
├── .github/workflows/       CI: tests, platform matrix, smoke test, documentation
├── setup.py, pyproject.toml, MANIFEST.in, requirements.txt     packaging
├── install.sh, install.bat, get-pinnstudio.sh, get-pinnstudio.ps1, select_torch_index.py   installers
└── README.md, CHANGELOG.md, CONTRIBUTING.md, CITATION.cff, LICENSE, LICENSES/, THIRD_PARTY_LICENSES.md
```

## `pinnstudio/core`

| Module | Role |
|---|---|
| `config.py` | `PINNConfig`, the dataclass holding a complete problem; `validate()`; `LBFGS_RECOMMENDED` ([Configuration schema](config.md)) |
| `codegen.py` | the two code generators: `generate_script()` (what **Solve** runs) and `generate_clean_script()` (**Export as DeepXDE Script...**), and their shared helpers ([Architecture](architecture.md)) |
| `runner.py` | `run_pinn()`: writes Solve's script to a temporary file, runs it in a subprocess and streams its output |
| `pyexe.py` | which Python runs the generated scripts, and their environment (UTF-8 mode, Windows process handling) |
| `constants.py` | named constants of the **Equations** page and their substitution into every expression |
| `deepxde_examples.py` | DeepXDE's documented examples as PINNStudio settings (`EXAMPLES`, `build_config`) |
| `pinnstudio_examples.py` | PINNStudio templates defined as plain settings, and optional local templates |
| `example_catalog.py` | the one list of every Quick Example (names, order, dimension, papers) shared by the window and the examples table of `benchmarks/examples.md` |
| `example_settings.py`, `paper_settings.py` | **Settings: Default / Tuned** of the examples, and the papers' own settings of the templates |
| `sweep_registry.py` | every setting a Parameter Sweep can vary, with get/set functions that respect the training architecture |
| `sweep_runner.py` | runs a sweep: builds the runs, validates and trains each, collects losses, errors and parameters, writes the summary |
| `bayes_opt.py` | the Gaussian-process Bayesian optimizer of **Bayesian search** ([theory](../theory/bayesian.md)) |
| `obs_reduce.py` | reduction of measured data to a grid or a random subset |
| `portable_paths.py` | reference-data paths in saved problems that work on any machine (`@REF/` prefix) |
| `ea_view3d.py`, `line_slice.py` | the 3D Error Analysis view, and where line plots cut a 2D/3D domain |
| `torch_index.py`, `update_info.py` | which PyTorch CUDA build fits the driver; how to update an installed copy |

## `pinnstudio/ui`

| Module | Role |
|---|---|
| `main_window.py` | `MainWindow`: every page, dialog and widget; `_build_config()` (widgets → `PINNConfig`) and `_apply_config()` (the reverse); the template set-up; Restore & Visualize script builders; `SolverThread`, `SweepThread` |
| `workspace.py` | the three-panel workspace (model tree, settings, graphics) |
| `results.py` | the Results tabs: live loss, Error analysis table, run history |
| `figure_style.py` | the **Plot Settings...** dialog |
| `sketch_canvas.py`, `sketch_edges.py`, `sketch3d.py`, `sketch_import.py` | the Sketch: drawing 2D/3D domains, numbering boundary edges and faces, DXF/SVG import |
| `view3d.py` | the 3D view of a 3D domain |
| `equation.py`, `sci_spinbox.py` | typeset equation boxes; number boxes for very small or large values |
| `start_page.py`, `theme.py`, `icons.py`, `export_tools.py` | Start tab, colour themes and stylesheet, icons, copy/save of results |

## Other folders

- **`reference_data/`** — `1D/`, `2D/`, `3D/` for PINNStudio's templates and `deepxde/` for DeepXDE's examples;
  one text file per reference time. The data are written by scripts in `tools/`
  (`make_exact_reference_data.py`, `make_deepxde_reference_data.py`, `make_poisson_reference_data.py`,
  `make_burgers2d_reference_data.py`, `make_phase_field_reference_data.py`); see
  [Reference data](../reproducibility/reference-data.md).
- **`tests/`** — `test_*.py`, most also runnable as plain scripts; `conftest.py` marks the slow ones
  ([Tests](tests.md)). `tests/local/` is ignored by git.
- **`tools/`** — checks (`check_export_equivalence.py`, `check_benchmarks_figures.py`, `smoke_gui.py`,
  `launch_check.py`, `all_examples_check.py`), example runners (`run_deepxde_examples.py`, `tuning_study.py`),
  page generators (`make_benchmarks_readme.py`, `make_tuning_page.py`, `update_source_links.py` for the source links
  [below](#where-each-feature-is-in-the-code)), figure
  scripts, and `tools/docs/` for this site (`WRITING.md`, `make_screenshots.py`, `make_config_reference.py`, and
  `make_gallery.py`, which writes the example gallery from `benchmarks/examples.md` and the example definitions).
- **`benchmarks/`** — the recorded runs and studies behind the benchmark pages (`README.md`, `examples.md`, `paper.md`,
  `tuning_study.md`, `training_methods.md`, `performance.md`) and their scripts; see
  [Benchmark collection](../reproducibility/benchmarks.md).

## Where each feature is in the code

The main functions behind each feature, with links to the code on GitHub (the `main` branch). The links are
written by `tools/update_source_links.py` from the `<!-- source: ... -->` comment above each line; run it after
changing the code they point to, so that the line numbers stay right (see [Contributing](contributing.md)).
The page named under each heading describes the feature itself.

#### ODE, 1D, 2D and 3D problems

<!-- source: pinnstudio/core/codegen.py::_ode_to_1d | ODEs ; pinnstudio/core/codegen.py::_integral_helper_code | integrals ; pinnstudio/core/codegen.py::_frac_helper_code | fractional equations ; pinnstudio/ui/main_window.py::MainWindow._on_dim_changed | dimension switch -->
**Source code:** [ODEs](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L4650-L4708) (`codegen.py`) · [integrals](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L4751-L4769) (`codegen.py`) · [fractional equations](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L4818-L4844) (`codegen.py`) · [dimension switch](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L5910-L6011) (`main_window.py`)

Described in [Problem](../guide/problem.md).

#### Forward and inverse problems

<!-- source: pinnstudio/core/codegen.py::_parse_inverse_variables | unknown parameters ; pinnstudio/core/codegen.py::_parse_inverse_obs_files | measured data ; pinnstudio/ui/main_window.py::MainWindow._build_inverse_panel | Inverse panel -->
**Source code:** [unknown parameters](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L639-L693) (`codegen.py`) · [measured data](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L746-L800) (`codegen.py`) · [Inverse panel](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L4207-L4405) (`main_window.py`)

Described in [Inverse problems and measured data](../guide/inverse.md).

#### Equations and coupled systems

<!-- source: pinnstudio/ui/main_window.py::MainWindow._build_pde_inputs | equation inputs ; pinnstudio/ui/equation.py | live equation preview ; pinnstudio/core/constants.py | constants -->
**Source code:** [equation inputs](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L7869-L7978) (`main_window.py`) · [live equation preview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/equation.py) (`equation.py`) · [constants](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/constants.py) (`constants.py`)

Described in [Equations, outputs and constants](../guide/equations.md).

#### Boundary and initial conditions

<!-- source: pinnstudio/core/codegen.py::_clean_bc_row_code | boundary conditions in the script ; pinnstudio/ui/main_window.py::MainWindow._add_custom_bc_entry | boundary-condition rows -->
**Source code:** [boundary conditions in the script](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L9278-L9468) (`codegen.py`) · [boundary-condition rows](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L8545-L9024) (`main_window.py`)

Described in [Boundary conditions](../guide/boundary-conditions.md).

#### Geometry, the Sketch and boundary edges

<!-- source: pinnstudio/core/codegen.py::_build_custom_geom_code | custom geometry ; pinnstudio/ui/sketch_canvas.py | Sketch ; pinnstudio/ui/sketch_edges.py | boundary edges ; pinnstudio/ui/sketch_import.py | DXF/SVG import -->
**Source code:** [custom geometry](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L366-L413) (`codegen.py`) · [Sketch](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_canvas.py) (`sketch_canvas.py`) · [boundary edges](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_edges.py) (`sketch_edges.py`) · [DXF/SVG import](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/sketch_import.py) (`sketch_import.py`)

Described in [Geometry and domain](../guide/geometry.md).

#### Collocation points and domain preview

<!-- source: pinnstudio/core/codegen.py::_anchor_points_code | extra training points ; pinnstudio/ui/main_window.py::MainWindow._preview_domain | domain preview ; pinnstudio/ui/view3d.py | 3D view -->
**Source code:** [extra training points](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L10516-L10553) (`codegen.py`) · [domain preview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L13059-L13348) (`main_window.py`) · [3D view](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/view3d.py) (`view3d.py`)

Described in [Collocation points](../guide/points.md).

#### Network architecture

<!-- source: pinnstudio/core/codegen.py::_net_construction_helper_code | networks (FNN, PFNN, MsFFN, STMsFFN) ; pinnstudio/ui/main_window.py::MainWindow._build_network_panel | Network panel -->
**Source code:** [networks (FNN, PFNN, MsFFN, STMsFFN)](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L1190-L1239) (`codegen.py`) · [Network panel](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L2835-L3056) (`main_window.py`)

Described in [Neural network](../guide/network.md).

#### Input and output transforms

<!-- source: pinnstudio/core/codegen.py::_net_transforms_code | input and output transforms -->
**Source code:** [input and output transforms](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L4538-L4604) (`codegen.py`)

Described in [Neural network](../guide/network.md).

#### Training phases, learning-rate schedules and loss weights

<!-- source: pinnstudio/core/codegen.py::_clean_phase_train_lines | training phases ; pinnstudio/core/codegen.py::_ic_pretrain_train_code | IC-guided pre-training ; pinnstudio/ui/main_window.py::MainWindow._add_scheduler_phase | phase rows -->
**Source code:** [training phases](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L9715-L9749) (`codegen.py`) · [IC-guided pre-training](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L11372-L11440) (`codegen.py`) · [phase rows](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L15924-L16024) (`main_window.py`)

Described in [Training phases](../guide/training.md).

#### Exact constraints and automatic weighting

<!-- source: pinnstudio/core/codegen.py::_exact_constraints | exact periodicity and initial condition ; pinnstudio/core/codegen.py::_training_methods_defs | causal training, RBA and SA-PINN -->
**Source code:** [exact periodicity and initial condition](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L4886-L4954) (`codegen.py`) · [causal training, RBA and SA-PINN](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L804-L1024) (`codegen.py`)

Described in [Loss weights](../guide/loss-weights.md).

#### Residual-based adaptive refinement (RAR)

<!-- source: pinnstudio/core/codegen.py::_gs_rar_loop_code | RAR loop -->
**Source code:** [RAR loop](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L1759-L2509) (`codegen.py`)

Described in [Adaptive training](../guide/adaptive.md).

#### Time-adaptive training

<!-- source: pinnstudio/core/codegen.py::_gs_time_adaptive_loop_code | time-adaptive loop ; pinnstudio/ui/main_window.py::MainWindow._add_ta_step_group | time-step groups -->
**Source code:** [time-adaptive loop](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L2665-L4504) (`codegen.py`) · [time-step groups](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L7351-L7405) (`main_window.py`)

Described in [Adaptive training](../guide/adaptive.md).

#### Training callbacks and monitors

<!-- source: pinnstudio/core/codegen.py::_build_train_cbs_code | callbacks ; pinnstudio/core/codegen.py::_training_monitor_runtime_code | training monitors -->
**Source code:** [callbacks](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L1041-L1119) (`codegen.py`) · [training monitors](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L1330-L1476) (`codegen.py`)

Described in [Training phases](../guide/training.md).

#### Inverse parameters during training

<!-- source: pinnstudio/core/codegen.py::_param_axis_helper_code | convergence plots ; pinnstudio/ui/main_window.py::MainWindow._build_restore_param_script | plots after training -->
**Source code:** [convergence plots](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L10245-L10397) (`codegen.py`) · [plots after training](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L17009-L17170) (`main_window.py`)

Described in [Results and plots](../guide/results.md).

#### Parameter sweeps and Bayesian search

<!-- source: pinnstudio/core/sweep_runner.py::run_sweep | sweep runner ; pinnstudio/core/sweep_registry.py::available_params | sweepable parameters ; pinnstudio/core/bayes_opt.py | Bayesian optimization -->
**Source code:** [sweep runner](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_runner.py#L514-L656) (`sweep_runner.py`) · [sweepable parameters](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_registry.py#L1367-L1395) (`sweep_registry.py`) · [Bayesian optimization](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/bayes_opt.py) (`bayes_opt.py`)

Described in [Parameter sweeps](../guide/sweeps.md).

#### Training log and live loss

<!-- source: pinnstudio/ui/results.py::RunProgress | progress ; pinnstudio/ui/results.py::LiveLossPlot | live loss ; pinnstudio/core/codegen.py::_loss_plot_runtime_code | saved loss figure -->
**Source code:** [progress](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/results.py#L87-L224) (`results.py`) · [live loss](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/results.py#L237-L367) (`results.py`) · [saved loss figure](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L1242-L1327) (`codegen.py`)

Described in [Solve and the training log](../guide/solve.md).

#### Run history and comparing runs

<!-- source: pinnstudio/ui/results.py::RunHistory | run history ; pinnstudio/ui/results.py::RunCompare | comparing runs -->
**Source code:** [run history](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/results.py#L750-L888) (`results.py`) · [comparing runs](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/results.py#L647-L747) (`results.py`)

Described in [Results and plots](../guide/results.md).

#### Copying and saving figures and tables

<!-- source: pinnstudio/ui/export_tools.py | copy and save -->
**Source code:** [copy and save](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/export_tools.py) (`export_tools.py`)

Described in [Results and plots](../guide/results.md).

#### Error Analysis

<!-- source: pinnstudio/core/codegen.py::_gs_ea_setup_code | error analysis ; pinnstudio/core/codegen.py::_formula_reference_code | exact-solution reference ; pinnstudio/ui/main_window.py::MainWindow._show_ea_dialog | Error Analysis dialog -->
**Source code:** [error analysis](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L2512-L2604) (`codegen.py`) · [exact-solution reference](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L10450-L10490) (`codegen.py`) · [Error Analysis dialog](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L15227-L15627) (`main_window.py`)

Described in [Error Analysis](../guide/error-analysis.md).

#### Plots and animations

<!-- source: pinnstudio/core/codegen.py::generate_clean_script._result_plot_parts | result plots ; pinnstudio/ui/figure_style.py::FigureStyleDialog | figure style -->
**Source code:** [result plots](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L12825-L13264) (`codegen.py`) · [figure style](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/figure_style.py#L113-L688) (`figure_style.py`)

Described in [Results and plots](../guide/results.md).

#### Data export and standalone scripts

<!-- source: pinnstudio/core/codegen.py::generate_clean_script | standalone script ; pinnstudio/ui/main_window.py::MainWindow._on_export_settings | data export -->
**Source code:** [standalone script](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py#L11593-L13743) (`codegen.py`) · [data export](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L13976-L14040) (`main_window.py`)

Described in [Exporting a script](../guide/export.md).

#### Saving problems, seeds and hardware

<!-- source: pinnstudio/core/config.py::PINNConfig | problem settings ; pinnstudio/ui/main_window.py::MainWindow._save_problem | save ; pinnstudio/ui/main_window.py::MainWindow._open_problem | open ; pinnstudio/ui/main_window.py::MainWindow._on_gpu_settings | GPU settings -->
**Source code:** [problem settings](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py#L12-L1601) (`config.py`) · [save](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L9975-L10033) (`main_window.py`) · [open](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L10113-L10196) (`main_window.py`) · [GPU settings](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L12473-L12564) (`main_window.py`)

Described in [Save, reopen and export](../getting-started/save-and-export.md).

#### Restore & visualize

<!-- source: pinnstudio/ui/main_window.py::MainWindow._on_restore | Restore & visualize ; pinnstudio/ui/main_window.py::MainWindow._build_restore_script | restore script -->
**Source code:** [Restore & visualize](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L16453-L16913) (`main_window.py`) · [restore script](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py#L17172-L17561) (`main_window.py`)

Described in [Restore & visualize](../guide/restore.md).

## Files not in git

`.gitignore` keeps out build output (`build/`, `dist/`, `*.egg-info/`), virtual environments, solution data,
and local material: `pinnstudio/core/local_examples.py`, `reference_data/local/`, `tools/local/` and
`tests/local/` (templates and tests kept for yourself).

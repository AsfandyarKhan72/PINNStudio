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
  page generators (`make_benchmarks_readme.py`, `make_tuning_page.py`), figure
  scripts, and `tools/docs/` for this site (`WRITING.md`, `make_screenshots.py`, `make_config_reference.py`,
  `make_gallery.py`, which writes the example gallery from `benchmarks/examples.md` and the example definitions, and
  `make_source_links.py` with its mapping `source_map.py` for the source links [below](#where-each-feature-is-in-the-code)).
- **`benchmarks/`** — the recorded runs and studies behind the benchmark pages (`README.md`, `examples.md`, `paper.md`,
  `tuning_study.md`, `training_methods.md`, `performance.md`) and their scripts; see
  [Benchmark collection](../reproducibility/benchmarks.md).

## Where each feature is in the code

The main functions behind each feature, the tests that exercise it most directly, and the pages that describe it.
The list is written by `tools/docs/make_source_links.py` from one mapping, `tools/docs/source_map.py`; the same
entries end each [Feature reference](../reference/index.md) page under *Implementation and source code*. The links
name a file and a function or class in it, not line numbers, so they stay right when the code moves. If you rename
or move a function named here, update `tools/docs/source_map.py` and run `python tools/docs/make_source_links.py`;
the documentation workflow and `tests/test_source_links.py` run it with `--check`, which fails when a file or a
name no longer exists (see [Contributing](contributing.md)).

--8<-- "_snippets/source/_overview.md"

## Files not in git

`.gitignore` keeps out build output (`build/`, `dist/`, `*.egg-info/`), virtual environments, solution data,
and local material: `pinnstudio/core/local_examples.py`, `reference_data/local/`, `tools/local/` and
`tests/local/` (templates and tests kept for yourself).

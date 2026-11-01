# Writing the PINNStudio documentation site (contributor guide)

Source: `website/` (MkDocs + Material), configuration `mkdocs.yml` at the repository root.
Build: `mkdocs build --strict -d /tmp/site_check` (must finish with no WARNING). Preview: `mkdocs serve`.

## Accuracy rules (these override everything else)

1. **Every control, option, value range and default you name must exist in the current code.** The GUI is
   `pinnstudio/ui/main_window.py` (widgets and their labels/tooltips; `_build_config()` maps widgets to
   `PINNConfig` fields); the configuration is `pinnstudio/core/config.py` (`PINNConfig`, `validate()`);
   what actually runs is `pinnstudio/core/codegen.py` (`generate_script` = Solve's script,
   `generate_clean_script` = "Export script"). Quote GUI labels exactly as the widget text shows them.
   If the README and the code disagree, the code wins — and note the contradiction in your report.
2. **No invented capabilities, controls, methods, examples or numbers.** Do not state accuracies, errors,
   timings or benchmark values unless they come from `benchmarks/` records that the Reproducibility section
   already uses — and preferably not at all outside that section ("expected results" in tutorials describe
   what the user sees qualitatively, or numbers you obtained yourself by running the workflow, stated as
   such: "on a laptop CPU this run reached L2 ≈ … in our test").
3. **Versions.** The public release is **v1.5.0** (PyPI, tag v1.5.0). These pages describe the
   development version that will become **v1.6.0** (audit branch `audit/v1.6.0`). When a feature or fix
   exists only in the development version, say so with:
   `!!! info "New in v1.6.0 (in development)"`. Never write install instructions for v1.6.0 as if it were
   released: `pip install pinnstudio` installs v1.5.0; the development version is installed from source.
4. **License, paper, DOI.** Do not describe the license (see `about/license.md`, written separately), do
   not invent a DOI, and refer to the paper only as "the PINNStudio paper" (in preparation); do not name a journal
   or say it has been submitted.
5. **Scope statement.** PINNStudio helps set up, train and evaluate PINNs; it does not guarantee
   convergence, physical validity or accuracy for an arbitrary problem. Say so where it matters.
6. If you find a software bug or a documentation contradiction, **do not change the software** — list it
   in your final report.

## Style

- Task-oriented, plain, precise. Academic tone; no marketing words ("powerful", "seamless", "cutting-edge").
- Headings in sentence case. Short paragraphs. Use lists for steps, tables for settings.
- GUI labels in **bold** exactly as shown (e.g. **Training phases**, **Add Phase**, **Solve**);
  model-tree pages as they appear in the tree (**Problem**, **Equations**, **Geometry**, **Collocation points**,
  **Initial conditions**, **Boundary conditions**, **Neural network**, **Training**, **Loss weights**,
  **Adaptive training**, **Parameter sweep**, **Results**, **Restore & visualize**); menus as
  **Settings › Random Seed...**.
- Equations: inline `\( u_t + u u_x = \nu u_{xx} \)`, display `\[ ... \]` (MathJax via arithmatex).
  The expression syntax users type in PINNStudio goes in code: `du_t + u*du_x - 0.01/pi*du_xx`.
- Admonitions: `!!! note`, `!!! tip`, `!!! warning`, `!!! info "New in v1.6.0 (in development)"`,
  collapsible `??? note "Details"`. Tabs: `=== "Linux / macOS"`.
- Screenshots: `![Alt text describing what is shown](../assets/screenshots/NAME.png){ .screenshot loading=lazy }`
  Alt text must describe the content. Only use screenshots of the real application (existing files in
  `website/assets/screenshots/`, or ones you create with a script under `tools/docs/` that drives the real
  `MainWindow` offscreen, like `tools/docs/make_screenshots.py`). Never edit or draw on screenshots.
- Links: relative Markdown links to other pages (`../reference/rar.md`, `#anchor`); they are checked by the
  strict build. External links: only stable URLs (DOIs, official docs, the GitHub repository).
- Citations: author–year in the text with a link to `about/references.md#key` (keys listed there), or a DOI.

## Site map (nav in mkdocs.yml — do not edit mkdocs.yml)

getting-started/ (installation, first-run, save-and-export, reporting-bugs) · guide/ (one page per model-tree
page and Solve/Results/Error Analysis/Restore/Export/Sweeps/Settings) · tutorials/01–17 · reference/ (one page
per feature) · examples/ (gallery + one generated page per example) · theory/ · reproducibility/ · help/ ·
developer/ · about/.

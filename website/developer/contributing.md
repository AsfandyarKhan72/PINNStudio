# Contributing and documentation

PINNStudio is a small research-software project and keeps the process light. The repository's
[`CONTRIBUTING.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/CONTRIBUTING.md) is the reference; this
page summarises it and adds how to work on this documentation site. The terms under which contributions are
accepted are stated in `CONTRIBUTING.md` (see also [License](../about/license.md)).

## Reporting bugs and requesting features

Open an issue with the templates provided:

- **Bug report**: the operating system, the Python version, the steps to reproduce and, if a run failed, the
  relevant lines of the Training Log. A saved problem file (`.pinn.json`) or the exported script makes a problem
  much easier to reproduce. See [Reporting a problem](../getting-started/reporting-bugs.md).
- **Feature request**: the PDE problem or workflow the feature would make possible, preferably with a concrete
  example (equation, boundary conditions, ...).

## Code contributions

1. Set up a development copy ([Developer documentation](index.md#development-setup)).
2. Make the change. A feature usually needs a `PINNConfig` field, a widget in `main_window.py` and the code
   path in `codegen.py` ([Adding a training method](add-method.md)). Follow the existing Qt patterns, prefer a
   configuration field over new function parameters, and keep the generated scripts readable: they are meant to be
   read and reused outside the window.
3. Before opening a pull request:
    - run the tests ([Tests](tests.md));
    - start the app (`python -m pinnstudio.main`) and check that it opens cleanly;
    - run at least one Quick Example end to end (Solve, then Stop or let it finish), so that script generation
      still produces a runnable script;
    - if you renamed, moved or removed code that a feature links to under
      [Where each feature is in the code](structure.md#where-each-feature-is-in-the-code), update
      `tools/docs/source_map.py` and run `python tools/docs/make_source_links.py` (`--check` and
      `tests/test_source_links.py` fail until you do);
    - do not edit generated text: the benchmark pages and the examples overview (`benchmarks/examples.md`) are
      written by `tools/make_benchmarks_readme.py` and `tools/make_tuning_page.py`, the example gallery on this site
      by `tools/docs/make_gallery.py` (from `benchmarks/examples.md`), and the configuration table by
      `tools/docs/make_config_reference.py`.
4. Keep a pull request to one feature or fix, and describe what changed and why; a screenshot or a snippet of
   the generated script helps for interface or code-generation changes.

Code or data from other projects can only be added when its license allows it; name its source and license in
`THIRD_PARTY_LICENSES.md`.

## Documentation contributions

The site's sources are in `website/` (MkDocs with the Material theme); the configuration and the navigation are in
`mkdocs.yml` at the repository root.

### Rules

`tools/docs/WRITING.md` is mandatory reading. Its accuracy rules override everything else:

- every control, option, value range and default named on a page must exist in the current code (the window in
  `pinnstudio/ui/main_window.py`, the configuration in `pinnstudio/core/config.py`, what runs in
  `pinnstudio/core/codegen.py`); GUI labels are quoted exactly as the widget shows them, in bold;
- if the README and the code disagree, the code wins, and the contradiction is reported;
- no invented capabilities, examples or numbers; results and timings only from the benchmark records, or measured
  by you and stated as such;
- features that exist only in the development version are marked `!!! info "New in v1.6.0 (in development)"`;
- the scope statement: PINNStudio does not guarantee convergence, physical validity or accuracy.

Style: task-oriented, plain and precise; headings in sentence case; equations in MathJax (`\( ... \)`,
`\[ ... \]`); the expression syntax users type in code font.

### Build and preview

```bash
pip install -r website/requirements.txt     # MkDocs and Material, pinned
mkdocs serve                                # live preview at http://127.0.0.1:8000
mkdocs build --strict                       # what CI runs: any warning fails
```

`--strict` turns every warning into an error, including a broken internal link or anchor and a page missing
from the navigation. The `docs.yml` workflow builds the site with `--strict` for every pull request and every push
to the documentation branch that changes `website/`, `mkdocs.yml` or the workflow, checks the links (internal and
external) of the built pages, and uploads the built site as a downloadable preview. It never deploys on a push, a pull
request or a release: deployment to GitHub Pages runs only when the workflow is started by hand on `main` with
deployment selected **and** the maintainers have approved publication (the repository variable
`DOCS_PUBLISH_APPROVED` set to `true`).

### Screenshots

Screenshots must show the real application. `tools/docs/make_screenshots.py` opens the real `MainWindow`
offscreen, loads shipped examples the way a user picks them, selects each page of the model tree and saves the
window; dialogs are captured while open. With `--solve` it also runs one short Solve of the 1D Burgers example
on the CPU (a few minutes) and captures the Results tabs.

```bash
QT_QPA_PLATFORM=offscreen python tools/docs/make_screenshots.py website/assets/screenshots [--solve]
```

Nothing is drawn on or edited in the images afterwards. Use them as
`![Alt text describing what is shown](../assets/screenshots/NAME.png){ .screenshot loading=lazy }`.

### Generated pages

The [configuration table](config.md#field-reference) is written by `tools/docs/make_config_reference.py` from
`pinnstudio/core/config.py`; run it after changing a field, and `--check` to test whether the page is up to date.

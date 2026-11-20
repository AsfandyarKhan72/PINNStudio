# Model, views and controllers

PINNStudio's code is split into tiers by what it may know of the others, in the way Qt applications
practise MVC: Qt's own widgets are view-controller pairs for their clicks and keys, and above them the
application's controllers take everything that changes state or computes, while a view keeps its
display logic (Fowler's *Supervising Controller*). `tests/test_mvc.py` checks the rules below on the
source at every test run.

## The tiers

| Tier | Paths | Holds |
|---|---|---|
| Domain (model) | `pinnstudio/core/` | the problem (`PINNConfig`), script generation, sweeps, the network designer's design model (`core/netdesign/`), every computation |
| State (model) | `pinnstudio/ui/state/` | the application's state that views observe, with undo and signals (`DesignState`) |
| Controller | `pinnstudio/ui/controllers/` | the reactions to the user's intent that change state or compute (`DesignerController`), and the `Prompter` port through which they ask the user |
| View | every other file under `pinnstudio/ui/` | widgets, layout, drawing, dialogs, drafts of an edit in progress |
| Launcher | `pinnstudio/__init__.py`, `__main__.py`, `main.py`, `errors.py`, `shortcut.py` | starting the app, outside the triad |

A new file gets its tier from the folder it is in.

## The rules

- **R1 The domain is UI-free.** A file under `pinnstudio/core/` imports nothing from PySide6, shiboken6,
  Matplotlib's Qt backends or `pinnstudio.ui`, at any depth. The generated scripts, the command-line
  checks and the tests use the domain without a display.
- **R2 The state tier is closed.** It imports from Qt only `PySide6.QtCore` (its signals are how views
  observe it), and from `pinnstudio.ui` only the state tier.
- **R3 Controllers draw nothing.** A controller imports from Qt only `PySide6.QtCore`, and from
  `pinnstudio.ui` only the state tier and other controllers. It asks the user only through `Prompter`.
- **R4 Views compute nothing.** A view imports from the domain only the names in `VIEW_SAFE`: constants,
  types, and pure functions on small values that do no I/O and fit nothing. Imports under
  `if TYPE_CHECKING:` are exempt.
- **R5 Views change no state directly.** A view never calls a state mutator (`MUTATORS` in
  `pinnstudio/ui/state/design_state.py`), nor hands one on as a slot; it calls its controller.
- **R7 No reaching into privates.** Inside `pinnstudio/ui`, no `x._name` unless `x` is `self`, `cls` or
  `super()`, and no `from M import _name` from outside the importer's own package.
- **R8 Composition.** Only `main_window.py` builds the state and the controllers and passes them to the views
  that need them. Among the views only `main_window.py` imports `pinnstudio.ui.controllers` or
  `pinnstudio.ui.state` at run time; the others import those names only under `TYPE_CHECKING`.

A view may read the state's public attributes and connect to its signals, hold a draft (an edit in
progress) and hand the finished value to its controller, and check one typed field with a `VIEW_SAFE`
helper (`pinnstudio.core.netdesign.params`).

## VIEW_SAFE

`pinnstudio.core.netdesign.design`, `.params` and `.layers` are safe as a whole (no I/O, no PyTorch, no
fitting), and so is the constant `pinnstudio.core.netdesign.presets.PRESETS`. A change to the list is
reviewed as a diff of `tests/test_mvc.py`.

## Code from before the rules

`pinnstudio/ui/main_window.py` and a few helpers around it predate these rules: the main window builds
the problem's configuration, writes files and starts training itself. Their violations are listed in
`tests/mvc_baseline.json`. The test fails on a violation that is not in the list, and also on a listed one
that no longer occurs, so the list can only shrink; new code adds nothing to it (`pinnstudio/core/`,
`ui/state/`, `ui/controllers/` and `ui/designer/` may have no entries at all).

### Moving the rest over, step by step

Each step moves one coherent part out of `main_window.py` and removes its baseline entries:

1. **Restore and error-analysis script builders** (`_build_restore_script`, `_build_restore_script_ta`,
   `_build_restore_param_script`, `_build_restore_ea_script`): pure functions of a configuration, to
   `core/restore_codegen.py`, with the main window calling them.
2. **Solve, Stop and the run threads** to a `RunController` with the run state (`ui/state/run_state.py`):
   the threads, the progress and the result paths; the Results panels observe it.
3. **Problem files** (new, open, save, recent files, unsaved changes) to a `ProblemController`.
4. **One settings page at a time** (Neural network, Training, Points, …): a view per page with
   `to_config`/`from_config` drafts, and a `ProblemState` holding the `PINNConfig` with undo; this ends the
   hand-written mirror between `_build_config` and `_apply_config`.
5. **Sweeps and Bayesian search** to a `SweepController`.

## How to

- **A new button that changes something.** The view collects its widgets' values and calls a controller
  method; the controller changes the state; the view redraws when the state signals.
- **A new question to the user.** Add a method to `Prompter` (`pinnstudio/ui/controllers/prompter.py`)
  together with `WindowPrompter`'s dialog and a test's fake answer; the controller calls the port.
- **A new file.** Put it in its tier's folder; `python tests/test_mvc.py` prints any violation by rule
  and file.

## The network designer, as an example

| Tier | File | Role |
|---|---|---|
| Domain | `core/netdesign/design.py`, `layers.py`, `params.py`, `shapes.py`, `presets.py`, `layout.py`, `runtime.py`, `export.py`, `library.py`, `trust.py`, `problem.py` | the design and its checks, widths and legal connections, ready-made designs, the PyTorch source, files |
| State | `ui/state/design_state.py` | the design, the level on view, the selection, undo and redo, widths and problems |
| Controller | `ui/controllers/designer.py`, `prompter.py` | every edit as one step of Undo, presets, library, export, trust |
| View | `ui/designer/canvas.py`, `palette.py`, `inspector.py`, `panel.py`, `window_prompter.py` | drawing and input; every change through the controller |

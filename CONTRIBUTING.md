# Contributing to PINNStudio

Thanks for considering a contribution. PINNStudio is a small research-tools project, so the process is kept light on purpose.

## Reporting bugs / requesting features

Please open an issue using the provided templates:

- **Bug report** — include your OS, Python version, the steps to reproduce, and (if the run failed) the relevant lines from the Training Log panel.
- **Feature request** — describe the PDE problem or workflow the feature would unblock; a concrete example (equation, boundary conditions, etc.) is very helpful.

## Development setup

```bash
git clone https://github.com/AsfandyarKhan72/pinnstudio.git
cd pinnstudio
python -m venv venv
source venv/bin/activate       # venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m pinnstudio.main
```

## Project layout

- `pinnstudio/ui/main_window.py` — the PyQt6 interface: every tab, dialog, and input widget.
- `pinnstudio/core/config.py` — `PINNConfig`, the dataclass that holds a full problem definition (domain, BCs/ICs, network, training schedule, etc.).
- `pinnstudio/core/codegen.py` — turns a `PINNConfig` into a standalone DeepXDE/PyTorch script.
- `pinnstudio/core/runner.py` — writes that script to a temp file, runs it as a subprocess, and streams stdout back into the GUI log.

If you're adding a feature, it usually touches all three: a new field on `PINNConfig`, a widget to set it in `main_window.py`, and the corresponding code path in `codegen.py`.

## Before opening a pull request

- Launch the app (`python -m pinnstudio.main`) and confirm it still starts cleanly.
- Run at least one built-in Quick Example end-to-end (Solve → Stop/finish) to confirm script generation still produces a runnable script.
- Keep PRs focused — one feature or fix per PR is easier to review than a bundle of unrelated changes.
- Describe *what changed and why* in the PR description; screenshots or a snippet of the generated script are welcome for UI or codegen changes.

## Code style

- Follow the existing PyQt6 patterns already used in `main_window.py` (naming, signal/slot wiring, dialog structure) rather than introducing a new UI pattern.
- Prefer adding a field to `PINNConfig` over threading new parameters through function signatures.
- Keep generated scripts (`codegen.py` output) readable — they're meant to be inspected and reused outside the GUI, not just executed.

## Questions

Open an issue, or reach out directly — contact details are in the [README](README.md#contact).

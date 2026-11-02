# Developer documentation

These pages are for people who want to change PINNStudio itself: fix a bug, add an example, a training method
or a geometry, or improve this documentation. They describe the development version that will become v1.6.0
(branch `audit/v1.6.0`); the released version is v1.5.0.

PINNStudio is a PySide6 (Qt for Python) application that writes and runs DeepXDE scripts. Almost every feature touches the
same three places: a field of the configuration (`PINNConfig`), a widget in the window, and the code
generators that turn the configuration into a training script.

| Page | Content |
|---|---|
| [Repository structure](structure.md) | What is where in the repository |
| [Architecture and code generation](architecture.md) | From the window to a running training script and back to the results |
| [Configuration schema](config.md) | `PINNConfig`, its validation, and a generated table of every field |
| [Adding an example](add-example.md) | How Quick Examples and templates are registered, with reference data and tests |
| [Adding a training method](add-method.md) | The full path of a setting, followed through an existing method (RBA) |
| [Geometry and boundary conditions](extend-geometry.md) | Where geometry types and boundary-condition rows are implemented |
| [Tests](tests.md) | The test suite, slow tests, the CI workflows, regression tests |
| [Building and packaging](packaging.md) | Package metadata, wheel and sdist, installers |
| [Contributing and documentation](contributing.md) | The contribution process and how to work on this site |
| [Releases](releases.md) | How a release is made |

## Development setup

```bash
git clone https://github.com/AsfandyarKhan72/PINNStudio.git
cd PINNStudio
python -m venv venv
source venv/bin/activate          # venv\Scripts\activate on Windows
pip install -e . pytest pytest-xdist
python -m pinnstudio              # or: python -m pinnstudio.main
```

This installs the development version from source. `pip install pinnstudio` installs the released v1.5.0
from PyPI instead. For a CPU-only or a specific CUDA build of PyTorch, install PyTorch first from the index
that fits the machine (see [Installation](../getting-started/installation.md)).

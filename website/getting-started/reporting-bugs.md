# Reporting a problem

Bug reports and feature requests go to the GitHub issue tracker:
[github.com/AsfandyarKhan72/PINNStudio/issues](https://github.com/AsfandyarKhan72/PINNStudio/issues).
Choose **New issue** and the **Bug report** or **Feature request** template.

Before opening an issue, have a look at [Troubleshooting](../help/troubleshooting.md) and the
[FAQ](../help/faq.md), and search the existing issues.

## What to include in a bug report

A problem can only be fixed when it can be reproduced. Please include as much of the following as applies.

**1. The PINNStudio version and how you installed it.** PINNStudio has no *About* dialog; find the version in one of
these ways:

- In a terminal, in the Python PINNStudio runs in:

    ```bash
    python -c "import importlib.metadata as m; print(m.version('pinnstudio'))"
    ```

    or `pip show pinnstudio` for a `pip` installation, or `uv tool list` for a copy installed with the one-command
    installer.
- In a saved problem: the entry `saved_with_pinnstudio_version` of any `.pinn.json` file you saved.
- The one-command installer prints it at the end (`PINNStudio <version> installed.`).

Say also how you installed it: one-command installer, `pip`, or from source.

For a source installation, give the commit as well: run `git rev-parse --short HEAD` in the `PINNStudio` folder.

**2. Your system.** Operating system and version, Python version, and the versions of PyTorch, DeepXDE and PySide6:

```bash
python -c "import sys, torch, deepxde, PySide6, PySide6.QtCore as q; print(sys.version); print(torch.__version__, torch.cuda.is_available(), deepxde.__version__, PySide6.__version__, q.qVersion())"
```

If you train on a GPU, name it, and give the driver's CUDA version (`nvidia-smi`).

**3. The problem file.** Save the problem (**Save**, Ctrl+S) and attach the `.pinn.json` file. It contains every
setting, so the problem can be opened exactly as you had it. If it uses data files of your own (measured data,
reference data, initial conditions or points from a file), attach them too, or say that the problem also occurs
without them. If the problem is with a built-in example, its name and the **Forward**/**Inverse** and
**Default**/**Tuned** choice are enough.

**4. The Training Log.** Copy the whole log, not only the last line: the error is often explained further up.
Select the text in the **Training Log** and copy it, or attach `training_log.txt` from the run's folder (in
`solution_results` inside the run folder, under **Save to**).

**5. The generated script, if the run failed in it.** **Export script** (Ctrl+E) writes the script for the current
settings, or use **Copy** on the **Script** tab. Running that script from a terminal
(`python my_problem.py`) shows the full error message and traceback, and tells whether the problem is in the
window or in the script.

**6. What you did and what happened.** The steps from opening PINNStudio to the problem, what you expected, and what
happened instead. A screenshot helps for problems with the window itself.

### When the window shows an error

An unexpected error in the window opens a message box titled **PINNStudio error**, which says that PINNStudio is still
running. Its **Show Details...** button shows the full message. The same message is written to the log file
`~/.pinnstudio/pinnstudio.log` in your home folder, which also receives PINNStudio's output when it was started from
its icon on Windows (no console). Attach the relevant part of that file.

### When PINNStudio does not start

Start it from a terminal (`pinnstudio`, or `./venv/bin/pinnstudio` for a source installation) and include everything
it prints, together with the installation method and the command you ran. See also
[PINNStudio does not start](../help/troubleshooting.md#pinnstudio-does-not-start).

## Feature requests

Describe the problem or workflow the feature would make possible. A concrete example (the equation, the domain and
conditions, or a sketch of the interface you have in mind) helps most.

## Contributing a fix

Pull requests are welcome. See [Contributing and documentation](../developer/contributing.md) and the repository's
[CONTRIBUTING.md](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/CONTRIBUTING.md).

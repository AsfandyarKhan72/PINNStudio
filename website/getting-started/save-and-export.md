# Save, reopen and export

A problem you set up in PINNStudio can be kept in two forms:

- a **saved problem** (`.pinn.json`), which PINNStudio opens again with every setting in place;
- an **exported script** (`.py`), a standalone DeepXDE/PyTorch program that trains the same model without the
  window.

The trained models and figures of each run are saved separately, in the run's results folder (see
[First run](first-run.md#results)).

## Where the commands are

| Action | Toolbar | Menu | Shortcut |
|---|---|---|---|
| Open a saved problem | **Open** | **Problem › Open Saved Problem...** | Ctrl+O |
| Open a recent problem | (Start tab, **RECENT**) | **Problem › Open Recent** | |
| Save | **Save** | **Problem › Save Problem** | Ctrl+S |
| Save under a new name | | **Problem › Save Problem As...** | Ctrl+Shift+S |
| Export the script | **Export script** | **Problem › Export as DeepXDE Script...** | Ctrl+E |
| Start a new problem | (Start tab, **NEW**) | **Problem › New** (1D problem, 2D problem, 3D problem, ODE system) | |

The **Start** tab also has **Open a saved problem...**, and the **Script** tab next to the Training Log has
**Export…** and **Copy** buttons.

## Saving a problem

Press **Save** (Ctrl+S). The first time, PINNStudio asks for a file name; the file type is
*PINNStudio Problem (\*.pinn.json)*, and `.pinn.json` is added if you leave it out. Later saves write to the same
file; **Save Problem As...** asks for a new name.

An example is never overwritten. Saving a loaded example always creates a new problem, with the proposed name
`<example name> (copy).pinn.json`, and the log says that the example itself is unchanged.

When you open another problem, load an example or start a new problem while the current one has unsaved changes,
PINNStudio asks **Save the changes to "..."?** with **Save**, **Discard** and **Cancel**.

### What a saved problem contains

A `.pinn.json` file is a plain JSON file with these entries:

| Entry | Content |
|---|---|
| `pinnstudio_problem_format` | The file format version (currently `1`) |
| `problem_name` | The file name without `.pinn.json` |
| `saved_with_pinnstudio_version` | The PINNStudio version that saved it |
| `saved_at` | Date and time of saving |
| `config` | Every setting of the problem (see below) |
| `template` | Only for a problem that started from an example: which example, so that switching between Forward and Inverse and finding the example's reference data keep working |

The `config` entry holds the complete problem definition, the same settings **Solve** uses: dimension and problem
type, outputs, equations and constants, geometry, collocation points, initial and boundary conditions, inverse
unknowns and measured-data files, network, training phases, loss weights, adaptive training, callbacks, parameter
sweep, plot settings, the error analysis set-up (reference files or exact-solution formula), and the options of the
**Settings** menu that belong to a run (precision, L-BFGS options, GPU / hardware, random seed). The field names are
those of PINNStudio's configuration (`PINNConfig`, see [Configuration schema](../developer/config.md)).

A saved problem does **not** contain trained models, results or the data files themselves: measured data and
reference files are stored as paths. Keep them where they are, or open the problem and point it at their new place.
Paths to your own files are saved as they are. Paths to files in PINNStudio's own `reference_data` folder (the
reference and measured data of the examples) are saved as `@REF/...`, relative to that folder, and opening a problem
points them at the `reference_data` folder of the installation that opens it. A problem saved from an example
therefore finds its data on another computer, and a full path into another computer's `reference_data` folder is
repaired when the file is opened.

## Opening a saved problem

Use **Open** (Ctrl+O), **Open a saved problem...** on the Start tab, or a file under **RECENT** or
**Problem › Open Recent**. Every setting is applied to the window, the example list goes back to **None**, and the
log says `✅ Problem loaded: <name>`.

PINNStudio checks the settings of the opened file. If some of them would stop **Solve** (for example an inverse
problem saved before a measured-data file was chosen), the file still opens, and the log lists them under
`⚠️ To fix before Solve:`. A file that is not valid JSON, or not a PINNStudio problem, is not opened; the log says
why (for example `❌ This doesn't look like a valid PINNStudio problem file.`).

Settings that the opening version does not know (from a newer version) are ignored. A problem saved by an older
version opens with today's defaults for settings that did not exist then.

## Exporting the script

**Export script** (Ctrl+E) writes the problem as a standalone Python script. PINNStudio proposes the problem's name
with the extension `.py`. The log confirms with `🐍 DeepXDE script exported: <file>`. The **Script** tab next to the
Training Log always shows the script that **Export script** would write for the current settings, and its
**Copy** button copies it.

The exported script

- imports DeepXDE (with the PyTorch backend), PyTorch, NumPy, SciPy and Matplotlib, not PINNStudio;
- contains only the features the problem uses, written as plain DeepXDE calls, with a short header that names the
  outputs and the domain;
- trains the model, saves it and the loss and solution figures, and runs the error analysis when it is set up;
- writes its results into a run folder inside **Save to**, named when the script was exported (the path is written
  into the script; running it again writes into the same folder).

Run it with a Python that has these packages, for example the one PINNStudio runs in:

```bash
python my_problem.py
```

What to keep in mind:

- **Paths are written into the script as they are on your computer**: the results folder and any data files
  (measured data, reference data, initial conditions or points from files). On another computer, edit these paths
  near the top of the script, or copy the files to the same places.
- The script is meant to train the same model as **Solve**. The project checks that both scripts train the same
  model for the built-in examples (see [Export equivalence](../reproducibility/export-equivalence.md)). With the
  time-window options growing time range, backward compatibility, initial-condition guided learning or resampling
  in each window, the export is the script **Solve** itself runs, with a note at the top.
- **Export solution data** (on the **Results** page) is written only by a run from the window. The exported script
  says so in its header when that option is on.

For the terms under which generated scripts can be used and shared, see [License](../about/license.md); the libraries
a script imports keep their own licenses ([Dependencies and attribution](../about/dependencies.md)).

More on the scripts: [Exporting a script](../guide/export.md) and [Generated scripts](../reference/scripts.md). If an
exported script fails outside PINNStudio, see
[Troubleshooting](../help/troubleshooting.md#an-exported-script-fails-outside-pinnstudio).

## Unsaved work after a crash

While a problem has unsaved changes, PINNStudio keeps a copy every 30 seconds in
`~/.pinnstudio/recovery/` (one file per running PINNStudio). The copy is removed when the problem is saved, when
another problem is opened or started, and when the window is closed normally.

If PINNStudio closes without that (a crash, a power cut, the computer restarting), the next start asks **Recover
unsaved work**:

- **Recover** opens the kept copy as an unsaved problem. A problem that was saved before keeps its name, and its
  file is not changed until you save.
- **Discard** removes the copy.
- **Later** keeps it and asks again at the next start.

Two PINNStudio windows at once each keep their own copy; one never offers the other's while it is running.

## Without the window: the command line

The `pinnstudio` command also works on a saved problem without opening the window. These commands do not load Qt,
so they run on a server or a cluster node without a display:

| Command | What it does |
|---|---|
| `pinnstudio problem.pinn.json` | Open the window with that problem (also what opening the file from the desktop does) |
| `pinnstudio check problem.pinn.json` | List the settings that would stop **Solve**, or print `OK` (exit code 1 when something is listed) |
| `pinnstudio export problem.pinn.json [-o script.py]` | Write the standalone script, as **Export script** does; `--full` writes Solve's own script instead |
| `pinnstudio run problem.pinn.json [--save-dir DIR]` | Train as **Solve** does, printing the Training Log; the results go into a new run folder `<problem>__<date>_<time>` in `DIR` (default: the problem's **Save to** folder) |
| `pinnstudio --version` | The version |

`pinnstudio run` refuses a [designed network](../guide/network-designer.md) whose custom code or saved modules
have not been trusted; trust them once in the window, or pass `--trust-code` for that run. The exit code is 0
when the run finished, so the command can be used in batch scripts and job schedulers.


---
search:
  boost: 2
---

# Generated scripts

## What it does

PINNStudio does not train inside the window: it writes a plain Python script for DeepXDE (PyTorch backend)
from the configuration and runs it. There are two generators:

- **Solve's script** (`codegen.generate_script`): what **Solve** runs. It handles every option PINNStudio
  offers, writes the run folder (models, plots, logs, Error Analysis, solution data) and prints the
  progress the window reads. It is written to a temporary file, run as a separate Python process, and
  deleted afterwards.
- **The exported script** (`codegen.generate_clean_script`): **Problem › Export as DeepXDE Script...**
  (Ctrl+E), the **Export script** toolbar button, or the **Script** tab (**Copy**, **Export…**). A shorter,
  tutorial-style script for exactly this problem: only the features that are on are written in, as plain
  DeepXDE calls, so it can be read, edited and run outside PINNStudio.

## When to use it

Export a script to see what your settings translate to in DeepXDE code, to run a problem on another machine
(a cluster, a GPU server) without the window, to keep a record of a configuration, or to extend a problem
beyond what the window offers.

## How to enable it

**Problem › Export as DeepXDE Script...** asks for a file name (`<problem>.py`) and writes the script; the
**Script** tab shows the same text live for the current settings.

## Settings

The scripts have no settings of their own; they follow the configuration. **Settings › GPU / Hardware...**
(device index, memory fraction, **Train on:** Automatic or CPU, CPU threads) and **Settings › Random Seed...**
are written into both.

## What each contains

| | Solve's script | Exported script |
|---|---|---|
| Problem, network, conditions, training phases, callbacks | yes | yes, only what is on |
| Inverse unknowns, measured data, parameter files and plots | yes | yes |
| RAR (with per-round diagnostics), IC pre-training, transforms, exact constraints | yes | yes |
| Time windows (transfer learning) | yes | yes |
| Time windows with backward compatibility, growing range, IC-guided learning or resampling | yes | the export **is** Solve's script, with a note at the top |
| Error Analysis (reference files or formula) | yes | yes |
| Solution data export (Results → Export Solution) | yes | no — a note says only Solve writes those files |
| Run folder bookkeeping (`training_log.txt`, `model_config.json`, …) | yes | plots, the model and the losses only |

Both start with `os.environ["DDE_BACKEND"] = "pytorch"`, set the default float type and the random seed,
and keep DeepXDE's internal L-BFGS step size in step with the L-BFGS options.

## Running outside the GUI

The exported script needs Python with DeepXDE, PyTorch, NumPy, SciPy and Matplotlib (PINNStudio's own
dependencies); it does not import PINNStudio. Run it with

```bash
python problem.py
```

`save_dir` near the top of the script sets the output folder; left empty, the script writes its plots and
files to `/tmp` (written literally into the script, so set `save_dir` on Windows). Paths to data files (measured data, reference files, IC files) are written as absolute paths of
the machine that exported it; edit them when you move the script. The header states the terms under which
the generated script may be used.

## Formulation

Both generators read the same configuration, after the same preparation step (named constants
substituted, exact constraints turned into transforms, an ODE turned into a stationary 1D problem in
\( x = t \), a non-box shape's bounding box used as the domain box). A check script,
`tools/check_export_equivalence.py`, loads each example, shortens its training, runs both scripts with the
same seed on the same device and compares the printed loss history line by line (and the identified
parameters of inverse problems), with a full-precision record of every `model.train()` call; see
[Export equivalence](../reproducibility/export-equivalence.md).

## Defaults and limitations

- Solve's script is not kept; export the script if you want a copy of what was run.
- The exported script does not write the solution data files.
- GPU runs are not bit-for-bit reproducible (cuDNN determinism is not forced), so two runs can differ
  slightly even with the same seed.

!!! info "New in v1.6.0 (in development)"
    - The exported script sets the GPU index and memory fraction of **Settings › GPU / Hardware...**, and
      notes when solution data export is on.
    - With bc-PINN, a growing time range, IC-guided learning or resampling rounds, the export is Solve's own
      script; with time windows the export draws the same random numbers before the windows as Solve's
      script, so the two train the same model.
    - Fixed: `NameError: _sol_dir` in the exported script with Model Checkpoint on.

## Supported problem types

All.

## Compatibility with other features

Every combination Solve accepts can be exported; the same checks (`validate()` and the code generator's
own) apply.

## Common mistakes and troubleshooting

- **`FileNotFoundError` when running an exported script elsewhere**: a data path from the exporting
  machine; edit the path.
- **Different results from Solve on a GPU**: GPU nondeterminism; compare on the CPU (**Train on:** CPU).

## Example

Export any example, e.g. **Poisson 1D – Dirichlet/Robin**, and compare it with the **Script** tab. See
[Tutorial 16: Standalone scripts](../tutorials/16-scripts.md) and the
[Exporting a script guide page](../guide/export.md).

## See also

- [Tutorial 16: standalone scripts](../tutorials/16-scripts.md)

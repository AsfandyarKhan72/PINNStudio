# Reporting and reproducing your own experiment

This page lists what to keep from a PINNStudio run so that you, a reviewer or a colleague can tell exactly what was
trained and repeat it. It is meant for your own problems; how the shipped benchmark studies are recorded is
described in [Benchmark collection](benchmarks.md).

!!! note "Benchmark records and tutorial results are different things"
    The published benchmark results are the records under `benchmarks/` in the repository and the results of the
    PINNStudio paper ([Reproducing the paper](paper.md)); they state the code version, settings, seeds and
    hardware of every run. The numbers in the [Tutorials](../tutorials/index.md) are illustrative: short runs made
    while writing the tutorials, to show what you should roughly see. Do not cite tutorial numbers as benchmark
    results, and do not expect your run to match them exactly.

## What to keep

### Software versions

- **PINNStudio version.** PINNStudio has no *About* dialog. For a release, `pip show pinnstudio` (or
  `uv tool list` for the one-command installer) gives the version, and every saved `.pinn.json` file records it as
  `saved_with_pinnstudio_version`. A source installation of the development version still reports `1.5.0` until
  v1.6.0 is released; record the commit instead, with `git rev-parse --short HEAD` in the `PINNStudio` folder (and
  `git status` to note local changes).
- **Python, PyTorch, DeepXDE.** PINNStudio does not write these into the run folder. Record them yourself, with the
  same Python PINNStudio runs in:

    ```bash
    python -c "import sys, torch, deepxde; print(sys.version); print('torch', torch.__version__, 'CUDA', torch.version.cuda); print('deepxde', deepxde.__version__)"
    ```

    `pip freeze > requirements-run.txt` keeps every package version.

### The problem and its settings

- **The problem file.** Save the problem (**Problem › Save Problem**) as a `.pinn.json` file. It is the complete
  `PINNConfig` ([Configuration schema](../developer/config.md)): equations, domain, conditions, network, the
  **Training phases**, loss weights, the adaptive methods and their settings, Error Analysis settings, the random
  seed and the precision. Keep the file you actually solved with; a problem changed after the run no longer
  describes it.
- **Seed and precision.** Both are in the problem file (`use_random_seed`, `random_seed`; `float_type`). State them
  in a report as well: **Settings › Random Seed...** (on by default, seed 2026) and the **Precision:** on the
  **Training** page ([Seeds, precision and hardware](environment.md)). A seed makes a run repeatable on the same
  hardware and software, not bit-identical across machines.
- **Data files.** The `.pinn.json` file stores data files as paths, not their content. Keep a copy of every
  measured-data file (**Inverse problem** page), reference file (**Error Analysis**), initial-condition file, Point
  Set file and extra-points file the problem uses. Files inside PINNStudio's own `reference_data` folder are saved
  relative to it.
- **The exported script.** **Problem › Export as DeepXDE Script...** writes a standalone script that trains the same
  model ([Export equivalence](export-equivalence.md)). The script Solve runs is a temporary file and is not kept in
  the run folder, so export the script if you want the code itself on record. It contains the full paths of the
  results folder and data files of the computer that exported it.

### The results

Each Solve writes a run folder under **Save to:** on the **Results** page
([Solve and the training log](../guide/solve.md)). Keep the whole folder; in particular

- `solution_results/training_log.txt`: the Training Log: the device it trained on (on a GPU also its name and the
  float type) and the loss of every recorded step;
- `solution_results/model_config.json`: the settings needed to restore the trained network (network, domain,
  outputs, precision, unknowns); it is not a copy of the whole problem, which is why the `.pinn.json` file is kept
  too;
- `solution_results/model_*.pt`: the trained network after each phase (and checkpoints, with **Model
  Checkpoint**); see [Checkpoints and restoring](../reference/checkpoints.md);
- `error_analysis/error_metrics.txt` and the comparison figures, when Error Analysis is set up;
- `<name>_convergence.txt` for every unknown of an inverse problem;
- `time_adaptive_steps/` or `solution_results/rar_rounds/` for time windows and RAR.

### Hardware

The device (CPU, or the GPU's name), the operating system and, for a GPU, the driver and CUDA versions
(`nvidia-smi`). The Training Log names the GPU; the rest is not recorded by PINNStudio. Training times are only
comparable on the same hardware.

### How the error is defined

State which error you report and at which points: PINNStudio's relative \( L_2 \) error, MSE and maximum error are
computed at the reference points of each comparison time, per output ([Error measures](../theory/metrics.md),
[Error metrics](../reference/metrics.md)). For an inverse problem, say whether you report the parameter error and
how it is defined. Name the reference solution and how it was obtained, including its resolution.

### Variability

A single run does not show how much a result varies. If a conclusion depends on a difference between two
settings, repeat the runs with other seeds (a [Parameter sweep](../guide/sweeps.md) over **Random seed**) and
report the spread. See also [Scope and limitations](../theory/limitations.md).

## Checklist

Copy this into your lab notes or the supplementary material of a report.

```text
PINNStudio experiment record
[ ] PINNStudio version (pip show pinnstudio) or git commit (git rev-parse --short HEAD), local changes noted
[ ] Python, PyTorch (+ CUDA), DeepXDE versions; pip freeze saved
[ ] Problem file (.pinn.json) exactly as solved
[ ] Random seed (on/off, value) and precision (float32/float64)
[ ] Training phases and adaptive methods (in the .pinn.json; summarised in the report)
[ ] Measured-data, reference, initial-condition, Point Set and extra-points files (copies)
[ ] Exported script (Problem > Export as DeepXDE Script...)
[ ] Run folder: training_log.txt, model_config.json, model_*.pt, error_analysis/error_metrics.txt,
    <name>_convergence.txt (inverse), time-window / RAR subfolders
[ ] Hardware: CPU or GPU model, OS, driver and CUDA version
[ ] Error definition, reference solution and its resolution, comparison times
[ ] Number of seeds and the spread, if results are compared
```

## Reproducing a recorded run

To repeat a run from such a record: install the same PINNStudio version (or check out the commit; see
[Installation](../getting-started/installation.md)) and the recorded package versions, put the data files where the
problem file expects them (or choose them again on the pages that use them), open the `.pinn.json` file with
**Problem › Open Saved Problem...** and press **Solve**, or run the exported script with the same Python. Compare
the new `error_metrics.txt` with the recorded one; on other hardware expect small differences
([Seeds, precision and hardware](environment.md)).

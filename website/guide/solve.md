# Solve and the training log

**Solve** (toolbar, or Ctrl+Enter / F5) trains the problem as it is set up in the window. This page
describes what happens when you press it, what the Training Log shows, and where the results go.

## What Solve does

1. **Checks the settings.** The same checks that mark tree entries while you work run once more
   (see [The interface](interface.md#settings-that-would-stop-solve)), together with a few optimizer
   checks (weight decay with L-BFGS or NNCG, AdamW without weight decay, NNCG on an old DeepXDE). If
   anything is wrong, nothing starts: the Training Log lists the problems under
   *Cannot start -- this configuration has invalid settings* and the first page to fix is selected.
2. **Starts a sweep instead**, if **Enable Parameter Sweep** is ticked ([Parameter sweeps](sweeps.md)).
3. **Writes the training script and runs it.** PINNStudio generates a complete DeepXDE/PyTorch script
   from the settings (the *Solve script*) and runs it as a separate Python process. The window stays
   usable while it trains.
4. **Shows the progress.** The Results tab switches to **Live loss**; the status bar shows the phase,
   the iteration, the loss, the elapsed time and an estimate of the time left; everything the script
   prints goes to the **Training Log**.
5. **Shows the results** when the run finishes: the loss and solution figures in **Overview**, the errors
   in **Error analysis** (when [Error Analysis](error-analysis.md) is set up), and the run in **History**.

**Stop** (Shift+F5) ends the training process; files it has already written stay in the run folder.

!!! note "Solve and Export script"
    Solve runs PINNStudio's own full script. **Export script** writes a shorter, standalone script for the
    same settings, meant to be read and run on its own; for most settings the two train the same model.
    See [Exporting a script](export.md).

## The Training Log

Before training, the log may say:

- what you changed relative to a loaded DeepXDE example (*Changed from the "…" example:*);
- that the measured data of an inverse problem is large (20,000 points or more) and training may be slow.

During training it shows what the script prints: the settings it uses, each phase's start, DeepXDE's
loss table every **Record every** iterations (**Plot settings**, **Loss** tab; default 1000), and, for time-adaptive training and
RAR, each window or round. Before training starts, the script also reports a boundary condition that
selects no training point, with its name and the reason.

At the end it says *Training complete!* and where the results were saved, or, when the run failed, the
error and a tip (for example that the GPU ran out of memory). The log keeps the last 20,000 lines.

![The end of a short 1D Burgers run: the Training Log lists the saved error metrics and comparison figures and ends with Training complete!, Results saved to the run folder and Error analysis ran inline; the status bar reads Finished with the run time, the final loss and the mean relative L₂ error](../assets/screenshots/solve_log.png){ .screenshot loading=lazy }

## Where the results go

Each run gets its own folder inside the **Save to** folder of the [Results](results.md) page, named after
the example (or the dimension and problem type, for your own problem) and the date and time, for example
`1D_Burgers__2026-10-30_15-47-03`. Runs are never overwritten. With **Save to** left empty nothing is kept.

| In the run folder | What it is |
|---|---|
| `solution_results/loss_plot.png` | The loss history |
| `solution_results/solution_plot.png` (or `.gif`) | The solution, in the plot type chosen on the Results page |
| `solution_results/model_*.pt` | The trained network (after each phase); used by [Restore & visualize](restore.md) |
| `solution_results/model_config.json` | The settings of the run |
| `solution_results/training_log.txt` | The Training Log of the run |
| `error_analysis/` | Error Analysis figures, `error_metrics.txt` and `error_spacetime.txt` |
| `<name>_convergence.txt` | Inverse problems: each unknown parameter's value during training |
| `time_adaptive_steps/step_…/` | Time-adaptive training: each window's model and settings |
| `solution_results/rar_rounds/` | RAR: each round's model and pictures |
| `solution_data/` | Only with **Export Solution** enabled: the solution as text files |

The **History** tab of the Results area lists every run folder in **Save to**; double-click one to show
its figures again ([Results and plots](results.md#look-at-earlier-runs)).

## When a run is slow or fails

- Many collocation or measured points, a large network or float64 make every step slower; small 1D
  problems often train faster on the CPU (**Settings › GPU / Hardware...**).
- *Check PDE/IC syntax*: an expression could not be evaluated; check the typeset preview under each
  equation, and use `*` for multiplication (`5*u`, not `5u`).
- See [Troubleshooting](../help/troubleshooting.md) for more.

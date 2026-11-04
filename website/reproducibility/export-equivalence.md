# Export equivalence

**Export script** writes a standalone DeepXDE script for the current problem; **Solve** runs a fuller script that also
streams progress to the window. The export-equivalence check verifies that the two train the same model.

## Protocol

For every example in the Examples list, in forward mode and, where the example has one, inverse mode, with its
Tuned settings,
[`tools/check_export_equivalence.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/check_export_equivalence.py):

1. builds the configuration the way the GUI does;
2. shortens training to the same small number of iterations for both scripts;
3. runs Solve's script and the exported script on the CPU with the same seed, recording the printed loss history,
   a full-precision history of every loss term, a SHA-256 hash of the network's weights and, for inverse problems,
   the identified parameters;
4. compares the records of the two scripts.

A configuration counts as identical when the printed loss history and the parameters match; whether the
full-precision records match as well is reported separately ("bitwise"). When the two scripts differ, Solve's script
is run a second time with the same seed as a control, to tell a difference between the scripts from run-to-run
variation of the CPU computation.

**Two kinds of identical configurations.** For most configurations the exported script comes from its own
generator (`generate_clean_script`), so an identical result compares two independently written scripts. When a
problem uses time windows with backward compatibility (bc-PINN), a growing time range, IC-guided learning or
resampling rounds, the export *is* Solve's script (`generate_clean_script` returns `generate_script`, with a header
saying so): these configurations are identical by construction, and the check shows only that the exported file
runs on its own and reproduces Solve's training. The record of the check names the one configuration of this kind
(found by that header; the exported scripts themselves are not kept in the repository).

**Latest result** (v1.6.0 code, commit 2ea54c3; Python 3.11.5, PyTorch 2.5.1 on the CPU, DeepXDE 1.15.0): of 93
example-mode configurations, 4 were excluded and **all 89 others were bitwise identical**. 88 of them compare two
independently generated scripts; 1 (1D Cahn-Hilliard of Mattey & Ghosh with the bc-PINN windows) is identical by
construction. No configuration differed or failed. A separate check of that example's current Tuned setting (38 time
windows with SA-PINN weights), which the exporter writes with its own generator, was also bitwise identical.

**What the check covers.** Each example's Tuned settings, training shortened as described above, CPU only, one
machine. The parameter-sweep example and the inverse mode of templates that have none are skipped. Default settings,
GPU runs, full-length training and problems of your own are not covered. The latest record is
[`benchmarks/export_check/README.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/export_check/README.md)
(commit, environment and result).

```bash
python tools/check_export_equivalence.py --out equivalence_results --device cpu
```

The command writes the records and the comparison of every configuration to the `--out` folder.

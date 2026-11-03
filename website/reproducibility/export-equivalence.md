# Export equivalence

**Export script** writes a standalone DeepXDE script for the current problem; **Solve** runs a fuller script that also
streams progress to the window. The export-equivalence check verifies that the two train the same model.

## Protocol

For every example in the Examples list, forward and inverse, with its Tuned settings,
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
variation of the CPU computation. Configurations whose exported script
deliberately differs (for example when the export is Solve's full script for an advanced time-window method) are
listed with the reason.

```bash
python tools/check_export_equivalence.py --out equivalence_results --device cpu
```

The command writes the records and the comparison of every configuration to the `--out` folder.

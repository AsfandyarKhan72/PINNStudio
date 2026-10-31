# Export equivalence

**Export script** writes a standalone DeepXDE script for the current problem; **Solve** runs a fuller script that also
streams progress to the window. The export-equivalence check verifies that the two train the same model.

## Protocol

For every example in the Examples list, forward and inverse, with its Tuned settings,
[`tools/check_export_equivalence.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/check_export_equivalence.py):

1. builds the configuration the way the GUI does;
2. shortens training to the same small number of iterations for both scripts;
3. runs Solve's script and the exported script on the CPU with the same seed, recording each loss evaluation (every
   recorded Adam step and the recorded L-BFGS steps) and, for inverse problems, the identified parameters;
4. compares the records value by value.

A configuration passes when every recorded loss term and parameter is identical. Configurations whose exported script
deliberately differs (for example when the export is Solve's full script for an advanced time-window method) are
listed with the reason.

```bash
python tools/check_export_equivalence.py --out equivalence_results --device cpu
```

!!! warning "Result for v1.6.0 pending"
    The v1.6.0 audit changed both generators. The check will be repeated on the final v1.6.0 release candidate, and its
    result is reported here only then.

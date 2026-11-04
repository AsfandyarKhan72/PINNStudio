# Export equivalence check (final, v1.6.0 candidate)

## Final run on the merged v1.6.0 code (`main` 2ea54c3, 2026-11-02/03)

Same tool and settings as the run below, on the commit merged into `main` (PySide6 interface). Records in
[`final_2ea54c3/`](final_2ea54c3/): `results.json`, `equivalence.log`, the environment, the commit and the job script.

- Commit `2ea54c363fd68d14476ad436684644d0496727ca`. Python 3.11.5, PyTorch 2.5.1+cu121 (run on the CPU),
  DeepXDE 1.15.0, NumPy 2.4.6, PySide6 6.11.3, Linux 6.8 (x86_64), 32 CPUs. Started 2026-11-02 19:52 MT, finished
  2026-11-03 00:59 MT (5 h 07 min), exit 0.
- Result: 93 example-mode configurations (the 64 examples in the mode each loads in, which includes DeepXDE's inverse-only examples, and 29 Inverse-mode entries, 3 of which do not exist and are skipped); **89 completed, all 89 bitwise identical**
  (loss histories, weight hashes after every training call, trained unknowns); 0 differ, 0 failed; 4 skipped (the
  same four as below).

| Kind | Count |
|---|---|
| Two independently generated scripts, bitwise identical | **88** (62 forward, 26 inverse) |
| Identical by construction: the export is Solve's script (time windows with backward compatibility) | **1**: 1D Cahn-Hilliard (Mattey & Ghosh), forward, whose Tuned setting at 2ea54c3 was the paper's bc-PINN |
| Skipped | 4 |
| Differ or failed | 0 |

**Separate check of the corrected Tuned preset (2026-11-03).** After the benchmark corrections, the Tuned
setting of 1D Cahn-Hilliard (Mattey & Ghosh) is `windows2` (38 time windows, SA-PINN, exact periodicity, no backward
compatibility), which the exporter writes with its own generator. Checked on its own, with the same tool and
settings, on the review commit `05c1047` (records in
[`ch1d_mattey_ghosh_windows2_05c1047/`](ch1d_mattey_ghosh_windows2_05c1047/)): **bitwise identical** — all 76
training calls (38 windows, Adam then L-BFGS), loss histories and weight hashes; Python 3.11.5, PyTorch 2.5.1+cu121
on the CPU, 45 min. This is reported separately and does not change the counts of the 93-configuration run above.

The by-construction configuration was identified as the only `export.py` of the run with the header "This problem uses time
windows with ...". The run below (af321c1) had the same Tuned setting for that example, so it also has 88
independent comparisons and one by construction.


`tools/check_export_equivalence.py` writes, for every example (forward, and inverse where it has one), the two
scripts PINNStudio generates from one configuration: the internal Solve script (`generate_script`) and the
exported standalone DeepXDE script (`generate_clean_script`). Both run with the example's Tuned settings, the
same random seed and the same device, with training shortened (every Adam phase to 200 steps, every L-BFGS
phase to 50 steps, IC pre-training and RAR rounds cut down the same way). An import hook records every
`model.train()` call at full precision: the loss history (every step and loss term), a SHA-256 of the network
weights after the call, and the trained unknowns.

- Tested commit: `af321c1` (integration/final; the release candidate `release/v1.6.0-prep` adds only the scipy
  dependency declaration). Environment: Python 3.11.5, PyTorch 2.6.0+cu124 (run on the CPU), DeepXDE 1.15.0,
  NumPy 2.4.6, Linux 6.8 (x86_64). Run 2026-10-30, about 4.6 h.
- Result: 93 example-mode configurations (the 64 examples in the mode each loads in, which includes DeepXDE's inverse-only examples, and 29 Inverse-mode entries, 3 of which do not exist and are skipped); 4 skipped; **89 completed, all 89 identical
  at full precision** (loss histories, weight hashes after every training call, and the values of the 33 sets of
  trained unknowns). No configuration differed and none failed. GUI smoke test passed (`smoke_gui.log`).
- Skipped: Helmholtz 2D hyperparameter search (a parameter-sweep example), and the three templates without an
  inverse mode (Cahn-Hilliard 1D of Wight & Zhao, Cahn-Hilliard 1D of Mattey & Ghosh, Poisson 3D cube with
  spherical cut-out) in the inverse list.
- Earlier run (`29c9dce`, before `6427cef`): 84 identical, 5 "differs", 4 skipped. The five differed only in the
  printed value of a trained unknown in the sixth or seventh significant digit (e.g. 2D Heat inverse, alpha
  0.591128 vs 0.591129) while the full-precision records were bitwise identical: the Solve script printed the
  unknowns with 6 significant digits and the export with 8. `6427cef` made both print 8; all five are now
  identical.

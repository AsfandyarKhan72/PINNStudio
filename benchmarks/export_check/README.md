# Export equivalence check (final, v1.6.0 candidate)

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
- Result: 93 example-mode configurations (64 forward, 29 inverse); 4 skipped; **89 completed, all 89 identical
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

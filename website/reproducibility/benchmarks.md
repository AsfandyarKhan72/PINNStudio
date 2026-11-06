# Benchmark collection

The benchmark collection lives in the repository's [`benchmarks/`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks) folder. Its
[README](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) lists every example with its reference settings and the best setting found, and
links each study to its write-up, plans and scripts.

## The studies

| Study | Question | Protocol | Scripts and plans |
|---|---|---|---|
| Reference runs | The error of every example with its own settings (DeepXDE's, the paper's, or the template's) | Each example as shipped, seed 2026 | `tools/tuning_study.py`, variant `base` |
| Tuning study | How far simple changes (more iterations, a larger network, more points, an L-BFGS phase) improve the examples with the largest errors; the selected setting becomes the example's **Tuned** preset | One change at a time from the example's settings, seed 2026; close calls repeated with seeds 1 and 2 | `tools/tuning_study.py`, `tools/make_tuning_page.py`; write-up [`tuning_study.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning_study.md) |
| Published methods | Whether PINNStudio reproduces the method of the paper each template comes from | The paper's network, points and training scheme as far as the paper states them (`pinnstudio/core/paper_settings.py`, one comment per problem with the paper's section and table) | `tools/tuning_study.py --variants plain,paper_*`; [`paper_comparison.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/paper_comparison.md) |
| Training-methods study | Which techniques (exact constraints, RBA, SA-PINN, causal training, MsFFN, time windows, bc-PINN, RAR) help on hard problems | Each example's own settings with one technique changed | `benchmarks/study_runner.py`, `benchmarks/plans/*.json`; [`training_methods.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/training_methods.md) |
| Controlled comparison | The same question under one protocol: the same network, points and iterations for every technique | Fixed per dimension; time windows with full or equal compute | `benchmarks/plans/fig7_v2.json`, `tools/make_methods_figure.py` |
| Export equivalence | Whether an exported script trains exactly the same model as Solve | See [Export equivalence](export-equivalence.md) | `tools/check_export_equivalence.py` |
| Performance | PINNStudio's overhead over a plain script, GPU against CPU, CPU thread scaling | Five problems, three repeats each | `benchmarks/perf_study.py`; [`performance.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/performance.md) |

## Error measures

The error of a forward run is the relative \(L_2\) error of the main output against the example's reference data, at
the reference points (see [Error measures](../theory/metrics.md)). For an inverse problem the accuracy is the relative
error of each learned parameter. Papers that report one error over all times are compared with that quantity.

## Seeds and selection

Runs use seed 2026 unless a study states more; repeated runs use seeds 1, 2 and 2026, and their median is reported.
Three selection rules are used, for different purposes:

- **The Tuned presets** (what **Settings › Tuned** loads, and the filled markers of the benchmark summary) use, for
  each example, the run with the lowest error at the last reference time (for Cahn–Hilliard problems the worse of
  the main output and μ; for inverse problems the median over seeds of each seed's largest parameter error), and of
  the runs within 25 % of that error the one that trains fastest; the starting settings are kept when they are
  within that margin and within 10 % of the fastest time. Candidates come from the tuning study, the
  training-methods best setups and the published-method runs. The rule is written out in
  [`benchmarks/README.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md).
- **The tuning-study page's "Chosen" column**
  ([`tuning_study.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning_study.md)) is the
  variant with the lowest final *training loss*, a choice that does not use the reference solution. It is not
  always the Tuned preset, and its problem files are labelled accordingly.
- **Helmholtz 2D – hyperparameter search** is its own search: PINNStudio's Bayesian search runs 50 configurations
  and selects the one with the lowest final training loss, without the reference solution; its error is measured
  only afterwards ([the search record](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/benchmarks/records/helmholtz_hpo_2026-10-25)).

Both the starting and the selected configuration of each example are recorded, so the effect of the selection can be separated from that of the problem.

## Known limitations of the comparisons

- Most results are single runs or medians of three seeds; differences smaller than the seed-to-seed spread are not
  meaningful.
- Times are wall-clock run times of the whole run (start-up, training and post-processing), measured on one GPU (an
  NVIDIA RTX 4090); they depend on hardware, drivers and other load.
- A paper's method is reproduced as far as the paper states it; settings a paper leaves out are filled in and marked
  in `paper_settings.py`.
- The selection rule uses the reference data it is evaluated on; the Tuned presets are a guide for similar problems,
  not tuned hyperparameters for new ones.

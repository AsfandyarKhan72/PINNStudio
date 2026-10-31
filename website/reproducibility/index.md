# Reproducibility

PINNStudio's examples double as a benchmark collection: every example ships with its settings and, for most, reference
data or an exact solution. This section explains what the benchmark collection contains, where the reference data come
from, how runs are made reproducible, and how to repeat the studies reported in the PINNStudio paper.

- [Benchmark collection](benchmarks.md): the studies, their protocols and records.
- [Reference data](reference-data.md): how each reference solution was produced and checked.
- [Seeds, precision and hardware](environment.md): what makes a run repeatable, and what does not.
- [Export equivalence](export-equivalence.md): the check that an exported script trains the same model as Solve.
- [Reproducing the paper](paper.md): the commands behind each table and figure.

!!! warning "Numerical results are being finalized for v1.6.0"
    The pre-release audit of v1.6.0 corrected code-generation and training behaviour (see the
    [release history](../about/changelog.md)) and regenerated the reference data of some templates. The affected
    benchmark runs are being repeated with the corrected code. Until that is complete, these pages describe the
    protocols and records but quote no numbers; the benchmark records in the repository are the source of the
    results, and the verified values will be added here with the v1.6.0 release.

## Versions

| | What it is | Used for |
|---|---|---|
| **v1.5.0** | The current public release (PyPI, tag `v1.5.0`) | Installing and using PINNStudio today |
| **Development code** | The repository source used for the benchmark runs; every record names the commit or branch it ran on | The benchmark collection and the paper's results |
| **v1.6.0** | The next release, being prepared from the audited development code | The published software version of the paper's results |

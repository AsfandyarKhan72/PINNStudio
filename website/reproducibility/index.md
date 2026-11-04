# Reproducibility

PINNStudio's examples double as a benchmark collection: every example ships with its settings and, for most, reference
data or an exact solution. This section explains what the benchmark collection contains, where the reference data come
from, how runs are made reproducible, and how to repeat the studies reported in the PINNStudio paper.

- [Benchmark collection](benchmarks.md): the studies, their protocols and records.
- [Reference data](reference-data.md): how each reference solution was produced and checked.
- [Seeds, precision and hardware](environment.md): what makes a run repeatable, and what does not.
- [Export equivalence](export-equivalence.md): the check that an exported script trains the same model as Solve.
- [Reproducing the paper](paper.md): the commands behind each table and figure.
- [Reporting your own experiment](own-experiments.md): what to keep from your own runs so that they can be
  reported and repeated, as a checklist.

These pages describe the protocols and records but quote no numbers; the benchmark records in the repository are the
source of the results.

## Versions

| | What it is | Used for |
|---|---|---|
| **PINNStudio 1.6.0** | The software these pages describe ([Installation](../getting-started/installation.md)) | Using PINNStudio; the software version the paper's results are reported for |
| **Repository commit** | The source a benchmark run used; every record names the commit or branch it ran on | Repeating a recorded run exactly |

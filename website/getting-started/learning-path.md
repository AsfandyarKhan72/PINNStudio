# Where should I start?

Three suggested orders through the documentation, depending on what you already know and what you want to do.
They are suggestions, not prerequisites: every page can be read on its own.

## New to PINNs or to PINNStudio

Work through these in order. The first four are the basis for everything else; the rest can be picked up as you
need them.

1. [Installation](installation.md): install the release with one command, or the development version from source.
2. [First run](first-run.md): load an example, press **Solve**, follow the Training Log and the Results tab.
3. [Tutorial 1: a first forward ODE](../tutorials/01-ode.md): outputs, equations, initial conditions, training
   phases, and the error against an exact solution.
4. [Tutorial 2: the 1D Burgers equation](../tutorials/02-burgers.md): the model tree of a time-dependent PDE,
   boundary-condition rows, reference files.
5. [Your own equations](../guide/equations.md): how to type a PDE, the derivative names, constants; then
   [Tutorial 18: your own PDE](../tutorials/18-own-problem.md) for the whole path with an equation of your own.
6. [Boundary conditions](../guide/boundary-conditions.md): the condition types and how **Where:** selects part
   of the boundary; [Tutorial 3](../tutorials/03-heat-2d.md) and [Tutorial 8](../tutorials/08-geometry.md) use
   them on 2D domains.
7. [Tutorial 6: inverse parameter estimation](../tutorials/06-inverse.md) and
   [Tutorial 7: importing observational data](../tutorials/07-data.md): unknown coefficients learned from data.
8. Advanced training: [Choosing a training strategy](../guide/choosing-a-strategy.md) first, then the tutorials on
   [exact constraints](../tutorials/09-exact.md), [RAR](../tutorials/10-rar.md),
   [RBA and SA-PINN](../tutorials/11-point-weights.md) and
   [causal and time-window training](../tutorials/12-causal-windows.md).
9. [Tutorial 17: error analysis](../tutorials/17-error-analysis.md) and the
   [Error Analysis](../guide/error-analysis.md) guide: how to judge a result against reference data.
10. Export and reproducibility: [Save, reopen and export](save-and-export.md),
    [Tutorial 16: standalone scripts](../tutorials/16-scripts.md) and
    [Reporting and reproducing your own experiment](../reproducibility/own-experiments.md).

Keep [Scope and limitations](../theory/limitations.md) in mind throughout: a small loss is not yet a correct
solution.

## Experienced with PINNs

If you know PINNs (for example from DeepXDE) and want to see what PINNStudio implements and how:

- [Feature reference](../reference/index.md): each feature's settings, the formulation the generated script uses,
  defaults, limitations and refused combinations; [All settings](../reference/settings-index.md) is a searchable
  table of every setting.
- [Theory](../theory/index.md): the formulations behind the methods, with references to the original papers.
- [Choosing a training strategy](../guide/choosing-a-strategy.md) and
  [Compatibility of methods](../reference/compatibility.md): which method targets which difficulty, and which
  combinations are supported, refused or untested.
- [Generated scripts](../reference/scripts.md): what Solve runs and what **Export as DeepXDE Script...** writes, to
  check the implementation or run it elsewhere.
- [Examples](../examples/index.md) and [Reproducibility](../reproducibility/index.md): the shipped benchmark
  problems, their settings and records, and how the studies of the PINNStudio paper are repeated.

## Developers

If you want to change PINNStudio itself:

- [Architecture and code generation](../developer/architecture.md): from the window to a running script and back.
- [Repository structure](../developer/structure.md): what is where, with links to the source of every feature.
- [Configuration schema](../developer/config.md): `PINNConfig`, its validation, and every field.
- [Tests](../developer/tests.md): the test suite and the CI workflows.
- [Contributing and documentation](../developer/contributing.md): the contribution process and how to work on this
  site; [Adding an example](../developer/add-example.md) and [Adding a training method](../developer/add-method.md)
  for the two most common changes.

# Help

Where to look when something does not work, or when you have a question the other sections do not answer.

- **[Troubleshooting](troubleshooting.md)**: PINNStudio does not start, the GPU is not used or runs out of memory,
  **Solve** refuses the settings, the loss is NaN, training does not converge, reference data or a checkpoint cannot
  be found, an exported script fails, training is slow. Each entry quotes the message PINNStudio shows and what to
  do about it.
- **[Diagnosing poor results](troubleshooting.md#diagnosing-poor-results)**: a run that finishes but whose result
  is not good enough, by symptom (error stays high while the loss falls, conditions not satisfied, inverse
  parameters that do not converge, late-time errors, ...), with the settings to look at and how to narrow the
  cause down.
- **[Compatibility of methods](../reference/compatibility.md)**: which training methods and problem types can be
  combined, which are refused and which are untested.
- **[Scope and limitations](../theory/limitations.md)**: what PINNStudio and PINNs in general can and cannot
  establish, and the known limitations of the implementation.
- **[FAQ](faq.md)**: short answers about what PINNStudio can do, hardware, results, reproducibility, citing and
  licensing.
- **[Installation](../getting-started/installation.md#common-installation-problems)**: problems while installing.

## Getting help

If the answer is not here:

1. Look through the [User guide](../guide/index.md) page of the setting concerned. The settings pages in PINNStudio
   also have a **?** button next to its title, which opens the matching page of the user guide.
2. Search the [GitHub issues](https://github.com/AsfandyarKhan72/PINNStudio/issues): someone may have had the same
   problem.
3. Open a new issue. [Reporting a problem](../getting-started/reporting-bugs.md) lists what to include so that the
   problem can be reproduced: the version, your system, the `.pinn.json` file, the Training Log and, if the run
   failed, the generated script.

Questions about the method itself (why a PINN does or does not converge on a given problem) are often answered in the
[Theory](../theory/index.md) section and in the papers listed under
[Scientific references](../about/references.md).

## A problem with this documentation

If a page is wrong, out of date, unclear or missing something, use
[Report a documentation problem](https://github.com/AsfandyarKhan72/PINNStudio/issues/new?template=documentation.md&labels=documentation) (also at the bottom of every page). It opens a GitHub issue with the
*documentation* label and asks for the page, what is wrong and your PINNStudio version. Each page also has
**Edit this page** and **View source of this page** buttons at the top right, for a direct correction through a
pull request ([Contributing and documentation](../developer/contributing.md)).

# Help

Where to look when something does not work, or when you have a question the other sections do not answer.

- **[Troubleshooting](troubleshooting.md)**: PINNStudio does not start, the GPU is not used or runs out of memory,
  **Solve** refuses the settings, the loss is NaN, training does not converge, reference data or a checkpoint cannot
  be found, an exported script fails, training is slow. Each entry quotes the message PINNStudio shows and what to
  do about it.
- **[FAQ](faq.md)**: short answers about what PINNStudio can do, hardware, results, reproducibility, citing and
  licensing.
- **[Installation](../getting-started/installation.md#common-installation-problems)**: problems while installing.

## Getting help

If the answer is not here:

1. Look through the [User guide](../guide/index.md) page of the setting concerned. The settings pages in PINNStudio
   also have a **?** button next to its title, which opens the matching section of the README on GitHub.
2. Search the [GitHub issues](https://github.com/AsfandyarKhan72/PINNStudio/issues): someone may have had the same
   problem.
3. Open a new issue. [Reporting a problem](../getting-started/reporting-bugs.md) lists what to include so that the
   problem can be reproduced: the version, your system, the `.pinn.json` file, the Training Log and, if the run
   failed, the generated script.

Questions about the method itself (why a PINN does or does not converge on a given problem) are often answered in the
[Theory](../theory/index.md) section and in the papers listed under
[Scientific references](../about/references.md).

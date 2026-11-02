# Security and supported versions

## Supported versions

PINNStudio is maintained by a small research group. Fixes, including security fixes, are made in the
development source and published in the next release; earlier releases are not patched. Please use the latest
release from PyPI (`pip install --upgrade pinnstudio`, or run the installer command again).

## Reporting a security problem

PINNStudio is a desktop application: it runs on your own computer, writes Python scripts from your settings and
runs them with your Python. It has no server component and does not send your data anywhere (it only checks PyPI
for a newer version). Problems that matter here are, for example, a problem file or data file that makes
PINNStudio run code you did not expect, or files written outside the folders you chose.

Please report such a problem privately by email to the maintainer (address in the README's *Contact* section)
rather than in a public issue, with the PINNStudio version and the steps or file that show it. You will get a
reply when the report has been read; how quickly it can be fixed depends on the problem.

Note that a problem file (`.pinn.json`) contains equations and formulas that PINNStudio turns into Python code
and runs, by design. Open problem files only from sources you trust, as you would a Python script.

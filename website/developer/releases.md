# Releases

!!! note
    Releases are made by the maintainers. Whether and when a version is released, and what it contains, is
    decided by them; a contributor's pull request is not released until they approve it. This page describes
    the procedure as it is recorded in the repository (its files, workflows and history); it is not a promise
    of a schedule.

The latest release is **v1.5.0** (git tag `v1.5.0`, on PyPI). The development version described on this site
will become **v1.6.0**.

## Version numbers

Versions are `MAJOR.MINOR.PATCH` and tags are `v` plus the version (`v1.4.0`, `v1.5.0`, ...). The version appears
in:

- `setup.py` (`version=...`): the number of the latest release, changed to the new number when it is published;
- `CITATION.cff`: in the development source it carries no version or release date; both are added for a release;
- `CHANGELOG.md`: one heading per released version.

The running application reads its own version from the installed package metadata
(`importlib.metadata.version("pinnstudio")`), and the "newer version available" notice compares it with the
latest version on PyPI.

## Changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Every change is recorded when it
is made, under **Unreleased**, in the sections *Added*, *Changed* and *Fixed*. At a release the *Unreleased*
entries move under a heading with the version number.

## Procedure

The steps below are those visible in the repository history and workflows for the past releases (for example
the commits leading to the `v1.5.0` tag):

1. **Tests on every platform.** The full suite runs on Windows, macOS and Linux for every push to the
   development branches (`platform-tests.yml`). Before a release the platform matrix (`platform-matrix.yml`) is
   started, by a commit whose message contains `[matrix]` or by hand: the suite and the window check on older and
   newer versions of each system and of Python, and the one-command installers ([Tests](tests.md)).
2. **Checks of the examples and results**, as the change requires: `tools/check_export_equivalence.py`,
   `tools/run_deepxde_examples.py`, and the benchmark pages and figures regenerated from the recorded runs
   (`tools/make_benchmarks_readme.py`, `tools/check_benchmarks_figures.py`).
3. **Version commit.** One commit sets the version in `setup.py` and `CITATION.cff` and moves the changelog
   entries under the new version heading.
4. **Merge into `main`.** The development branch is merged into `main` (for 1.5.0: "Merge branch 'cleanup' into
   main (release 1.5.0)").
5. **Tag.** An annotated tag `vX.Y.Z` with the message "PINNStudio X.Y.Z" on the merge commit.
6. **PyPI.** The sdist and wheel are built from the tagged commit (`python -m build`, [Building and
   packaging](packaging.md)) and uploaded to PyPI. The repository has no workflow that publishes to PyPI; the
   upload is done by a maintainer. From then on `pip install pinnstudio` and the one-command installers install
   the new version.
7. **Documentation.** Publishing a release does not deploy this site. The `docs.yml` workflow deploys to GitHub
   Pages only when a maintainer starts it by hand on `main` with deployment selected, and only once publication
   has been approved (repository variable `DOCS_PUBLISH_APPROVED`).

## After a release

The development source continues under a new *Unreleased* section of the changelog. Problems saved by the
released version must keep opening in later versions: a configuration field that is renamed or removed needs a
fallback when a saved problem is loaded (see the legacy fields in [Configuration schema](config.md)).

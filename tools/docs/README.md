# Documentation tools

Everything under `website/assets/screenshots/` is made by these scripts from the real application;
nothing is drawn or edited afterwards. Rerun them after any change to the interface.

Each run needs its own, new home folder (`_docs_home.py`): runs never share result folders and no tool
deletes anything. For the published images use a neutral home so no build-machine path appears:

```bash
export QT_QPA_PLATFORM=offscreen
# 1. interface pages and a short Burgers Solve for the Results tabs (a few minutes on a CPU)
PINNSTUDIO_DOCS_HOME=/home/researcher python tools/docs/make_screenshots.py website/assets/screenshots --solve
mv /home/researcher /home/researcher.step1      # keep step 1's run; the next step needs a new folder
# 2. user-guide pages; --run-folder: the Burgers run folder written by step 1
PINNSTUDIO_DOCS_HOME=/home/researcher python tools/docs/shots_guide.py website/assets/screenshots \
    --run-folder /home/researcher.step1/PINNStudio_Results/<run folder>
mv /home/researcher /home/researcher.step2
# 3. tutorials; --model: the Burgers model saved by step 1
PINNSTUDIO_DOCS_HOME=/home/researcher python tools/docs/shots_tutorials.py website/assets/screenshots \
    --solve --model /home/researcher.step1/PINNStudio_Results/<run folder>/<model>.pt
```

Each `PINNSTUDIO_DOCS_HOME` must be new or empty, otherwise the script stops; that is why the
folder is moved aside between steps (so every image shows the same neutral path). Without it a private
`/tmp` folder is used, which is fine for previews but shows a `/tmp` path in the status bar.

Other generators: `make_gallery.py` (example gallery pages from the shipped examples),
`make_config_reference.py` (the configuration reference) and `make_source_links.py` (below).

## Source-code links

Every feature page ends with *Implementation and source code*: the entry points, helpers and tests behind the
feature, linked on GitHub. They come from one mapping, `source_map.py` (feature id → file, function or class, role,
note). `make_source_links.py` checks that every file exists and every function, class or method is defined in it
(Python's parser; `MainWindow._build_config` is a method), and writes one snippet per feature into
`website/_snippets/source/<id>.md` (not built as pages: `exclude_docs` in `mkdocs.yml`), which the page includes with
`--8<-- "_snippets/source/<id>.md"`; `_overview.md` lists them all on the developer page *Repository structure*.

```bash
python tools/docs/make_source_links.py           # after changing source_map.py or renaming code it names
python tools/docs/make_source_links.py --check   # CI (docs.yml) and tests/test_source_links.py
python tools/docs/make_source_links.py --table   # coverage: feature, code, tests, pages that include it
```

Links name the file and the function, never line numbers. The base of every generated link is set once, in
`mkdocs.yml`: `extra.repo_blob` (the repository's `blob/` address) and `extra.source_ref` (now the branch `main`).
**At a release, `extra.source_ref` changes to the release tag (for example `v1.6.0`) once the tag exists**; then rerun
`make_source_links.py` and `make_gallery.py` (the gallery's repository links use the same two values) and commit.
The edit buttons use `edit_uri` (the branch the site's sources are edited on, now `docs/official-site`; `main` after
the merge, see `DEPLOY.md`). *Report a documentation problem* in the footer opens `extra.docs_issue_url`, a new issue
from `.github/ISSUE_TEMPLATE/documentation.md`.

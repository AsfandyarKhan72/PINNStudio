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

Other generators: `make_gallery.py` (example gallery pages from the shipped examples) and
`make_config_reference.py` (the configuration reference).

# The old project page (`docs/`) during integration

The public address https://asfandyarkhan72.github.io/PINNStudio/ is served from `main`'s `docs/` folder
(Settings → Pages → Deploy from a branch, see `tools/docs/DEPLOY.md`). So that merging the v1.6.0 work into
`main` does not change the public page, `docs/` is kept exactly as it is on `main` (the v1.5.0 page, MIT).

The integration branch had updated that page (application icon in the navigation and as favicon, a cover image
for link previews, new screenshots, and a license line that said GPLv3). Those changes are kept in the history:

    git show 351c21f:docs/index.html                 # the updated page
    git diff 351c21f~0 -- docs/                      # or: git checkout 351c21f -- docs/ to restore all of it

`tools/make_app_icon.py` and `tools/make_social_preview.py` write `docs/assets/pinnstudio_icon.png` and
`docs/assets/social_preview.png` when run. The documentation website in `website/` replaces this page when its
deployment is approved; the page stays in `main` as the rollback until then (DEPLOY.md, *Rollback*).

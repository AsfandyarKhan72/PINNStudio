# Publishing the documentation site

The documentation site is published at **https://asfandyarkhan72.github.io/PINNStudio/**. It is the MkDocs
Material site built from `website/` and `mkdocs.yml` by `.github/workflows/docs.yml` and deployed with the
"GitHub Actions" Pages source. It is the project's only website; the earlier single-page site that was served from
`main`'s `docs/` folder has been retired (see *Rollback*).

## Current state (checked 2026-11-03)

- **Live:** the MkDocs site (generator `mkdocs-1.6.1, mkdocs-material-9.7.7`), deployed by the Documentation
  workflow from `main`. Deployments to the `github-pages` environment: `b05ef50` (2026-11-03 18:26 UTC), `2ea54c3`
  (2026-11-03 04:25 UTC). `pages-build-deployment` (the branch source) last ran on 2026-11-03 02:52 on `2ea54c3`
  and did not run for later `main` commits that changed `docs/index.html`, so the branch source is no longer used.
- **Pages source:** "GitHub Actions" (inferred: `actions/deploy-pages` succeeds only with that source; the setting
  itself is visible only to a maintainer, under Settings → Pages).
- `site_url` in `mkdocs.yml` is `https://asfandyarkhan72.github.io/PINNStudio/`, so pages, assets, search and the
  404 page work under `/PINNStudio/`; `edit_uri` is `edit/main/website/`.

## Why nothing deploys by accident

The deploy job runs only if **all** of these hold:

1. the workflow is started by hand (Actions → Documentation → Run workflow), and
2. on the `main` branch, and
3. with the input `deploy` ticked, and
4. the repository variable `DOCS_PUBLISH_APPROVED` is `true`, and
5. the Pages source is "GitHub Actions".

Keep `DOCS_PUBLISH_APPROVED` at `false` between approved deployments (Settings → Secrets and variables → Actions →
Variables), so that each deployment is a deliberate act. Pushes, pull requests, releases and tags never deploy:
pull requests that touch `website/**`, `mkdocs.yml` or the workflow run the strict build and the link checks only.

## Deploying (each time, after approval)

1. Merge the documentation changes into `main` through a pull request (it runs `mkdocs build --strict`, the
   source-link and settings-index checks and the link check).
2. Set `DOCS_PUBLISH_APPROVED` to `true`.
3. Actions → Documentation → Run workflow → branch `main`, tick `deploy` → Run. The `deploy` job prints the URL.
4. Check the live site: the home page, one page of each section, search, an equation page, an example page, the
   404 page (any wrong path), on a desktop and a phone browser.
5. Set `DOCS_PUBLISH_APPROVED` back to `false`.

## Rollback

- **To an earlier version of the site:** revert the documentation commit(s) on `main` (`git revert`, through a pull
  request), then deploy as above. Every deployment is built from `main`, so the site always matches a commit of
  `main`.
- **To the retired single-page site (only if ever needed):** its files are in the history; restore them with
  `git checkout 78bf1a9 -- docs/` (the last commit with them; 78bf1a9 has the corrected funding line), commit, and
  set Settings → Pages → Source to "Deploy from a branch", `main`, `/docs`. Switching the source back to "GitHub
  Actions" and deploying returns to the MkDocs site.

## Releases

At each release: set `extra.source_ref` in `mkdocs.yml` to the release tag once it exists (so the source-code links
point at the released code; rerun `tools/docs/make_source_links.py` and `tools/docs/make_gallery.py`), update the
installation page and the changelog page, merge, and deploy as above. See "Documentation" in the release checklist
(`website/developer/releases.md`).

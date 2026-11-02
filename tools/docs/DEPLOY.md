# Publishing the documentation site

The site is published at **https://asfandyarkhan72.github.io/PINNStudio/**, the same address as today's
single-page site. Nothing below has been done yet; every step needs the maintainers' approval.

## What serves that address today

- `main` contains `docs/index.html` (one hand-written landing page) and `docs/assets/` (its images).
- `main` has no Pages workflow and the repository has no `gh-pages` branch, so the page is served by
  **Settings → Pages → Build and deployment → Source: Deploy from a branch, `main`, folder `/docs`**.
  (Inferred from the repository; the Pages settings themselves can only be read by a maintainer.
  Check this first, step 1 below.)
- Links to the address found in the repository: the README's "Website" link (to the root) and the landing
  page's own `og:image` (`/PINNStudio/assets/interface.png`). No other page paths are linked from
  the repository, `setup.py`, `pyproject.toml` or `CITATION.cff`.

## What will serve it

The MkDocs Material site built from `website/` by `.github/workflows/docs.yml`, deployed with GitHub's
"GitHub Actions" Pages source. `site_url` in `mkdocs.yml` is
`https://asfandyarkhan72.github.io/PINNStudio/`, so all pages, assets, search and the 404 page work
under the `/PINNStudio/` base path (checked by serving the built site under `/PINNStudio/` locally and
by the workflow's link check).

There is one site: the old landing page is replaced, not run alongside. `docs/index.html` stays in
the repository until the new site has been live for a while, because it is the rollback.

## Why nothing can deploy by accident

The deploy job runs only if **all** of these hold:

1. the workflow is started by hand (Actions → Documentation → Run workflow), and
2. on the `main` branch, and
3. with the input `deploy` ticked, and
4. the repository variable `DOCS_PUBLISH_APPROVED` is `true` (it does not exist yet), and
5. the Pages source is "GitHub Actions" (with "Deploy from a branch" the deploy step is refused by GitHub).

Pushes, pull requests, releases and tags never deploy: the workflow has no `release` or `push: main`
trigger, and on a push to `docs/official-site` it only builds, checks links and uploads a preview.

## Deployment, step by step (after approval)

1. **Check the current settings.** Settings → Pages: note the source (expected: branch `main`, `/docs`)
   and whether a custom domain is set (expected: none). Take a screenshot for the record.
2. **Merge.** Merge `docs/official-site` into `main` through a pull request. The pull request runs the
   strict build and link check; merging changes nothing on the live site (the source is still the
   `/docs` folder, which the merge does not touch). In the same pull request (or right after it), change
   `edit_uri` in `mkdocs.yml` from `edit/docs/official-site/website/` to `edit/main/website/`, so that
   **Edit this page** and **View source of this page** open the files on `main`.
3. **Allow deployment.** Settings → Secrets and variables → Actions → Variables → New repository
   variable `DOCS_PUBLISH_APPROVED` = `true`.
4. **Switch the source.** Settings → Pages → Source: **GitHub Actions**. From this moment until step 5
   finishes, the address shows GitHub's 404 page (normally a few minutes), so do 4 and 5 together.
5. **Deploy.** Actions → Documentation → Run workflow → branch `main`, tick `deploy` → Run. The `deploy`
   job prints the published URL.
6. **Check the live site.** Home page, one page of each section, search, an equation page, a gallery page,
   the 404 page (any wrong path), on a desktop and a phone browser.
7. **Turn deployment off again** if you want each later deployment to be a deliberate act: set
   `DOCS_PUBLISH_APPROVED` to `false` (re-enable it for the next release).

## Rollback

- **Back to the old landing page (minutes):** Settings → Pages → Source: Deploy from a branch, `main`,
  `/docs`. The old `docs/index.html` is still in `main`, so it is served again as before.
- **Back to an earlier version of the new site:** Actions → Documentation → Run workflow with `deploy` on
  a commit of `main` that had the earlier site (or revert the docs commit on `main` and deploy).
- Keep `docs/index.html` and `docs/assets/` in `main` until the new site is accepted; remove them in a
  later, separate commit.

## Later releases

Each release: update the version banner (`extra.version_label`, `extra.released_version` in
`mkdocs.yml`), set `extra.source_ref` to the release tag once it exists (so the source-code links point at the
released code; rerun `tools/docs/make_source_links.py` and `tools/docs/make_gallery.py`), update the installation
page and the changelog page, merge, then run step 5 again. See
"Documentation" in the release checklist (`website/developer/releases.md`).

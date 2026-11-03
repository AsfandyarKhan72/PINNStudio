"""MkDocs hooks for the documentation site (mkdocs.yml: hooks).

Pages written by a generator have no "Edit this page" button: an edit to the page file would be overwritten the next
time the generator runs. The page says which tool writes it; the source to change is that tool's input.
"""

# page path (relative to website/) or folder prefix -> the tool that writes it
GENERATED = {
    "examples/": "tools/docs/make_gallery.py",
    "reference/settings-index.md": "tools/docs/make_settings_index.py",
    "developer/config.md": "tools/docs/make_config_reference.py",
}


def is_generated(src_uri):
    return any(src_uri == k or (k.endswith("/") and src_uri.startswith(k)) for k in GENERATED)


def on_page_context(context, page, config, nav):
    if is_generated(page.file.src_uri):
        page.edit_url = None
    return context

#!/usr/bin/env python3
"""The "All settings" page of the documentation site, built from the settings tables of the other pages.

    python tools/docs/make_settings_index.py            # write website/reference/settings-index.md
    python tools/docs/make_settings_index.py --check    # exit 1 if the page is out of date
    python tools/docs/make_settings_index.py --verify   # also compare the tables with the code (report only)

Sources. Every Markdown table in website/guide/*.md and website/reference/*.md whose header has a label column
("Label", "GUI label", "Group / label", "Labels", "Main settings (labels)") is read. The other columns are found by
their header: "Meaning", "Default..." or "Recommended value", "Range..." or "Choices", and the configuration column
("Config field...", "Config fields", "Saved as", "Key in `custom_bc_json`"). A first column that is not the label
("Dialog", "Shape", "Tab") qualifies the label; a blank cell repeats the one above. Cells are copied as written,
with their relative links rewritten for the index page.

One row per setting. A row's key is its configuration field: the first backticked name in the configuration cell
that is a field of PINNConfig (read from pinnstudio/core/config.py with Python's parser), or for a
`custom_bc_json`, `scheduler_phases`, `inverse_variables_json` or `inverse_obs_files_json` entry that field and its
key. Rows of different pages with the same key and the same location are one setting, however their labels are
worded; a row whose label says it lives on another page ("(Results page)") joins the row of its field wherever
that is. Rows of one page stay separate (one page may list the same field under several labels, e.g. the decay
settings of each schedule). Rows without a field are merged by their label text. The text shown comes from the Feature reference row when there is one
(it has the range), otherwise from the User guide row; every page that documents the setting is linked, with the
anchor of the heading above its table.

Where the setting is in the window comes from the guide page the row is on (one guide page per model-tree page),
from the label's "(... page)" note, or, for a setting only in the Feature reference, from LOCATION below.

--verify (nothing is written) reports, for review by hand:
  * bold labels that do not occur as text in pinnstudio/ui/*.py,
  * rows whose default does not mention the PINNConfig default of their field (often correct: the window's
    default for a new problem can differ from the dataclass default that an old file without the key gets),
  * merged rows whose pages give different defaults.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(ROOT, "website")
CONFIG = os.path.join(ROOT, "pinnstudio", "core", "config.py")
UI_DIR = os.path.join(ROOT, "pinnstudio", "ui")
OUT_REL = "reference/settings-index.md"
OUT = os.path.join(WEB, OUT_REL)

# guide pages in model-tree order: (page, location of its settings)
GUIDE = [
    ("guide/problem.md", "Problem page"),
    ("guide/equations.md", "Equations page"),
    ("guide/geometry.md", "Geometry page"),
    ("guide/points.md", "Collocation points page"),
    ("guide/initial-conditions.md", "Initial conditions page"),
    ("guide/boundary-conditions.md", "Boundary conditions page"),
    ("guide/inverse.md", "Inverse problem page"),
    ("guide/network.md", "Neural network page"),
    ("guide/training.md", "Training page"),
    ("guide/loss-weights.md", "Loss weights page"),
    ("guide/adaptive.md", "Adaptive training page"),
    ("guide/sweeps.md", "Parameter sweep page"),
    ("guide/results.md", "Results page"),
    ("guide/export.md", "Results page"),
    ("guide/error-analysis.md", "Error Analysis dialog"),
    ("guide/restore.md", "Restore & visualize page"),
    ("guide/settings.md", "Settings menu"),
]
CALLBACKS = "Training page: Training Callbacks"
PLOT_SETTINGS = "Plot settings dialog"
# order of the sections on the index page
SECTIONS = ["Problem page", "Equations page", "Geometry page", "Collocation points page", "Initial conditions page",
            "Boundary conditions page", "Inverse problem page", "Neural network page", "Training page", CALLBACKS,
            "Loss weights page", "Adaptive training page", "Parameter sweep page", "Results page", PLOT_SETTINGS,
            "Error Analysis dialog", "Restore & visualize page", "Settings menu"]
# Feature reference pages in navigation order; location of a table's settings when no guide row has the same
# field: (page, heading of the table or None for any) -> location
REFERENCE = [
    "reference/forward.md", "reference/inverse.md", "reference/equations.md", "reference/coupled.md",
    "reference/geometry.md", "reference/boundary-conditions.md", "reference/initial-conditions.md",
    "reference/exact-constraints.md", "reference/networks.md", "reference/optimizers.md",
    "reference/loss-weights.md", "reference/pretraining.md", "reference/rar.md", "reference/point-weights.md",
    "reference/causal.md", "reference/time-windows.md", "reference/observations.md", "reference/metrics.md",
    "reference/plots.md", "reference/checkpoints.md", "reference/scripts.md", "reference/sweeps.md",
]
LOCATION = {
    ("reference/forward.md", None): "Problem page",
    ("reference/inverse.md", None): "Inverse problem page",
    ("reference/equations.md", None): "Equations page",
    ("reference/coupled.md", None): "Equations page",
    ("reference/geometry.md", None): "Geometry page",
    ("reference/boundary-conditions.md", None): "Boundary conditions page",
    ("reference/initial-conditions.md", None): "Initial conditions page",
    ("reference/exact-constraints.md", None): "Initial conditions page",
    ("reference/networks.md", None): "Neural network page",
    ("reference/optimizers.md", "Training phases"): "Training page",
    ("reference/optimizers.md", "L-BFGS options"): "Training page",
    ("reference/optimizers.md", "Callbacks"): CALLBACKS,
    ("reference/loss-weights.md", None): "Loss weights page",
    ("reference/pretraining.md", "IC pre-training"): "Training page",
    ("reference/pretraining.md", "IC-guided learning"): "Adaptive training page",
    ("reference/rar.md", None): "Adaptive training page",
    ("reference/time-windows.md", None): "Adaptive training page",
    ("reference/point-weights.md", None): "Loss weights page",
    ("reference/causal.md", None): "Loss weights page",
    ("reference/observations.md", None): "Inverse problem page",
    ("reference/metrics.md", None): "Error Analysis dialog",
    ("reference/plots.md", "Plot type"): "Results page",
    ("reference/plots.md", "Plot settings"): PLOT_SETTINGS,
    ("reference/plots.md", "Export Solution"): "Results page",
    ("reference/checkpoints.md", "Model Checkpoint"): CALLBACKS,
    ("reference/checkpoints.md", "Restore"): "Restore & visualize page",
    ("reference/sweeps.md", None): "Parameter sweep page",
}
PAGE_NOTE = re.compile(r"\(([A-Z][\w &]*?) (page|dialog)\)")   # "(Results page)" in a label cell
# the keys of a Training Phases entry, which guide/training.md lists without the `scheduler_phases` prefix
PHASE_KEYS = {"optimizer", "iterations", "lr", "decay_type", "decay_p1", "decay_p2", "loss"}
JSON_LIST_FIELDS = ("custom_bc_json", "scheduler_phases", "inverse_variables_json", "inverse_obs_files_json")
LABEL_HEADERS = ("label", "gui label", "group / label", "labels", "main settings (labels)")


# ---------------------------------------------------------------- reading


def config_defaults(path=CONFIG):
    """{field: default source text} of the PINNConfig dataclass."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PINNConfig")
    out = {}
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            v = n.value
            if isinstance(v, ast.Call) and getattr(v.func, "id", "") == "field":
                kw = {k.arg: k.value for k in v.keywords}
                v = kw.get("default") or getattr(kw.get("default_factory"), "body", None)
            out[n.target.id] = ast.get_source_segment(src, v) if v is not None else ""
    return out


def split_row(line):
    """Cells of a Markdown table row; a | inside `code` or escaped as \\| does not split."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, in_code, i = [], "", False, 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur += "\\|"
            i += 2
            continue
        if c == "`":
            in_code = not in_code
        if c == "|" and not in_code:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += c
        i += 1
    cells.append(cur.strip())
    return cells


def _plain_heading(text):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[*`]", "", text).strip()


def slugify(text, used):
    """The anchor MkDocs (Python-Markdown's toc) gives a heading, with its _1, _2 suffixes for repeats."""
    import unicodedata
    value = unicodedata.normalize("NFKD", _plain_heading(text)).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    slug = re.sub(r"[-\s]+", "-", value)
    base, n = slug, 0
    while slug in used:
        n += 1
        slug = f"{base}_{n}"
    used.add(slug)
    return slug


def read_tables(page):
    """[(heading text, anchor, header cells, [row cells])] for the settings tables of one page, and its title."""
    with open(os.path.join(WEB, page), encoding="utf-8") as f:
        lines = f.read().split("\n")
    out, used, heading, anchor, title, fence = [], set(), "", "", "", False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            text = re.sub(r"\s*\{[^}]*\}\s*$", "", m.group(2))
            slug = slugify(text, used)
            if len(m.group(1)) == 1 and not title:
                title = _plain_heading(text)
            else:
                heading, anchor = _plain_heading(text), slug
        if (not fence and line.startswith("|") and i + 1 < len(lines)
                and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1])):
            header = split_row(line)
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            if any(h.lower() in LABEL_HEADERS for h in header):
                out.append((heading, anchor, header, rows))
            i = j
            continue
        i += 1
    return title, out


def columns(header):
    """{role: column index} for label, qualifier, meaning, default, range, config."""
    cols = {}
    for k, h in enumerate(header):
        hl = h.lower()
        if hl in LABEL_HEADERS:
            cols.setdefault("label", k)
        elif hl.startswith("meaning"):
            cols["meaning"] = k
        elif hl.startswith("default") or hl.startswith("recommended"):
            cols["default"] = k
        elif hl.startswith("range") or hl.startswith("choices"):
            cols["range"] = k
        elif hl.startswith("config field") or hl.startswith("saved as") or hl.startswith("key in"):
            cols["config"] = k
    if cols.get("label", 0) > 0:
        cols["qualifier"] = 0
    return cols


def rewrite_links(cell, page):
    """Relative links in a cell of `page`, rewritten for the index page."""
    src_dir = os.path.dirname(page)
    out_dir = os.path.dirname(OUT_REL)

    def fix(m):
        target = m.group(2)
        if re.match(r"[a-z]+:", target) or target.startswith("/"):
            return m.group(0)
        path, _, frag = target.partition("#")
        full = page if not path else os.path.normpath(os.path.join(src_dir, path))
        rel = os.path.relpath(full, out_dir).replace(os.sep, "/")
        return f"{m.group(1)}({rel}{'#' + frag if frag else ''})"

    return re.sub(r"(\]\s?)\(([^)\s]+)\)", fix, cell)


def bold_texts(cell):
    return [b.strip() for b in re.findall(r"\*\*(.+?)\*\*", cell)]


def config_key(cell, fields, json_entry=None):
    """The first configuration field named in a config cell (JSON-list field plus its key when present)."""
    names = re.findall(r"`([A-Za-z_][\w.]*)`", cell)
    if json_entry and names:
        if names[0] in fields and not (json_entry == "scheduler_phases" and names[0] in PHASE_KEYS):
            return names[0]
        return f"{json_entry}.{names[0]}"
    for k, n in enumerate(names):
        if n in fields:
            if n in JSON_LIST_FIELDS and k + 1 < len(names) and names[k + 1] not in fields:
                return f"{n}.{names[k + 1]}"
            return n
    return None


# ---------------------------------------------------------------- collecting


def collect():
    fields = config_defaults()
    rows, by_key, by_field, titles = [], {}, {}, {}
    pages = [(p, loc) for p, loc in GUIDE] + [(p, None) for p in REFERENCE]
    for page, page_loc in pages:
        title, tables = read_tables(page)
        titles[page] = title
        for heading, anchor, header, trs in tables:
            cols = columns(header)
            if "label" not in cols:
                continue
            cfg_header = header[cols["config"]] if "config" in cols else ""
            json_entry = next((f for f in ("custom_bc_json", "scheduler_phases") if f in cfg_header), None)
            loc = page_loc
            if loc is None:
                loc = next((v for (p, h), v in LOCATION.items() if p == page and h and heading.startswith(h)),
                           LOCATION.get((page, None)))
                if loc is None:
                    raise SystemExit(f"{page}, table under '{heading}': no location in LOCATION")
            qualifier, cursor = "", None
            for r in trs:
                r = r + [""] * (len(header) - len(r))
                get = lambda role: r[cols[role]] if role in cols else ""
                if "qualifier" in cols and get("qualifier"):
                    qualifier = get("qualifier")
                label = get("label")
                cfg = get("config")
                key_field = config_key(cfg, fields, json_entry) if cfg else None
                if json_entry and "own field" in cfg:
                    key_field = config_key(cfg, fields)
                if page == "guide/training.md" and cfg:
                    first = re.findall(r"`([A-Za-z_]\w*)`", cfg)[:1]
                    if first and first[0] in PHASE_KEYS:
                        key_field = "scheduler_phases." + first[0]
                note = PAGE_NOTE.search(label)
                row_loc = loc
                if note:
                    row_loc = note.group(1) + " " + note.group(2)
                    if row_loc == "Training page" and key_field and key_field.startswith("cb_"):
                        row_loc = CALLBACKS
                elif page == "guide/training.md" and key_field and (
                        key_field.startswith("cb_") or key_field.startswith("training_monitors")):
                    row_loc = CALLBACKS
                if row_loc not in SECTIONS:
                    raise SystemExit(f"{page}: unknown location '{row_loc}' (label {label})")
                bold = " ".join(bold_texts(label)) or re.sub(r"\s+", " ", label)
                link = (page, anchor, heading)
                entry = {
                    "label": label, "qualifier": qualifier if "qualifier" in cols else "",
                    "meaning": get("meaning"), "default": get("default"), "range": get("range"),
                    "config": cfg if cfg else "not saved", "field": key_field, "loc": row_loc,
                    "links": [link], "src": page, "reference": page.startswith("reference/"),
                    "defaults_seen": [(page, get("default"))],
                }
                qual = entry["qualifier"] if page != "guide/settings.md" else ""
                plain = re.sub(r"\s+", " ", label)
                key = (key_field, bold, qual) if key_field else (None, plain, row_loc, qual)
                target = by_key.get(key)
                # the same field documented on another page (with the same location, or a cross-reference
                # such as "(Results page)") is the same setting, even when the label is worded differently
                summary = entry["reference"] and "qualifier" in cols
                other = by_field.get(key_field) if key_field else None
                if (target is None and other is not None and page not in {l[0] for l in other["links"]}
                        and (note or other["loc"] == row_loc)):
                    target = other
                if target is None:
                    by_key[key] = entry
                    if key_field and key_field not in by_field:
                        by_field[key_field] = entry
                    # placed after the previous row of the same table, so a table's rows stay together
                    pos = rows.index(cursor) + 1 if cursor is not None and cursor["loc"] == row_loc else len(rows)
                    rows.insert(pos, entry)
                    cursor = entry
                    continue
                if target["loc"] == row_loc:
                    cursor = target
                if link not in target["links"]:
                    target["links"].append(link)
                target["defaults_seen"].append((page, get("default")))
                if entry["reference"] and not target["reference"] and not note and not summary:
                    # the Feature reference row has the range: show its text
                    for k in ("label", "meaning", "default", "range", "config"):
                        target[k] = entry[k]
                    target["reference"], target["src"] = True, page
    return rows, titles, fields


# ---------------------------------------------------------------- writing


def render(rows, titles):
    def cell(text, src):
        text = rewrite_links(text, src).replace("\n", " ")
        return text if text.strip() else "—"

    out = [
        "---",
        "search:",
        "  boost: 2",
        "---",
        "",
        "<!-- Generated by tools/docs/make_settings_index.py from the settings tables of website/guide/ and",
        "     website/reference/. Do not edit: change the source table and run the script again. -->",
        "",
        "# All settings",
        "",
        "Every setting of the window that the User guide and the Feature reference describe in a settings table, "
        "grouped by where it is: the model-tree page, the dialog or the menu. Use the search box (or your "
        "browser's find) with the label as it appears in the window, or with the configuration field saved in "
        "the `.pinn.json` file.",
        "",
        "Each row gives the label, a short meaning, the default, the allowed values and the configuration field. "
        "What the setting does in detail, its mathematics, how it interacts with other settings and its "
        "limitations are on the pages linked under **Explained in**. Which methods can be combined is on "
        "[Compatibility of methods](compatibility.md).",
        "",
        "!!! note \"Defaults\"",
        "    **Default** is the value the window shows for a new problem (a loaded example sets its own values). "
        "When an older problem file lacks a field, the field takes the `PINNConfig` default listed on "
        "[Configuration schema](../developer/config.md#field-reference); the two can differ.",
        "",
        "This page is generated by `tools/docs/make_settings_index.py` from the settings tables of the other "
        "pages, so it changes when they do.",
        "",
    ]
    for sec in SECTIONS:
        sec_rows = [r for r in rows if r["loc"] == sec]
        if not sec_rows:
            continue
        out += [f"## {sec}", "", "| Setting | Meaning | Default | Allowed values | Config field | Explained in |",
                "|---|---|---|---|---|---|"]
        for r in sec_rows:
            label = r["label"]
            if r["qualifier"]:
                label = f"{r['qualifier']}: {label}"
            links = []
            for page, anchor, _h in r["links"]:
                href = os.path.relpath(page, os.path.dirname(OUT_REL)).replace(os.sep, "/")
                href += f"#{anchor}" if anchor else ""
                links.append(f"[{titles[page]}]({href})")
            out.append("| " + " | ".join([
                cell(label, r["src"]), cell(r["meaning"], r["src"]), cell(r["default"], r["src"]),
                cell(r["range"], r["src"]), cell(r["config"], r["src"]), ", ".join(links)]) + " |")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- verifying


def _norm_num(s):
    return re.sub(r"(?<=\d),(?=\d{3})", "", s)


def default_mentions(doc_default, cfg_default):
    d = _norm_num(doc_default).lower()
    c = cfg_default.strip().strip("\"'").lower()
    if c in ("true", "false"):
        return ("on" in d or "true" in d or "ticked" in d) if c == "true" else (
            "off" in d or "false" in d or "none" in d)
    try:
        val = float(c)
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", d.replace("10⁻⁸", "1e-8"))]
        return any(abs(n - val) <= 1e-12 * max(1, abs(val)) for n in nums) or (val == 0 and (
            "off" in d or "none" in d or "auto" in d or "blank" in d))
    except ValueError:
        pass
    if c in ("", "none"):
        return any(w in d for w in ("blank", "none", "empty", "—", "off", "automatic", "not set"))
    return c in d


def verify(rows, fields):
    ui = ""
    for name in sorted(os.listdir(UI_DIR)):
        if name.endswith(".py"):
            with open(os.path.join(UI_DIR, name), encoding="utf-8") as f:
                ui += f.read()
    ui_plain = ui.replace("\\u2026", "…")
    report = []
    for r in rows:
        for b in bold_texts(r["label"]):
            t = b.replace("*", "").replace("&#124;", "|").strip()
            if "…" in t or " n " in f" {t} " or " N " in f" {t} " or t.lower() in ("x", "y", "z"):
                continue
            if t not in ui_plain:
                report.append(f"label not in pinnstudio/ui: {t!r}  ({r['src']})")
        f = r["field"]
        if f and f in fields and r["default"]:
            if not default_mentions(r["default"], fields[f]):
                report.append(f"default differs: {f} = {fields[f]} in config.py; docs say {r['default']!r} "
                              f"({r['src']})")
        seen = {re.sub(r"\s+", " ", _norm_num(d)).strip().lower() for _p, d in r["defaults_seen"] if d}
        if len(seen) > 1:
            report.append(f"pages differ on the default of {r['field'] or r['label']}: "
                          + "; ".join(f"{p}: {d!r}" for p, d in r["defaults_seen"]))
    return report


def main(argv):
    rows, titles, fields = collect()
    page = render(rows, titles)
    if "--verify" in argv:
        for line in verify(rows, fields):
            print(line)
        print(f"{len(rows)} settings")
        return 0
    old = open(OUT, encoding="utf-8").read() if os.path.isfile(OUT) else ""
    if "--check" in argv:
        if old != page:
            print(f"website/{OUT_REL} is out of date: run tools/docs/make_settings_index.py")
            return 1
        print(f"settings index is up to date ({len(rows)} settings)")
        return 0
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote website/{OUT_REL} ({len(rows)} settings)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

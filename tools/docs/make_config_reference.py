#!/usr/bin/env python3
"""Configuration reference table for the documentation site.

    python tools/docs/make_config_reference.py [--check]

Reads pinnstudio/core/config.py (without importing it), lists every field of
the PINNConfig dataclass in source order -- name, type annotation, default
value -- and a one-line description taken from the source comments, and
writes the result into website/developer/config.md between the markers

    <!-- BEGIN GENERATED CONFIG TABLE -->
    <!-- END GENERATED CONFIG TABLE -->

Everything outside the markers is left as it is. The description of a field
is the first sentence of the comment directly above it (or of its inline
comment; in a comment that lists several fields as "- name: ...", that
field's own item); a field without a comment of its own gets the first
sentence of a nearby comment block that names it, and "--" when there is
none. The window control is found in MainWindow._build_config()
(pinnstudio/ui/main_window.py): the widget the field is read from, and the
label created with that widget; blank when it cannot be found. Short
one-line comments such as "# Neural network" become the table's section
headings. With --check nothing is written; the exit status is 1 when the page
is out of date.
"""
import ast
import io
import os
import re
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(ROOT, "pinnstudio", "core", "config.py")
WINDOW = os.path.join(ROOT, "pinnstudio", "ui", "main_window.py")
PAGE = os.path.join(ROOT, "website", "developer", "config.md")
BEGIN = "<!-- BEGIN GENERATED CONFIG TABLE -->"
END = "<!-- END GENERATED CONFIG TABLE -->"
MAX_DESC = 240          # characters of a description before it is cut at a word
HEADING_MAX_WORDS = 6   # a single-line comment this short above a field starts a section


def _comments_by_line(src):
    """{line number: comment text without '#'} for every comment token."""
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            out[tok.start[0]] = tok.string.lstrip("#").strip()
    return out


def _bullet_for(name, text):
    """The '- name: ...' item of a comment that describes several fields, or None."""
    for piece in re.split(r"(?:^|\s)-\s+(?=[A-Za-z_]\w*:)", text):
        if piece.startswith(name + ":"):
            return piece
    return None


def _first_sentence(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text, maxsplit=1)
    s = parts[0].strip()
    if len(s) > MAX_DESC:
        s = s[:MAX_DESC].rsplit(" ", 1)[0].rstrip(",;:(") + " ..."
    return s


def _cell(text):
    """Text for a Markdown table cell: pipes and angle brackets escaped."""
    return (text.replace("\\", "\\\\").replace("|", "\\|")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _code(text):
    text = text.replace("|", "\\|")
    tick = "``" if "`" in text else "`"
    pad = " " if tick == "``" else ""
    return f"{tick}{pad}{text}{pad}{tick}"


def _default_text(node, src):
    """Source text of a field's default; field(default_factory=lambda: X) gives X."""
    if node is None:
        return ""
    if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "field"):
        for kw in node.keywords:
            if kw.arg == "default_factory":
                v = kw.value
                if isinstance(v, ast.Lambda):
                    return ast.get_source_segment(src, v.body)
                return ast.get_source_segment(src, v) + "()"
            if kw.arg == "default":
                return ast.get_source_segment(src, kw.value)
    return ast.get_source_segment(src, node)


def collect(src):
    """[(kind, payload)] in source order: ("heading", text) or
    ("field", (name, type, default, description))."""
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PINNConfig")
    lines = src.splitlines()
    comments = _comments_by_line(src)
    fields = [n for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]

    # comment blocks: runs of comment-only lines, keyed by their last line
    blocks = []          # (first line, last line, text)
    i = 1
    while i <= len(lines):
        if lines[i - 1].strip().startswith("#"):
            j = i
            while j + 1 <= len(lines) and lines[j].strip().startswith("#"):
                j += 1
            blocks.append((i, j, " ".join(comments.get(k, "") for k in range(i, j + 1))))
            i = j + 1
        else:
            i += 1
    block_ending_at = {b[1]: b for b in blocks}

    out = []
    for f in fields:
        name = f.target.id
        ann = ast.get_source_segment(src, f.annotation)
        default = _default_text(f.value, src)
        desc = ""
        own = block_ending_at.get(f.lineno - 1)
        if own is not None:
            text = own[2]
            if own[0] == own[1] and len(text.split()) <= HEADING_MAX_WORDS and not text.endswith("."):
                out.append(("heading", text))
            else:
                desc = _first_sentence(_bullet_for(name, text) or text)
        end_line = getattr(f, "end_lineno", f.lineno)
        if not desc and end_line in comments and not lines[end_line - 1].strip().startswith("#"):
            desc = _first_sentence(comments[end_line])
        if not desc:
            # a nearby block (within 60 lines above) that names the field
            pat = re.compile(r"(?<![\w.])" + re.escape(name) + r"(?!\w)")
            for b in reversed([b for b in blocks if f.lineno - 60 <= b[1] < f.lineno]):
                if pat.search(b[2]):
                    hit = _bullet_for(name, b[2])
                    if hit is None:
                        pieces = re.split(r"(?<=[.!?;])\s+|\s+-\s+(?=\w+:)", b[2])
                        hit = next((p for p in pieces if pat.search(p)), "")
                    desc = _first_sentence(hit.lstrip("- ")) if hit else ""
                    break
        out.append(("field", (name, ann, default, desc)))
    return out


def window_labels(src):
    """{field: label text} from MainWindow._build_config(): `field=self.<widget>.value()` (or
    isChecked/currentData/currentText/text), and the label made together with that widget."""
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if "def _build_config(self)" in l)
    stop = next(i for i in range(start + 1, len(lines)) if re.match(r"    def ", lines[i]))
    body = "\n".join(lines[start:stop])
    widget_of = {}
    for fld, w in re.findall(r"\b(\w+)\s*=\s*self\.(\w+)\.(?:value|isChecked|currentData|currentText|text)\(\)", body):
        widget_of.setdefault(fld, w)
    out = {}
    for fld, w in widget_of.items():
        W = re.escape(w)
        found = None
        for pat in (rf'self\.{W}\s*=\s*QCheckBox\("([^"]+)"', rf'self\.{W}\s*=\s*_\w*row\("([^"]+)"',
                    rf'_num_row\(\w+,\s*"([^"]+)",\s*self\.{W}\)', rf'\("([^"]+)",\s*self\.{W}\)'):
            m = re.search(pat, src)
            if m:
                found = m.group(1)
                break
        if found is None:
            for i, l in enumerate(lines):
                if re.search(rf"self\.{W}\s*=\s*\w", l):
                    for j in range(max(0, i - 2), i):
                        m = re.search(r'addWidget\(QLabel\("([^"]+)"\)\)', lines[j])
                        if m:
                            found = m.group(1)
                    break
        if found:
            out[fld] = found.strip()
    return out


def render(items, labels=None):
    labels = labels or {}
    n_fields = sum(1 for k, _ in items if k == "field")
    rows = [f"{n_fields} fields, in the order they are defined in `pinnstudio/core/config.py`.", ""]
    header = ["| Field | Type | Default | Window control | Description (from the source comment) |",
              "|---|---|---|---|---|"]
    rows.append("### General")
    rows.append("")
    rows += header
    for kind, payload in items:
        if kind == "heading":
            rows += ["", f"### {payload[0].upper() + payload[1:]}", ""] + header
            continue
        name, ann, default, desc = payload
        lab = f"**{_cell(labels[name])}**" if name in labels else ""
        rows.append(f"| {_code(name)} | {_code(ann)} | {_code(default) if default else ''} | {lab} | "
                    f"{_cell(desc) if desc else '--'} |")
    return "\n".join(rows) + "\n"


def main(argv):
    check = "--check" in argv
    with open(CONFIG, encoding="utf-8") as fh:
        src = fh.read()
    with open(WINDOW, encoding="utf-8") as fh:
        labels = window_labels(fh.read())
    table = render(collect(src), labels)
    with open(PAGE, encoding="utf-8") as fh:
        page = fh.read()
    if BEGIN not in page or END not in page:
        sys.exit(f"{PAGE}: the markers {BEGIN} / {END} are missing")
    head, rest = page.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    new = head + BEGIN + "\n\n" + table + "\n" + END + tail
    if check:
        if new != page:
            print(f"{os.path.relpath(PAGE, ROOT)} is out of date: run tools/docs/make_config_reference.py")
            return 1
        print("configuration reference is up to date")
        return 0
    with open(PAGE, "w", encoding="utf-8") as fh:
        fh.write(new)
    print(f"wrote {os.path.relpath(PAGE, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

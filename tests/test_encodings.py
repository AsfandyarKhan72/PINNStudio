"""Text files and generated scripts are UTF-8 on every system. Windows'
default text encoding (cp1252) cannot hold the symbols PINNStudio writes
(check marks, L2 subscripts, ...): Restore failed with "UnicodeEncodeError:
'charmap' codec can't encode characters" writing its script."""
import ast
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [f for f in glob.glob(os.path.join(ROOT, "pinnstudio", "**", "*.py"), recursive=True)
         if not f.endswith("local_examples.py")]


def _calls():
    for path in FILES:
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                yield os.path.relpath(path, ROOT), node, name


def _mode(node):
    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        return node.args[1].value
    for k in node.keywords:
        if k.arg == "mode" and isinstance(k.value, ast.Constant):
            return k.value.value
    return None


def test_text_files_have_an_encoding():
    bad = []
    for path, node, name in _calls():
        if name not in ("open", "NamedTemporaryFile") or isinstance(node.func, ast.Attribute) and name == "open" \
                and getattr(node.func.value, "id", "") not in ("io",):
            continue
        mode = _mode(node)
        if name == "NamedTemporaryFile" and mode is None:
            continue                       # binary by default
        if mode and "b" in mode:
            continue
        if node.args and isinstance(node.args[0], ast.Attribute) and node.args[0].attr == "devnull":
            continue
        if not any(k.arg == "encoding" for k in node.keywords):
            bad.append(f"{path}:{node.lineno}")
    assert not bad, "open() without encoding= at " + ", ".join(bad)


def test_scripts_run_in_utf8_mode():
    bad = [f"{path}:{node.lineno}" for path, node, name in _calls()
           if name == "Popen" and not any(k.arg == "env" for k in node.keywords)
           and "shortcut" not in path]
    assert not bad, "subprocess.Popen without env=_pyexe.script_env() at " + ", ".join(bad)

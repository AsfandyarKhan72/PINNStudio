"""Reference files on Windows: "@REF/" in an example's settings becomes a path
with backslashes (C:\\Users\\...). Inside JSON and Python-literal settings it
must stay valid, and the generated scripts must still be valid Python."""
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core.codegen import generate_script, generate_clean_script

WIN_ROOT = "C:\\Users\\Ünïcode User\\AppData\\Roaming\\uv\\tools\\pinnstudio\\Lib\\site-packages\\reference_data"
WIN_SAVE = "C:\\Users\\new user\\PINNStudio_Results"   # \\U and \\n would be escapes if not escaped
JSON_FIELDS = ("custom_bc_json", "inverse_variables_json", "inverse_obs_files_json", "constants",
               "geom_custom_shapes_json", "scheduler_phases")


def _configs(monkeypatch):
    monkeypatch.setattr(dx.os, "sep", "\\")
    for ex in dx.EXAMPLES:
        for inv in ((False, True) if ex.inverse else (False,)):
            yield ex, inv, dx.build_config(ex, WIN_ROOT, inverse=inv)


def test_settings_parse_with_windows_paths(monkeypatch):
    n = 0
    for ex, inv, cfg in _configs(monkeypatch):
        for f in JSON_FIELDS:
            v = getattr(cfg, f, "")
            if isinstance(v, str) and v.strip() and v.strip()[0] in "[{":
                json.loads(v)
        files = ast.literal_eval(cfg.ea_files or "[]")
        for entry in files:
            if WIN_ROOT in entry[1]:
                n += 1
                assert entry[1].startswith(WIN_ROOT + "\\"), (ex.name, entry[1])
        for f in JSON_FIELDS:
            assert "@REF/" not in (getattr(cfg, f, "") or ""), (ex.name, f)
    assert n > 20


def test_generated_scripts_parse_with_windows_paths(monkeypatch):
    """Every path goes into a script as an escaped string literal: the raw
    path (single backslashes) never appears in the source, which would be a
    syntax error ("\\U...") or silently a different path ("\\n")."""
    for ex, inv, cfg in _configs(monkeypatch):
        cfg.save_dir = WIN_SAVE
        for gen in (generate_script, generate_clean_script):
            src = gen(cfg)
            try:
                ast.parse(src)
            except SyntaxError as e:
                raise AssertionError(f"{ex.name} (inverse={inv}), {gen.__name__}: {e}")
            for raw in (WIN_ROOT, WIN_SAVE):
                bad = [l.strip() for l in src.split("\n") if raw in l]
                assert not bad, f"{ex.name} (inverse={inv}), {gen.__name__}: unescaped path in {bad[0][:120]}"


def test_scripts_compile_before_python_312(monkeypatch, tmp_path):
    """Python 3.10 and 3.11 refuse a backslash inside an f-string expression,
    which 3.12 allows: a Windows path written into one broke the exported
    inverse script there. Checked with an older Python when one is installed."""
    import shutil
    import subprocess
    old = shutil.which("python3.11") or shutil.which("python3.10")
    if not old:
        return
    files = []
    for ex, inv, cfg in _configs(monkeypatch):
        cfg.save_dir = WIN_SAVE
        for gen in (generate_script, generate_clean_script):
            f = tmp_path / f"s{len(files)}.py"
            f.write_text(gen(cfg), encoding="utf-8")
            files.append((str(f), ex.name, inv, gen.__name__))
    code = ("import sys\nfor p in sys.argv[1:]:\n"
            "    try: compile(open(p, encoding='utf-8').read(), p, 'exec')\n"
            "    except SyntaxError as e: print(p, e)\n")
    out = subprocess.run([old, "-S", "-c", code] + [f[0] for f in files],
                         capture_output=True, text=True, timeout=300).stdout
    bad = [f"{n} (inverse={i}), {g}: {out.split(p, 1)[1].splitlines()[0]}" for p, n, i, g in files if p + " " in out]
    assert not bad, "\n".join(bad[:5])

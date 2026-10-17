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
    for ex, inv, cfg in _configs(monkeypatch):
        for gen in (generate_script, generate_clean_script):
            src = gen(cfg)
            try:
                ast.parse(src)
            except SyntaxError as e:
                raise AssertionError(f"{ex.name} (inverse={inv}), {gen.__name__}: {e}")

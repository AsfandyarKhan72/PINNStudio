"""The documentation's source-code links (tools/docs/make_source_links.py, from the mapping tools/docs/source_map.py):
every file and every function, class or method the mapping names still exists, and the snippets the site includes
(website/_snippets/source/) are up to date. A renamed or deleted function fails here, so the mapping is updated with
the code. The site is on the documentation branch; without website/ the checks that need it are skipped."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "docs", "make_source_links.py")

needs_tool = pytest.mark.skipif(not os.path.isfile(TOOL), reason="tools/docs/make_source_links.py is not in this checkout")
needs_site = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "website")),
                                reason="website/ is not in this checkout")


def _tool():
    spec = importlib.util.spec_from_file_location("make_source_links", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_tool
@needs_site
def test_every_name_exists():
    msl = _tool()
    features = msl.load_map()
    assert len(features) >= 20
    assert msl.validate(features) == []


@needs_tool
@needs_site
def test_snippets_up_to_date(capsys):
    assert _tool().main(["--check"]) == 0, capsys.readouterr().out


@needs_tool
def test_find():
    msl = _tool()
    msl.find("pinnstudio/core/codegen.py", "generate_clean_script")
    msl.find("pinnstudio/core/codegen.py", "generate_clean_script._result_plot_parts")     # nested function
    msl.find("pinnstudio/core/config.py", "PINNConfig.validate")                           # method
    msl.find("pinnstudio/ui/main_window.py", "MainWindow.INVERSE_AUTO_VARS")               # class attribute
    msl.find("pinnstudio/core/deepxde_examples.py", "EXAMPLES")                            # module assignment
    for path, name in [("pinnstudio/core/codegen.py", "no_such_function"),
                       ("pinnstudio/core/config.py", "PINNConfig.no_such_method"),
                       ("pinnstudio/no_such_file.py", None)]:
        with pytest.raises(LookupError):
            msl.find(path, name)


@needs_tool
def test_links_use_the_base_and_no_line_numbers():
    msl = _tool()
    feat = {"id": "x", "title": "X", "pages": [], "entries": [
        {"path": "pinnstudio/core/codegen.py", "symbol": "generate_script", "role": "entry point", "note": "n"},
        {"path": "tests/test_source_links.py", "symbol": None, "role": "test", "note": ""}]}
    text = msl.render(feat, "https://example.org/r/blob/", "v9.9.9")
    assert "(https://example.org/r/blob/v9.9.9/pinnstudio/core/codegen.py)" in text
    assert "`generate_script`" in text and "#L" not in text
    assert "release `v9.9.9`" in text and text.index("**Entry points**") < text.index("**Tests**")


@needs_tool
def test_without_the_site(monkeypatch, capsys):
    msl = _tool()
    monkeypatch.setattr(msl, "WEB", os.path.join(ROOT, "no-such-folder"))
    assert msl.main(["--check"]) == 0
    assert "not in this checkout" in capsys.readouterr().out

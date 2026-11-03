"""Examples -> Settings: Default (DeepXDE's own settings, or the paper's for a
template from a paper) and Tuned (PINNStudio's best, resources/tuned_settings.json)."""
import json

from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core import example_settings as es
from pinnstudio.core import paper_settings as ps

REF = "/ref"


def _dx(name, inverse=False):
    return dx.build_config(dx.BY_NAME[name], REF, inverse=inverse)


def test_default_keeps_a_deepxde_example():
    cfg = _dx("Euler beam")
    out, msg = es.settings_for(cfg, "Euler beam", False, es.DEFAULT)
    assert out is cfg and msg == ""


def test_tuned_applies_the_stored_change():
    table = es.tuned_table()
    assert table, "resources/tuned_settings.json is shipped"
    key = next(k for k, v in table.items() if k in dx.BY_NAME and v.get("forward", {}).get("variant") == "lbfgs")
    cfg = _dx(key)
    out, msg = es.settings_for(cfg, key, False, es.TUNED)
    phases = json.loads(out.scheduler_phases)
    assert any(p["optimizer"].startswith("l") for p in phases) and "L-BFGS" in msg


def test_tuned_without_an_entry_says_so():
    cfg = _dx("Poisson 1D – Dirichlet")
    out, msg = es.settings_for(cfg, "Poisson 1D – Dirichlet", False, es.TUNED)
    assert out is cfg and "already accurate" in msg


def test_every_stored_entry_applies():
    for key, ent in es.tuned_table().items():
        for mode in ("forward", "inverse"):
            t = ent.get(mode)
            if t and t.get("variant") not in es.VARIANT_TEXT and t.get("variant") not in ("base", None):
                raise AssertionError(f"{key} {mode}: unknown change {t.get('variant')}")


def test_paper_templates_have_their_papers_method():
    for key, v in ps.TEMPLATE_PAPER.items():
        assert (key, v) in ps.PAPER, key


def test_no_tuned_choice_where_default_is_best():
    """Tuned is offered only where it differs from Default."""
    from pinnstudio.core import example_settings as es
    assert es.has_tuned("1D Allen-Cahn (whole interval)", False)        # paper's method vs the template's own
    assert not es.has_tuned("Helmholtz equation over a 2D square domain: Hyper-parameter optimization", False) \
        or es.tuned_table().get("Helmholtz equation over a 2D square domain: Hyper-parameter optimization")

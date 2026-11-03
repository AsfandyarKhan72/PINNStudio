"""Export as DeepXDE script must train the same model as Solve. The short export script does not write
backward compatibility, a growing time range, IC-guided learning or per-window resampling yet, so with any
of them on the export is Solve's own script."""
import dataclasses

from pinnstudio.core import codegen
from pinnstudio.core.config import PINNConfig


def _ta(**kw):
    c = PINNConfig()
    return dataclasses.replace(c, time_adaptive=True, adapt_method="Time Adaptive", ta_transfer_learning=True, **kw)


def test_plain_time_windows_keep_the_short_export():
    c = _ta()
    assert codegen._ta_clean_unsupported(c) == []
    assert not codegen.generate_clean_script(c).startswith("# Exported from PINNStudio. This problem uses time windows")


def test_window_options_export_solves_own_script():
    for kw, word in ((dict(ta_backward=True), "backward"), (dict(ta_growing=True), "growing"),
                     (dict(ta_icgl_fraction=0.1), "guided"), (dict(ta_resample_rounds=2), "resampling")):
        c = _ta(**kw)
        out = codegen.generate_clean_script(c)
        assert word in out.splitlines()[0], kw
        assert out.endswith(codegen.generate_script(c)), kw


def test_short_export_uses_up_random_numbers_like_solve():
    """Solve's script builds the whole-interval model before the time windows; the short export does the same
    (unused), so the first window starts from the same points and weights and the two train the same model."""
    out = codegen.generate_clean_script(_ta())
    i_full, i_loop = out.index("_gt_full = dde.geometry.GeometryXTime("), out.index("for step_i, (t0, t1) in enumerate(intervals):")
    assert i_full < i_loop
    block = out[i_full:i_loop]
    assert "dde.data.TimePDE(_gt_full, pde, _c_full," in block and "_make_net(" in block
    compile(out, "export.py", "exec")

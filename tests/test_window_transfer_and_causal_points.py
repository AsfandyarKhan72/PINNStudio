"""Two training-method corrections (v1.6.0 review):

1. Time windows with resampling rounds and Transfer optimizer "Adam": the next window started from the network
   *before* the window's resampling rounds, while its initial condition came from the network *after* them. It now
   starts from the network after them (the rounds are first-order training too). Checked by running a tiny
   two-window script and comparing the transferred weights with the window's final weights.
2. Causal training recognised the training points by their number only, so another point set of the same size
   (RAR or resampling candidates, test points) was weighted too and overwrote the training weights. It now uses a
   checksum of the point set, refreshed before every step (a point resampler replaces the points mid-run).
"""
import dataclasses
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import codegen  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402

_SUM = "float(sum(float(_v.double().sum()) for _v in model_i.net.state_dict().values()))"


def _window_cfg(**kw):
    c = PINNConfig(time_adaptive=True, adapt_method="Time Adaptive", ta_num_steps=2, ta_transfer_learning=True,
                   ta_transfer_optimizer="adam", ta_resample_rounds=1, ta_resample_points=10,
                   ta_resample_candidates=200, ta_resample_iters=30, iterations=40, num_domain=200, num_boundary=20,
                   num_initial=20, num_test=50, layers=[2, 12, 12, 1], random_seed=1,
                   use_random_seed=True)
    return dataclasses.replace(c, **kw)


def _instrument(script):
    """Prints a checksum of the network at three places: after the resampling rounds (the window's final
    network), and right after the next window has loaded the transferred weights."""
    a = "        if _ta_rs_rounds > 0:\n            # the resampling rounds are first-order training too"
    assert a in script
    script = script.replace(a, f"        print('CHK after-resampling', step_i, repr({_SUM}), flush=True)\n" + a, 1)
    b = "                model_i.net.load_state_dict(_cur_state)\n"
    assert b in script
    script = script.replace(b, b + f"                print('CHK window-start', step_i, repr({_SUM}), flush=True)\n", 1)
    return script


def _run(cfg):
    with tempfile.TemporaryDirectory() as d:
        cfg = dataclasses.replace(cfg, save_dir=d)
        assert not cfg.validate(), cfg.validate()
        sp = os.path.join(d, "run.py")
        with open(sp, "w", encoding="utf-8") as f:
            f.write(_instrument(codegen.generate_script(cfg)))
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=ROOT)
        p = subprocess.run([sys.executable, sp], cwd=d, capture_output=True, text=True, timeout=600, env=env)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        return {(m.group(1), int(m.group(2))): float(m.group(3))
                for m in re.finditer(r"CHK (\S+) (\d+) (\S+)", p.stdout)}


def test_next_window_starts_from_the_network_after_resampling():
    chk = _run(_window_cfg())
    assert ("after-resampling", 0) in chk and ("window-start", 1) in chk, chk
    assert chk[("window-start", 1)] == chk[("after-resampling", 0)], chk


def test_lbfgs_transfer_also_starts_from_the_final_network():
    chk = _run(_window_cfg(ta_transfer_optimizer="lbfgs"))
    assert chk[("window-start", 1)] == chk[("after-resampling", 0)], chk


def test_causal_weights_recognise_the_training_points_by_a_checksum():
    c = PINNConfig(causal_training=True, causal_anneal=True)
    s = codegen.generate_script(c)
    assert "def _causal_is_train(x):" in s and 'if not _causal_is_train(x):' in s
    assert '_CAUSAL["n"]' not in s                       # the count-only test is gone
    assert "def on_batch_begin(self):" in s.split("class _CausalMonitor")[1].split("def on_batch_end")[0]
    compile(s, "run.py", "exec")


def test_causal_point_check_distinguishes_sets_of_the_same_size():
    """The checksum helpers as written into the script: the training points are recognised; another set of the
    same size, the same points reordered, and the set before a resampler replaced it are not."""
    import numpy as np
    import torch
    s = codegen.generate_script(PINNConfig(causal_training=True))
    src = s[s.index("def _causal_sums(X):"):s.index("def _causal_wrap(_f):")]
    ns = {"torch": torch, "_CAUSAL": {"fp": None}}
    exec(src, ns)
    rng = np.random.default_rng(0)
    train = rng.uniform(-1, 1, (500, 2)).astype(np.float32)
    other = rng.uniform(-1, 1, (500, 2)).astype(np.float32)        # e.g. RAR candidates of the same size
    assert ns["_causal_is_train"](torch.as_tensor(train))           # nothing marked yet: weigh, as before
    ns["_CAUSAL"]["fp"] = ns["_causal_sums"](train)
    assert ns["_causal_is_train"](torch.as_tensor(train))
    assert not ns["_causal_is_train"](torch.as_tensor(other))
    assert not ns["_causal_is_train"](torch.as_tensor(train[::-1].copy()))
    assert not ns["_causal_is_train"](torch.as_tensor(train[:400]))
    ns["_CAUSAL"]["fp"] = ns["_causal_sums"](other)                 # a resampler replaced the points: re-marked
    assert ns["_causal_is_train"](torch.as_tensor(other)) and not ns["_causal_is_train"](torch.as_tensor(train))

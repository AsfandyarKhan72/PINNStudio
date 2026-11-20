"""The network designer's domain: the design model, widths, legality of connections, presets, layout.

No Qt and no PyTorch here (see test_netdesign_torch.py for the network itself).
"""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio.core.netdesign import design as D  # noqa: E402
from pinnstudio.core.netdesign import presets as P  # noqa: E402
from pinnstudio.core.netdesign import shapes as S  # noqa: E402
from pinnstudio.core.netdesign.layers import LAYERS, check_params, stack_width  # noqa: E402
from pinnstudio.core.netdesign.params import ParamError  # noqa: E402

IN, OUT = ["x", "t"], ["u", "v"]


def _base():
    return P.fnn([16, 16])


def test_round_trip_and_hash_ignores_layout():
    d = _base()
    again = D.check(json.loads(D.to_json(d)))
    assert again == d
    moved = copy.deepcopy(d)
    moved["blocks"][0]["x"] += 500
    assert D.design_hash(moved) == D.design_hash(d)
    changed = copy.deepcopy(d)
    changed["blocks"][1]["layers"][0]["params"]["units"] = 17
    assert D.design_hash(changed) != D.design_hash(d)
    assert D.check("") == D.empty()


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["blocks"].append(dict(d["blocks"][0])), "used twice"),
    (lambda d: d["blocks"][0].update(kind="widget"), "unknown block kind"),
    (lambda d: d["blocks"][1].update(colour="red"), "unknown setting 'colour'"),
    (lambda d: d["edges"].append({"src": "s1", "dst": "nope"}), "not there"),
    (lambda d: d["edges"].append({"src": "s1", "dst": "s1"}), "connected to itself"),
    (lambda d: d["edges"].append({"src": "h1", "dst": "s1"}), "cannot start at"),
    (lambda d: d["edges"].append(dict(d["edges"][0])), "there twice"),
    (lambda d: d["blocks"][1]["layers"].append({"type": "dense", "params": {"units": 0}}), "outside 1 to"),
    (lambda d: d["blocks"][1]["layers"].append({"type": "conv", "params": {}}), "unknown layer"),
    (lambda d: d["blocks"][1].update(merge="xor"), "merge must be one of"),
    (lambda d: d["blocks"].append({"id": "po1", "kind": "port_out"}), "inside a group"),
    (lambda d: d["blocks"].append({"id": "sh1", "kind": "shared", "ref": "h1"}), "can share a stack"),
    (lambda d: d.update(colour=1), "unknown key"),
])
def test_validation_names_the_problem(mutate, message):
    d = copy.deepcopy(_base())
    mutate(d)
    with pytest.raises(ParamError, match=message):
        D.check(d)


def test_group_needs_ports():
    d = _base()
    d["blocks"].append({"id": "g1", "kind": "group", "inner": {"blocks": [{"id": "pi1", "kind": "port_in"}],
                                                             "edges": []}})
    with pytest.raises(ParamError, match="exactly one Port out"):
        D.check(d)


def test_widths_and_problems():
    d = _base()
    widths, problems = S.infer(d, IN, OUT)
    assert widths == {"in1": 2, "s1": 16, "h1": 2} and problems == []
    d2 = copy.deepcopy(d)
    d2["blocks"][0]["inputs"] = ["x", "q"]
    msgs = [m for _p, _b, m in S.problems(d2, IN, OUT)]
    assert any("no input 'q'" in m for m in msgs)
    d3 = copy.deepcopy(d)
    d3["blocks"][2]["outputs"] = ["u"]
    assert any("no head gives output 'v'" in m for _p, _b, m in S.problems(d3, IN, OUT))
    d4 = copy.deepcopy(d)
    d4["edges"] = []
    msgs = [m for _p, _b, m in S.problems(d4, IN, OUT)]
    assert any("nothing is connected" in m for m in msgs) and any("goes nowhere" in m for m in msgs)


def test_merge_rules():
    d = P.resnet(8, 1)
    assert S.problems(d, IN, OUT) == []
    bad = copy.deepcopy(d)
    lift = next(b for b in bad["blocks"] if b["title"] == "Lift")
    lift["layers"][0]["params"]["units"] = 9          # the skip now adds 9 to 8
    msgs = [m for _p, _b, m in S.problems(bad, IN, OUT)]
    assert any("add needs inputs of one width" in m for m in msgs), msgs


@pytest.mark.parametrize("src, dst, reason", [
    ("s1", "s1", "its own output"),
    ("h1", "s1", "where values end"),
    ("s1", "in1", "where values start"),
    ("in1", "s1", "connected already"),
    ("s1", "in1x", "no such block"),
])
def test_can_connect_reasons(src, dst, reason):
    d = _base()
    base = S.problems(d, IN, OUT)
    ok, why = S.can_connect(d, (), src, dst, "in", base, IN, OUT)
    assert not ok and reason in why


def test_can_connect_loop_and_breaking_another_block():
    d = P.resnet(8, 1)
    base = S.problems(d, IN, OUT)
    skip = next(b["id"] for b in d["blocks"] if b["kind"] == "merge")
    lift = next(b["id"] for b in d["blocks"] if b["title"] == "Lift")
    ok, why = S.can_connect(d, (), skip, lift, "in", base, IN, OUT)
    assert not ok and "loop" in why
    # the Input (width 2) into the add merge (width 8): breaks the merge itself
    ok, why = S.can_connect(d, (), "in1", skip, "in", base, IN, OUT)
    assert not ok and "one width" in why
    # an extra input into the Lift stack widens it: fine for the stack, legal
    d2 = P.fnn([4])
    d2["blocks"].append({"id": "in2", "kind": "input", "inputs": ["x"]})
    d2 = D.check(d2)
    ok, why = S.can_connect(d2, (), "in2", "s1", "in", S.problems(d2, IN, OUT), IN, OUT)
    assert ok, why


def test_baseline_problems_do_not_block():
    d = _base()
    d["blocks"].append({"id": "m1", "kind": "merge", "merge": "add"})     # unconnected: a problem already
    d = D.check(d)
    base = S.problems(d, IN, OUT)
    assert base
    ok, why = S.can_connect(d, (), "in1", "m1", "in", base, IN, OUT)
    assert ok, why


def test_presets_are_valid_for_every_problem_shape():
    from pinnstudio.core.config import PINNConfig
    c = PINNConfig()
    c.layers = [3, 32, 32, 32, 3]
    for ins, outs in ((["x", "y", "t"], ["u", "v", "p"]), (["t"], ["y1"]), (["x"], ["u"])):
        for name in P.PRESETS:
            if name == "stmsffn" and (len(ins) < 2 or "t" not in ins):
                with pytest.raises(ValueError):
                    P.make(name, c, ins, outs)
                continue
            d = P.make(name, c, ins, outs)
            assert S.problems(d, ins, outs) == [], (name, ins)
            xs = [b["x"] for b in d["blocks"]]
            heads = [b["x"] for b in d["blocks"] if b["kind"] == "head"]
            inputs = [b["x"] for b in d["blocks"] if b["kind"] == "input"]
            assert min(inputs) == min(xs) and max(heads) == max(xs), "inputs left, heads right"


def test_from_settings_follows_the_network_type():
    from pinnstudio.core.config import PINNConfig
    c = PINNConfig()
    c.layers = [2, 20, 20, 1]
    for kind, expect in (("FNN", "FNN"), ("PFNN", "PFNN"), ("MsFFN", "MsFFN"), ("STMsFFN", "STMsFFN")):
        c.network_type = kind
        assert P.from_settings(c, ["x", "t"], ["u"])["name"] == expect


def test_layer_widths():
    assert stack_width([{"type": "fourier", "params": {"features": 5}}], 3) == 10
    assert stack_width([{"type": "periodic", "params": {"harmonics": 2}}], 3) == 12
    assert stack_width([{"type": "custom", "params": {"width": "7"}}], 3) == 7
    assert stack_width([{"type": "custom", "params": {}}], 3) == 3
    with pytest.raises(ParamError):
        check_params("dense", {"units": "many"})
    assert set(LAYERS) >= {"dense", "activation", "layer_norm", "scale_shift", "fourier", "periodic",
                           "adaptive", "custom"}


def test_new_ids_are_unique_across_groups():
    d = _base()
    d["blocks"].append({"id": "g1", "kind": "group", "inner": {
        "blocks": [{"id": "pi1", "kind": "port_in"}, {"id": "s2", "kind": "stack"}, {"id": "po1", "kind": "port_out"}],
        "edges": [{"src": "pi1", "dst": "s2"}, {"src": "s2", "dst": "po1"}]}})
    d = D.check(d)
    assert D.new_id(d, "stack") == "s3"
    assert D.find(d, "s2")[0] == ("g1",)

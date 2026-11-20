"""The width every block gives, what keeps a design from running, and whether a connection is legal.

`infer` never stops at the first problem, so every block that cannot run is marked with its reason.
`can_connect` answers the canvas while a connection is dragged: cheap structural checks first, then the
design with the trial connection is inferred again, and only a problem it did not have before (the
`baseline`) makes the connection illegal; one on another block reads "it would break …". Pure (VIEW_SAFE).
"""
from __future__ import annotations

import copy

from pinnstudio.core.netdesign import design as D
from pinnstudio.core.netdesign.layers import stack_width
from pinnstudio.core.netdesign.params import ParamError


def combine(widths: list, how: str):
    """(width, problem) of several connections into one in-port."""
    if not widths:
        return None, ""
    if any(w is None for w in widths):
        return None, ""          # an upstream block already has its own problem
    if len(widths) == 1 or how == "concatenate":
        return sum(widths), ""
    if len(set(widths)) != 1:
        return None, f"{how} needs inputs of one width; they are {', '.join(map(str, widths))}"
    return widths[0], ""


class _Infer:
    def __init__(self, d, input_names, output_names):
        self.d = d
        self.inputs = list(input_names)
        self.outputs = list(output_names)
        self.widths = {}           # block id -> width it gives (None: cannot be worked out)
        self.in_widths = {}        # block id -> {port: width it takes}
        self.problems = []         # (path, block id, message); block id "" for the design as a whole

    def bad(self, path, bid, msg):
        self.problems.append((tuple(path), bid, msg))

    def level(self, level, path, port_widths=None):
        order, cycle = D.topological(level)
        for bid in cycle:
            self.bad(path, bid, "it is part of a loop")
            self.widths[bid] = None
        blocks = {b["id"]: b for b in level["blocks"]}
        for bid in order:
            b = blocks[bid]
            self.widths[bid] = self.block(level, b, path, port_widths or {})
        for b in level["blocks"]:
            if D.has_out(b["kind"]) and not D.outgoing(level, b["id"]) and b["kind"] != "port_in":
                if b["kind"] == "input" or self.widths.get(b["id"]) is not None:
                    self.bad(path, b["id"], "its output goes nowhere: connect it, or delete it")

    def port_in(self, level, b, path):
        out = {}
        for port in D.ports_of(b):
            edges = D.incoming(level, b["id"], port)
            if not edges:
                self.bad(path, b["id"], "nothing is connected to it" if port == "in"
                         else f"nothing is connected to its port {port!r}")
                out[port] = None
                continue
            w, msg = combine([self.widths.get(e["src"]) for e in edges], b.get("merge", "concatenate"))
            if msg:
                self.bad(path, b["id"], msg)
            out[port] = w
        self.in_widths[b["id"]] = out
        return out

    def block(self, level, b, path, port_widths):
        kind = b["kind"]
        if kind == "input":
            names = self.inputs if b["inputs"] == [D.ALL] else b["inputs"]
            missing = [n for n in names if n not in self.inputs]
            if missing:
                self.bad(path, b["id"], f"the network has no input {missing[0]!r}; it has "
                         f"{', '.join(self.inputs) or 'none'}")
                return None
            if not names:
                self.bad(path, b["id"], "the network has no inputs")
                return None
            return len(names)
        if kind == "port_in":
            return port_widths.get(b["port"])
        ins = self.port_in(level, b, path)
        w = ins.get("in")
        if kind == "merge" or kind == "port_out":
            return w
        if kind == "stack":
            if w is None:
                return None
            try:
                return stack_width(b["layers"], w)
            except ParamError as exc:
                self.bad(path, b["id"], str(exc))
                return None
        if kind == "head":
            names = self.outputs if b["outputs"] == [D.ALL] else b["outputs"]
            missing = [n for n in names if n not in self.outputs]
            if missing:
                self.bad(path, b["id"], f"there is no output {missing[0]!r}; the outputs are "
                         f"{', '.join(self.outputs)}")
                return None
            if b["direct"] and w is not None and w != len(names):
                self.bad(path, b["id"], f"with 'values as they come' it needs {len(names)} value(s), and "
                         f"{w} reach it")
            return len(names)
        if kind == "saved":
            if w is not None and w != b["width_in"]:
                self.bad(path, b["id"], f"the saved module takes {b['width_in']} value(s); {w} reach it")
            return b["width_out"]
        if kind == "group":
            self.level(b["inner"], tuple(path) + (b["id"],), ins)
            out = next(x for x in b["inner"]["blocks"] if x["kind"] == "port_out")
            return self.widths.get(out["id"])
        if kind == "shared":
            ref = next((x for x in level["blocks"] if x["id"] == b["ref"]), None)
            if ref is None:
                return None
            ref_in = self.in_widths.get(ref["id"])
            if ref_in is None:   # the original comes later in the order: work out what it takes
                ref_in = {p: combine([self.widths.get(e["src"]) for e in D.incoming(level, ref["id"], p)],
                                     ref.get("merge", "concatenate"))[0] for p in D.ports_of(ref)}
            mine = ins
            if ref["kind"] == "saved":
                expect = {"in": ref["width_in"]}
            elif ref["kind"] == "group":
                gports = D.ports_of(ref)
                if len(gports) != 1:      # a shared block has one in-port
                    self.bad(path, b["id"], f"it calls {D.label(ref)}, which has {len(gports)} Ports in; "
                                            "a shared copy can call a group with one Port in only")
                    return None
                expect = {"in": ref_in.get(gports[0])}
            else:
                expect = ref_in
            for port, want in expect.items():
                got = mine.get(port)
                if got is not None and want is not None and got != want:
                    self.bad(path, b["id"], f"it calls {D.label(ref)}, which takes {want} value(s); {got} reach it")
                    return None
            if ref["kind"] == "stack":
                w0 = mine.get("in") if mine.get("in") is not None else expect.get("in")
                try:
                    return stack_width(ref["layers"], w0) if w0 is not None else None
                except ParamError:
                    return None
            if ref["kind"] == "saved":
                return ref["width_out"]
            return self.widths.get(ref["id"])
        return None

    def run(self):
        top = self.d
        self.level(top, ())
        kinds = [b["kind"] for b in top["blocks"]]
        if "input" not in kinds:
            self.bad((), "", "the design needs an Input block")
        if "head" not in kinds:
            self.bad((), "", "the design needs a Head block for the outputs")
        covered = {}
        for b in top["blocks"]:
            if b["kind"] != "head":
                continue
            names = self.outputs if b["outputs"] == [D.ALL] else b["outputs"]
            for n in names:
                if n in covered:
                    self.bad((), b["id"], f"output {n!r} is also given by {D.label(covered[n])}")
                else:
                    covered[n] = b
        if "head" in kinds:
            for n in self.outputs:
                if n not in covered:
                    self.bad((), "", f"no head gives output {n!r}")
        return self.widths, self.problems


def infer(d: dict, input_names, output_names):
    """(widths by block id, problems as (path, block id, message)) of a checked design."""
    return _Infer(d, input_names, output_names).run()


def problems(d: dict, input_names, output_names) -> list:
    return infer(d, input_names, output_names)[1]


def _kind_text(kind: str) -> str:
    name = {"input": "an Input", "port_in": "a Port in", "head": "a Head", "port_out": "a Port out"}
    return name.get(kind, f"a {kind}")


def can_connect(d: dict, path, src: str, dst: str, port: str, baseline, input_names, output_names):
    """(legal, reason) for a new connection src → dst.port at the level `path`. `baseline` is the set of
    problems the design had when the drag started (they do not block other connections)."""
    level = D.level_of(d, tuple(path))
    blocks = {b["id"]: b for b in level["blocks"]}
    s, t = blocks.get(src), blocks.get(dst)
    if s is None or t is None:
        return False, "no such block here"
    if src == dst:
        return False, "a block cannot take its own output"
    if not D.has_out(s["kind"]):
        return False, f"{D.label(s)} gives nothing: {_kind_text(s['kind'])} is where values end"
    if not D.has_in(t["kind"]):
        return False, f"{D.label(t)} takes nothing: {_kind_text(t['kind'])} is where values start"
    if port not in D.ports_of(t):
        return False, f"{D.label(t)} has no port {port!r}"
    if any(e["src"] == src and e["dst"] == dst and e["port"] == port for e in level["edges"]):
        return False, "they are connected already"
    if D.reaches(level, dst, src):
        return False, "it would close a loop"
    trial = copy.deepcopy(d)
    D.level_of(trial, tuple(path))["edges"].append({"src": src, "dst": dst, "port": port})
    base = {(tuple(p), b, m) for p, b, m in (baseline or ())}
    new = [(p, b, m) for p, b, m in problems(trial, input_names, output_names)
           if (tuple(p), b, m) not in base and "nothing is connected" not in m and "goes nowhere" not in m]
    if not new:
        return True, ""
    p, b, m = new[0]
    if b == dst and tuple(p) == tuple(path):
        return False, m
    _path, other = D.find(trial, b) if b else (None, None)
    return False, f"it would break {D.label(other) if other else 'the design'}: {m}"

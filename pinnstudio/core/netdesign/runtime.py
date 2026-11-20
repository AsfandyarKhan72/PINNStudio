"""The PyTorch code of a designed network, as the generated scripts embed it.

`resolve` checks a design against the problem's input and output names and writes into a copy what
running it needs: each Input block's input indices, each Head's output indices, every block's input and
output widths, and each level's order. `net_source` gives the script's source: the resolved design as a
literal (`_DESIGN`), the design's own code (custom layers), and `_DesignedNet`, a `dde.nn.NN` that builds
one module per block and runs them in order. It applies DeepXDE's input and output transforms as the
other networks do (`_MsFFN` in codegen.py), so exact constraints, periodic features and rescaling work
unchanged. The same source serves Solve, the exported script and every restore, so they build identical
parameters.
"""
from __future__ import annotations

import copy
import json

from pinnstudio.core.netdesign import design as D
from pinnstudio.core.netdesign import shapes as S
from pinnstudio.core.netdesign.params import ParamError


def resolve(d, input_names, output_names) -> dict:
    """A checked copy of design `d` with what running it needs; ParamError naming the first problem."""
    d = D.check(d)
    widths, problems = S.infer(d, input_names, output_names)
    if problems:
        path, bid, msg = problems[0]
        _p, b = D.find(d, bid) if bid else (None, None)
        where = f"{D.label(b)}: " if b else ""
        raise ParamError(f"Network design: {where}{msg}")
    inf = S._Infer(d, input_names, output_names)
    inf.run()
    out = copy.deepcopy(d)
    for _path, level in D.walk_levels(out):
        level["_order"] = D.topological(level)[0]
        for b in level["blocks"]:
            b["_out"] = inf.widths.get(b["id"])
            b["_in"] = {p: w for p, w in (inf.in_widths.get(b["id"]) or {}).items()}
            if b["kind"] == "input":
                names = list(input_names) if b["inputs"] == [D.ALL] else b["inputs"]
                b["_idx"] = [list(input_names).index(n) for n in names]
            elif b["kind"] == "head":
                names = list(output_names) if b["outputs"] == [D.ALL] else b["outputs"]
                b["_idx"] = [list(output_names).index(n) for n in names]
    out["_n_in"] = len(input_names)
    out["_n_out"] = len(output_names)
    return out


def summary_lines(resolved: dict) -> list:
    """A readable list of the design's blocks, for a comment above the network in the script."""
    lines = []

    def level(lv, indent):
        blocks = {b["id"]: b for b in lv["blocks"]}
        for bid in lv["_order"]:
            b = blocks[bid]
            srcs = ", ".join(e["src"] for e in lv["edges"] if e["dst"] == bid)
            what = b["kind"]
            if b["kind"] == "stack":
                what = " → ".join(_layer_text(sp) for sp in b["layers"]) or "identity"
            elif b["kind"] in ("merge", ):
                what = b["merge"]
            elif b["kind"] == "shared":
                what = f"calls {b['ref']} again (same weights)"
            elif b["kind"] == "input":
                what = f"inputs {b['_idx']}"
            elif b["kind"] == "head":
                what = f"outputs {b['_idx']}" + (" (as they come)" if b["direct"] else " (linear)")
            title = f" '{b['title']}'" if b["title"] else ""
            lines.append(f"{indent}{bid}{title}: {what}" + (f"  <- {srcs}" if srcs else "")
                         + f"  [width {b['_out']}]")
            if b["kind"] == "group":
                level(b["inner"], indent + "    ")
    level(resolved, "")
    return lines


def _layer_text(sp):
    p = sp["params"]
    t = sp["type"]
    if t == "dense":
        return f"Dense({p['units']}, {p['activation']})"
    if t == "fourier":
        return f"Fourier({p['features']}, σ={p['sigma']:g})"
    if t == "periodic":
        return f"Periodic(K={p['harmonics']}, L={p['period']:g})"
    if t == "scale_shift":
        return f"×{p['scale']:g}+{p['shift']:g}"
    if t == "adaptive":
        return f"Adaptive({p['activation']})"
    if t == "custom":
        return f"{p['class_name']}(...)"
    return t


_RUNTIME = r'''
def _dn_act(name, default):
    _n = default if name in (None, "", "inherit") else name
    if str(_n).lower() == "linear":
        return lambda _x: _x
    return dde.nn.activations.get(str(_n).lower())


def _dn_init(name, default):
    return dde.nn.initializers.get(default if name in (None, "", "inherit") else name)


class _DnLinear(torch.nn.Module):
    def __init__(self, w_in, units, act, init, bias=True):
        super().__init__()
        self.lin = torch.nn.Linear(w_in, units, bias=bias, dtype=dde.config.real(torch))
        init(self.lin.weight)
        if bias:
            dde.nn.initializers.get("zeros")(self.lin.bias)
        self.act = act

    def forward(self, x):
        return self.act(self.lin(x))


class _DnFn(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class _DnFourier(torch.nn.Module):
    # [cos(x B), sin(x B)], B ~ N(0, sigma^2): fixed (a buffer, saved with the weights) or trained
    def __init__(self, w_in, m, sigma, trainable):
        super().__init__()
        _B = torch.randn(w_in, m, dtype=dde.config.real(torch)) * float(sigma)
        if trainable:
            self.B = torch.nn.Parameter(_B)
        else:
            self.register_buffer("B", _B)

    def forward(self, x):
        _p = x @ self.B
        return torch.cat([torch.cos(_p), torch.sin(_p)], dim=1)


class _DnPeriodic(torch.nn.Module):
    # cos(2 pi k x / L) and sin(2 pi k x / L) for k = 1..K: periodic with period L in every input
    def __init__(self, harmonics, period):
        super().__init__()
        self.k = list(range(1, int(harmonics) + 1))
        self.w = 2.0 * 3.141592653589793 / float(period)

    def forward(self, x):
        _parts = []
        for _k in self.k:
            _parts += [torch.cos(_k * self.w * x), torch.sin(_k * self.w * x)]
        return torch.cat(_parts, dim=1)


class _DnAdaptive(torch.nn.Module):
    # sigma(n a x) with a learned slope a (Jagtap, Kawaguchi & Karniadakis 2020)
    def __init__(self, act, n, a0):
        super().__init__()
        self.act, self.n = act, float(n)
        self.a = torch.nn.Parameter(torch.tensor(float(a0), dtype=dde.config.real(torch)))

    def forward(self, x):
        return self.act(self.n * self.a * x)


def _dn_layer(spec, w_in, activation, kernel_init):
    _t, _p = spec["type"], spec["params"]
    if _t == "dense":
        return _DnLinear(w_in, _p["units"], _dn_act(_p["activation"], activation),
                         _dn_init(_p["init"], kernel_init), _p["bias"])
    if _t == "activation":
        return _DnFn(_dn_act(_p["activation"], activation))
    if _t == "layer_norm":
        return torch.nn.LayerNorm(w_in, dtype=dde.config.real(torch))
    if _t == "scale_shift":
        _s, _b = float(_p["scale"]), float(_p["shift"])
        return _DnFn(lambda _x: _x * _s + _b)
    if _t == "fourier":
        return _DnFourier(w_in, _p["features"], _p["sigma"], _p["trainable"])
    if _t == "periodic":
        return _DnPeriodic(_p["harmonics"], _p["period"])
    if _t == "adaptive":
        return _DnAdaptive(_dn_act(_p["activation"], activation), _p["scale"], _p["init"])
    if _t == "custom":
        _cls = globals().get(_p["class_name"])
        if _cls is None:
            raise RuntimeError(f"Network design: the class {_p['class_name']} is not in the design's code")
        return _cls(**json.loads(_p["kwargs"] or "{}"))
    raise RuntimeError(f"Network design: unknown layer {_t}")


class _DnStack(torch.nn.Module):
    def __init__(self, block, activation, kernel_init):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.name = block.get("title") or block["id"]
        self.widths = []          # the width each layer declares (a custom layer's is checked once)
        self.checked = False
        _w = block["_in"]["in"]
        for _spec in block["layers"]:
            self.layers.append(_dn_layer(_spec, _w, activation, kernel_init))
            _w = _dn_width(_spec, _w)
            self.widths.append((_spec["type"], _spec["params"].get("class_name", ""), _w))
        if not block.get("trainable", True):
            for _q in self.parameters():
                _q.requires_grad_(False)

    def forward(self, x):
        for _l, (_t, _cls, _w) in zip(self.layers, self.widths):
            x = _l(x)
            if not self.checked and (x.dim() != 2 or x.shape[1] != _w):
                raise RuntimeError(f"Network design: in {self.name}, the layer {_cls or _t} gives "
                                   f"{tuple(x.shape[1:])} values per point, but declares {_w} (set its "
                                   f"Output width on the Network tab)")
        self.checked = True
        return x


def _dn_width(spec, w):
    _t, _p = spec["type"], spec["params"]
    if _t == "dense":
        return _p["units"]
    if _t == "fourier":
        return 2 * _p["features"]
    if _t == "periodic":
        return 2 * _p["harmonics"] * w
    if _t == "custom" and str(_p.get("width", "same")).strip().lower() not in ("", "same"):
        return int(_p["width"])
    return w


def _dn_saved(block):
    import hashlib as _hl
    with open(block["path"], "rb") as _fh:
        _sha = _hl.sha256(_fh.read()).hexdigest()
    if block["sha256"] and _sha != block["sha256"]:
        raise RuntimeError(f"Network design: {block['path']} has changed since it was trusted "
                           f"(SHA-256 {_sha[:16]}..., expected {block['sha256'][:16]}...)")
    # on the device DeepXDE trains on (it sets torch's default device: cuda, mps or cpu)
    _m = torch.jit.load(block["path"], map_location=torch.empty(0).device)
    for _q in _m.parameters():
        _q.requires_grad_(bool(block.get("fine_tune")))
    return _m


def _dn_merge(values, how):
    if len(values) == 1:
        return values[0]
    if how == "concatenate":
        return torch.cat(values, dim=1)
    _out = values[0]
    for _v in values[1:]:
        _out = _out + _v if how in ("add", "average") else _out * _v
    return _out / len(values) if how == "average" else _out


class _DnLevel(torch.nn.Module):
    # One level of the design (the top, or a group's inside): a module per block. What runs, in which
    # order, is written out below this class (_dn_level_top and one _dn_level_<group> per group).
    def __init__(self, level, activation, kernel_init):
        super().__init__()
        self.level = level
        self.blocks = {_b["id"]: _b for _b in level["blocks"]}
        self.mods = torch.nn.ModuleDict()
        for _b in level["blocks"]:
            _k = _b["kind"]
            if _k == "stack":
                self.mods[_b["id"]] = _DnStack(_b, activation, kernel_init)
            elif _k == "group":
                self.mods[_b["id"]] = _DnLevel(_b["inner"], activation, kernel_init)
                if not _b.get("trainable", True):
                    for _q in self.mods[_b["id"]].parameters():
                        _q.requires_grad_(False)
            elif _k == "saved":
                self.mods[_b["id"]] = _dn_saved(_b)
            elif _k == "head" and not _b["direct"]:
                self.mods[_b["id"]] = _DnLinear(_b["_in"]["in"], len(_b["_idx"]), lambda _x: _x,
                                                dde.nn.initializers.get(kernel_init))


class _DesignedNet(dde.nn.NN):
    # A network drawn as blocks on PINNStudio's Network tab (the design is _DESIGN above).
    def __init__(self, design, activation, kernel_initializer, weight_decay=0.0):
        super().__init__()
        self.design = design
        self.top = _DnLevel(design, activation, kernel_initializer)
        self.heads = [_b for _b in design["blocks"] if _b["kind"] == "head"]
        self.n_out = design["_n_out"]
        if weight_decay and weight_decay > 0:
            self.regularizer = ["l2", float(weight_decay)]

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        y = _dn_level_top(self.top.mods, x)
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
'''


def _var(bid: str) -> str:
    return f"v_{bid}"


def _merge_expr(srcs, how) -> str:
    vs = [_var(s) for s in srcs]
    if len(vs) == 1:
        return vs[0]
    if how == "concatenate":
        return f"torch.cat([{', '.join(vs)}], dim=1)"
    if how == "add":
        return " + ".join(vs)
    if how == "multiply":
        return " * ".join(vs)
    return f"({' + '.join(vs)}) / {len(vs)}"


def forward_source(resolved: dict) -> str:
    """The design's forward pass written out: one line per block, in order, and one function per group,
    so the generated scripts show what the network computes."""
    funcs = []

    def level(lv, name, is_top):
        blocks = {b["id"]: b for b in lv["blocks"]}
        lines = []
        for bid in lv["_order"]:
            b = blocks[bid]
            k = b["kind"]
            title = f"  # {b['title']}" if b.get("title") else ""
            if k == "input":
                lines.append(f"{_var(bid)} = x[:, {b['_idx']}]{title}")
                continue
            if k == "port_in":
                lines.append(f"{_var(bid)} = ports[{b['port']!r}]{title}")
                continue
            ins = {p: _merge_expr([e["src"] for e in lv["edges"] if e["dst"] == bid and e["port"] == p],
                                  b.get("merge", "concatenate")) for p in b["_in"]}
            arg = ins.get("in", "")
            if k in ("merge", "port_out"):
                expr = arg
            elif k in ("stack", "saved"):
                expr = f"m[{bid!r}]({arg})"
            elif k == "head":
                expr = arg if b["direct"] else f"m[{bid!r}]({arg})"
            elif k == "group":
                expr = f"_dn_level_{bid}(m[{bid!r}].mods, {{{', '.join(f'{p!r}: {v}' for p, v in ins.items())}}})"
            elif k == "shared":
                ref = blocks[b["ref"]]
                if ref["kind"] == "group":
                    expr = (f"_dn_level_{ref['id']}(m[{ref['id']!r}].mods, "
                            f"{{{', '.join(f'{p!r}: {v}' for p, v in ins.items())}}})")
                else:
                    expr = f"m[{ref['id']!r}]({arg})"
            else:
                continue
            lines.append(f"{_var(bid)} = {expr}{title}")
            if k == "group":
                level(b["inner"], bid, False)
        if is_top:
            cols = {}
            for b in lv["blocks"]:
                if b["kind"] == "head":
                    for j, o in enumerate(b["_idx"]):
                        cols[o] = f"{_var(b['id'])}[:, {j}:{j + 1}]"
            outs = [cols[o] for o in sorted(cols)]
            heads = [b for b in lv["blocks"] if b["kind"] == "head"]
            if len(heads) == 1 and heads[0]["_idx"] == list(range(len(outs))):
                outs = [_var(heads[0]["id"])]      # one head gives every output, in order
            lines.append(f"return torch.cat([{', '.join(outs)}], dim=1)" if len(outs) > 1 else f"return {outs[0]}")
            head = "def _dn_level_top(m, x):\n    # the network's inputs x -> its outputs, in the outputs' order"
        else:
            out = next(b for b in lv["blocks"] if b["kind"] == "port_out")
            lines.append(f"return {_var(out['id'])}")
            head = f"def _dn_level_{name}(m, ports):\n    # the inside of group {name}"
        funcs.append(head + "\n" + "\n".join("    " + ln for ln in lines) + "\n")

    level(resolved, "top", True)
    return "\n\n".join(reversed(funcs))


def net_source(design, input_names, output_names) -> str:
    """The source a script embeds for a designed network: imports, `_DESIGN`, the design's own code and
    `_DesignedNet`. ParamError when the design cannot run with these inputs and outputs."""
    resolved = resolve(design, input_names, output_names)
    code = resolved.get("code", "") or ""
    summary = "\n".join("#   " + ln for ln in summary_lines(resolved))
    return ("import json\n"
            "# ── Designed network (Network tab): the blocks in the order they run ──\n"
            f"{summary}\n"
            f"_DESIGN = json.loads({json.dumps(json.dumps(resolved))})\n"
            + (f"\n# ── The design's own code (custom layers) ──\n{code.rstrip()}\n" if code.strip() else "")
            + _RUNTIME + "\n\n" + forward_source(resolved))

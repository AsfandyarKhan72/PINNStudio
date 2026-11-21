"""Ready-made designs: today's architectures as blocks, and a few more to start from.

`from_settings` turns the Neural network page's form (FNN, PFNN, MsFFN, STMsFFN with its widths) into a
design that computes the same network, so **Edit as blocks** starts from what the problem has. The
other presets are architectures from the PINN literature that the form cannot express.
"""
from __future__ import annotations

from pinnstudio.core.netdesign import design as D
from pinnstudio.core.netdesign.layout import organize

PRESETS = {
    "fnn": "Fully connected (FNN)",
    "pfnn": "One sub-network per output (PFNN)",
    "msffn": "Multi-scale Fourier features (MsFFN)",
    "stmsffn": "Space-time multi-scale Fourier features (STMsFFN)",
    "resnet": "Residual network (skip connections)",
    "modified_mlp": "Modified MLP (Wang, Teng & Perdikaris 2021)",
}


class _B:
    """Builds a design level by level with fresh ids."""

    def __init__(self):
        self.d = D.empty()

    def add(self, kind, title="", **kw):
        b = {"id": D.new_id(self.d, kind), "kind": kind, "title": title, "x": 0.0, "y": 0.0}
        b.update(kw)
        self.d["blocks"].append(b)
        return b["id"]

    def link(self, src, dst, port="in"):
        self.d["edges"].append({"src": src, "dst": dst, "port": port})

    def done(self, name=""):
        if name:
            self.d["name"] = name
        out = D.check(self.d)
        organize(out)
        return out


def _dense(units, activation="inherit"):
    return {"type": "dense", "params": {"units": int(units), "activation": activation}}


def fnn(widths, name="FNN"):
    """Input → Dense × len(widths) → Head: DeepXDE's FNN."""
    b = _B()
    i = b.add("input", "Inputs", inputs=[D.ALL])
    s = b.add("stack", "Hidden layers", layers=[_dense(w) for w in widths])
    h = b.add("head", "Outputs", outputs=[D.ALL])
    b.link(i, s)
    b.link(s, h)
    return b.done(name)


def pfnn(widths, outputs, name="PFNN"):
    """One sub-network and head per output: DeepXDE's PFNN with a separate branch for each output."""
    b = _B()
    i = b.add("input", "Inputs", inputs=[D.ALL])
    for o in outputs:
        s = b.add("stack", f"Branch {o}", layers=[_dense(w) for w in widths])
        h = b.add("head", f"Output {o}", outputs=[o])
        b.link(i, s)
        b.link(s, h)
    return b.done(name)


def msffn(widths, sigmas, name="MsFFN"):
    """Random Fourier features of the inputs at each scale σ, the same hidden layers for every scale
    (one stack, shared), the branches side by side, and a linear output: PINNStudio's MsFFN."""
    widths = list(widths)
    m = max(1, widths[0] // 2)
    b = _B()
    i = b.add("input", "Inputs", inputs=[D.ALL])
    hidden = None
    branches = []
    for k, s in enumerate(sigmas):
        f = b.add("stack", f"Fourier σ={s:g}",
                  layers=[{"type": "fourier", "params": {"features": m, "sigma": float(s), "trainable": False}}])
        b.link(i, f)
        if hidden is None:
            hidden = b.add("stack", "Hidden layers", layers=[_dense(w) for w in widths[1:]] or [])
            node = hidden
        else:
            node = b.add("shared", f"Hidden layers (σ={s:g})", ref=hidden)
        b.link(f, node)
        branches.append(node)
    cat = b.add("merge", "Side by side", merge="concatenate")
    for n in branches:
        b.link(n, cat)
    h = b.add("head", "Outputs", outputs=[D.ALL])
    b.link(cat, h)
    return b.done(name)


def stmsffn(widths, sigmas, sigmas_t, space_names, name="STMsFFN"):
    """Space and time get Fourier features of their own; every (space, time) pair of branches, through the
    same hidden layers, is multiplied point-wise; the products side by side; a linear output."""
    widths = list(widths)
    m = max(1, widths[0] // 2)
    b = _B()
    ix = b.add("input", "Space", inputs=list(space_names))
    it = b.add("input", "Time", inputs=["t"])
    hidden = None
    xs, ts = [], []
    for src, sig, tag, out in ((ix, sigmas, "x", xs), (it, sigmas_t, "t", ts)):
        for s in sig:
            f = b.add("stack", f"Fourier {tag} σ={s:g}",
                      layers=[{"type": "fourier", "params": {"features": m, "sigma": float(s), "trainable": False}}])
            b.link(src, f)
            if hidden is None:
                hidden = b.add("stack", "Hidden layers", layers=[_dense(w) for w in widths[1:]])
                node = hidden
            else:
                node = b.add("shared", f"Hidden layers ({tag}, σ={s:g})", ref=hidden)
            b.link(f, node)
            out.append(node)
    cat = b.add("merge", "Side by side", merge="concatenate")
    for a in xs:
        for c in ts:
            p = b.add("merge", "×", merge="multiply")
            b.link(a, p)
            b.link(c, p)
            b.link(p, cat)
    h = b.add("head", "Outputs", outputs=[D.ALL])
    b.link(cat, h)
    return b.done(name)


def resnet(width, blocks=3, name="Residual network"):
    """A lifting layer, then residual blocks: two Dense layers whose output is added to their input."""
    b = _B()
    i = b.add("input", "Inputs", inputs=[D.ALL])
    prev = b.add("stack", "Lift", layers=[_dense(width)])
    b.link(i, prev)
    for k in range(blocks):
        s = b.add("stack", f"Block {k + 1}", layers=[_dense(width), _dense(width)])
        a = b.add("merge", f"Skip {k + 1}", merge="add")
        b.link(prev, s)
        b.link(s, a)
        b.link(prev, a)
        prev = a
    h = b.add("head", "Outputs", outputs=[D.ALL])
    b.link(prev, h)
    return b.done(name)


def modified_mlp(width, depth=3, name="Modified MLP"):
    """Wang, Teng & Perdikaris (2021): U = σ(XW_U), V = σ(XW_V), H₁ = σ(XW₁), and per layer
    Z = σ(H W), H ← (1 − Z) ⊙ U + Z ⊙ V; a linear output. Gates by Multiply and Add merges, 1 − Z by
    Scale and shift."""
    b = _B()
    i = b.add("input", "Inputs", inputs=[D.ALL])
    u = b.add("stack", "U", layers=[_dense(width)])
    v = b.add("stack", "V", layers=[_dense(width)])
    h = b.add("stack", "H1", layers=[_dense(width)])
    for n in (u, v, h):
        b.link(i, n)
    for k in range(depth):
        z = b.add("stack", f"Z{k + 1}", layers=[_dense(width)])
        omz = b.add("stack", f"1 − Z{k + 1}", layers=[{"type": "scale_shift", "params": {"scale": -1.0, "shift": 1.0}}])
        gu = b.add("merge", f"(1 − Z{k + 1}) ⊙ U", merge="multiply")
        gv = b.add("merge", f"Z{k + 1} ⊙ V", merge="multiply")
        hn = b.add("merge", f"H{k + 2}", merge="add")
        b.link(h, z)
        b.link(z, omz)
        b.link(omz, gu)
        b.link(u, gu)
        b.link(z, gv)
        b.link(v, gv)
        b.link(gu, hn)
        b.link(gv, hn)
        h = hn
    head = b.add("head", "Outputs", outputs=[D.ALL])
    b.link(h, head)
    return b.done(name)


def _sigmas(text, default):
    try:
        vals = [float(s) for s in str(text or "").replace(";", ",").split(",") if s.strip()]
    except ValueError:
        vals = []
    return vals or default


def from_settings(config, input_names, output_names) -> dict:
    """The design that computes what the Neural network page's settings make."""
    layers = list(getattr(config, "layers", None) or [len(input_names), 64, 64, 64, len(output_names)])
    widths = [int(w) for w in layers[1:-1]] or [64]
    kind = getattr(config, "network_type", "FNN") or "FNN"
    if kind == "PFNN":
        return pfnn(widths, output_names)
    if kind == "MsFFN":
        return msffn(widths, _sigmas(getattr(config, "msffn_sigmas", ""), [1.0, 10.0]))
    if kind == "STMsFFN" and "t" in input_names and len(input_names) > 1:
        return stmsffn(widths, _sigmas(getattr(config, "msffn_sigmas", ""), [1.0, 10.0]),
                       _sigmas(getattr(config, "msffn_sigmas_t", ""), [1.0, 10.0]),
                       [n for n in input_names if n != "t"])
    return fnn(widths)


def make(name: str, config, input_names, output_names) -> dict:
    """Preset `name` (a key of PRESETS) sized from the problem's current settings."""
    layers = list(getattr(config, "layers", None) or [0, 64, 64, 64, 0])
    widths = [int(w) for w in layers[1:-1]] or [64]
    w = widths[0]
    if name == "fnn":
        return fnn(widths)
    if name == "pfnn":
        return pfnn(widths, output_names)
    if name == "msffn":
        return msffn(widths, _sigmas(getattr(config, "msffn_sigmas", ""), [1.0, 10.0]))
    if name == "stmsffn":
        space = [n for n in input_names if n != "t"]
        if "t" not in input_names or not space:
            raise ValueError("STMsFFN needs a time-dependent problem with space (inputs x … and t)")
        return stmsffn(widths, _sigmas(getattr(config, "msffn_sigmas", ""), [1.0, 10.0]),
                       _sigmas(getattr(config, "msffn_sigmas_t", ""), [1.0, 10.0]), space)
    if name == "resnet":
        return resnet(w, max(1, len(widths) // 2))
    if name == "modified_mlp":
        return modified_mlp(w, max(1, len(widths)))
    raise ValueError(f"unknown preset {name!r}")

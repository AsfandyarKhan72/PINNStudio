"""A network design: blocks on a canvas and the connections between them.

Kinds of block:

- ``input`` reads network inputs by name (x, y, z, t, or the features of an input transform);
  ``["@all"]`` reads every one, in order;
- ``stack`` runs its layers in order (`layers.py`);
- ``merge`` combines what reaches it: concatenate, add, multiply or average;
- ``head`` gives outputs by name (``["@all"]``: every output): a linear layer to that many values, or,
  with ``direct``, the values that reach it as they are;
- ``shared`` calls a stack's, group's or saved module's network again, with the same weights;
- ``group`` holds a design of its own, with ``port_in`` blocks (its in-ports, by name) and one
  ``port_out`` block (its output), used as one block;
- ``saved`` is a TorchScript module from a file, frozen or fine-tuned.

Connections run from a block's out-port to another block's in-port; a group has one in-port per Port in
inside it, every other block one, named "in". Several connections into one in-port are combined by the
block's ``merge``. Block ids are unique in the whole design, nested groups included.

The stored form is a plain dict (PINNConfig.network_design holds it as JSON); `check()` validates it and
`Design` gives lookups over it. Pure: no I/O, no PyTorch, safe for views (VIEW_SAFE).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from pinnstudio.core.netdesign.layers import check_params
from pinnstudio.core.netdesign.params import ParamError

VERSION = 1
KINDS = ("input", "stack", "merge", "head", "shared", "group", "port_in", "port_out", "saved")
MERGES = ("concatenate", "add", "multiply", "average")
ALL = "@all"
PORT_NAME = re.compile(r"^[A-Za-z0-9_]+$")
CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# What each kind keeps besides id, kind, title, x and y, with defaults.
KIND_KEYS = {
    "input": {"inputs": [ALL]},
    "stack": {"layers": [], "merge": "concatenate", "trainable": True},
    "merge": {"merge": "concatenate"},
    "head": {"outputs": [ALL], "direct": False, "merge": "concatenate"},
    "shared": {"ref": "", "merge": "concatenate"},
    "group": {"inner": None, "merge": "concatenate", "trainable": True},
    "port_in": {"port": "in"},
    "port_out": {},
    "saved": {"path": "", "sha256": "", "width_in": 1, "width_out": 1, "fine_tune": False,
              "merge": "concatenate"},
}
COMMON = ("id", "kind", "title", "x", "y")
LAYOUT_KEYS = ("x", "y")          # where a block is drawn: left out of the hash
PREFIX = {"input": "in", "stack": "s", "merge": "m", "head": "h", "shared": "sh", "group": "g",
          "port_in": "pi", "port_out": "po", "saved": "sv"}
TOP_ONLY = ("input", "head")      # only at the top level
INNER_ONLY = ("port_in", "port_out")


def empty() -> dict:
    return {"version": VERSION, "blocks": [], "edges": [], "code": ""}


def has_in(kind: str) -> bool:
    return kind not in ("input", "port_in")


def has_out(kind: str) -> bool:
    return kind not in ("head", "port_out")


def ports_of(block: dict) -> list:
    """The in-ports of a block: a group's Port in names, "in" for the others, none for an input."""
    if not has_in(block.get("kind", "")):
        return []
    if block.get("kind") == "group":
        return [b.get("port", "in") for b in (block.get("inner") or {}).get("blocks", []) if b.get("kind") == "port_in"]
    return ["in"]


def label(block: dict) -> str:
    return block.get("title") or block.get("id", "")


# ── checking and completing ──────────────────────────────────────────────
def check(d) -> dict:
    """The design `d` (a dict, or its JSON) completed with defaults, checked; ParamError naming the problem.
    Structural errors only (unknown kinds, bad references, ports, loops are left to shapes.infer)."""
    if isinstance(d, str):
        try:
            d = json.loads(d) if d.strip() else empty()
        except ValueError as exc:
            raise ParamError(f"the network design is not valid JSON ({exc})")
    if not isinstance(d, dict):
        raise ParamError("the network design must be a JSON object")
    unknown = [k for k in d if k not in ("version", "blocks", "edges", "code", "name")]
    if unknown:
        raise ParamError(f"unknown key {unknown[0]!r} in the network design")
    if int(d.get("version", VERSION)) > VERSION:
        raise ParamError(f"this network design was made by a newer PINNStudio (version {d.get('version')})")
    code = d.get("code", "") or ""
    if not isinstance(code, str):
        raise ParamError("the design's code must be text")
    bad = code_problems(code)
    if bad:
        raise ParamError(bad[0])
    out = {"version": VERSION, "blocks": [], "edges": [], "code": code}
    if d.get("name"):
        out["name"] = str(d["name"])
    _level(d, out, path=(), seen=set())
    _check_refs(out)
    return out


def _level(src: dict, out: dict, path: tuple, seen: set) -> None:
    blocks = src.get("blocks", [])
    edges = src.get("edges", [])
    if not isinstance(blocks, list) or not isinstance(edges, list):
        raise ParamError("blocks and edges must be lists")
    where = f"in group {path[-1]}: " if path else ""
    for raw in blocks:
        b = _block(raw, path, where)
        if b["id"] in seen:
            raise ParamError(f"{where}block id {b['id']!r} is used twice")
        seen.add(b["id"])
        if b["kind"] == "group":
            inner = {"blocks": [], "edges": []}
            _level(raw.get("inner") or {"blocks": [], "edges": []}, inner, path + (b["id"],), seen)
            b["inner"] = inner
            names = [x["port"] for x in inner["blocks"] if x["kind"] == "port_in"]
            if len(set(names)) != len(names):
                raise ParamError(f"group {label(b)}: two Ports in have the same name")
            n_out = sum(1 for x in inner["blocks"] if x["kind"] == "port_out")
            if n_out != 1:
                raise ParamError(f"group {label(b)} needs exactly one Port out (it has {n_out})")
            if not names:
                raise ParamError(f"group {label(b)} needs a Port in")
        out["blocks"].append(b)
    ids = {b["id"]: b for b in out["blocks"]}
    keys = set()
    for raw in edges:
        if not isinstance(raw, dict):
            raise ParamError(f"{where}a connection must be an object")
        e = {"src": str(raw.get("src", "")), "dst": str(raw.get("dst", "")), "port": str(raw.get("port", "in"))}
        s, t = ids.get(e["src"]), ids.get(e["dst"])
        if s is None or t is None:
            raise ParamError(f"{where}a connection names a block that is not there: {e['src']} → {e['dst']}")
        if e["src"] == e["dst"]:
            raise ParamError(f"{where}{label(s)} is connected to itself")
        if not has_out(s["kind"]):
            raise ParamError(f"{where}a connection cannot start at {label(s)} ({s['kind']} has no output)")
        if e["port"] not in ports_of(t):
            raise ParamError(f"{where}{label(t)} has no in-port {e['port']!r}")
        key = (e["src"], e["dst"], e["port"])
        if key in keys:
            raise ParamError(f"{where}the connection {e['src']} → {e['dst']} is there twice")
        keys.add(key)
        out["edges"].append(e)


def _block(raw, path, where) -> dict:
    if not isinstance(raw, dict):
        raise ParamError(f"{where}a block must be an object")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ParamError(f"{where}unknown block kind {kind!r}")
    bid = str(raw.get("id", ""))
    if not PORT_NAME.match(bid):
        raise ParamError(f"{where}block id {bid!r}: letters, digits and _ only")
    if kind in TOP_ONLY and path:
        raise ParamError(f"{where}{kind} blocks belong at the top level, not inside a group")
    if kind in INNER_ONLY and not path:
        raise ParamError(f"{kind.replace('_', ' ').capitalize()} blocks belong inside a group")
    allowed = set(COMMON) | set(KIND_KEYS[kind])
    unknown = [k for k in raw if k not in allowed]
    if unknown:
        raise ParamError(f"{where}{bid}: unknown setting {unknown[0]!r} for a {kind} block")
    b = {"id": bid, "kind": kind, "title": str(raw.get("title", "") or ""),
         "x": float(raw.get("x", 0.0) or 0.0), "y": float(raw.get("y", 0.0) or 0.0)}
    for k, default in KIND_KEYS[kind].items():
        b[k] = copy.deepcopy(raw.get(k, default))
    name = label(b)
    if "merge" in b and b["merge"] not in MERGES:
        raise ParamError(f"{name}: merge must be one of {', '.join(MERGES)}")
    if kind == "input":
        b["inputs"] = _names(b["inputs"], f"{name}: inputs")
    elif kind == "head":
        b["outputs"] = _names(b["outputs"], f"{name}: outputs")
        b["direct"] = bool(b["direct"])
    elif kind == "stack":
        if not isinstance(b["layers"], list):
            raise ParamError(f"{name}: layers must be a list")
        layers = []
        for i, spec in enumerate(b["layers"]):
            if not isinstance(spec, dict) or "type" not in spec:
                raise ParamError(f"{name}: layer {i + 1} needs a type")
            try:
                params = check_params(spec["type"], spec.get("params") or {})
            except ParamError as exc:
                raise ParamError(f"{name}, layer {i + 1}: {exc}")
            layers.append({"type": spec["type"], "params": params})
        b["layers"] = layers
        b["trainable"] = bool(b["trainable"])
    elif kind == "port_in":
        if not PORT_NAME.match(str(b["port"])):
            raise ParamError(f"{name}: a port name is letters, digits and _ only")
    elif kind == "shared":
        b["ref"] = str(b["ref"])
    elif kind == "saved":
        for k in ("width_in", "width_out"):
            if not isinstance(b[k], int) or isinstance(b[k], bool) or b[k] < 1:
                raise ParamError(f"{name}: {k.replace('_', ' ')} must be a whole number of at least 1")
        b["path"], b["sha256"], b["fine_tune"] = str(b["path"]), str(b["sha256"]), bool(b["fine_tune"])
    elif kind == "group":
        b["trainable"] = bool(b["trainable"])
    return b


def _names(v, what):
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",") if x.strip()]
    if not isinstance(v, list) or not v or not all(isinstance(x, str) and x for x in v):
        raise ParamError(f"{what}: at least one name is needed")
    if len(set(v)) != len(v):
        raise ParamError(f"{what}: a name is listed twice")
    if ALL in v and len(v) > 1:
        raise ParamError(f"{what}: {ALL} stands for every name on its own")
    return list(v)


def _check_refs(d: dict) -> None:
    """A shared block calls a stack, group or saved module at its own level."""
    for path, level in walk_levels(d):
        ids = {b["id"]: b for b in level["blocks"]}
        for b in level["blocks"]:
            if b["kind"] != "shared":
                continue
            ref = ids.get(b["ref"])
            if ref is None:
                raise ParamError(f"{label(b)} shares {b['ref'] or '(nothing)'}, which is not at its level")
            if ref["kind"] not in ("stack", "group", "saved"):
                raise ParamError(f"{label(b)} can share a stack, group or saved module, not a {ref['kind']}")


# ── lookups ─────────────────────────────────────────────────────────────
def walk_levels(d: dict, path: tuple = ()):
    """(path, level) for the top level and every group's inside, depth first."""
    yield path, d
    for b in d.get("blocks", []):
        if b.get("kind") == "group" and b.get("inner"):
            yield from walk_levels(b["inner"], path + (b["id"],))


def all_ids(d: dict) -> set:
    return {b["id"] for _p, level in walk_levels(d) for b in level["blocks"]}


def level_of(d: dict, path) -> dict:
    """The level (top, or a group's inside) at `path`, a tuple of group ids: the dict itself, to edit."""
    level = d
    for gid in path:
        g = next((b for b in level["blocks"] if b["id"] == gid and b["kind"] == "group"), None)
        if g is None:
            raise ParamError(f"no group {gid!r} here")
        level = g["inner"]
    return level


def find(d: dict, bid: str):
    """(path, block) of block `bid` anywhere in the design, or (None, None)."""
    for path, level in walk_levels(d):
        for b in level["blocks"]:
            if b["id"] == bid:
                return path, b
    return None, None


def incoming(level: dict, bid: str, port: str | None = None) -> list:
    return [e for e in level["edges"] if e["dst"] == bid and (port is None or e["port"] == port)]


def outgoing(level: dict, bid: str) -> list:
    return [e for e in level["edges"] if e["src"] == bid]


def new_id(d: dict, kind: str) -> str:
    used = all_ids(d)
    k = 1
    while f"{PREFIX[kind]}{k}" in used:
        k += 1
    return f"{PREFIX[kind]}{k}"


def topological(level: dict):
    """(order, blocks on a cycle) of a level's blocks (Kahn's algorithm, ties in list order)."""
    ids = [b["id"] for b in level["blocks"]]
    indeg = {i: 0 for i in ids}
    for e in level["edges"]:
        if e["dst"] in indeg and e["src"] in indeg:
            indeg[e["dst"]] += 1
    order, ready = [], [i for i in ids if indeg[i] == 0]
    while ready:
        n = ready.pop(0)
        order.append(n)
        for e in level["edges"]:
            if e["src"] == n and e["dst"] in indeg:
                indeg[e["dst"]] -= 1
                if indeg[e["dst"]] == 0:
                    ready.append(e["dst"])
    left = [i for i in ids if i not in order]
    return order, left


def reaches(level: dict, src: str, dst: str) -> bool:
    """Is there a path src → … → dst at this level?"""
    todo, seen = [src], set()
    while todo:
        n = todo.pop()
        if n == dst:
            return True
        if n in seen:
            continue
        seen.add(n)
        todo += [e["dst"] for e in level["edges"] if e["src"] == n]
    return False


# ── identity ────────────────────────────────────────────────────────────
def canonical(d: dict) -> dict:
    """The design without where its blocks are drawn: moving a block changes no result."""
    out = copy.deepcopy(d)
    for _p, level in walk_levels(out):
        for b in level["blocks"]:
            for k in LAYOUT_KEYS:
                b.pop(k, None)
    return out


def design_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(canonical(d), sort_keys=True).encode()).hexdigest()[:16]


def to_json(d: dict) -> str:
    return json.dumps(d, sort_keys=False, separators=(",", ":"))


def class_names(code: str) -> list:
    """The top-level classes defined in the design's code (for the custom layers' Class choices)."""
    return re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(:]", code or "", flags=re.M)


# Names a generated script defines or imports at the top level: the design's code is pasted among them,
# so it may not define any of these (nor anything starting with an underscore, the scripts' own helpers).
SCRIPT_NAMES = frozenset({
    "os", "sys", "json", "math", "time", "np", "torch", "dde", "plt", "matplotlib", "scipy", "pd",
    "model", "net", "data", "pde", "geom", "geomtime", "timedomain", "config", "constraints", "bcs", "ic",
    "losshistory", "loss_history", "train_state", "save_dir", "sol_dir", "X", "y", "inputs", "outputs",
    "build_net", "load_trained", "ACTIVATION", "KERNEL_INITIALIZER", "WEIGHT_DECAY",
})


def code_problems(code: str) -> list:
    """What is wrong with the design's code before it runs: a syntax error, or a top-level name the
    scripts already use."""
    import ast
    if not (code or "").strip():
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"the design's code does not parse: line {exc.lineno}: {exc.msg}"]
    out = []
    for node in tree.body:
        names = []
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [(a.asname or a.name).split(".")[0] for a in node.names]
            names = [n for n in names if n not in ("torch", "np", "numpy", "math", "json")]
        for n in names:
            if n in SCRIPT_NAMES or n.startswith("_"):
                out.append(f"the design's code defines {n!r}, a name the generated scripts use: rename it")
    return out


def uses_code(d: dict) -> bool:
    """Does the design run code of its own (a custom layer) or load a file (a saved module)?"""
    for _p, level in walk_levels(d):
        for b in level["blocks"]:
            if b["kind"] == "saved":
                return True
            if b["kind"] == "stack" and any(sp["type"] == "custom" for sp in b["layers"]):
                return True
    return False

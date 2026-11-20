"""Saved blocks and designs: the personal block library and design files.

A block file (``*.pinn-block.json``) holds one block (a group with its inside), with the code its
custom layers use; a design file (``*.pinn-design.json``) holds a whole design. Block files in the
library folder, ``~/.pinnstudio/blocks`` (``$PINNSTUDIO_BLOCKS`` overrides it), show under *My blocks*
in the palette. Inserting a block gives every block in it a fresh id.
"""
from __future__ import annotations

import copy
import json
import os

from pinnstudio.core.netdesign import design as D
from pinnstudio.core.netdesign.params import ParamError

BLOCK_EXT = ".pinn-block.json"
DESIGN_EXT = ".pinn-design.json"


def blocks_dir() -> str:
    from pinnstudio.core.netdesign.trust import home
    return os.environ.get("PINNSTUDIO_BLOCKS") or os.path.join(home(), "blocks")


def list_blocks() -> list:
    """[(name, path)] of the block files in the library folder, by name."""
    folder = blocks_dir()
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith(BLOCK_EXT))
    except OSError:
        return []
    return [(n[:-len(BLOCK_EXT)], os.path.join(folder, n)) for n in names]


def _write(path, data) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def save_block(d: dict, bid: str, path: str) -> None:
    _path, b = D.find(d, bid)
    if b is None:
        raise ParamError(f"no block {bid!r}")
    if b["kind"] in ("input", "head", "port_in", "port_out", "shared"):
        raise ParamError(f"a {b['kind'].replace('_', ' ')} block is part of its design; save a stack, a "
                         "merge, a group or a saved module")
    block = copy.deepcopy(b)
    if block["kind"] == "saved" and block.get("path") and not os.path.isabs(block["path"]):
        block["path"] = os.path.abspath(block["path"])
    _write(path, {"pinnstudio_block": 1, "block": block, "code": d.get("code", "")})


def save_design(d: dict, path: str) -> None:
    _write(path, {"pinnstudio_design": 1, "design": D.check(d)})


def read(path: str):
    """("block", block, code) or ("design", design, "") from a library or design file."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ParamError(f"{os.path.basename(path)} cannot be read ({exc})")
    if isinstance(data, dict) and "pinnstudio_block" in data and isinstance(data.get("block"), dict):
        return "block", data["block"], str(data.get("code", "") or "")
    if isinstance(data, dict) and "pinnstudio_design" in data:
        return "design", D.check(data["design"]), ""
    raise ParamError(f"{os.path.basename(path)} is not a PINNStudio block or design file")


def insert_block(d: dict, path, block: dict, code: str, x: float, y: float) -> dict:
    """`d` with `block` added at level `path` (fresh ids throughout, its code appended with any class name
    that clashes renamed Name_2, ...). Returns the new design (checked)."""
    out = copy.deepcopy(d)
    b = copy.deepcopy(block)
    mapping = {}

    # ids reserved as they are handed out, so a group's inner blocks do not collide
    used = D.all_ids(out)

    def new_id(kind):
        k = 1
        while f"{D.PREFIX[kind]}{k}" in used:
            k += 1
        used.add(f"{D.PREFIX[kind]}{k}")
        return f"{D.PREFIX[kind]}{k}"

    def renew(blk):
        old = blk["id"]
        blk["id"] = new_id(blk["kind"])
        mapping[old] = blk["id"]
        if blk["kind"] == "group" and blk.get("inner"):
            for inner in blk["inner"]["blocks"]:
                renew(inner)
            for e in blk["inner"]["edges"]:
                e["src"], e["dst"] = mapping[e["src"]], mapping[e["dst"]]
            for inner in blk["inner"]["blocks"]:
                if inner["kind"] == "shared":
                    inner["ref"] = mapping.get(inner["ref"], inner["ref"])

    renew(b)
    b["x"], b["y"] = float(x), float(y)
    renames = {}
    if code.strip():
        have = set(D.class_names(out.get("code", "")))
        for name in D.class_names(code):
            if name in have:
                k = 2
                while f"{name}_{k}" in have:
                    k += 1
                renames[name] = f"{name}_{k}"
        import re
        for old, new in renames.items():
            code = re.sub(rf"\b{re.escape(old)}\b", new, code)
        for _p, lv in D.walk_levels({"blocks": [b], "edges": []}):
            for blk in lv["blocks"]:
                if blk["kind"] == "stack":
                    for sp in blk["layers"]:
                        if sp["type"] == "custom" and sp["params"].get("class_name") in renames:
                            sp["params"]["class_name"] = renames[sp["params"]["class_name"]]
        out["code"] = ((out.get("code", "") or "").rstrip() + "\n\n\n" + code.strip() + "\n").lstrip()
    D.level_of(out, tuple(path))["blocks"].append(b)
    return D.check(out)


def relocate_paths(design_json: str, base_dir: str, relative: bool) -> str:
    """The design with its saved modules' file paths relative to `base_dir` (a problem file's folder, when
    saving: the folder can move to another computer as a whole) or absolute again (when opening)."""
    if not design_json or '"saved"' not in design_json:
        return design_json
    try:
        d = json.loads(design_json)
    except ValueError:
        return design_json
    for _p, level in D.walk_levels(d):
        for b in level.get("blocks", []):
            if b.get("kind") != "saved" or not b.get("path"):
                continue
            p = b["path"]
            if relative and os.path.isabs(p):
                try:
                    b["path"] = os.path.relpath(p, base_dir).replace(os.sep, "/")
                except ValueError:       # another drive on Windows: keep it absolute
                    pass
            elif not relative and not os.path.isabs(p):
                b["path"] = os.path.normpath(os.path.join(base_dir, p))
    return json.dumps(d, separators=(",", ":"))

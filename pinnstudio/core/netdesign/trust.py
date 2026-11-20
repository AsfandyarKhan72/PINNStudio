"""Which code and files the user has trusted, by SHA-256.

A design can run code of its own (custom layers) and load a TorchScript file (a saved module); both run
with the user's rights when the network is built. Nothing of that runs until the user has trusted exactly
this code and these files: a changed character or byte asks again. Decisions are kept in
``~/.pinnstudio/trusted.json`` (``$PINNSTUDIO_HOME`` overrides the folder). Trust is a decision, not a
sandbox.
"""
from __future__ import annotations

import hashlib
import json
import os

from pinnstudio.core.netdesign import design as D


def home() -> str:
    return os.environ.get("PINNSTUDIO_HOME") or os.path.join(os.path.expanduser("~"), ".pinnstudio")


def store_path() -> str:
    return os.path.join(home(), "trusted.json")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load() -> dict:
    try:
        with open(store_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def needed(d: dict) -> list:
    """What design `d` runs or loads, as items {kind: "code"|"file", name, sha256, size, text, path,
    problem}; `problem` says why a file cannot be trusted (missing, or changed since it was added)."""
    items = []
    code = d.get("code", "") or ""
    if code.strip():           # the scripts embed the code whenever there is any, used by a layer or not
        items.append({"kind": "code", "name": "the design's code", "sha256": sha256_text(code),
                      "size": len(code.encode("utf-8")), "text": code, "path": "", "problem": ""})
    for _p, lv in D.walk_levels(d):
        for b in lv["blocks"]:
            if b["kind"] != "saved":
                continue
            item = {"kind": "file", "name": D.label(b), "sha256": b.get("sha256", ""), "size": 0, "text": "",
                    "path": b.get("path", ""), "problem": ""}
            try:
                real = sha256_file(b["path"])
                item["size"] = os.path.getsize(b["path"])
                if b.get("sha256") and real != b["sha256"]:
                    item["problem"] = "the file has changed since it was added to the design"
                item["sha256"] = real
            except OSError:
                item["problem"] = f"the file {b.get('path') or '(none)'} cannot be read"
            items.append(item)
    return items


def untrusted(d: dict) -> list:
    """The items of `needed(d)` not trusted yet (or that cannot be trusted)."""
    known = _load()
    return [it for it in needed(d) if it["problem"] or not known.get(it["sha256"])]


def remember(items) -> None:
    """Trust these items (by their SHA-256) from now on."""
    data = _load()
    for it in items:
        if it.get("sha256") and not it.get("problem"):
            data[it["sha256"]] = {"kind": it["kind"], "name": it["name"], "path": it.get("path", "")}
    os.makedirs(home(), exist_ok=True)
    tmp = store_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, store_path())

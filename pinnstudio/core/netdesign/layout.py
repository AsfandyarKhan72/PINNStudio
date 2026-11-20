"""Automatic layout of a design: inputs on the left, heads on the right.

Layers by longest path from the inputs (a block one column right of everything that feeds it), then a
few barycentre sweeps to order each column so connections cross less (Sugiyama, Tagawa & Toda 1981, in
its simplest form). Writes each block's x and y in place, at every level of the design.
"""
from __future__ import annotations

from pinnstudio.core.netdesign import design as D

COL = 240.0     # column spacing
ROW = 120.0     # row spacing


def organize(d: dict) -> dict:
    for _path, level in D.walk_levels(d):
        organize_level(level)
    return d


def organize_level(level: dict) -> None:
    """Lay out one level (the top, or a group's inside) in place."""
    blocks = level["blocks"]
    if not blocks:
        return
    order, cycle = D.topological(level)
    col = {}
    for bid in order + cycle:
        srcs = [e["src"] for e in level["edges"] if e["dst"] == bid and e["src"] in col]
        col[bid] = (max(col[s] for s in srcs) + 1) if srcs else 0
    last = max(col.values())
    for b in blocks:                         # heads and Ports out in the last column
        if b["kind"] in ("head", "port_out"):
            col[b["id"]] = last
    columns = {}
    for b in blocks:
        columns.setdefault(col[b["id"]], []).append(b["id"])
    pos = {}
    for c in sorted(columns):
        for r, bid in enumerate(columns[c]):
            pos[bid] = r
    for _sweep in range(4):
        for c in sorted(columns)[1:]:
            def bary(bid):
                srcs = [pos[e["src"]] for e in level["edges"] if e["dst"] == bid and e["src"] in pos]
                return sum(srcs) / len(srcs) if srcs else pos[bid]
            columns[c].sort(key=bary)
            for r, bid in enumerate(columns[c]):
                pos[bid] = r
    for b in blocks:
        c = col[b["id"]]
        n = len(columns[c])
        b["x"] = c * COL
        b["y"] = (pos[b["id"]] - (n - 1) / 2.0) * ROW
